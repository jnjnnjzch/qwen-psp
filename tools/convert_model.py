import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType, dequantize

ROOT = Path(__file__).resolve().parents[1]
QK = 32


def q4_0_bytes(values: np.ndarray) -> bytes:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] % QK:
        raise ValueError(f"Q4 shape must be [rows, cols divisible by {QK}], got {matrix.shape}")
    rows, cols = matrix.shape
    blocks = np.ascontiguousarray(matrix).reshape(rows, cols // QK, QK)
    maxima = np.argmax(np.abs(blocks), axis=2)
    signed_max = np.take_along_axis(blocks, maxima[..., None], axis=2)[..., 0]
    scales = signed_max / -8.0
    inverse = np.zeros_like(scales)
    np.divide(1.0, scales, out=inverse, where=scales != 0.0)
    quants = np.trunc(blocks * inverse[..., None] + 8.5).astype(np.int16)
    quants = np.clip(quants, 0, 15).astype(np.uint8)
    packed = quants[..., :16] | (quants[..., 16:] << 4)
    output = np.empty((rows, cols // QK, 18), dtype=np.uint8)
    output[..., :2] = scales.astype("<f2").view(np.uint8).reshape(rows, cols // QK, 2)
    output[..., 2:] = packed
    return output.tobytes()


def bytes_to_unicode() -> dict[int, str]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, unicode_values)))


def align_file(file, alignment: int = 64) -> None:
    padding = (-file.tell()) % alignment
    if padding:
        file.write(b"\0" * padding)


def tokenizer_section(reader):
    tokens = reader.fields["tokenizer.ggml.tokens"].contents()
    types = reader.fields["tokenizer.ggml.token_type"].contents()
    vocab = {text: index for index, text in enumerate(tokens)}
    decoder = {char: value for value, char in bytes_to_unicode().items()}
    pieces = [bytes(decoder[char] for char in text) for text in tokens]
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    blob = b"".join(pieces)
    ranks = [0xFFFFFFFF] * len(tokens)
    for rank, merge in enumerate(reader.fields["tokenizer.ggml.merges"].contents()):
        left, right = merge.split(" ", 1)
        ranks[vocab[left + right]] = rank
    sorted_ids = sorted(range(len(tokens)), key=pieces.__getitem__)
    specials = [index for index, kind in enumerate(types) if kind in (3, 4)]
    offsets_off = 52
    pieces_off = offsets_off + 4 * (len(tokens) + 1)
    sorted_off = (pieces_off + len(blob) + 3) & ~3
    ranks_off = sorted_off + 4 * len(tokens)
    byte_off = ranks_off + 4 * len(tokens)
    special_off = byte_off + 256 * 4
    section = bytearray(special_off + len(specials) * 4)
    struct.pack_into("<4s12I", section, 0, b"TOK3", 1, len(tokens), len(blob),
                     offsets_off, pieces_off, sorted_off, ranks_off, byte_off,
                     len(section), max(map(len, pieces)), len(specials), special_off)
    struct.pack_into(f"<{len(offsets)}I", section, offsets_off, *offsets)
    section[pieces_off:pieces_off + len(blob)] = blob
    struct.pack_into(f"<{len(tokens)}I", section, sorted_off, *sorted_ids)
    struct.pack_into(f"<{len(tokens)}I", section, ranks_off, *ranks)
    encoder = bytes_to_unicode()
    struct.pack_into("<256I", section, byte_off, *(vocab[encoder[i]] for i in range(256)))
    struct.pack_into(f"<{len(specials)}I", section, special_off, *specials)
    return bytes(section), vocab


def convert(source, destination, context):
    reader = GGUFReader(source)
    field = lambda name: reader.fields[name].contents()
    expected = {
        "general.architecture": "qwen3", "qwen3.block_count": 28,
        "qwen3.embedding_length": 1024, "qwen3.feed_forward_length": 3072,
        "qwen3.attention.head_count": 16, "qwen3.attention.head_count_kv": 8,
        "qwen3.attention.key_length": 128, "qwen3.attention.value_length": 128,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2",
    }
    for name, value in expected.items():
        if field(name) != value:
            raise ValueError(f"{name}: {field(name)!r}, expected {value!r}")
    if not 32 <= context <= 128:
        raise ValueError("PSP-1000 context must be 32..128")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    tokenizer, vocab = tokenizer_section(reader)
    if len(vocab) != 151936:
        raise ValueError("expected 151936 vocabulary entries")
    manifest = {"source": str(source), "context": context, "tensors": {}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w+b") as output:
        output.write(bytes(256))
        output.write(tokenizer)
        align_file(output)
        weights_offset = output.tell()

        def write(name, shape, quantized):
            tensor = tensors[name]
            if tuple(tensor.shape[::-1]) != shape:
                raise ValueError(f"{name}: unexpected shape {tensor.shape}")
            entry = {"offset": output.tell(), "shape": list(shape),
                     "type": "Q4_0" if quantized else "F32",
                     "source_type": tensor.tensor_type.name}
            if not quantized:
                if tensor.tensor_type != GGMLQuantizationType.F32:
                    raise ValueError(f"{name}: normalization must be F32")
                output.write(tensor.data.astype("<f4", copy=False).tobytes())
            elif tensor.tensor_type == GGMLQuantizationType.Q4_0:
                output.write(tensor.data.tobytes())
            else:
                # 分块转换混合量化的矩阵，复用原项目的 Q4_0 量化器。
                for begin in range(0, shape[0], 1024):
                    values = dequantize(tensor.data[begin:begin + 1024], tensor.tensor_type)
                    output.write(q4_0_bytes(values.reshape(-1, shape[1])))
            entry["bytes"] = output.tell() - entry["offset"]
            manifest["tensors"][name] = entry
            print(f"{name}: {tensor.tensor_type.name} -> {entry['type']}", flush=True)

        write("token_embd.weight", (151936, 1024), True)
        for layer in range(28):
            prefix = f"blk.{layer}."
            for name, shape, quantized in (
                ("attn_norm.weight", (1024,), False),
                ("attn_q.weight", (2048, 1024), True),
                ("attn_k.weight", (1024, 1024), True),
                ("attn_v.weight", (1024, 1024), True),
                ("attn_output.weight", (1024, 2048), True),
                ("attn_q_norm.weight", (128,), False),
                ("attn_k_norm.weight", (128,), False),
                ("ffn_norm.weight", (1024,), False),
                ("ffn_gate.weight", (3072, 1024), True),
                ("ffn_up.weight", (3072, 1024), True),
                ("ffn_down.weight", (1024, 3072), True),
            ):
                write(prefix + name, shape, quantized)
        write("output_norm.weight", (1024,), False)
        if set(tensors) != set(manifest["tensors"]):
            raise ValueError("unconverted tensors in GGUF")
        size = output.tell()
        header = bytearray(256)
        struct.pack_into("<4sIIIQQQ", header, 0, b"FHQ4", 2, 256, 0,
                         size, 256, weights_offset)
        integers = (
            1024, 3072, 28, 16, 8, 128, 151936,
            field("qwen3.context_length"), context,
            field("tokenizer.ggml.padding_token_id"),
            field("tokenizer.ggml.eos_token_id"), vocab["<|endoftext|>"],
            vocab["<|im_start|>"], vocab["<|im_end|>"], vocab["<|endoftext|>"], 0xFFFFFFFF,
            0, 0, 0, 0, 0, 0, 0, 32, 0,
        )
        struct.pack_into("<25I", header, 40, *integers)
        struct.pack_into("<4f", header, 140, field("qwen3.rope.freq_base"),
                         field("qwen3.attention.layer_norm_rms_epsilon"), 1.0, 1.0)
        output.seek(0)
        output.write(header)
    with destination.open("rb") as model_file:
        checksum = hashlib.file_digest(model_file, "sha256").hexdigest()
    manifest.update({"bytes": size, "tokenizer_bytes": len(tokenizer),
                     "weights_offset": weights_offset, "sha256": checksum})
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {destination}: {size} bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / ".work/Qwen3-0.6B-Q4_0.gguf")
    parser.add_argument("--output", type=Path, default=ROOT / "model.fhq4")
    parser.add_argument("--context", type=int, default=128)
    args = parser.parse_args()
    convert(args.source, args.output, args.context)
