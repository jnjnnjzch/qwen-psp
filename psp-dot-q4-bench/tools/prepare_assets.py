import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib

from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "LLMPSP"
QWEN = UPSTREAM / "qwen/q4_0"
sys.path.insert(0, str(UPSTREAM / "tools"))
from convert_falcon_h1 import q4_0_bytes
PROMPT = (
    "<|im_start|>user\nExplain how a small language model performs matrix multiplication "
    "on a portable game console, using clear examples and practical measurements."
    "<|im_end|>\n<|im_start|>assistant\n"
)


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def rms_norm(values, weight):
    mean = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
    return values / np.sqrt(mean + np.float32(1e-6)) * weight


def first_layer_inputs(read_tensor, tokens):
    embedding = read_tensor("token_embd.weight", tokens)
    attention_input = rms_norm(embedding, read_tensor("blk.0.attn_norm.weight"))
    q = (attention_input @ read_tensor("blk.0.attn_q.weight").T).reshape(16, 16, 128)
    k = (attention_input @ read_tensor("blk.0.attn_k.weight").T).reshape(16, 8, 128)
    v = (attention_input @ read_tensor("blk.0.attn_v.weight").T).reshape(16, 8, 128)
    q = rms_norm(q, read_tensor("blk.0.attn_q_norm.weight"))
    k = rms_norm(k, read_tensor("blk.0.attn_k_norm.weight"))
    angles = np.arange(16, dtype=np.float32)[:, None] * np.power(
        np.float32(1e6), -np.arange(64, dtype=np.float32) / np.float32(64))[None, :]
    cosine, sine = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]

    def rope(values):
        left, right = values[..., :64], values[..., 64:]
        return np.concatenate((left * cosine - right * sine,
                               right * cosine + left * sine), axis=-1)

    q, k = rope(q), np.repeat(rope(k), 2, axis=1)
    v = np.repeat(v, 2, axis=1)
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(np.float32(128))
    scores[:, np.triu_indices(16, 1)[0], np.triu_indices(16, 1)[1]] = -np.inf
    probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    context = np.einsum("hts,shd->thd", probabilities, v).reshape(16, 2048)
    hidden = embedding + context @ read_tensor("blk.0.attn_output.weight").T
    ffn_input = rms_norm(hidden, read_tensor("blk.0.ffn_norm.weight"))
    gate = ffn_input @ read_tensor("blk.0.ffn_gate.weight").T
    up = ffn_input @ read_tensor("blk.0.ffn_up.weight").T
    # SiLU 分开处理正负值，指数参数始终小于等于零。
    exponential = np.exp(-np.abs(gate))
    sigmoid = np.where(gate >= 0, 1 / (1 + exponential), exponential / (1 + exponential))
    down_input = gate * sigmoid * up
    if not np.isfinite(attention_input).all() or not np.isfinite(down_input).all():
        raise ValueError("第一层激活包含非有限值")
    return attention_input.astype("<f4"), down_input.astype("<f4")


def make_icon(destination):
    image = Image.new("RGB", (144, 80))
    draw = ImageDraw.Draw(image)
    for y in range(80):
        draw.line((0, y, 144, y), fill=(10 + y // 8, 22 + y // 5, 45 + y // 3))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    draw.text((12, 9), "DOT / Q4", font=font, fill=(242, 247, 255))
    draw.text((13, 28), "PSP KERNEL BENCH", font=small, fill=(146, 173, 200))
    for y, width, color in ((48, 112, (148, 170, 197)), (57, 76, (88, 212, 231)),
                             (66, 35, (224, 185, 114))):
        draw.rounded_rectangle((13, y, 13 + width, y + 4), radius=2, fill=color)
    image.save(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "assets/generated")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((UPSTREAM / "qwen/download-manifest-q4_0.json").read_text())
    gguf_path = UPSTREAM / "qwen" / source_manifest["model_file"]
    model_path = QWEN / "model.fhq4"
    metadata = json.loads((QWEN / "model.fhq4.json").read_text())
    if sha256(gguf_path) != source_manifest["sha256"] or sha256(model_path) != metadata["sha256"]:
        raise ValueError("模型 SHA-256 与来源记录不同")
    reader = GGUFReader(gguf_path)
    gguf_tensors = {tensor.name: tensor for tensor in reader.tensors}
    if reader.fields["general.architecture"].contents() != "qwen3":
        raise ValueError("需要 Qwen3 模型")
    tokenizer = Tokenizer.from_file(str(QWEN / ".work/tokenizer.json"))
    tokens = tokenizer.encode(PROMPT, add_special_tokens=False).ids[:16]
    if len(tokens) != 16:
        raise ValueError("真实文本需要至少 16 个 token")

    def read_bytes(name, rows=None):
        entry = metadata["tensors"][name]
        with model_path.open("rb") as file:
            if rows is None:
                file.seek(entry["offset"])
                content = file.read(entry["bytes"])
            else:
                stride = entry["shape"][1] // 32 * 18
                chunks = []
                for row in rows:
                    file.seek(entry["offset"] + row * stride)
                    chunks.append(file.read(stride))
                content = b"".join(chunks)
        return content

    def read_tensor(name, rows=None):
        entry = metadata["tensors"][name]
        content = read_bytes(name, rows)
        if entry["type"] == "F32":
            return np.frombuffer(content, dtype="<f4").reshape(entry["shape"])
        columns = entry["shape"][1]
        packed = np.frombuffer(content, dtype=np.uint8).reshape(-1, columns // 32 * 18)
        return dequantize(packed, GGMLQuantizationType.Q4_0).astype(np.float32)

    names = ("blk.0.attn_q.weight", "blk.0.ffn_down.weight")
    packed_weights, weights = [], []
    scale_stats = {}
    for name in names:
        content = read_bytes(name)
        source = gguf_tensors[name]
        if source.tensor_type == GGMLQuantizationType.Q4_0:
            expected_bytes = source.data.tobytes()
        else:
            expected_bytes = q4_0_bytes(dequantize(source.data, source.tensor_type))
        if expected_bytes != content:
            raise ValueError(f"{name} 的字节与原始模型及既有量化器不同")
        packed_weights.append(content)
        weights.append(read_tensor(name))
        blocks = np.frombuffer(content, dtype=np.uint8).reshape(-1, 18)
        scales = np.ascontiguousarray(blocks[:, :2]).view("<f2").reshape(-1)
        subnormals = (scales != 0) & (np.abs(scales) < np.finfo(np.float16).tiny)
        scale_stats[name] = {"blocks": len(scales), "subnormal_scales": int(subnormals.sum())}
        if subnormals.any() or not np.isfinite(scales).all():
            raise ValueError("选中权重的 scale 超出原始 VFPU 算子的精确转换范围")
    inputs = first_layer_inputs(read_tensor, tokens)
    cases = json.loads((ROOT / "cases.json").read_text())
    case_record = struct.Struct("<32s9I")
    if not 1 <= len(cases) <= 32:
        raise ValueError("测试数量必须在 1 到 32 之间")
    blob = bytearray(64 + case_record.size * len(cases))

    def append(content):
        blob.extend(bytes((-len(blob)) % 16))
        offset = len(blob)
        blob.extend(content)
        return offset

    q_offset, down_offset = (append(content) for content in packed_weights)
    xq_offset, xd_offset = (append(values.tobytes()) for values in inputs)
    for index, case in enumerate(cases):
        component, rows, columns, batch = (case[key] for key in ("component", "rows", "columns", "batch"))
        row_start, column_start = case.get("row_start", 0), case.get("column_start", 0)
        if component not in (0, 1) or not 1 <= batch <= 16 or rows < 1 or columns < 32 or \
                column_start < 0 or column_start % 32 or columns % 32 or row_start < 0 or \
                row_start + rows > weights[component].shape[0] or \
                column_start + columns > weights[component].shape[1] or len(case["name"].encode()) >= 32:
            raise ValueError(f"真实权重切片范围无效: {case['name']}")
        selected_w = weights[component][row_start:row_start + rows,
                                        column_start:column_start + columns].astype(np.float64)
        selected_x = inputs[component][:batch, column_start:column_start + columns].astype(np.float64)
        reference = selected_x @ selected_w.T
        bounds = np.abs(selected_x) @ np.abs(selected_w).T
        reference_offset = append(reference.astype("<f4").tobytes())
        bound_offset = append(bounds.astype("<f4").tobytes())
        case_record.pack_into(blob, 64 + index * case_record.size, case["name"].encode(),
                              component, rows, columns, batch, reference_offset, bound_offset,
                              row_start, column_start, int(case.get("full", False)))
    struct.pack_into("<16I", blob, 0, 0x42345144, 2, len(cases), 16, 64,
                     q_offset, len(packed_weights[0]), down_offset, len(packed_weights[1]),
                     xq_offset, xd_offset, 1024, 3072, zlib.crc32(blob[64:]), len(blob), case_record.size)
    output = args.output / "BENCH.DAT"
    output.write_bytes(blob)
    manifest = {
        "dataset_version": 2, "interface": "matrix_q4", "model": "Qwen/Qwen3-0.6B", "gguf_sha256": source_manifest["sha256"],
        "quantized_repository": source_manifest["quantized_repository"], "revision": source_manifest["revision"],
        "fhq4_sha256": metadata["sha256"], "dataset_sha256": sha256(output),
        "dataset_crc32": f"{zlib.crc32(blob[64:]):08x}", "dataset_bytes": len(blob),
        "component_names": list(names), "scale_stats": scale_stats,
        "source_quantization": {name: gguf_tensors[name].tensor_type.name for name in names},
        "benchmark_quantization": "Q4_0; existing LLMPSP quantizer converts Q4_1 FFN down",
        "component_sha256": {name: hashlib.sha256(content).hexdigest()
                             for name, content in zip(names, packed_weights)},
        "activation_model": "FHQ4 Q4_0 layer 0, FP32 NumPy causal GQA, Q/K RMSNorm, RoPE, SiLU",
        "activation_positions": list(range(16)), "prompt": PROMPT, "token_ids": tokens,
        "token_text": tokenizer.decode(tokens, skip_special_tokens=False),
        "reference": "GGUF dequantize Q4_0; FP64 matrix product rounded once to FP32",
        "tolerance": "abs(error) <= 2e-5 + 2e-5 * sum(abs(x_i * w_i))",
        "cases": cases,
        "upstream_kernel_sha256": sha256(UPSTREAM / "src/falcon_q4_psp.c"),
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    make_icon(args.output / "ICON0.PNG")
    print(f"Real Qwen assets: {len(blob):,} bytes, {len(cases)} cases, sha256={manifest['dataset_sha256']}")


if __name__ == "__main__":
    main()
