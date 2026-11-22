import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from gguf import GGMLQuantizationType, dequantize
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model.fhq4"
BUILD = HERE / "build"


def probe(mode, text, *extra):
    output = subprocess.check_output(
        [str(BUILD / "probe"), str(MODEL), mode, text, *map(str, extra)], text=True)
    return list(map(int, output.split()))


class Int8Cache(DynamicCache):
    def __init__(self):
        super().__init__()
        self.inputs = []

    @staticmethod
    def quantized(values):
        scale = values.abs().amax(dim=-1, keepdim=True) / 127
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (values / scale).round().clamp(-127, 127) * scale

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.inputs.append(np.stack([key_states.detach().numpy().reshape(8, 128),
                                     value_states.detach().numpy().reshape(8, 128)]))
        return super().update(self.quantized(key_states), self.quantized(value_states),
                              layer_idx, cache_kwargs)


def reference_model(manifest):
    config = Qwen3Config(
        vocab_size=151936, hidden_size=1024, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, max_position_embeddings=40960, rope_theta=1000000.0,
        rms_norm_eps=1e-6, tie_word_embeddings=True, attention_bias=False,
        bos_token_id=151643, eos_token_id=151645,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    names = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
    }
    suffixes = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }
    for layer in range(28):
        names.update({f"blk.{layer}.{name}": f"model.layers.{layer}.{target}"
                      for name, target in suffixes.items()})
    with torch.no_grad():
        for name, entry in manifest["tensors"].items():
            shape = tuple(entry["shape"])
            if entry["type"] == "Q4_0":
                data = np.memmap(MODEL, mode="r", dtype=np.uint8,
                                 offset=entry["offset"], shape=(shape[0], shape[1] // 32 * 18))
                values = dequantize(data, GGMLQuantizationType.Q4_0)
            else:
                values = np.array(np.memmap(MODEL, mode="r", dtype="<f4",
                                            offset=entry["offset"], shape=shape))
            model.get_parameter(names[name]).copy_(torch.from_numpy(values))
    return model


def run(reference):
    BUILD.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(HERE / ".work", local_files_only=True)
    report = {"tokenizer": [], "generation": []}
    texts = (
        "Hello world", "What is 2+2?", "你好，请用中文介绍一下自己。",
        "I'm learning PSP inference, and I've got 32 MB.",
        "Can you write print('hello')?\nExplain each line.",
        "\tHello   world  \r\nNext line.",
        "  <|im_start|>user\n你好<|im_end|>\n",
        "Let's use <think> and </think> in a sentence.",
        "東京でAIを勉強しています。", "🙂 PSP-1000 支持 C 语言吗？",
    )
    for text in texts:
        expected = tokenizer.encode(text, add_special_tokens=False)
        actual = probe("tokenize", text)
        if actual != expected:
            raise AssertionError(f"tokenizer mismatch for {text!r}: {actual} != {expected}")
        report["tokenizer"].append({"text": text, "ids": actual})
    for prompt in ("What is 2+2?", "你好，请用中文回答。"):
        expected = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        if probe("chat", prompt) != expected:
            raise AssertionError("chat template mismatch")
    print("Official tokenizer and chat template: PASS", flush=True)
    if reference:
        torch.set_num_threads(4)
        manifest = json.loads(MODEL.with_suffix(".fhq4.json").read_text())
        model = reference_model(manifest)
        trace = []

        def trace_layer(module, inputs):
            trace.append(inputs[0].detach().numpy().reshape(1024).copy())

        for layer in model.model.layers:
            layer.register_forward_pre_hook(trace_layer)
        model.model.norm.register_forward_pre_hook(trace_layer)
        text = "Hello world"
        ids = probe("logits", text, BUILD / "c-logits.f32")
        cache = Int8Cache()
        with torch.no_grad():
            for token in ids:
                logits = model(torch.tensor([[token]]), past_key_values=cache,
                               use_cache=True).logits[0, -1]
        actual = np.fromfile(BUILD / "c-logits.f32", dtype="<f4")
        expected = logits.numpy()
        maximum = float(np.max(np.abs(actual - expected)))
        c_kv = np.fromfile(BUILD / "c-logits.f32.kv", dtype="<f4").reshape(-1, 2, 8, 128)
        reference_kv = np.array(cache.inputs)

        def q8(values):
            scale = np.max(np.abs(values), axis=-1, keepdims=True) / np.float32(127)
            return np.rint(values / np.where(scale > 0, scale, 1)).astype(np.int8)

        cache_difference = q8(c_kv) != q8(reference_kv)
        changes = np.argwhere(cache_difference)
        first_rounding = None
        if len(changes):
            step, kind, head, channel = map(int, changes[0])
            first_rounding = {"position": step // 28, "layer": step % 28,
                              "cache": "key" if kind == 0 else "value",
                              "max_projection_error": float(np.max(np.abs(c_kv[step] - reference_kv[step]))),
                              "different_int8_values": int(cache_difference[step].sum())}
            print("First int8 rounding difference:", first_rounding, flush=True)
        c_trace = np.fromfile(BUILD / "c-logits.f32.layers", dtype="<f4").reshape(-1, 1024)
        errors = np.max(np.abs(c_trace - np.array(trace)), axis=1).reshape(len(ids), 29)
        cosine = float(np.dot(actual, expected) / np.linalg.norm(actual) / np.linalg.norm(expected))
        log_actual = actual - np.logaddexp.reduce(actual)
        log_expected = expected - np.logaddexp.reduce(expected)
        kl = float(np.sum(np.exp(log_expected) * (log_expected - log_actual)))
        # FP32 累加顺序会影响 int8 KV 的舍入，核验输出分布及实际生成的 token。
        if kl > 0.001:
            raise AssertionError(f"reference distribution mismatch: cosine={cosine}, KL={kl}")
        if actual.argmax() != expected.argmax():
            raise AssertionError("reference argmax mismatch")
        report["reference"] = {"implementation": "transformers Qwen3ForCausalLM",
                               "max_abs_logit_error": maximum,
                               "cosine_similarity": cosine, "kl_divergence": kl,
                               "first_int8_rounding_difference": first_rounding,
                               "layer_max_abs_errors": errors.tolist(),
                               "next_token": int(actual.argmax())}
        print(f"Hugging Face logits: PASS (max error {maximum:.7f})", flush=True)
    for prompt, limit in (("What is 2+2?", 8), ("你好，请用中文回答：1加1等于多少？", 16)):
        answers = []
        for cache_mb in (0, 6):
            command = [str(BUILD / "qwen_cli"), str(MODEL), prompt, str(limit), str(cache_mb)]
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            answers.append(result.stdout)
            (BUILD / f"host-{len(report['generation'])}-cache{cache_mb}.log").write_text(
                result.stderr + result.stdout)
        if answers[0] != answers[1] or not answers[0].strip():
            raise AssertionError("cache equivalence or generation failed")
        report["generation"].append({"prompt": prompt, "answer": answers[0].strip(),
                                      "cache_mib": [0, 6]})
        print(f"Generation cache 0/6: PASS {answers[0].strip()!r}", flush=True)
        if reference:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            cache = Int8Cache()
            generated = []
            with torch.no_grad():
                for token in input_ids:
                    logits = model(torch.tensor([[token]]), past_key_values=cache,
                                   use_cache=True).logits[0, -1]
                for _ in range(limit):
                    token = int(logits.argmax())
                    if token in (151643, 151645):
                        break
                    generated.append(token)
                    logits = model(torch.tensor([[token]]), past_key_values=cache,
                                   use_cache=True).logits[0, -1]
            answer = tokenizer.decode(generated, skip_special_tokens=False)
            if answer != answers[0].rstrip("\n"):
                raise AssertionError(f"Hugging Face reply mismatch: {answer!r}")
            report["generation"][-1]["reference_ids"] = generated
            print("Hugging Face generated token sequence: PASS", flush=True)
    (BUILD / "host-verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    run(parser.parse_args().reference)
