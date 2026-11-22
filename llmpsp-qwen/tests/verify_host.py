import json
from pathlib import Path
import subprocess

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "LLMPSP/qwen/q4_0"
MODEL = ROOT / "models/qwen3-0.6b-q4_0.fhq4"
BUILD = ROOT / "build"


def probe(executable, mode, text, *extra):
    output = subprocess.check_output(
        [str(executable), str(MODEL), mode, text, *map(str, extra)], text=True)
    return list(map(int, output.split()))


def main():
    # 官方 tokenizer 的文件和已验证权重沿用此前下载的真实资源。
    tokenizer = AutoTokenizer.from_pretrained(UPSTREAM / ".work", local_files_only=True)
    previous = json.loads((UPSTREAM / "build/host-verification.json").read_text(encoding="utf-8"))
    report = {"tokenizer": [], "generation": []}
    for example in previous["tokenizer"]:
        text = example["text"]
        actual = probe(BUILD / "probe", "tokenize", text)
        expected = tokenizer.encode(text, add_special_tokens=False)
        if actual != expected:
            raise AssertionError(f"tokenizer mismatch: {text!r}")
        report["tokenizer"].append({"text": text, "ids": actual})
    for text in ("What is 2+2?", "你好，请用中文回答。"):
        expected = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        if probe(BUILD / "probe", "chat", text) != expected:
            raise AssertionError("chat template mismatch")
    print("official tokenizer and chat templates: PASS", flush=True)

    # 比较重组前后相同真实输入的每层输出和完整词表分数。
    probe(UPSTREAM / "build/probe", "logits", "Hello world", BUILD / "previous-logits.f32")
    probe(BUILD / "probe", "logits", "Hello world", BUILD / "current-logits.f32")
    for suffix in ("", ".layers"):
        expected = np.fromfile(str(BUILD / "previous-logits.f32") + suffix, dtype="<f4")
        actual = np.fromfile(str(BUILD / "current-logits.f32") + suffix, dtype="<f4")
        np.testing.assert_array_equal(actual.view("<u4"), expected.view("<u4"))
    report["forward"] = {"layer_outputs_bitwise_equal": True, "logits_bitwise_equal": True}
    print("all decoder layers and complete vocabulary logits: bitwise equal", flush=True)

    for index, (example, limit) in enumerate(zip(previous["generation"], (8, 16))):
        prompt, answers = example["prompt"], []
        for cache_mib in (0, 6):
            result = subprocess.run([str(BUILD / "qwen_cli"), str(MODEL), prompt,
                                     str(limit), str(cache_mib)], capture_output=True, text=True, check=True)
            answers.append(result.stdout.rstrip("\n"))
            (BUILD / f"host-{index}-cache{cache_mib}.log").write_text(
                result.stderr + result.stdout, encoding="utf-8")
        if answers[0] != example["answer"] or answers[1] != example["answer"]:
            raise AssertionError(f"generation mismatch: {answers}")
        report["generation"].append({"prompt": prompt, "answer": answers[0],
                                      "reference_ids": example["reference_ids"], "cache_mib": [0, 6]})
        print(f"generation cache 0/6: PASS {answers[0]!r}", flush=True)
    result = subprocess.run([str(BUILD / "multiturn"), str(MODEL)], capture_output=True,
                            text=True, check=True)
    print(result.stdout.strip(), flush=True)
    report["multiturn"] = result.stdout.strip()
    demo = subprocess.check_output([str(BUILD / "tokenizer_demo"), str(MODEL), "Hello world"], text=True)
    if "重新拼接 token 的字节：Hello world" not in demo:
        raise AssertionError("tokenizer walkthrough decode mismatch")
    (BUILD / "tokenizer-demo.txt").write_text(demo, encoding="utf-8")
    (BUILD / "host-verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
