import configparser
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
EMULATOR = ROOT.parent / "LLMPSP/qwen/q4_0/.work/ppsspp-build/PPSSPPHeadless"


def main():
    config = configparser.ConfigParser()
    config.read(ROOT / "psp1000.ini")
    if config.getint("SystemParam", "PSPModel") != 0:
        raise ValueError("PSP-1000 configuration required")
    host = json.loads((BUILD / "host-verification.json").read_text(encoding="utf-8"))
    report = {"psp_model": "PSP-1000", "physical_ram_mib": 32, "runs": []}
    game = BUILD / "Qwen3Test"
    for index, cache_mib, limit in ((0, 0, 8), (1, 6, 16), (0, 6, 8)):
        expected = host["generation"][index]
        (game / "qwen_test.txt").write_text(
            f"{cache_mib} {limit}\n{expected['prompt']}\n", encoding="utf-8")
        log_path = BUILD / f"psp-{index}-cache{cache_mib}.log"
        command = [str(EMULATOR), str(game / "EBOOT.PBP"),
                   "--appendconfig=" + str(ROOT / "psp1000.ini"),
                   "--memstick=" + str(BUILD / "test-memstick"),
                   "--root=" + str(game), "--timeout-wall=1800"]
        with log_path.open("wb") as log:
            subprocess.run(command, cwd=game, stdout=log, stderr=subprocess.STDOUT, check=True)
        output = log_path.read_bytes()
        if b"[pass]" not in output or b"[fail]" in output:
            raise AssertionError(f"PSP generation failed: {log_path}")
        token_ids = [int(value) for value in re.findall(rb"\[token\] id=(\d+)", output)]
        if token_ids != expected["reference_ids"]:
            raise AssertionError(f"PSP token mismatch: {token_ids}")
        attention = re.search(
            rb"\[attention\] head_dim=128 forwards=(\d+) score_max_abs=(\S+) output_max_abs=(\S+)", output)
        if attention is None or int(attention.group(1)) == 0:
            raise AssertionError("128-dimensional attention checks missing")
        attention_report = {"head_dim": 128, "forward_checks": int(attention.group(1)),
                            "score_max_abs": float(attention.group(2)),
                            "output_max_abs": float(attention.group(3))}
        memory_line = re.search(rb"\[memory\] ([^\r\n]+)", output)
        memory = {key.decode(): int(value) for key, value in
                  re.findall(rb"([a-z_]+)=(\d+)", memory_line.group(1))}
        if memory["resident_bytes"] >= 20 * 1048576:
            raise AssertionError("runtime exceeds the 20 MiB resident target")
        report["runs"].append({"prompt": expected["prompt"], "answer": expected["answer"],
                                "token_ids": token_ids, "cache_mib": cache_mib, "memory": memory,
                                "attention": attention_report,
                                "log": str(log_path.relative_to(ROOT))})
        print(f"PSP-1000 cache={cache_mib}: PASS {expected['answer']!r}, "
              f"resident={memory['resident_bytes'] / 1048576:.2f} MiB", flush=True)
    (BUILD / "psp-verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
