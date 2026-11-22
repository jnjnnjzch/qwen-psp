import argparse
import configparser
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"


def run(emulator):
    config = configparser.ConfigParser()
    config.read(HERE / "psp1000.ini")
    if config.getint("SystemParam", "PSPModel") != 0:
        raise ValueError("PSP-1000 configuration required")
    host = json.loads((BUILD / "host-verification.json").read_text(encoding="utf-8"))
    report = {"psp_model": "PSP-1000", "physical_ram_mib": 32, "runs": []}
    game = BUILD / "Qwen3Test"
    for index, cache_mb, max_new in ((0, 0, 8), (1, 6, 16), (0, 6, 8)):
        expected = host["generation"][index]
        settings = f"{cache_mb} {max_new}\n{expected['prompt']}\n"
        (game / "qwen_test.txt").write_text(settings, encoding="utf-8")
        log_path = BUILD / f"psp-{index}-cache{cache_mb}.log"
        command = [str(emulator), str(game / "EBOOT.PBP"),
                   "--appendconfig=" + str(HERE / "psp1000.ini"),
                   "--memstick=" + str(BUILD / "memstick"),
                   "--root=" + str(game), "--timeout-wall=1800"]
        with log_path.open("wb") as log:
            subprocess.run(command, cwd=game, stdout=log, stderr=subprocess.STDOUT, check=True)
        output = log_path.read_bytes()
        if b"[pass]" not in output or b"[fail]" in output:
            raise AssertionError(f"PSP generation failed: {log_path}")
        ids = [int(value) for value in re.findall(rb"\[token\] id=(\d+)", output)]
        if ids != expected["reference_ids"]:
            raise AssertionError(f"PSP token mismatch: {ids} != {expected['reference_ids']}")
        memory_line = re.search(rb"\[memory\] ([^\r\n]+)", output)
        memory = {key.decode(): int(value) for key, value in
                  re.findall(rb"([a-z_]+)=(\d+)", memory_line.group(1))}
        if memory["resident_bytes"] >= 20 * 1048576:
            raise AssertionError("PSP runtime exceeds the 20 MiB resident target")
        elapsed = re.search(rb"elapsed_us=(\d+)", output)
        report["runs"].append({"prompt": expected["prompt"], "cache_mib": cache_mb,
                                "answer": expected["answer"], "token_ids": ids,
                                "memory": memory, "elapsed_emulated_us": int(elapsed.group(1)),
                                "log": str(log_path.relative_to(HERE))})
        print(f"PSP-1000 cache={cache_mb}: PASS {expected['answer']!r}, "
              f"resident={memory['resident_bytes'] / 1048576:.2f} MiB", flush=True)
    (BUILD / "psp-verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emulator", type=Path,
                        default=HERE / ".work/ppsspp-build/PPSSPPHeadless")
    run(parser.parse_args().emulator.resolve())
