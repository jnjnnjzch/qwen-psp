import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
VERIFY = BUILD / "verify"
EMULATOR = ROOT.parent / "LLMPSP/qwen/q4_0/.work/ppsspp-build/PPSSPPHeadless"
CASES = json.loads((ROOT / "cases.json").read_text())
KERNELS = ["C / SCALAR", "LLMPSP / VFPU", "VH2F / VFPU", "CUSTOM / VMMUL", "VH2F / VMMUL"]
IMPLEMENTATION_ID = "five-kernels-1"
sys.path.insert(0, str(ROOT / "tests"))
from iso_layout import verify_iso_layout


def run_logged(command, path, env):
    with path.open("w", encoding="utf-8") as log:
        subprocess.run([str(value) for value in command], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
    return path.read_text(encoding="utf-8")


def run_psp(name, target, presses, exit_time, env):
    memstick = VERIFY / f"{name}-ms"
    if memstick.exists():
        # 保留每次验证产生的实际输出，重复执行使用独立目录。
        suffix = 1
        while (VERIFY / f"{name}-ms-{suffix}").exists():
            suffix += 1
        memstick = VERIFY / f"{name}-ms-{suffix}"
    memstick.mkdir(parents=True)
    if target.suffix.upper() == ".PBP":
        installed = memstick / "PSP/GAME/DOTQ4BENCH"
        shutil.copytree(target.parent, installed)
        target = installed / "EBOOT.PBP"
    mount_root = target.parent if target.suffix.upper() == ".PBP" else BUILD
    command = [EMULATOR, target, "--appendconfig=" + str(ROOT / "tests/psp1000.ini"),
               "--memstick=" + str(memstick), "--root=" + str(mount_root),
               "--graphics=software", "--timeout-wall=180", f"--timeout-emulated={exit_time + 5}",
               "--press=" + presses]
    output = run_logged(command, VERIFY / f"{name}.log", env)
    if f"[boot] cases={len(CASES)}" not in output or "custom=1" not in output or \
            f"implementation={IMPLEMENTATION_ID}" not in output or \
            "[exit]" not in output or "[fail]" in output:
        raise AssertionError(f"PSP 启动或执行失败: {name}")
    screenshots = sorted((memstick / "PSP/SAVEDATA/DOTQ4BENCH").glob("screen-*.bmp"))
    if not screenshots:
        raise AssertionError("程序未保存实际画面")
    for screenshot in screenshots:
        with Image.open(screenshot) as native_image:
            if native_image.size != (480, 272):
                raise AssertionError("PSP 截图尺寸错误")
    with Image.open(screenshots[-1]) as native_image:
        native_image.save(VERIFY / f"{name}.png")
    if name == "count":
        with Image.open(screenshots[0]) as native_image:
            native_image.save(VERIFY / "settings.png")
    print(f"[pass] {name}: boot, controller input, display, exit", flush=True)
    return output, memstick


def read_results(memstick):
    files = sorted((memstick / "PSP/SAVEDATA/DOTQ4BENCH").glob("*.csv"))
    result = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as file:
            result.append((path, list(csv.DictReader(file))))
    return result


def validate_csv(rows, cases, mode, clocks, samples, target_repetitions=None):
    expected = {(case["name"], kernel) for case in cases for kernel in KERNELS}
    geometry = {case["name"]: case for case in cases}
    groups = {}
    manifest = json.loads((ROOT / "assets/generated/MANIFEST.json").read_text())
    for row in rows:
        groups.setdefault((row["case"], row["kernel"]), []).append(row)
        if row["dataset_crc32"] != manifest["dataset_crc32"] or row["mode"] != mode:
            raise AssertionError("CSV 数据或停止条件不同")
        case = geometry[row["case"]]
        if row["interface"] != "matrix_q4" or row["implementation_id"] != IMPLEMENTATION_ID or \
                (int(row["B"]), int(row["N"]), int(row["K"])) != (case["batch"], case["rows"], case["columns"]) or \
                (int(row["row_start"]), int(row["column_start"])) != \
                (case.get("row_start", 0), case.get("column_start", 0)):
            raise AssertionError("矩阵接口或真实切片范围不同")
        requested_cpu = int(row["requested_cpu_mhz"])
        actual_cpu, actual_bus = float(row["cpu_mhz"]), float(row["bus_mhz"])
        if requested_cpu not in clocks or not math.isfinite(actual_cpu) or not math.isfinite(actual_bus) or \
                abs(actual_cpu - requested_cpu) >= 1 or abs(actual_bus - requested_cpu / 2) >= 1:
            raise AssertionError("CPU/总线频率不同")
        for field in ("us_per_operation", "median_us", "gflops", "speedup"):
            if not math.isfinite(float(row[field])) or float(row[field]) <= 0:
                raise AssertionError(f"无效计时数值: {field}")
        if float(row["max_normalized_error"]) > 1:
            raise AssertionError("计算误差超过阈值")
        count, elapsed = int(row["iterations"]), int(row["elapsed_us"])
        if not math.isclose(float(row["us_per_operation"]), elapsed / count, abs_tol=1e-7):
            raise AssertionError("平均耗时与原始测量不同")
        if mode == "TIME" and elapsed < int(row["target_ms"]) * 1000:
            raise AssertionError("目标计算时间未完成")
        if mode == "COUNT" and count != target_repetitions:
            raise AssertionError("目标执行次数未完成")
    if set(groups) != expected:
        raise AssertionError("结果缺失或包含未实现的算子")
    for key, values in groups.items():
        if len(values) != samples or {int(row["sample"]) for row in values} != set(range(1, samples + 1)):
            raise AssertionError(f"采样数量不同: {key}")
        median = statistics.median(float(row["us_per_operation"]) for row in values)
        if not math.isclose(median, float(values[0]["median_us"]), abs_tol=1e-7):
            raise AssertionError("中位数与样本不同")
    return {"comparisons": len(groups), "sample_rows": len(rows),
            "max_absolute_error": max(float(row["max_abs_error"]) for row in rows),
            "max_normalized_error": max(float(row["max_normalized_error"]) for row in rows)}


def main():
    VERIFY.mkdir(parents=True, exist_ok=True)
    layout = verify_iso_layout(BUILD / "DOTQ4BENCH.iso")
    print("[pass] PSP ISO9660: exact file identifiers", flush=True)
    env = dict(os.environ)
    env["TMPDIR"] = str(ROOT / ".work/tmp")
    for variable, directory in (("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache"),
                                ("XDG_RUNTIME_DIR", "runtime")):
        path = ROOT / ".work" / directory
        path.mkdir(parents=True, exist_ok=True)
        env[variable] = str(path)
    run_logged(["gcc", "-O3", "-std=c99", "-Wall", "-Wextra", "-Werror",
                "-Iinclude", "tests/host.c", "src/engine.c", "src/scalar.c", "src/custom_vfpu.c",
                "-lm", "-lz", "-o", BUILD / "host-check"], VERIFY / "host-build.log", env)
    output = run_logged([BUILD / "host-check", ROOT / "assets/generated/BENCH.DAT"],
                        VERIFY / "host.log", env)
    expected_outputs = sum(case["rows"] * case["batch"] for case in CASES)
    if f"[host-pass] cases={len(CASES)} outputs={expected_outputs} custom=0" not in output:
        raise AssertionError("Host 真实权重计算未完成")
    print(f"[pass] host: {expected_outputs:,} real outputs", flush=True)
    cases = CASES
    iso = BUILD / "DOTQ4BENCH.iso"
    _, _ = run_psp("workloads", iso, "1:0x8000:0.15,2:0x1:0.15", 2, env)
    presses = ("0.5:0x200:0.1,0.8:0x40:0.1,1.1:0x80:0.1,1.4:0x40:0.1,"
               "1.7:0x40:0.1,2:0x80:0.1,2.3:0x40:0.1,2.6:0x40:0.1,2.9:0x20:0.1,"
               "3.2:0x8:0.15,190:0x8000:0.15,191:0x1:0.15")
    output, memstick = run_psp("suite", iso, presses, 191, env)
    if f"[suite-complete] cases={len(cases)} scope=FULL" not in output or "[unimplemented]" in output:
        raise AssertionError("完整测试集合未完成")
    suite_files = read_results(memstick)
    if len(suite_files) != 1:
        raise AssertionError("完整集合需要导出一个 CSV")
    report = {"psp_model": "PSP-1000", "validation_environment": "PPSSPP; timing is emulated",
              "physical_psp_tested": False, "host_outputs": expected_outputs,
              "interface": "matrix_q4", "custom_vmmul": True,
              "dataset_bytes": json.loads((ROOT / "assets/generated/MANIFEST.json").read_text())["dataset_bytes"],
              "free_user_kib": int(re.search(r"free_user_kib=(\d+)", output).group(1)),
              "suite": validate_csv(suite_files[0][1], cases, "TIME", {333}, 1),
              "suite_csv": str(suite_files[0][0].relative_to(ROOT))}
    build_report = json.loads((BUILD / "build-report.json").read_text())
    report["custom_quant_conversion"] = build_report["custom_quant_conversion"]
    report["custom_scale_conversion"] = build_report["custom_scale_conversion"]
    report["custom_workspace_bytes"] = build_report["custom_workspace_bytes"]
    report["custom_scalar_float_arithmetic"] = build_report["custom_scalar_float_arithmetic"]
    report["kernels"] = build_report["kernels"]
    report["implementation_id"] = IMPLEMENTATION_ID
    quick = [case for case in cases if not case.get("full", False)]
    output, memstick = run_psp("quick", iso, "1:0x8:0.15,80:0x8000:0.15,81:0x1:0.15", 81, env)
    if f"[suite-complete] cases={len(quick)} scope=QUICK" not in output:
        raise AssertionError("默认快速集合未完成")
    last_quick = max(index for index, case in enumerate(cases) if not case.get("full", False))
    if f"case={last_quick} valid=1" not in output:
        raise AssertionError("快速集合结果页未选中已完成的项目")
    files = read_results(memstick)
    if len(files) != 1:
        raise AssertionError("快速集合需要导出一个 CSV")
    report["quick"] = validate_csv(files[0][1], quick, "TIME", {333}, 3)
    presses = ("0.5:0x200:0.1,0.8:0x20:0.1,1.1:0x40:0.1,1.4:0x40:0.1,"
               "1.7:0x80:0.1,2:0x40:0.1,2.3:0x20:0.1,2.6:0x40:0.1,2.9:0x20:0.1,"
               "3.2:0x8000:0.1,3.5:0x4000:0.1,4.5:0x100:0.1,4.8:0x20:0.1,5.2:0x4000:0.1,"
               "6.5:0x8000:0.1,7:0x1:0.1")
    output, memstick = run_psp("count", BUILD / "PSP/GAME/DOTQ4BENCH/EBOOT.PBP", presses, 7, env)
    files = read_results(memstick)
    if len(files) != 2:
        raise AssertionError("次数与频率交互需要导出两次 CSV")
    report["count"] = []
    for (path, rows), clock in zip(files, (222, 266)):
        stats = validate_csv(rows, cases[:1], "COUNT", {clock}, 5, target_repetitions=10)
        stats.update({"requested_cpu_mhz": clock, "cpu_mhz": float(rows[0]["cpu_mhz"]),
                      "bus_mhz": float(rows[0]["bus_mhz"]), "csv": str(path.relative_to(ROOT))})
        report["count"].append(stats)
    output, memstick = run_psp("cancel", iso, "1:0x8:0.1,1.3:0x2000:0.2,2.5:0x8000:0.1,3:0x1:0.1", 3, env)
    if "[cancel] index=0" not in output or "[suite-complete]" in output or "[result]" in output:
        raise AssertionError("取消操作未终止当前比较")
    files = read_results(memstick)
    if len(files) != 1 or files[0][1]:
        raise AssertionError("取消操作导出了未完成的测量")
    report["cancel_incomplete_measurement_discarded"] = True
    report["iso_and_pbp_boot"] = True
    report["iso_layout"] = layout
    report["artifact_sha256"] = {}
    for path in (iso, BUILD / "PSP/GAME/DOTQ4BENCH/EBOOT.PBP"):
        with path.open("rb") as file:
            report["artifact_sha256"][str(path.relative_to(ROOT))] = hashlib.file_digest(file, "sha256").hexdigest()
    report["screenshots"] = ["build/verify/workloads.png", "build/verify/settings.png", "build/verify/suite.png", "build/verify/quick.png"]
    (BUILD / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"[pass] ISO/PBP, {len(cases)} workloads, QUICK/FULL, TIME/COUNT, 222/266/333 MHz, CSV, cancellation", flush=True)


if __name__ == "__main__":
    main()
