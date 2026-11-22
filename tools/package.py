import hashlib
import json
from pathlib import Path
import re
import shutil

import pycdlib

from iso_layout import verify_iso_layout

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
ASSETS = BUILD / "assets"
IMPLEMENTATION_ID = "five-kernels-1"
KERNELS = [
    {"name": "C / SCALAR", "scale_conversion": "C half_to_float", "disassembly": "scalar-disassembly.txt"},
    {"name": "LLMPSP / VFPU", "scale_conversion": "MIPS integer bits + sw/lv.s",
     "disassembly": "vfpu-disassembly.txt"},
    {"name": "VH2F / VFPU", "scale_conversion": "lhu + mtv + vh2f.s",
     "disassembly": "dot-vh2f-disassembly.txt"},
    {"name": "CUSTOM / VMMUL", "scale_conversion": "MIPS integer bits + mtv",
     "disassembly": "custom-disassembly.txt"},
    {"name": "VH2F / VMMUL", "scale_conversion": "lhu + packing + mtv + vh2f.p",
     "disassembly": "custom-vh2f-disassembly.txt"},
]


def instructions(filename):
    return re.findall(r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+([a-z0-9.]+)",
                      (BUILD / filename).read_text(), re.MULTILINE)


def main():
    compiled = [instructions(kernel["disassembly"]) for kernel in KERNELS]
    scalar, vfpu, dot_vh2f, custom, custom_vh2f = compiled
    forbidden = [mnemonic for mnemonic in scalar
                 if mnemonic.startswith(("v", "lv.", "sv.", "ulv.", "usv."))]
    scalar_float = {KERNELS[index]["name"]: [mnemonic for mnemonic in compiled[index]
                    if mnemonic in {"mul.s", "add.s", "sub.s", "div.s", "cvt.s.w", "cvt.s.d"}]
                    for index in (3, 4)}
    if not all(compiled) or forbidden or any(scalar_float.values()) or \
            not {"vc2i.s", "vi2f.q", "vmul.q"}.issubset(vfpu) or \
            not {"vc2i.s", "vi2f.q", "vmul.q", "vh2f.s"}.issubset(dot_vh2f) or \
            not {"vmmul.q", "vc2i.s", "vi2f.q"}.issubset(custom) or \
            not {"vmmul.q", "vc2i.s", "vi2f.q", "vh2f.p"}.issubset(custom_vh2f) or \
            any(mnemonic.startswith("vh2f") for mnemonic in vfpu + custom):
        raise ValueError("普通 C 或 VFPU 反汇编检查失败")
    # 同组实现保留相同的 VFPU 计算顺序，差异限定在 scale 转换。
    def arithmetic(sequence):
        return [mnemonic for mnemonic in sequence if mnemonic.startswith("v") and
                not mnemonic.startswith("vh2f")]

    if arithmetic(vfpu) != arithmetic(dot_vh2f) or arithmetic(custom) != arithmetic(custom_vh2f):
        raise ValueError("对应实现的 VFPU 计算顺序不同")
    game = BUILD / "PSP/GAME/DOTQ4BENCH"
    iso_root = BUILD / "iso"
    usrdir = iso_root / "PSP_GAME/USRDIR"
    game.mkdir(parents=True, exist_ok=True)
    usrdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BUILD / "psp/EBOOT.PBP", game / "EBOOT.PBP")
    for destination in (game, usrdir):
        for filename in ("BENCH.DAT", "MANIFEST.json"):
            shutil.copyfile(ASSETS / filename, destination / filename)
        shutil.copyfile(ROOT / "LICENSE", destination / "LLMPSP_LICENSE.txt")
        shutil.copyfile(ROOT / "MODEL-LICENSE.txt", destination / "MODEL_LICENSE.txt")
    shutil.copyfile(ASSETS / "ICON0.PNG", iso_root / "PSP_GAME/ICON0.PNG")
    for name in ("EBOOT.BIN", "BOOT.BIN"):
        shutil.copyfile(BUILD / "psp/DotQ4Bench.prx", iso_root / "PSP_GAME/SYSDIR" / name)
    (iso_root / "UMD_DATA.BIN").write_bytes(b"DQTB-00001|0000000000000000|0001|G\x00")
    image = pycdlib.PyCdlib()
    image.new(interchange_level=2, vol_ident="DOTQ4BENCH", sys_ident="PSP GAME")
    for path in sorted(iso_root.rglob("*")):
        relative = path.relative_to(iso_root)
        iso_path = "/" + "/".join(part.upper() for part in relative.parts)
        if path.is_dir():
            image.add_directory(iso_path=iso_path)
        else:
            image.add_file(str(path), iso_path=iso_path)
    iso_path = BUILD / "DOTQ4BENCH.iso"
    image.write(str(iso_path))
    image.close()
    layout = verify_iso_layout(iso_path)
    # 使用 ISO 库重新读取最终镜像，确认程序和计算数据的字节。
    image.open(str(iso_path))
    for target, source in (("/PSP_GAME/SYSDIR/EBOOT.BIN", BUILD / "psp/DotQ4Bench.prx"),
                           ("/PSP_GAME/USRDIR/BENCH.DAT", ASSETS / "BENCH.DAT")):
        with image.open_file_from_iso(iso_path=target) as file:
            if file.read() != source.read_bytes():
                raise ValueError(f"ISO 内容不同: {target}")
    image.close()
    checksums = []
    for path in (iso_path, game / "EBOOT.PBP", game / "BENCH.DAT"):
        with path.open("rb") as file:
            checksums.append(f"{hashlib.file_digest(file, 'sha256').hexdigest()}  {path.relative_to(BUILD)}")
    (BUILD / "CHECKSUMS.sha256").write_text("\n".join(checksums) + "\n")
    report = {"scalar_vfpu_instructions": forbidden, "scalar_instruction_count": len(scalar),
              "upstream_vfpu_present": True, "custom_vmmul_present": True, "interface": "matrix_q4",
              "implementation_id": IMPLEMENTATION_ID, "kernels": KERNELS,
              "matched_pair_vfpu_arithmetic_verified": True,
              "custom_workspace_bytes": 4 * 4 * 4,
              "custom_quant_conversion": "VFPU vc2i.s + vi2f.q",
              "custom_scale_conversion": {kernel["name"]: kernel["scale_conversion"] for kernel in KERNELS[3:]},
              "custom_scalar_float_arithmetic": scalar_float,
              "custom_accumulator_lifetime": "full K in VFPU registers",
              "custom_compiler_frame_usage": {
                  "CUSTOM / VMMUL": (BUILD / "psp/objects/custom_vfpu.su").read_text(),
                  "VH2F / VMMUL": (BUILD / "psp/objects/custom_vh2f.su").read_text()},
              "psp_size": (BUILD / "psp-size.txt").read_text(), "iso_payload_verified": True,
              "target_psp_firmware": "5.50", "target_custom_firmware": "5.50 Prome-4",
              "iso_layout": layout,
              "iso_bytes": iso_path.stat().st_size,
              "compiler": (BUILD / "compiler.txt").read_text(),
              "image": "ghcr.io/pspdev/pspdev@sha256:ee385df3aa783189d5a033af5eb1d4a9cdd93718fb8bf652ec7bb48e54038a8b"}
    (BUILD / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Packaged {iso_path}: {iso_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
