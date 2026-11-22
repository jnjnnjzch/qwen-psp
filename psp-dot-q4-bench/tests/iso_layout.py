import json
from pathlib import Path
import sys

import pycdlib

REQUIRED = {
    "/": {"PSP_GAME", "UMD_DATA.BIN"},
    "/PSP_GAME": {"PARAM.SFO", "ICON0.PNG", "SYSDIR", "USRDIR"},
    "/PSP_GAME/SYSDIR": {"EBOOT.BIN", "BOOT.BIN"},
    "/PSP_GAME/USRDIR": {"BENCH.DAT", "MANIFEST.JSON", "LLMPSP_LICENSE.TXT", "MODEL_LICENSE.TXT"},
}


def verify_iso_layout(path):
    image = pycdlib.PyCdlib()
    image.open(str(path))
    directories = {}
    # 直接检查 ISO9660 标识符，按 PSP 固件的查找名称严格匹配。
    for directory, required in REQUIRED.items():
        names = {record.file_identifier().decode("ascii")
                 for record in image.list_children(iso_path=directory)} - {".", ".."}
        if not required.issubset(names):
            raise AssertionError(f"PSP 文件名无法匹配: {directory}: {sorted(names)}")
        if any(";" in name for name in names):
            raise AssertionError(f"PSP 文件标识符包含版本后缀: {directory}")
        directories[directory] = sorted(names)
    if image.has_rock_ridge() or image.has_joliet() or image.has_udf():
        raise AssertionError("PSP 镜像需要使用基础 ISO9660 目录")
    image.close()
    return {"psp_exact_filenames": True, "directories": directories}


if __name__ == "__main__":
    print(json.dumps(verify_iso_layout(Path(sys.argv[1])), indent=2))
