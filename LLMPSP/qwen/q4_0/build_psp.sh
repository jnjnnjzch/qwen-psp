set -eu

QWEN_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$QWEN_DIR"
export TMPDIR="$QWEN_DIR/.work/tmp"
mkdir -p "$TMPDIR" build/LLMPSP-Qwen3 build/Qwen3Test
sh build_pcre2_psp.sh

make -C build/LLMPSP-Qwen3 -f ../../Makefile.psp all
make -C build/Qwen3Test -f ../../Makefile.psp AUTOTEST=1 all
cp model.fhq4 build/LLMPSP-Qwen3/model.fhq4
cp model.fhq4 build/Qwen3Test/model.fhq4
cp qwen_test.txt build/Qwen3Test/qwen_test.txt
cp llmpsp_config.cfg build/LLMPSP-Qwen3/llmpsp_config.cfg
cp ../../LICENSE build/LLMPSP-Qwen3/LICENSE
cp .work/pcre2-10.49/LICENCE.md build/LLMPSP-Qwen3/PCRE2-LICENSE.md
mkdir -p build/PSP/GAME/LLMPSP-Qwen3
cp build/LLMPSP-Qwen3/EBOOT.PBP model.fhq4 llmpsp_config.cfg \
    MODEL_NOTICE.txt MODEL-LICENSE.txt build/LLMPSP-Qwen3/LICENSE \
    build/LLMPSP-Qwen3/PCRE2-LICENSE.md build/PSP/GAME/LLMPSP-Qwen3/
cd build/PSP/GAME/LLMPSP-Qwen3
sha256sum EBOOT.PBP model.fhq4 llmpsp_config.cfg > CHECKSUMS.sha256
printf 'PSP package: %s/build/PSP/GAME/LLMPSP-Qwen3\n' "$QWEN_DIR"
