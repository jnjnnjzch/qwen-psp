set -eu

PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT"
export TMPDIR="$PROJECT/.work/tmp"
mkdir -p "$TMPDIR" build/psp-app build/Qwen3Test build/PSP/GAME/LLMPSP-Qwen3
sh ../LLMPSP/qwen/q4_0/build_pcre2_psp.sh
make -C build/psp-app -f ../../Makefile.psp all
make -C build/Qwen3Test -f ../../Makefile.psp AUTOTEST=1 all
cp models/qwen3-0.6b-q4_0.fhq4 build/Qwen3Test/model.fhq4
cp tests/qwen_test.txt build/Qwen3Test/qwen_test.txt
cp build/psp-app/EBOOT.PBP build/PSP/GAME/LLMPSP-Qwen3/
cp models/qwen3-0.6b-q4_0.fhq4 build/PSP/GAME/LLMPSP-Qwen3/model.fhq4
cp llmpsp_config.cfg LICENSE build/PSP/GAME/LLMPSP-Qwen3/
cp ../LLMPSP/qwen/q4_0/MODEL-LICENSE.txt ../LLMPSP/qwen/q4_0/MODEL_NOTICE.txt \
 ../LLMPSP/qwen/q4_0/.work/pcre2-10.49/LICENCE.md build/PSP/GAME/LLMPSP-Qwen3/
cd build/PSP/GAME/LLMPSP-Qwen3
sha256sum EBOOT.PBP model.fhq4 llmpsp_config.cfg > CHECKSUMS.sha256
