set -eu

QWEN_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export TMPDIR="$QWEN_DIR/.work/tmp"
mkdir -p "$TMPDIR" "$QWEN_DIR/.work/pcre2-psp" "$QWEN_DIR/.work/psp-deps"
cd "$QWEN_DIR/.work/pcre2-psp"
CC=psp-gcc AR=psp-ar RANLIB=psp-ranlib CFLAGS='-O2 -G0' ../pcre2-10.49/configure \
    --host=psp --prefix="$QWEN_DIR/.work/psp-deps" \
    --disable-shared --enable-static --enable-unicode \
    --disable-jit
make -j2 libpcre2-8.la
mkdir -p "$QWEN_DIR/.work/psp-deps/lib" "$QWEN_DIR/.work/psp-deps/include"
cp .libs/libpcre2-8.a "$QWEN_DIR/.work/psp-deps/lib/"
cp src/pcre2.h "$QWEN_DIR/.work/psp-deps/include/"
