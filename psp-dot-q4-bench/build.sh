set -eu

PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORKSPACE=$(dirname "$PROJECT")
cd "$PROJECT"
export TMPDIR="$PROJECT/.work/tmp"
export PYTHONPATH="$PROJECT/.work/python${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=2
mkdir -p "$TMPDIR" build/psp
PSP_IMAGE=ghcr.io/pspdev/pspdev@sha256:ee385df3aa783189d5a033af5eb1d4a9cdd93718fb8bf652ec7bb48e54038a8b

python tools/prepare_assets.py
docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
    --mount "type=bind,source=$PROJECT,target=/workspace/psp-dot-q4-bench" \
    --mount "type=bind,source=$WORKSPACE/LLMPSP,target=/workspace/LLMPSP,readonly" \
    -e TMPDIR=/workspace/psp-dot-q4-bench/.work/tmp -w /workspace/psp-dot-q4-bench \
    --entrypoint /bin/sh "$PSP_IMAGE" build-container.sh
python tools/package.py
printf 'ISO: %s/build/DOTQ4BENCH.iso\n' "$PROJECT"
printf 'PBP: %s/build/PSP/GAME/DOTQ4BENCH/EBOOT.PBP\n' "$PROJECT"
