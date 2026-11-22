set -eu

cd /workspace/psp-dot-q4-bench
export TMPDIR=/workspace/psp-dot-q4-bench/.work/tmp
make -C build/psp -f ../../Makefile.psp -j2 all
psp-objdump -d build/psp/objects/scalar.o > build/scalar-disassembly.txt
psp-objdump -d --disassemble=falcon_dot_q4_vfpu build/psp/objects/llmpsp_vfpu.o > build/vfpu-disassembly.txt
psp-objdump -d build/psp/objects/custom_vfpu.o > build/custom-disassembly.txt
psp-objdump -d build/psp/objects/custom_vh2f.o > build/custom-vh2f-disassembly.txt
psp-objdump -d build/psp/objects/dot_vh2f.o > build/dot-vh2f-disassembly.txt
psp-size build/psp/DotQ4Bench.elf > build/psp-size.txt
psp-gcc --version > build/compiler.txt
psp-config --pspsdk-path >> build/compiler.txt
mkdir -p build/iso/PSP_GAME/SYSDIR
mksfoex -s CATEGORY=UG -s DISC_ID=DQTB00001 -s DISC_VERSION=1.00 \
    -s PSP_SYSTEM_VER=5.50 -d BOOTABLE=1 -d PARENTAL_LEVEL=1 -d REGION=32768 \
    'DOT Q4 / PSP Compute Lab' build/iso/PSP_GAME/PARAM.SFO
