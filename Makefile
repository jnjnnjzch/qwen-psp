CC ?= cc
CFLAGS := -O3 -std=c99 -Wall -Wextra -fno-math-errno -MMD -MP
CPPFLAGS := -Iinclude -Isrc
LDLIBS := -lpcre2-8 -lm
export TMPDIR := $(CURDIR)/.work/tmp
.DEFAULT_GOAL := all

include tools/sources.mk
OBJECTS := $(CORE_SOURCES:%.c=build/host/%.o)

.PHONY: all host psp package psp-test
all: psp
host: build/qwen_cli build/tokenizer_demo build/probe build/multiturn

build/qwen_cli: $(OBJECTS) build/host/tools/qwen_cli.o
	$(CC) $^ $(LDLIBS) -o $@

build/tokenizer_demo: $(OBJECTS) build/host/tools/tokenizer_demo.o
	$(CC) $^ $(LDLIBS) -o $@

build/probe: $(OBJECTS) build/host/tests/probe.o
	$(CC) $^ $(LDLIBS) -o $@

build/multiturn: $(OBJECTS) build/host/tests/multiturn.o
	$(CC) $^ $(LDLIBS) -o $@

build/host/%.o: %.c
	mkdir -p $(dir $@) .work/tmp
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

psp:
	sh tools/build_pcre2_psp.sh
	mkdir -p build/psp
	$(MAKE) -C build/psp -f ../../Makefile.psp

package: psp
	mkdir -p build/PSP/GAME/LLMPSP-Qwen3
	cp build/psp/EBOOT.PBP model.fhq4 llmpsp_config.cfg LICENSE MODEL-LICENSE.txt MODEL_NOTICE.txt build/PSP/GAME/LLMPSP-Qwen3/
	cp .work/pcre2-10.49/LICENCE.md build/PSP/GAME/LLMPSP-Qwen3/PCRE2-LICENSE.md
	cd build/PSP/GAME/LLMPSP-Qwen3 && sha256sum EBOOT.PBP model.fhq4 llmpsp_config.cfg > CHECKSUMS.sha256

psp-test: psp
	mkdir -p build/Qwen3Test
	$(MAKE) -C build/Qwen3Test -f ../../Makefile.psp AUTOTEST=1
	cp model.fhq4 tests/qwen_test.txt build/Qwen3Test/

-include $(OBJECTS:.o=.d) build/host/tools/qwen_cli.d build/host/tools/tokenizer_demo.d \
 build/host/tests/probe.d build/host/tests/multiturn.d
