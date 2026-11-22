BUILD := build
CC ?= cc
CFLAGS := -O3 -std=c99 -Wall -Wextra -fno-math-errno
CPPFLAGS := -DFALCON_QWEN3 -DFALCON_SCALAR_ATTENTION -Iinclude
SOURCES := src/falcon_h1.c src/falcon_tokenizer.c src/falcon_sampler.c
LDLIBS := -lpcre2-8 -lm
export TMPDIR := $(CURDIR)/.work/tmp
.DEFAULT_GOAL := all

.PHONY: all host psp package psp-test
all: psp
host: $(BUILD)/qwen_cli $(BUILD)/probe

$(BUILD) .work/tmp:
	mkdir -p $@

$(BUILD)/qwen_cli: $(SOURCES) tools/qwen_cli.c include/falcon_h1.h | $(BUILD) .work/tmp
	$(CC) $(CFLAGS) $(CPPFLAGS) $(SOURCES) tools/qwen_cli.c $(LDLIBS) -o $@

$(BUILD)/probe: $(SOURCES) tests/probe.c include/falcon_h1.h | $(BUILD) .work/tmp
	$(CC) $(CFLAGS) $(CPPFLAGS) $(SOURCES) tests/probe.c $(LDLIBS) -o $@

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
