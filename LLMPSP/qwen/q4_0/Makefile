ROOT := ../..
BUILD := build
CC ?= cc
CFLAGS := -O3 -std=c99 -Wall -Wextra -fno-math-errno
CPPFLAGS := -DFALCON_QWEN3 -DFALCON_SCALAR_ATTENTION -I$(ROOT)/include
SOURCES := $(ROOT)/src/falcon_h1.c $(ROOT)/src/falcon_tokenizer.c $(ROOT)/src/falcon_sampler.c
LDLIBS := -lpcre2-8 -lm
export TMPDIR := $(CURDIR)/.work/tmp

.PHONY: host convert psp
host: $(BUILD)/qwen_cli $(BUILD)/probe

$(BUILD) .work/tmp:
	mkdir -p $@

$(BUILD)/qwen_cli: $(SOURCES) $(ROOT)/tools/falcon_host_cli.c model.inc tokenizer.inc $(ROOT)/include/falcon_h1.h | $(BUILD) .work/tmp
	$(CC) $(CFLAGS) $(CPPFLAGS) $(SOURCES) $(ROOT)/tools/falcon_host_cli.c $(LDLIBS) -o $@

$(BUILD)/probe: $(SOURCES) probe.c model.inc tokenizer.inc $(ROOT)/include/falcon_h1.h | $(BUILD) .work/tmp
	$(CC) $(CFLAGS) $(CPPFLAGS) $(SOURCES) probe.c $(LDLIBS) -o $@

convert:
	python convert.py

psp:
	sh build_psp.sh
