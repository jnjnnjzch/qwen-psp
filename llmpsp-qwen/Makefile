CC ?= cc
CFLAGS := -O3 -std=c99 -Wall -Wextra -fno-math-errno -MMD -MP
CPPFLAGS := -Iinclude -Isrc
LDLIBS := -lpcre2-8 -lm
export TMPDIR := $(CURDIR)/.work/tmp

include sources.mk
OBJECTS := $(CORE_SOURCES:%.c=build/host/%.o)

.PHONY: all host
all: host
host: build/qwen_cli build/tokenizer_demo build/probe build/multiturn

build/qwen_cli: $(OBJECTS) build/host/app/main.o
	$(CC) $^ $(LDLIBS) -o $@

build/tokenizer_demo: $(OBJECTS) build/host/app/tokenizer_demo.o
	$(CC) $^ $(LDLIBS) -o $@

build/probe: $(OBJECTS) build/host/tests/probe.o
	$(CC) $^ $(LDLIBS) -o $@

build/multiturn: $(OBJECTS) build/host/tests/multiturn.o
	$(CC) $^ $(LDLIBS) -o $@

build/host/%.o: %.c
	mkdir -p $(dir $@) .work/tmp
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

-include $(OBJECTS:.o=.d) build/host/app/main.d build/host/app/tokenizer_demo.d \
 build/host/tests/probe.d build/host/tests/multiturn.d
