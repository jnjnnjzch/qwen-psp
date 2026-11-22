.PHONY: all package test

all: package

package:
	sh tools/build_psp.sh

test: package
	TMPDIR="$(CURDIR)/.work/tmp" PYTHONPATH="$(CURDIR)/.work/python" python tools/verify.py
