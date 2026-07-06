# CDNA3 Based 2nd-order Taylor-feature causal linear attention + fp64 oracle.
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
all: based.out
based.out: based.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	HIP_VISIBLE_DEVICES=0 ./based.out
clean:
	rm -f *.out
.PHONY: all test clean
