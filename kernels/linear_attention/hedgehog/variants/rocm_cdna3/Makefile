# CDNA3 Hedgehog hybrid linear+exact attention (feature maps + windowed) + oracle.
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
all: hedgehog.out
hedgehog.out: hedgehog.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	HIP_VISIBLE_DEVICES=0 ./hedgehog.out
clean:
	rm -f *.out
.PHONY: all test clean
