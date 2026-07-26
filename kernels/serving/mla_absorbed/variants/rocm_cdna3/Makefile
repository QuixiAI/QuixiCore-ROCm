# CDNA3 quantized MLA decode against an absorbed kv_b (dense + sparse/DSA).
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value
HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: mla_absorbed.out
mla_absorbed.out: mla_absorbed.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	./mla_absorbed.out
clean:
	rm -f *.out
.PHONY: all test clean
