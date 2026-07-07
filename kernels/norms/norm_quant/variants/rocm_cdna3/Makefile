# CDNA3 norm+quant epilogues and int8 quantizers.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: norm_quant_test.out

norm_quant_test.out: norm_quant_test.cu tm_norm_quant_kernels.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: norm_quant_test.out
	HIP_VISIBLE_DEVICES=0 ./norm_quant_test.out

bench: test

clean:
	rm -f norm_quant_test.out

.PHONY: all test bench clean
