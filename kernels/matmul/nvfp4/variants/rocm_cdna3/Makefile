GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: nvfp4_gemm.out

nvfp4_gemm.out: nvfp4_gemm.cu ../../../../quantization/qgemm/variants/rocm_cdna3/quant_formats.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: nvfp4_gemm.out
	HIP_VISIBLE_DEVICES=0 ./nvfp4_gemm.out 64 64 256

bench: nvfp4_gemm.out
	HIP_VISIBLE_DEVICES=0 ./nvfp4_gemm.out 256 256 512

clean:
	rm -f nvfp4_gemm.out

.PHONY: all test bench clean
