GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: mxfp8_gemm.out

mxfp8_gemm.out: mxfp8_gemm.cu ../../../../quantization/qgemm/variants/rocm_cdna3/quant_formats.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: mxfp8_gemm.out
	HIP_VISIBLE_DEVICES=0 ./mxfp8_gemm.out 64 64 256

bench: mxfp8_gemm.out
	HIP_VISIBLE_DEVICES=0 ./mxfp8_gemm.out 256 256 512

clean:
	rm -f mxfp8_gemm.out

.PHONY: all test bench clean
