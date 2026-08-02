# CDNA3 MXFP4 (GGML_TYPE_MXFP4) GGUF-layout kernels: dequant, q8_1 GEMV/MoE
# and the q8_1 tiled GEMM.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: mxfp4_gguf_test.out mxfp4_mmq_test.out

mxfp4_gguf_test.out: mxfp4_gguf_test.cu mxfp4_gguf_kernels.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

mxfp4_mmq_test.out: mxfp4_mmq_test.cu mxfp4_mmq_kernels.cuh mxfp4_gguf_kernels.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./mxfp4_gguf_test.out
	HIP_VISIBLE_DEVICES=0 ./mxfp4_mmq_test.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./mxfp4_gguf_test.out --bench
	HIP_VISIBLE_DEVICES=0 ./mxfp4_mmq_test.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
