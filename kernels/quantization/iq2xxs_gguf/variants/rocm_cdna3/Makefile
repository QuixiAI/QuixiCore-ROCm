# CDNA3 IQ2_XXS (GGML_TYPE_IQ2_XXS) GGUF-layout kernels: dequant + q8_1 tiled GEMM.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: iq2xxs_mmq_test.out

iq2xxs_mmq_test.out: iq2xxs_mmq_test.cu iq2xxs_mmq_kernels.cuh quant_tables.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./iq2xxs_mmq_test.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./iq2xxs_mmq_test.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
