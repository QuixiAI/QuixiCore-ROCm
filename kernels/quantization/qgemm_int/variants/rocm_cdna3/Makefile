# CDNA3 qgemm_int: W8A8/W8A8-AZP/W2A8 prefill GEMM.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: qgemm_int.out

qgemm_int.out: qgemm_int.cu ../../../qgemm/variants/rocm_cdna3/quant_formats.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./qgemm_int.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./qgemm_int.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
