# CDNA3 gemm_staged: LDS-staged f32/bf16 GEMM contract surface.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: gemm_staged.out

gemm_staged.out: gemm_staged.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./gemm_staged.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./gemm_staged.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
