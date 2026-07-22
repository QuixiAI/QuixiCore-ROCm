# CDNA3 qkv_proj_fused: fused combined Q/K/V weight-only Q4_0 MFMA projection.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: qkv_proj_fused.out

qkv_proj_fused.out: qkv_proj_fused.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./qkv_proj_fused.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./qkv_proj_fused.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
