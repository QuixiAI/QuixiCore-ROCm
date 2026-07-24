# CDNA3 qk_norm_rope: packed QKV Q/K RMSNorm + RoPE, V copy-through.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: qk_norm_rope.out

qk_norm_rope.out: qk_norm_rope.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./qk_norm_rope.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./qk_norm_rope.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
