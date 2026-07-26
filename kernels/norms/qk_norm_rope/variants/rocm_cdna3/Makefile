# CDNA3 qk_norm_rope: packed QKV Q/K RMSNorm + RoPE, V copy-through.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value
HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: qk_norm_rope.out qk_norm_rope_kv_f16.out

qk_norm_rope.out: qk_norm_rope.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

qk_norm_rope_kv_f16.out: qk_norm_rope_kv_f16.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	./qk_norm_rope.out
	./qk_norm_rope_kv_f16.out

test-kv-f16: qk_norm_rope_kv_f16.out
	./qk_norm_rope_kv_f16.out

bench: all
	./qk_norm_rope.out --bench
	./qk_norm_rope_kv_f16.out --bench

bench-kv-f16: qk_norm_rope_kv_f16.out
	./qk_norm_rope_kv_f16.out --bench

clean:
	rm -f *.out

.PHONY: all test test-kv-f16 bench bench-kv-f16 clean
