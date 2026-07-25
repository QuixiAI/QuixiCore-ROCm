# CDNA3 RoPE variants: rotary_positioned, mrope, rope_table,
# rope_interleaved_to_split, rope_backward, qk_norm_rope_positioned,
# qk_norm_rope_split, rope_q_norm.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

# Inherited from the environment when set, so perf/harness/run_kernel_bench.sh
# can pin a run to an idle GPU and have the recorded device index be true.
HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: rope_variants.out

rope_variants.out: rope_variants.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	./rope_variants.out

bench: all
	./rope_variants.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
