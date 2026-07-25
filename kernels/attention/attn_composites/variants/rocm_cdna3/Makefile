# CDNA3 composite attention: swin_attention_d32, decode_cache_attention,
# cascade_attention_multi.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

# Inherited from the environment when set, so perf/harness/run_kernel_bench.sh
# can pin a run to an idle GPU and have the recorded device index be true.
HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: attn_composites.out

attn_composites.out: attn_composites.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	./attn_composites.out

bench: all
	./attn_composites.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
