# CDNA3 Q8_0 paged KV-cache codec: scatter, gather, copy_blocks, paged attention.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

# Inherited from the environment when set, so perf/harness/run_kernel_bench.sh
# can pin a run to an idle GPU and have the recorded device index be true.
HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: kv_cache_q8_0.out

kv_cache_q8_0.out: kv_cache_q8_0.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	./kv_cache_q8_0.out

bench: all
	./kv_cache_q8_0.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
