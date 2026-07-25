# CDNA3 cross-attention: independent query/key lengths, per-batch valid-key
# count, optional additive bias, optional score softcap.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

# Inherited from the environment when set, so perf/harness/run_kernel_bench.sh
# can pin a run to an idle GPU and have the recorded device index be true.
HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: cross_attention.out

cross_attention.out: cross_attention.cu ../../../../common/cdna3_harness.cuh ../../../../common/cdna3_mfma.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	./cross_attention.out

bench: all
	./cross_attention.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
