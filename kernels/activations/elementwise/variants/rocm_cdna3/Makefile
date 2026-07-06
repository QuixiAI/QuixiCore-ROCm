# CDNA3 elementwise/norm family: standalone HIP oracle + perf harnesses.
# Ported from ../../../../../../QuixiCore-CUDA/kernels/elementwise (plain-CUDA
# tm_* kernels) via hipify + CDNA3 warp adaptations (mask-free __shfl_*).
#
#   make            # build both harnesses
#   make test       # fp64-oracle correctness on GPU 0
#   make bench      # focused perf run (32-lane vs 64-lane wavefront) on GPU 0
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: elementwise_test.out bench.out

elementwise_test.out: elementwise_test.cu tm_elementwise_kernels.cuh tm_warp.cuh tm_rng.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

bench.out: bench.cu tm_elementwise_kernels.cuh tm_warp.cuh tm_rng.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: elementwise_test.out
	HIP_VISIBLE_DEVICES=0 ./elementwise_test.out

bench: bench.out
	HIP_VISIBLE_DEVICES=0 ./bench.out

clean:
	rm -f elementwise_test.out bench.out

.PHONY: all test bench clean
