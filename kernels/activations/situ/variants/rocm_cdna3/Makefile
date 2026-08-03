# CDNA3 SITU (SituGLU) gated activation: standalone HIP oracle + perf harness.
# The gated activation Kimi K3 uses in both its dense MLP and its 896-expert
# MoE; the masked form is the one the MoE calls.
#
#   make            # build both harnesses
#   make test       # fp64-oracle correctness on GPU 0
#   make bench      # bandwidth-vs-roofline run on GPU 0
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: situ_test.out bench.out

situ_test.out: situ_test.cu tm_situ_kernels.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

bench.out: bench.cu tm_situ_kernels.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: situ_test.out
	HIP_VISIBLE_DEVICES=0 ./situ_test.out

bench: bench.out
	HIP_VISIBLE_DEVICES=0 ./bench.out

clean:
	rm -f situ_test.out bench.out

.PHONY: all test bench clean
