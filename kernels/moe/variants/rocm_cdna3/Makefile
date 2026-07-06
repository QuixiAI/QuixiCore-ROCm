# CDNA3 MoE family: standalone self-checking harness.
# Ported from ../QuixiCore-CUDA/kernels/moe (plain-CUDA tm_moe_kernels) via
# hipify + mask-free __shfl_*. moe_quant depends on the tensor-core mma16816
# (tm_qmm.cuh) and is ported with the MFMA pass, not here.
GPU_TARGET ?= CDNA3
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
all: moe_test.out
moe_test.out: moe_test.cu tm_moe_kernels.cuh tm_warp.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	HIP_VISIBLE_DEVICES=0 ./moe_test.out
clean:
	rm -f *.out
.PHONY: all test clean
