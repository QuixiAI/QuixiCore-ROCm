# CDNA3 quantized MoE grouped GEMMs (MFMA). The 3 tensor-core grouped GEMMs
# (fp8/nvfp4/wna16) use v_mfma_f32_16x16x16_f16 via tm_qmm_mfma.cuh (rewritten
# from the CUDA PTX mma.sync path); silu/quant/routing kernels are plain ports.
# Self-checking harness (fp64 refs, no golden).
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
all: moe_quant_test.out
moe_quant_test.out: moe_quant_test.cu tm_moe_quant_kernels.cuh tm_qmm_mfma.cuh quant_formats.cuh quant_formats_tables.cuh quant_tables.cuh tm_warp.cuh tm_rng.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	HIP_VISIBLE_DEVICES=0 ./moe_quant_test.out
clean:
	rm -f *.out
.PHONY: all test clean
