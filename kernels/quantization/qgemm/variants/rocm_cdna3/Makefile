# CDNA3 quantized tensor-core GEMM (MFMA). Rewrites the CUDA PTX
# mma.sync.m16n8k16 path (tm_qmm.cuh) to v_mfma_f32_16x16x16_f16 (tm_qmm_mfma.cuh)
# on a full 64-wide wavefront. Correctness vs the checked-in golden Y_ref.
GPU_TARGET ?= CDNA3
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
GOLDEN ?= ../../../../../../QuixiCore-CUDA/kernels/quant/golden
all: qgemm.out qflux.out
qflux.out: qflux.cu tm_qmm_mfma.cuh quant_formats.cuh quant_formats_tables.cuh quant_tables.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

qgemm.out: qgemm.cu tm_qmm_mfma.cuh quant_formats.cuh quant_formats_tables.cuh quant_tables.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	@for d in $(GOLDEN)/*/; do HIP_VISIBLE_DEVICES=0 ./qgemm.out "$$d"; HIP_VISIBLE_DEVICES=0 ./qflux.out "$$d"; done
clean:
	rm -f *.out
.PHONY: all test clean

qgemm_bench.out: qgemm_bench.cu tm_qmm_mfma.cuh quant_formats.cuh quant_formats_tables.cuh quant_tables.cuh
	$(HIPCC) -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value $< -o $@
bench: qgemm_bench.out
	HIP_VISIBLE_DEVICES=0 ./qgemm_bench.out 64 4096 4096
	HIP_VISIBLE_DEVICES=0 ./qgemm_bench.out 256 4096 4096
	HIP_VISIBLE_DEVICES=0 ./qgemm_bench.out 2048 4096 4096
.PHONY: bench
