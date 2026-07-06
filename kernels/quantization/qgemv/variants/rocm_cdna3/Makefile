# CDNA3 quantized dequant + GEMV harnesses.
# Ported from ../QuixiCore-CUDA/kernels/quant (plain-CUDA tm_* format layer) via
# hipify + CDNA3 adaptations: __dp4a -> __builtin_amdgcn_sdot4 (gfx942 V_DOT4),
# mask-free __shfl_*. Correctness is checked against the CUDA repo's checked-in
# golden data (quant.py byte-exact references); point GOLDEN/GOLDEN_INT at it.
#
#   make            # build all harnesses
#   make test       # dequant+GEMV over all formats + int GEMV + runtime-quant
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

GOLDEN     ?= ../../../../../../QuixiCore-CUDA/kernels/quant/golden
GOLDEN_INT ?= ../../../../../../QuixiCore-CUDA/kernels/quant/golden_int

HDRS = quant_formats.cuh quant_formats_tables.cuh quant_tables.cuh tm_rng.cuh

all: qgemv.out qgemv_int.out quant_rt.out

qgemv.out: qgemv.cu $(HDRS)
	$(HIPCC) $(HIPFLAGS) $< -o $@
qgemv_int.out: qgemv_int.cu $(HDRS)
	$(HIPCC) $(HIPFLAGS) $< -o $@
quant_rt.out: quant_rt.cu $(HDRS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	@echo "== dequant + GEMV over all golden formats =="; \
	for d in $(GOLDEN)/*/; do HIP_VISIBLE_DEVICES=0 ./qgemv.out "$$d"; done
	@echo "== int8 GEMV (w8a8, w2a8) =="; HIP_VISIBLE_DEVICES=0 ./qgemv_int.out $(GOLDEN_INT)
	@echo "== runtime quantize (per-token/tensor, int8/fp8) =="; HIP_VISIBLE_DEVICES=0 ./quant_rt.out

clean:
	rm -f *.out

.PHONY: all test clean
