# CDNA3 lm_head: vocab-projection argmax/categorical sampling over fp16 and
# quantized weights. Scalar GEMV-argmax (no MFMA); plain-CUDA port.
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
GOLDEN ?= ../../../../../../QuixiCore-CUDA/kernels/quant/golden
all: lm_head.out
lm_head.out: lm_head.cu quant_formats.cuh quant_formats_tables.cuh quant_tables.cuh tm_rng.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	@for f in q8_0 q4_0 mxfp4 nvfp4 mxfp8; do HIP_VISIBLE_DEVICES=0 ./lm_head.out $(GOLDEN)/$$f; done
clean:
	rm -f *.out
.PHONY: all test clean
