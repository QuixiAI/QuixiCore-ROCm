# CDNA3 quant primitives: turboquant (FWHT rotate, permute, moe_lora) +
# mixed-format primitives (fp4/e8m0 roundtrip, scan, nvfp4 swizzle).
# Plain-CUDA port (hipify + mask-free __shfl_*); no MFMA.
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
all: turboquant_test.out mf_primitives_test.out
%.out: %.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	HIP_VISIBLE_DEVICES=0 ./turboquant_test.out
	HIP_VISIBLE_DEVICES=0 ./mf_primitives_test.out
clean:
	rm -f *.out
.PHONY: all test clean
