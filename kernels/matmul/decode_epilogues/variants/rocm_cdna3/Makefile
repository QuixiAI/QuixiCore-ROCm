# CDNA3 Phase 3 decode/epilogue matmul ports.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: decode_epilogues.out

decode_epilogues.out: decode_epilogues.cu \
	../../../../common/cdna3_harness.cuh \
	../../../../quantization/qgemv/variants/rocm_cdna3/quant_formats.cuh \
	../../../../quantization/qgemv/variants/rocm_cdna3/quant_formats_tables.cuh \
	../../../../quantization/qgemv/variants/rocm_cdna3/quant_tables.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./decode_epilogues.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./decode_epilogues.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
