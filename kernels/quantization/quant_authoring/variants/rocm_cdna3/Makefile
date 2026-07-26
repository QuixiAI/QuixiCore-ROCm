# CDNA3 Phase 6 quant authoring, fake-quant, embedding, and MXFP4 parity ports.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: quant_authoring.out

quant_authoring.out: quant_authoring.cu ../../../../common/cdna3_harness.cuh ../../../qgemv/variants/rocm_cdna3/quant_formats.cuh ../../../qgemv/variants/rocm_cdna3/quant_formats_tables.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./quant_authoring.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./quant_authoring.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
