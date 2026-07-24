# CDNA3 runtime quantization group variants.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: quant_rt_group.out

quant_rt_group.out: quant_rt_group.cu ../../../qgemm/variants/rocm_cdna3/quant_formats.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./quant_rt_group.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./quant_rt_group.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
