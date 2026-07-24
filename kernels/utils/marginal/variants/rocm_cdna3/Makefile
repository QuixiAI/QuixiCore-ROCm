# CDNA3 marginal utilities: tau_tail, packbits, segment_packbits, permute_cols.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: marginal.out

marginal.out: marginal.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./marginal.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./marginal.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
