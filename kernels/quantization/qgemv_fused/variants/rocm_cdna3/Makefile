# CDNA3 fused packed-Q4_0 decode GEMV variants.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: qgemv_fused.out

qgemv_fused.out: qgemv_fused.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	./qgemv_fused.out

bench: all
	./qgemv_fused.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
