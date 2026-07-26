# CDNA3 rms_norm_residual_next: bf16 residual seam.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: rms_norm_residual_next.out

rms_norm_residual_next.out: rms_norm_residual_next.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	./rms_norm_residual_next.out

bench: all
	./rms_norm_residual_next.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
