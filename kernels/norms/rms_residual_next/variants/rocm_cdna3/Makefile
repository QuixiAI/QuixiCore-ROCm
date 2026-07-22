# CDNA3 rms_residual_next: fused residual-add + double RMSNorm (register-cached).
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: rms_residual_next.out

rms_residual_next.out: rms_residual_next.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./rms_residual_next.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./rms_residual_next.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
