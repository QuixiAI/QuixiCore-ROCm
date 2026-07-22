# CDNA3 pool_mean_rms_l2: sentence-embedding pooling head
# (masked mean-pool of per-token RMSNorm, then L2-normalize).
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: pool_mean_rms_l2.out

pool_mean_rms_l2.out: pool_mean_rms_l2.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./pool_mean_rms_l2.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./pool_mean_rms_l2.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
