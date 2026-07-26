# CDNA3 mean_pool_rms_l2: bf16 mean-pool -> RMSNorm -> L2.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: mean_pool_rms_l2.out

mean_pool_rms_l2.out: mean_pool_rms_l2.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	./mean_pool_rms_l2.out

bench: all
	./mean_pool_rms_l2.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
