# CDNA3 Phase 10 training, distillation, and optimizer parity ports.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: phase10_training.out

phase10_training.out: phase10_training.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./phase10_training.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./phase10_training.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
