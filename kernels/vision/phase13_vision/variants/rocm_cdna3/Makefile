# CDNA3 Phase 13 vision parity ports.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: phase13_vision.out

phase13_vision.out: phase13_vision.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./phase13_vision.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./phase13_vision.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
