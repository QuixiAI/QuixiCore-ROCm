# CDNA3 Phase 5 BaseQ canonical-family ports.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: base_q.out

base_q.out: base_q.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./base_q.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./base_q.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
