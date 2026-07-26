# CDNA3 Phase 7 sampling, embedding, and root-collective stragglers.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: phase7_stragglers.out

phase7_stragglers.out: phase7_stragglers.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./phase7_stragglers.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./phase7_stragglers.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
