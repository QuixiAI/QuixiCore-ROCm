# Self-test for the shared CDNA3 kernel harness (cdna3_harness.cuh).
# Standalone and torch-free, like every kernel harness in this repo.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

# Inherited from the environment when set, so a run can be pinned to an idle GPU.
HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: harness_selftest.out

harness_selftest.out: harness_selftest.cu cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	./harness_selftest.out

bench: all
	./harness_selftest.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
