# CDNA3 Phase 9 state-space and hyper-connection parity ports.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: phase9_ssm.out

phase9_ssm.out: phase9_ssm.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./phase9_ssm.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./phase9_ssm.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
