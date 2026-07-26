# CDNA3 Phase 11 elementwise and tensor-op parity ports.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: phase11_tensor_ops.out

phase11_tensor_ops.out: phase11_tensor_ops.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./phase11_tensor_ops.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./phase11_tensor_ops.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
