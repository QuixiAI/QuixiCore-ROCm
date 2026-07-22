# CDNA3 qgemm_q4q8: Q4_0-weight x Q8_0-activation sdot4 prefill GEMM.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: qgemm_q4q8.out

qgemm_q4q8.out: qgemm_q4q8.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./qgemm_q4q8.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./qgemm_q4q8.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
