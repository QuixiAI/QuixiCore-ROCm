GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: int8_gemm.out

int8_gemm.out: int8_gemm.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: int8_gemm.out
	HIP_VISIBLE_DEVICES=0 ./int8_gemm.out 256 256 512

bench: int8_gemm.out
	HIP_VISIBLE_DEVICES=0 ./int8_gemm.out 512 512 1024

clean:
	rm -f int8_gemm.out

.PHONY: all test bench clean
