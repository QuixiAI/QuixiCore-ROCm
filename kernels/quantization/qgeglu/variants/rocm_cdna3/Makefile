# CDNA3 qgeglu: quantized gated GeGLU (dual Q4_0 MFMA proj + gelu(gate)*up).
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: qgeglu.out

qgeglu.out: qgeglu.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./qgeglu.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./qgeglu.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
