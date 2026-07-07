# CDNA3 flux: dense bf16 matmul + fused epilogue (gelu+bias / gate) via MFMA.
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
all: flux.out
flux.out: flux.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	HIP_VISIBLE_DEVICES=0 ./flux.out
clean:
	rm -f *.out
.PHONY: all test clean

bench:
	$(HIPCC) $(HIPFLAGS) flux_bench.cu -o flux_bench.out
	HIP_VISIBLE_DEVICES=0 ./flux_bench.out 2048 2048 2048
	HIP_VISIBLE_DEVICES=0 ./flux_bench.out 4096 4096 4096
.PHONY: bench
