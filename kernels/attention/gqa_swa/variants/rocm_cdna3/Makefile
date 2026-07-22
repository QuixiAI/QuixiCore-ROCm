# CDNA3 gqa_swa: symmetric sliding-window GQA flash attention.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: gqa_swa.out

gqa_swa.out: gqa_swa.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./gqa_swa.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./gqa_swa.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
