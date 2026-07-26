# CDNA3 attn_fwd_sg_d256: fp32 Q/O, fp16 K/V, bidirectional GQA.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: attn_fwd_sg_d256.out

attn_fwd_sg_d256.out: attn_fwd_sg_d256.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	./attn_fwd_sg_d256.out

bench: all
	./attn_fwd_sg_d256.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
