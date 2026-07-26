# CDNA3 Phase 12 convolution, pooling, and audio parity ports.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

all: phase12_conv_audio.out

phase12_conv_audio.out: phase12_conv_audio.cu ../../../../common/cdna3_harness.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./phase12_conv_audio.out

bench: all
	HIP_VISIBLE_DEVICES=0 ./phase12_conv_audio.out --bench

clean:
	rm -f *.out

.PHONY: all test bench clean
