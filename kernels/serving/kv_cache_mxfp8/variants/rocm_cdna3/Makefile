# CDNA3 MXFP8 paged KV-cache codec: scatter, gather.
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: kv_cache_mxfp8.out paged_attention_mxfp8.out

kv_cache_mxfp8.out: kv_cache_mxfp8.cu mxfp8_common.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

paged_attention_mxfp8.out: paged_attention_mxfp8.cu mxfp8_common.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	./kv_cache_mxfp8.out
	./paged_attention_mxfp8.out

clean:
	rm -f *.out

.PHONY: all test clean
