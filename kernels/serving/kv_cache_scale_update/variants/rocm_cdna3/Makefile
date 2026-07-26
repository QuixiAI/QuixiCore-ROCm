# CDNA3 running KV-cache scale update (amax/240, monotone max against old).
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value
HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: kv_cache_scale_update.out
kv_cache_scale_update.out: kv_cache_scale_update.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	./kv_cache_scale_update.out
clean:
	rm -f *.out
.PHONY: all test clean
