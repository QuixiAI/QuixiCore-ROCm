# CDNA3 TurboQuant query transform: sign flip, unnormalized FWHT, 1/sqrt(hs).
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value
HIP_VISIBLE_DEVICES ?= 0
export HIP_VISIBLE_DEVICES

all: turboquant_query_transform.out
turboquant_query_transform.out: turboquant_query_transform.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	./turboquant_query_transform.out
clean:
	rm -f *.out
.PHONY: all test clean
