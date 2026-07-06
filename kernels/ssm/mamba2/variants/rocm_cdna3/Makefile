# CDNA3 mamba2 selective-scan (varlen + APC). Plain-CUDA port.
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
all: selective_scan_test.out mamba2_ssd.out
selective_scan_test.out: selective_scan_test.cu selective_scan_kernels.cuh tm_warp.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	HIP_VISIBLE_DEVICES=0 ./selective_scan_test.out
	HIP_VISIBLE_DEVICES=0 ./mamba2_ssd.out
clean:
	rm -f *.out
.PHONY: all test clean

mamba2_ssd.out: mamba2_ssd.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@
