# CDNA3 fused collective+GEMM (correct semantics: RCCL collective o local GEMM).
# NOTE: multi-GPU run requires a clean RCCL environment (no stuck comms).
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -I/opt/rocm/include/rccl
all: gemm_ar.out gemm_collectives.out
gemm_ar.out: gemm_ar.cpp
	$(HIPCC) $(HIPFLAGS) $< -lrccl -o $@
gemm_collectives.out: gemm_collectives.cpp
	$(HIPCC) $(HIPFLAGS) $< -lrccl -o $@
test: all
	HIP_VISIBLE_DEVICES=0,1,2,3 ./gemm_ar.out
clean:
	rm -f *.out
.PHONY: all test clean
