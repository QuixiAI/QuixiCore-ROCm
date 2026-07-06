# CDNA3 collectives via RCCL (NCCL-API compatible). Replaces NVIDIA multimem/NVLS.
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 -I/opt/rocm/include/rccl
all: rccl.out
rccl.out: rccl_collectives.cpp
	$(HIPCC) $(HIPFLAGS) $< -lrccl -o $@
test: all
	HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./rccl.out
clean:
	rm -f *.out
.PHONY: all test clean
