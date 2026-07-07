# CDNA3 fused collective+GEMM. The correct multi-GPU run is one process per GPU
# via torchrun (torch.distributed NCCL/RCCL backend); the C++ files are reference.
GPUS ?= 0,1,2,3,4,5,6,7
NPROC ?= 8
PY ?= $(HOME)/.venvs/rocm-torch/bin/torchrun
test:
	HIP_VISIBLE_DEVICES=$(GPUS) $(PY) --nproc_per_node=$(NPROC) --master_port=29520 torch_gemm_collectives.py
# reference single-process C++ (see README: single-process-multi-device is fragile)
HIPCC ?= /opt/rocm/bin/hipcc
cxx:
	$(HIPCC) -std=c++17 -O3 --offload-arch=gfx942 -I/opt/rocm/include/rccl gemm_collectives.cpp -lrccl -o gc.out
clean:
	rm -f *.out
.PHONY: test cxx clean
