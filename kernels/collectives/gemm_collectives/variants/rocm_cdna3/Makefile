# CDNA3 fused collective+GEMM. The correct multi-GPU run is one process per GPU
# via torchrun (torch.distributed NCCL/RCCL backend); the C++ files are reference.
GPUS ?= 0,1,2,3,4,5,6,7
NPROC ?= 8
PY ?= $(HOME)/.venvs/rocm-torch/bin/torchrun
FP8_GPUS ?= 0,1
FP8_NPROC ?= 2
FP8_PORT ?= 29534
FP8_M ?= 256
FP8_N ?= 256
FP8_K ?= 512
ROCM_TORCHRUN ?= $(HOME)/QuixiCore/QuixiCore-ROCm/.venv/bin/torchrun
test:
	HIP_VISIBLE_DEVICES=$(GPUS) $(PY) --nproc_per_node=$(NPROC) --master_port=29520 torch_gemm_collectives.py
test-fp8:
	HIP_VISIBLE_DEVICES=$(FP8_GPUS) $(ROCM_TORCHRUN) --nproc_per_node=$(FP8_NPROC) --master_port=$(FP8_PORT) torch_gemm_collectives_fp8.py $(FP8_M) $(FP8_N) $(FP8_K)
# reference single-process C++ (see README: single-process-multi-device is fragile)
HIPCC ?= /opt/rocm/bin/hipcc
cxx:
	$(HIPCC) -std=c++17 -O3 --offload-arch=gfx942 -I/opt/rocm/include/rccl gemm_collectives.cpp -lrccl -o gc.out
clean:
	rm -f *.out
.PHONY: test test-fp8 cxx clean
