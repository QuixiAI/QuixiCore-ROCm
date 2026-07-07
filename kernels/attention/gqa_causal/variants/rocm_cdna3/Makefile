GPU_TARGET ?= CDNA3
TARGET ?= tk_kernel
PYTHON ?= python3.12

ATTN_B ?= 16
ATTN_H ?= 64
ATTN_H_KV ?= 8
ATTN_N ?= 2048
ATTN_D ?= 128

CPPFLAGS += -DATTN_B=$(ATTN_B)
CPPFLAGS += -DATTN_H=$(ATTN_H)
CPPFLAGS += -DATTN_H_KV=$(ATTN_H_KV)
CPPFLAGS += -DATTN_N=$(ATTN_N)

ifeq ($(ATTN_D),64)
SRC ?= kernel_d64.cpp
else
SRC ?= kernel.cpp
endif

EXTRA_HIPFLAGS += -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
EXTRA_ICXXFLAGS += --save-temps

include ../../../../common.mk
CPPFLAGS += -DATTN_CAUSAL=1

# --- standalone MFMA-attention correctness + perf harnesses (self-checking) ---
SA_FLAGS := -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
BENCH_SHAPE := -DATTN_CAUSAL=1 -DATTN_B=4 -DATTN_H=32 -DATTN_H_KV=8 -DATTN_N=2048 -DATTN_D=128
test:
	/opt/rocm/bin/hipcc $(SA_FLAGS) attn_mfma.cu -o attn_mfma.out
	HIP_VISIBLE_DEVICES=0 ./attn_mfma.out
bench:
	/opt/rocm/bin/hipcc $(SA_FLAGS) $(BENCH_SHAPE) attn_bench.cu -o attn_bench.out
	HIP_VISIBLE_DEVICES=0 ./attn_bench.out
.PHONY: test bench
