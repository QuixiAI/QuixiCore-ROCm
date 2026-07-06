# CDNA3 GQA attention backward (dQ/dK/dV). Correct wavefront/online kernels +
# standalone fp64 analytic oracle. Non-causal by default; -DCAUSAL_=1 for causal.
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
CAUSAL ?= 1
all: attn_bwd.out
attn_bwd.out: attn_bwd.cu
	$(HIPCC) $(HIPFLAGS) -DCAUSAL_=$(CAUSAL) $< -o $@
test: all
	HIP_VISIBLE_DEVICES=0 ./attn_bwd.out
clean:
	rm -f *.out
.PHONY: all test clean
