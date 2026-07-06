# CDNA3 linear-attention TM kernels (GDN / gated-deltanet + linear attention +
# chunked scan + complex matmul). Plain-CUDA port (hipify + mask-free __shfl_*).
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
all: gdn_test.out linattn_test.out decay_linear_attn.out
%.out: %.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	HIP_VISIBLE_DEVICES=0 ./gdn_test.out
	HIP_VISIBLE_DEVICES=0 ./linattn_test.out
	HIP_VISIBLE_DEVICES=0 ./decay_linear_attn.out
clean:
	rm -f *.out
.PHONY: all test clean

decay_linear_attn.out: decay_linear_attn.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@
