# CDNA3 serving family: standalone self-checking HIP harnesses.
# Ported from ../QuixiCore-CUDA/kernels/serving (plain-CUDA tm_* kernels) via
# hipify + CDNA3 warp adaptations (mask-free __shfl_*, __dp4a->sdot4 for the
# fp8-KV dequant path). Each harness generates its own data and checks against
# an in-process fp64/replay reference (no golden files).
#
#   make          # build all harnesses
#   make test     # run all harnesses on GPU 0
GPU_TARGET ?= CDNA3
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=$(OFFLOAD_ARCH) -Wno-unused-value

HARNESSES = attn_q attn_varlen beam_xcache eagle_test kv_cache logits_proc_test \
            mla paged_attn_v2 rope_kv sampling sparse_serving_test spec_beam

OUTS = $(addsuffix .out,$(HARNESSES))

all: $(OUTS)

%.out: %.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	@for h in $(HARNESSES); do echo "== $$h =="; HIP_VISIBLE_DEVICES=0 ./$$h.out || exit 1; done

clean:
	rm -f *.out

.PHONY: all test clean
