# CDNA3 followups: cross-cutting integration test (turboquant encode,
# selective_scan APC, EAGLE, sparse serving, merge_attn_states). Reuses the
# already-ported serving/mamba2/turboquant headers.
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
all: followups_test.out
followups_test.out: followups_test.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	HIP_VISIBLE_DEVICES=0 ./followups_test.out
clean:
	rm -f *.out
.PHONY: all test clean
