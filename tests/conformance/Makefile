# Conformance of this backend's decoders against the umbrella test vectors.
OFFLOAD_ARCH ?= gfx942
HIPCC ?= /opt/rocm/bin/hipcc
QC_INCLUDE ?= ../../include/cdna3/common
VECTORS ?= ../../../test-vectors/quant
HIPFLAGS ?= -std=c++17 -O2 --offload-arch=$(OFFLOAD_ARCH) -I$(QC_INCLUDE)

all: quant_conformance.out

quant_conformance.out: quant_conformance.cu $(QC_INCLUDE)/quant_primitives.cuh
	$(HIPCC) $(HIPFLAGS) $< -o $@

test: all
	HIP_VISIBLE_DEVICES=0 ./quant_conformance.out $(VECTORS)

clean:
	rm -f *.out

.PHONY: all test clean
