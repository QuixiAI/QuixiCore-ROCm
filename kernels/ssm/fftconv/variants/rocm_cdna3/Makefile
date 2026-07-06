# CDNA3 fftconv (complex circular convolution = ifft(fft(u)*fft(k))) + fp64 oracle.
HIPCC ?= /opt/rocm/bin/hipcc
HIPFLAGS ?= -std=c++17 -O3 --offload-arch=gfx942 -Wno-unused-value
all: fftconv.out
fftconv.out: fftconv.cu
	$(HIPCC) $(HIPFLAGS) $< -o $@
test: all
	HIP_VISIBLE_DEVICES=0 ./fftconv.out
clean:
	rm -f *.out
.PHONY: all test clean
