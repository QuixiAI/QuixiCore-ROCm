GPU_TARGET ?= CDNA3
COMP_LEVEL ?= profile
BUILD_MODE := standalone
TARGET ?= tk_kernel
SRC ?= 4_wave.cu

EXTRA_HIPFLAGS += -fopenmp -Wl,--allow-multiple-definition
EXTRA_HIPFLAGS += -Rpass-analysis=kernel-resource-usage

include ../../../../../common.mk
