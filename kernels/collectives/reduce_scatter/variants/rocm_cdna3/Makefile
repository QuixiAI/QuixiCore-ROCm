GPUS ?= 0,1
NPROC ?= 2
PORT ?= 29533
TORCHRUN ?= $(HOME)/QuixiCore/QuixiCore-ROCm/.venv/bin/torchrun
N ?= 1048576

test:
	HIP_VISIBLE_DEVICES=$(GPUS) $(TORCHRUN) --nproc_per_node=$(NPROC) --master_port=$(PORT) torch_reduce_scatter.py $(N)

clean:
	rm -f *.out

.PHONY: test clean
