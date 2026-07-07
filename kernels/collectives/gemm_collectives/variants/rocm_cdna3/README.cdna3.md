# Fused collective+GEMM — CDNA3 (gfx942): correctness-valid (multi-GPU)

Correct ROCm semantics for the CUDA parallel/{gemm_ar,ag_gemm,gemm_rs} kernels,
which fuse NVIDIA multimem (NVLS) collectives with a GEMM. gfx942 has no NVLS, so
these compose an RCCL collective with a local GEMM:

- `gemm_ar`  : K-parallel local GEMM + all_reduce(sum)  -> Y = A@B
- `ag_gemm`  : all_gather(A row-shards) + full GEMM
- `gemm_rs`  : K-parallel local GEMM + reduce_scatter over M

## Validated — one process per GPU (production model)

`torch_gemm_collectives.py` runs all three across the MI300X GPUs via
`torchrun` + `torch.distributed` (NCCL/RCCL backend, torch's bundled RCCL) and
checks each against a single-GPU reference. **gemm_ar / ag_gemm / gemm_rs all
PASS on 4 and 8 GPUs.**

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ~/.venvs/rocm-torch/bin/torchrun --nproc_per_node=8 torch_gemm_collectives.py
```

This is the correct multi-GPU model — one process per GPU, each owning one device
— exactly how PyTorch DDP / Megatron tensor-parallel run.

## Note on the single-process C++ harnesses

`gemm_collectives.cpp` / `gemm_ar.cpp` (RCCL + a HIP GEMM in one process driving
all devices) are kept for reference but are NOT the way to run this: single-
process-multi-device RCCL **deadlocks** when a compute kernel precedes the
collective (single host thread), and the thread-per-device variant hits an
`invalid resource handle` (RCCL comms created in one thread used from another).
The plain collectives (`../../all_reduce/variants/rocm_cdna3`, all_reduce +
reduce_scatter) do pass single-process because they never launch a compute
kernel first — but the general fused pattern needs one process per GPU. RCCL's
first collective on a fresh single-process communicator also has a ~2 min warmup
on this box; torchrun avoids all of that.

## Remaining (perf / scope)

Compute/comm OVERLAP (streamed GEMM tiles, or the repo's Iris/XGMI framework) and
the device-initiated one-sided kernels (ring_attn, ulysses_attn,
moe_dispatch_gemm) are the performance follow-up (task #12). Correctness of the
tensor/sequence-parallel GEMM math is established here.
