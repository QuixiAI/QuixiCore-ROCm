# Collectives — CDNA3 (gfx942): RCCL path

The CUDA `parallel/*` kernels (all_reduce, all_gather, reduce_scatter, all_to_all,
+ fused ag_gemm / gemm_rs / gemm_ar / moe_dispatch / ring_attn / ulysses_attn)
use NVIDIA **multimem** (NVLink multicast / NVLS) and device-side cross-GPU
barriers over a multicast pointer — hardware CDNA3/MI300X does not have.

The correct ROCm mapping:
- **Plain collectives** (all_reduce/all_gather/reduce_scatter/all_to_all) ->
  **RCCL** (NCCL-API compatible, `librccl`). Demonstrated here: `rccl_collectives.cpp`
  runs all_reduce + reduce_scatter (sum) across all 8 MI300X and verifies the
  analytic result. ALL PASS.
- **Fused collective+GEMM / device-initiated one-sided** (ag_gemm, gemm_rs,
  gemm_ar, moe_dispatch_gemm, ring/ulysses attention) -> the repo's **Iris**
  framework (`distributed-kernels/`, XGMI peer access) for compute/comm overlap,
  or RCCL + streamed GEMM tiles. These need a multi-GPU perf design and are the
  distributed follow-up.

    make test    # all_reduce + reduce_scatter across 8 GPUs
