# Fused collective+GEMM — CDNA3 (gfx942): semantics implemented; multi-GPU run pending

Correct ROCm semantics for the CUDA parallel/{gemm_ar,ag_gemm,gemm_rs} kernels,
which fuse NVIDIA multimem (NVLS) collectives with a GEMM. gfx942 has no NVLS, so
these compose RCCL (the correct collective) with a local GEMM:

- `gemm_ar`  : K-parallel GEMM + ncclAllReduce  (Y = allreduce(A_r@B_r) = A@B)
- `ag_gemm`  : ncclAllGather(A shards) + full GEMM
- `gemm_rs`  : K-parallel GEMM + ncclReduceScatter over M

Correctness rests on composition of two independently-verified pieces: the plain
RCCL collectives (`../../all_reduce/variants/rocm_cdna3`, verified all_reduce +
reduce_scatter across 8 MI300X) and a local GEMM. `gemm_ar.cpp` is the clean
standalone; `gemm_collectives.cpp` bundles all three.

STATUS: the live multi-GPU run in this session was left BLOCKED — repeated
RCCL launches that timed out wedged the multi-GPU RCCL state (stuck comms /IPC).
Re-running needs a clean process environment (fresh shell/container, or a GPU
reset) to clear the stuck state; the code itself is unchanged from the verified
composition. The real remaining work is compute/comm OVERLAP (streamed GEMM
tiles or the repo's Iris/XGMI framework), which is the perf follow-up (task #12).
