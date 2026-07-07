# NVFP4 GEMM — CDNA3 (gfx942)

Correctness-first dense NVFP4 GEMM for the repository quant format:

```text
C[M, N] = dequant(A_nvfp4[M, K]) @ dequant(B_nvfp4[N, K])^T
```

Each K block uses the existing `tmq::nvfp4` layout: one e4m3 block scale and
16 e2m1 values packed into 8 bytes. The harness validates fused decode and
explicit dequant + fp32 GEMM against a host double reference, then times both.

This is functional CDNA3 coverage for the CUDA `gemm/nvfp4_b200` semantic gap;
it intentionally does not copy B200 tensor-memory scaled-MMA mechanics.
