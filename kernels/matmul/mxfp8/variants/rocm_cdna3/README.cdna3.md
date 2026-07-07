# MXFP8 GEMM — CDNA3 (gfx942)

Correctness-first dense MXFP8 GEMM for the repository quant format:

```text
C[M, N] = dequant(A_mxfp8[M, K]) @ dequant(B_mxfp8[N, K])^T
```

Each K block uses the existing `tmq::mxfp8` layout: one e8m0 scale byte and
32 e4m3 codes. The harness validates both fused decode and explicit dequant +
fp32 GEMM against a host double reference, then times both routes.

Current measured small-shape result keeps explicit dequant + fp32 GEMM as the
baseline CDNA3 route.
