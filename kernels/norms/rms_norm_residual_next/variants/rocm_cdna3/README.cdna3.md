# rms_norm_residual_next - CDNA3

Port of the current QuixiCore-Metal `kernels/norms/rms_norm_residual_next`
contract. It is the bf16 public variant of the residual-stream seam and is
separate from the older ROCm `rms_residual_next` embeddinggemma harness, which
is shape-locked to 768-wide fp32 residual storage.

Contract:

```text
x, residual    : bf16 (M, D)
post_weight    : bf16 (D)
next_weight    : bf16 (D)
res_out        : bf16 (M, D)
next_out       : bf16 (M, D)
D in {256,512,768,1024}

pinv = rsqrt(mean(x^2) + eps)
res = residual + x * pinv * post_weight
rinv = rsqrt(mean(res^2) + eps)
next = res * rinv * next_weight
```

The candidate assigns one wave64 to a row, keeps the unrounded residual in
registers between the two reductions, writes `res_out` as bf16, and computes
`next_out` from the unrounded residual just like the Metal kernel. The harness
compares both outputs against an fp64 oracle and benchmarks against a scalar
one-thread-per-row baseline.
