# INT8 GEMM — CDNA3 (gfx942)

Correctness-first dense int8 GEMM for the CUDA `gemm/int8_*` semantic surface:

```text
C[M, N] = A[M, K] @ B[N, K]^T
A, B: signed int8
C: signed int32
```

The kept CDNA3 path uses `__builtin_amdgcn_sdot4` (`V_DOT4`) and is validated
against a scalar int32 reference. The harness also times a scalar GPU baseline
on the same shape.

```bash
make test
make bench
```
