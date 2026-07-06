# MoE family — CDNA3 (gfx942)

Native CDNA3 port of `../QuixiCore-CUDA/kernels/moe` (plain-CUDA `tm_moe`
routing + grouped-GEMM MoE MLP). QuixiCore-internal port. Self-checking harness
(fp64 references, no golden files).

## Kernels

top-k routing (`moe_route_topk`), expert histogram + offset scan, token
scatter/gather (+inverse, +zero-padded), pad-offset build, grouped GEMM, and the
end-to-end MoE MLP.

## Port notes (CUDA -> CDNA3)

- `hipify-perl`: headers, `__nv_bfloat16` -> `__hip_bfloat16`, runtime API.
- `__shfl_*_sync(0xffffffffu, ...)` -> mask-free `__shfl_*` (64-bit-mask rule on
  the 64-wide wavefront; reductions stay within 32-lane groups).

`moe_quant` (fp8/nvfp4 expert weights) is **not** here: it depends on the
tensor-core `mma16816` primitive in `tm_qmm.cuh` (PTX `mma.sync`) and is ported
with the MFMA pass alongside qgemm.

## Build / run

```bash
make test    # -> "ALL PASS (0 failures)"
```

## Result (MI300X, 2026-07-06)

All 8 checks pass, incl. end-to-end MoE MLP vs fp64 (err 5.9e-07) and grouped
GEMM (err 4.0e-07). Raw: `perf/results/2026-07-06/moe/`.
