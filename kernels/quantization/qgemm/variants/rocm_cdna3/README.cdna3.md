# Quantized tensor-core GEMM — CDNA3 (gfx942) MFMA

Native CDNA3 port of the tensor-core quantized-matmul path from
`../QuixiCore-CUDA/kernels/quant`. The CUDA version used the NVIDIA PTX
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (32-lane fragment layout in
`tm_qmm.cuh`); this rewrites it to gfx942 `v_mfma_f32_16x16x16_f16` on a full
64-wide wavefront (`tm_qmm_mfma.cuh`).

## What's here

- `tm_qmm_mfma.cuh` — the MFMA primitive + fragment loaders (the validated
  replacement for the CUDA `mma16816`/`load_wfrag`/`load_xfrag`).
- `qgemm.cu` — weight-only quantized GEMM `Y = X @ dequant(W)^T`, fragment path
  + full-dequant route for 256-superblock k/i-quants + K-split, wide-N, and
  prefill-scale CTA-LDS variants.
- `qgemm_variants.cu` — GPTQ act-order and fp8_raw block-scale variants.
- `qflux.cu` — fused `gelu_tanh(X @ dequant(W)^T + bias)`.

## MFMA lane layout (v_mfma_f32_16x16x16_f16, 64 lanes, 4 regs/lane)

```
A[M=16,K=16] : lane l, reg v -> A[m = l%16][k = 4*(l/16) + v]
B[K=16,N=16] : lane l, reg v -> B[k = 4*(l/16) + v][n = l%16]
D[M=16,N=16] : lane l, reg v -> D[m = 4*(l/16) + v][n = l%16]
```

For `Y = X @ W^T` we set A=X and B=W^T, so each lane reads 4 contiguous K of one
X row and 4 contiguous K of one W row (n = n0 + l%16), does one MFMA per K=16
step, and owns output column n over 4 rows. This is simpler than the PTX path
(one 16x16 MFMA vs two 16x8 mma; per-lane single bias in qflux vs four). The 4
per-lane K values (offsets 0/4/8/12) stay inside one quant block for every
format (all block_k are multiples of 16).

## Port notes

- fragment loaders + the two-mma 16x16 tile -> single `mma_16x16x16`; launch
  geometry 32 threads -> 64 (full wavefront); MFMA accumulator store layout.
- `__half` and `__fp16` share IEEE fp16 bits, so the X fragment is a raw aligned
  `half4` reinterpret; W is dequantized to float then narrowed to `__fp16`.
- Golden `meta.txt` fallback (dir basename + N=512/K=4096) as in the qgemv port.

## Build / run

```bash
make test    # qgemm + qflux over all 29 golden formats vs Y_ref / Yflux_ref
```

## Result (MI300X, 2026-07-06/07)

- `qgemm`: PASS across the golden format matrix for base, K-split, wide-N, and
  CTA-LDS variants vs `Y_ref`, ~0.017-0.02% rel err. Fragment path ~1.8 TFLOP/s
  at M=64 N=512 K=4096; K-split ~10-13 TFLOP/s (fills the device at this decode
  shape). CTA-LDS is a prefill-scale candidate: it loses at decode/small M but
  reaches ~86-91 TFLOP/s on fp16_raw M>=2048,N=K=4096 by reusing W across four
  wavefronts; packed q8_0/q4_0 M=1024,N=K=4096 also confirm the win. Broader
  format routing and qflux mirroring are still deferred.
- `qflux`: 29/29 PASS vs `Yflux_ref` (fused GELU+bias).
- `qgemm_variants`: act-order q4_0 PASS vs fp64 gathered reference
  (8.628e-07 rel); blockscale fp8_raw PASS vs fp64 tiled-scale reference
  (1.982e-05 rel).

Follow-ups (still on task #8): port the remaining tensor-core consumers that
reuse this primitive — `lm_head`(+topkp), `moe_quant` (grouped quantized GEMM),
`followups_test`, `mf_primitives_test`, `turboquant`. Perf: LDS staging /
cp.async-equivalent prefetch, wider output tiles, bf16 MFMA variant.
