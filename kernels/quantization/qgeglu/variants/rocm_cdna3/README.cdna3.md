# Quantized gated GeGLU — CDNA3 (gfx942) fused Q4_0 MFMA

Shape/format-named port of embeddinggemma.c `q4_mfma_up_gate_gelu_f16_kernel`
(`~/embeddinggemma-bench/src/engine_rocm.hip`). One wavefront computes **two**
weight-only Q4_0 projections of the same fp16 activation — `up` and `gate` —
accumulating both with `v_mfma_f32_16x16x16_f16` (Q4_0 dequantized to fp16, the
same MFMA primitive as `qgemm`/`qflux`), then applies the gated GeGLU epilogue
`gelu_tanh(gate) * up` in registers before a single fp16 store.

This fuses what the existing epilogue kernels do not:

- `qflux` — single quantized projection + `gelu + bias` (one GEMM).
- `flux` `flux_gate` — dense GEMM times a *precomputed* elementwise gate.
- **`qgeglu` (this) — dual quantized projection where the GELU is applied to one
  of the two on-chip products.** Neither intermediate round-trips device memory.

## Contract

- `Wup, Wgate (N,K)` Q4_0 blocks; `X (M,K)` fp16 activations, token-major.
- `Y (M,N)` fp16, `Y[m,n] = gelu_tanh(gate[m,n]) * up[m,n]`.
- 16x16 output tile per wave64; edge rows/tokens masked. tanh-approx GELU.

## Files

- `qgeglu.cu` — self-contained: fused kernel, the unfused baseline pieces (two
  MFMA projections + a gelu-mul over fp32 buffers), fp64 host reference,
  correctness check, and the fused-vs-unfused A/B (`--bench`).

## Build / run

```bash
make test     # correctness vs fp64 host ref
make bench    # fused vs unfused (two proj + gelu-mul)
```

## Result (MI300X, gfx942, ROCm/HIP 7.2)

Correctness PASS, rel ~3.3e-4 vs the fp64 host reference (fp16 MFMA accumulation);
the fused and unfused paths are numerically identical.

Focused A/B (HIP-event median, warmup 10 / iters 50), N=1152 (n_ff), K=768
(n_embd); FLOPs counted for both projections:

| M (tokens) | unfused (base) | fused (cand) | speedup | decision |
|---:|---:|---:|---:|---|
| 64 | 6.25 TFLOP/s | 7.45 TFLOP/s | 1.19x | KEEP |
| 256 | 17.57 TFLOP/s | 21.68 TFLOP/s | 1.23x | KEEP |

KEEP: fusion removes the two fp32 intermediate round-trips through device memory
for a 1.19–1.23x win at identical numerics. Follow-up: LDS-stage the shared X
fragment across the up/gate MFMA chains, wider N-tiles (mirror the qgemm/qflux
wide-N X-reuse win).
