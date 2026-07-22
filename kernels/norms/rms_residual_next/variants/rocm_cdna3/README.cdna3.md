# Fused residual-add + double RMSNorm — CDNA3 (gfx942) register-cached

Shape-named port of embeddinggemma.c `rms_residual_next_f32_cached` /
`rms_residual_next_f16_cached` (register-cached) and `rms_residual_next_f16`
(general `n_cols`) in `~/embeddinggemma-bench/src/engine_rocm.hip`; the math
matches the CPU oracle `ei_rms_norm_residual_inplace` + `ei_rms_norm` in
`~/embeddinggemma-bench/src/kernels.c`. Matching Metal/CUDA/XPU siblings.

One launch (one 32-lane warp per token row) does, over `n_cols = EI_N_EMBD = 768`:

1. `projected_inv = rsqrt( mean(input²) + eps )`
2. `residual[c] += input[c] · post_weight[c] · projected_inv`  (updated stream)
3. `residual_inv = rsqrt( mean(residual²) + eps )`
4. `next_out[c]  = residual[c] · next_weight[c] · residual_inv`

collapsing the ~4 norm launches per layer to 2. The cached variant keeps the
`EI_N_EMBD/32 = 24` residual values per lane in **registers** between steps 2 and
4, so the updated residual stream feeds the next RMS without a device-memory
round-trip. RMSNorm form matches the model: `out = x · w · rsqrt(mean(x²)+eps)`
(any `(1+γ)` offset folded into the stored weight upstream).

## Caveat (shape-lock)

The register-cache depth (`items_per_lane`) is shape-locked to
`EI_N_EMBD/32 = 24`. Self-parity vs the separate-launch route has a documented
~0.999997 ceiling (accumulation order + f16 output rounding), so validation is
against the fp64 oracle at tolerance, not bitwise.

## Contract

- `input (M,768)` f32 (post-projection), `residual (M,768)` f32 (in/out, mutated),
  `post_weight (768)`, `next_weight (768)` f32.
- `next_output (M,768)` f32 (`_f32_cached`, the next projection input) or f16
  (`_f16_cached`).
- One warp (32 lanes) per token row; `kThreads=256` → 8 rows/block.

## Files

- `rms_residual_next.cu` — self-contained: the f32 and f16 register-cached fused
  kernels, the unfused baseline pieces (`rms_residual_add_kernel` +
  `rms_norm_to_f32/f16_kernel` — the separate-launch route), fp64 host oracle
  (residual + next), correctness + cosine self-parity, and the A/B (`--bench`).
- `Makefile` — `make test` / `make bench`.

## Build / run

```bash
make test     # correctness vs fp64 oracle (residual stream + next input)
make bench    # fused (1 launch) vs unfused (2 separate norm launches)
```

## AGENTS gate record

- GPU: AMD Instinct MI300X (gfx942, sramecc+:xnack-).
- ROCm 7.2.4; HIP 7.2.53211-97f5574fe2; AMD clang 22.0.0git (roc-7.2.4).
- Build: `hipcc -std=c++17 -O3 --offload-arch=gfx942`.
- Measurement: HIP-event median, warmup 10 / iters 50, `HIP_VISIBLE_DEVICES=0`.
  Timed region is pure kernel launches on both sides (no host↔device copies), so
  the ratio reflects the launch + residual DRAM round-trip the fusion removes.
- Command: `make bench`. Run date 2026-07-22.

### Correctness (vs fp64 oracle)

| output | rel | max abs | note |
|---|---:|---:|---|
| fused_f32 residual | ~3.5e-8 | ~9.5e-7 | fp32 machine precision |
| fused_f32 next     | ~4.8e-8 | ~9.5e-7 | fp32 machine precision |
| fused_f16 next     | ~1.76e-4 | ~1.9e-3 | f16 output rounding |

`fused_f32 vs unfused_f32 next` cosine = 1.0000000; `fused_f32 vs oracle`
cosine = 1.0000000 — well within the ~0.999997 self-parity ceiling. Both the f32
and f16 unfused baselines PASS identically (same math, split across two launches).

### Focused A/B (fused 1 launch vs unfused 2 launches), N=768

| M (tokens) | f32 unfused | f32 fused | f32 speedup | f16 speedup | decision |
|---:|---:|---:|---:|---:|---|
| 64   | 0.02331 ms | 0.01166 ms | 2.000x | 1.957x | KEEP |
| 256  | 0.02400 ms | 0.01214 ms | 1.977x | 1.917x | KEEP |
| 512  | 0.02440 ms | 0.01230 ms | 1.984x | 1.896x | KEEP |
| 1024 | 0.02616 ms | 0.01422 ms | 1.839x | 1.813x | KEEP |
| 2048 | 0.03249 ms | 0.01899 ms | 1.711x | 1.717x | KEEP |

KEEP: the fusion halves the launch count (2→1) and removes the residual-stream
DRAM round-trip between the two norms, for ~1.7–2.0x at identical numerics
(cosine 1.0). The ratio is near the 2:1 launch ratio at small M and tapers toward
memory-bandwidth limits at large M (1.71x at M=2048), reported honestly.
