# Fused combined Q/K/V projection — CDNA3 (gfx942) weight-only Q4_0 MFMA

Shape/format-named port of embeddinggemma.c `q4_mfma_qkv_projection_kernel`
(with its `_wide` sibling and `qkv_mfma_target` host dispatch,
`~/embeddinggemma-bench/src/engine_rocm.hip`). One launch computes the Q, K and V
weight-only Q4_0 projections of the **same** fp16 activation, accumulating each
16x16 output tile with `v_mfma_f32_16x16x16_f16` (Q4_0 dequantized to fp16, the
same MFMA primitive as `qgemm`/`qflux`/`qgeglu`).

This is the exact structural analog of `qgeglu` (up+gate fusion) applied to the
attention projection. The distinguishing feature vs `qgeglu`: **heterogeneous
output-row counts** — Q has `EI_N_EMBD = 768` output rows, K and V have
`EI_HEAD_DIM = 256` each — all sharing one activation fragment over the same
`K = 768` contraction. A combined 16-row output tile is routed to whichever of
the three Q4_0 weight matrices it lands in (`qkv_target`, faithful port of
`qkv_mfma_target`); the unfused route is three separate projection launches.

## Combined projection only — no norm / RoPE

This ports the **combined projection only**. Norm and RoPE are deliberately not
folded in: the QKV+norm+RoPE mega-fusion (`EI_ROCM_NATIVE_Q4_DIRECT_FP16_QKV`)
was measured at 0.997–1.008x on CDNA3 — coarse fusion collapses occupancy. The
fine-grained projection fusion below (fewer launches, one shared X load) is the
part that wins.

## Contract

- `Wq (768,768)`, `Wk (256,768)`, `Wv (256,768)` Q4_0 blocks, row-major over the
  output-feature dimension N.
- `X (M,768)` fp16 activations, token-major.
- `Q (M,768)`, `K (M,256)`, `V (M,256)` f32, token-major.
- 16x16 output tile per wave64; combined row space `768 + 2*256 = 1280`; edge
  rows/tokens masked. K, N, M multiples of 16 on the fast path.

## Files

- `qkv_proj_fused.cu` — self-contained: fused combined kernel (`qkv_target`
  routing + one shared X fragment), the unfused baseline piece
  (`q4_mfma_proj_kernel`, run once per Q/K/V), fp64 host reference, correctness
  check, fused-vs-unfused parity, and the A/B (`--bench`).
- `Makefile` — `make test` / `make bench`.

## Build / run

```bash
make test     # correctness vs fp64 host ref (three separate dequant-MFMA proj)
make bench    # fused (1 launch) vs unfused (3 proj launches)
```

## AGENTS gate record

- GPU: AMD Instinct MI300X (gfx942, sramecc+:xnack-).
- ROCm 7.2.4; HIP 7.2.53211-97f5574fe2; AMD clang 22.0.0git (roc-7.2.4).
- Build: `hipcc -std=c++17 -O3 --offload-arch=gfx942`.
- Measurement: HIP-event median, warmup 10 / iters 50, `HIP_VISIBLE_DEVICES=0`.
- Command: `make bench`. Run date 2026-07-22.

### Correctness

vs fp64 host reference (three separate Q4_0 dequant-MFMA projections): all of
Q/K/V PASS at rel ~1.6e-4, max ~1.1e-3 (fp16 MFMA accumulation error, same order
as `qgeglu`). The fused and unfused device paths are **bit-identical**
(max abs diff 0.000e+00) at every M — the fusion reorders launches, not math.

### Focused A/B (fused vs three separate projections)

FLOPs counted for all three projections: `2 · M · 768 · (768 + 2·256)`.

| M (tokens) | unfused (base) | fused (cand) | speedup | decision |
|---:|---:|---:|---:|---|
| 32   | 1.27 TFLOP/s  | 3.55 TFLOP/s  | 2.803x | KEEP |
| 64   | 2.55 TFLOP/s  | 6.42 TFLOP/s  | 2.523x | KEEP |
| 128  | 4.93 TFLOP/s  | 12.20 TFLOP/s | 2.476x | KEEP |
| 256  | 9.65 TFLOP/s  | 16.34 TFLOP/s | 1.693x | KEEP |
| 368  | 11.97 TFLOP/s | 17.59 TFLOP/s | 1.469x | KEEP |
| 512  | 14.13 TFLOP/s | 18.71 TFLOP/s | 1.324x | (above band) |
| 1024 | 18.40 TFLOP/s | 20.75 TFLOP/s | 1.128x | (above band) |

KEEP: in the ~32–368 token native-MFMA band the fusion is 1.47–2.80x — the win
is dominated by collapsing three launches into one (plus one shared X load) while
the tiny per-projection tiles are launch/occupancy-bound. Reported honestly, the
advantage tapers as M grows compute-bound: 1.32x at 512, 1.13x at 1024. Numerics
identical to the separate-projection route. Follow-up (not landed here): the
`_wide` template (multiple output tiles per block for X-reuse), mirroring the
qgemm/qflux wide-N win.
