# Q4_0 x Q8_0 integer GEMM — CDNA3 (gfx942) sdot4

Shape/format-named port of the embeddinggemma.c `q4_q8_projection` path
(`~/embeddinggemma-bench/src/engine_rocm.hip`: `q4_q8_dot` /
`q4_q8_projection_kernel`). This is the llama.cpp **Q4_0 weight x Q8_0
activation** route, distinct from the sibling quantized GEMMs:

- `qgemm` — Q4_0 dequantized to fp16, fp16 activations, `v_mfma_f32_16x16x16_f16`.
- `qgemm_int` — W8A8 / W8A8-AZP / BitNet W2A8 (8-bit or 2-bit weights).
- **`qgemm_q4q8` (this) — 4-bit packed Q4_0 weight (−8 zero point, per-block
  fp16 scale) x int8 Q8_0 activation (per-block fp16 scale + int16 code sum),
  integer dot via gfx942 `v_dot4` (sdot4).**

## Format / algorithm

- `block_q4_0` = fp16 `d` + 16 bytes (32 nibbles, value = `d*(code−8)`), 18 B.
- `activation_q8` = fp16 `d` + int16 `sum` + 32 int8, 36 B. `sum` is the block
  code sum, precomputed so the Q4_0 `−8` zero point is corrected once per block:
  `dot -= 8 * q8.sum` (avoids re-adding 8 to every nibble).
- Per output: one 32-lane wavefront strides over the row's blocks; per block four
  `sdot4` pairs (low/high nibbles vs the two int8 halves), then scale by
  `q4.d * q8.d` and a wavefront sum. Layout is token-major:
  `W (N,K)`, `X (M,K)`, `Y (M,N) f32`; K divisible by 32.

## Files

- `qgemm_q4q8.cu` — self-contained: kernels, host quantizers, fp64 reference,
  correctness check, and the sdot4-vs-float-dequant A/B (`--bench`).

## Build / run

```bash
make test     # correctness vs fp64 host ref (ragged + FFN/QKV/decode shapes)
make bench    # sdot4 candidate vs float-dequant baseline
```

## Result (MI300X, gfx942, ROCm/HIP 7.2)

Correctness PASS on all shapes, rel ~6e-8 vs the fp64 host reference (the device
integer math is exact; only fp16 scale rounding differs).

Focused A/B (HIP-event median, warmup 10 / iters 50), candidate = sdot4 integer
path, baseline = float-dequant scalar dot (same launch geometry):

| Shape (N,K,M) | baseline | candidate (sdot4) | speedup | decision |
|---|---:|---:|---:|---|
| 1152,768,64 (FFN up/gate prefill) | 5.69 TOPS | 8.46 TOPS | 1.49x | KEEP |
| 768,1152,64 (down proj) | 5.39 TOPS | 8.46 TOPS | 1.57x | KEEP |
| 768,768,1 (decode) | 0.0059 ms | 0.0058 ms | 1.02x | KEEP (mem-bound parity) |

KEEP: the sdot4 integer route is 1.49–1.57x over float-dequant at prefill and
neutral at the memory-bound decode singleton, at exact correctness. Follow-up:
MFMA-tile the int8 path (`v_mfma_i32_16x16x16_i8`) for larger token blocks.
