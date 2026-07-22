# Symmetric sliding-window GQA — CDNA3 (gfx942) MFMA flash

Shape-named port of embeddinggemma.c `mfma_attention_f16_kernel`
(`~/embeddinggemma-bench/src/engine_rocm.hip`), which the source itself notes was
"adapted from QuixiCore's BQ=BK=16 GQA kernel." The specialization this adds over
the repo's `gqa` / `gqa_causal` forward is a **symmetric (bidirectional,
centered) sliding window**: an encoder-style query at position `q` attends keys
in `[q − window/2, q + window/2]` with **no causal mask**. `window == 0` recovers
full bidirectional attention. This is the EmbeddingGemma / Gemma-encoder attention
shape; `gqa` only exposed `ATTN_CAUSAL` (backward-looking), so the centered band
was previously unreachable.

## Approach

Same online-softmax MFMA flash as the shipped `gqa` kernel: one wave64 per
(16-query block, head); K/V reused across the block; `QK^T` and `P@V` on
`v_mfma_f32_16x16x16_f16`; softmax over an LDS transpose of the scores. GQA with
a single shared KV head (`H_KV = 1`, the embeddinggemma layout). The window logic
is two-level: a per-tile `[union_first, union_last)` KV bound prunes whole 16-key
tiles, and a per-query `[first, last)` band masks inside the tile — so at long
context only the banded tiles are visited. Q is expected pre-scaled by
`1/sqrt(D)` (Gemma folds the attention scale into Q).

Layout: Q/O `[T, H*D]` (head-major within a token), K/V `[T, D]` (shared KV head).

## Files

- `gqa_swa.cu` — self-contained: kernel, fp64 host reference with the symmetric
  window mask, correctness check, and the full-vs-windowed A/B (`--bench`).

## Build / run

```bash
make test     # correctness vs fp64 host ref (full + windowed, ragged + aligned T)
make bench    # full (window=0) vs symmetric-window bands at T=2048
```

## Result (MI300X, gfx942, ROCm/HIP 7.2, D=256, H=3, H_KV=1)

Correctness PASS on all shapes, rel ~1.4e-4 to 1.9e-4 vs the fp64 host reference
(full attention and windows 16/128/256/512; ragged T=37 and aligned T=512).

Focused A/B (HIP-event median, warmup 10 / iters 50), T=2048:

| window | time | vs full | decision |
|---:|---:|---:|---|
| 0 (full O(T^2)) | 1.3639 ms | 1.0x | baseline |
| 256 | 0.1995 ms | 6.84x | KEEP |
| 512 | 0.3764 ms | 3.62x | KEEP |

KEEP: the symmetric-window specialization keeps flash-attention numerics
identical inside the band while pruning the out-of-band KV tiles, for a 3.6–6.8x
reduction at 2K context (grows with T). Follow-up: LDS-stage K/V, larger query
blocks, and a packed-QKV entry to match the embeddinggemma fused front end.
