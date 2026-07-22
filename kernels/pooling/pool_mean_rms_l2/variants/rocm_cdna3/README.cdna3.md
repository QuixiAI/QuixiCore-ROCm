# Sentence-embedding pooling head — CDNA3 (gfx942)

Shape-named port of embeddinggemma.c `pool_kernel`
(`~/embeddinggemma-bench/src/engine_rocm.hip`, ~L2463); the fp64 oracle is the
CPU reference `ei_mean_pool_rms_l2` (`src/kernels.c`, ~L417). Named by the
D-vector shape (`D ∈ {256, 512, 768, 1024}`), never by model. QuixiCore-ROCm had
no pooling family; this adds one.

The head that turns a sequence of token embeddings into one sentence vector:

```
for each token t in [start, stop):               # per sequence, masked by offsets
    ss   = sum_d x[t][d]^2
    inv  = 1 / sqrt(ss / D + eps)                 # per-token RMSNorm scale
    m[d] += x[t][d] * inv * w[d]                  # learned RMS gain w
m[d] *= 1 / n_tokens                              # masked mean-pool
l2   = sum_d m[d]^2
y[d] = m[d] * (l2 == 0 ? 1 : rsqrt(l2))           # L2-normalize
```

The RMSNorm is applied **per token, before averaging** (verified against both
the HIP source and the CPU reference — not mean-then-norm). The learned gain `w`
is applied multiplicatively exactly as the reference `ei_norm_scale`
(`out = x*scale*w`); Gemma's stored "(1+w)" gain is baked into the exported
weight upstream at load, so both reference paths use plain `w` and this kernel
carries no `(1+w)`. `eps` is the model `rms_eps` (embeddinggemma default `1e-6`).

Only the plain mean-pool → RMS → L2 head is ported. embeddinggemma's
`final_singleton_pool` fused-singleton epilogue is **intentionally not** ported —
it fails parity on GEMM shapes (cosine 0.770).

## Approach

One **wave64 owns a whole sequence's D-vector**: lane `l` owns dims
`l, l+64, …, l+(D−64)`, i.e. `ITEMS = D/64` registers per lane. Each per-token
sum-of-squares is a single wavefront `shfl_xor` butterfly all-reduce (no LDS, no
block sync), and the normalized rows accumulate into registers across the token
loop, so the token matrix is read exactly once. Loads are 64-wide coalesced
(256 B/wave per item). Sequences are packed with an `offsets[batch+1]` index, so
ragged batches carry no padding.

Layout: input `X [total_tokens, D]` (token-major), `weight [D]`,
`output Y [batch, D]`.

## Files

- `pool_mean_rms_l2.cu` — self-contained: templated candidate kernel, the
  naive-composed baseline (two passes through a global temp), fp64 host oracle
  (= `ei_mean_pool_rms_l2`), correctness sweep, and the fused-vs-naive `--bench`.
- `Makefile` — `make test` (correctness) / `make bench` (A/B).

## Build / run

```bash
make test     # fp64-oracle correctness across D ∈ {256,512,768,1024}
make bench    # fused vs naive-composed A/B on this GPU
```

## Result (MI300X, gfx942, ROCm/HIP 7.2, hipcc 22.0.0git roc-7.2.4)

**Correctness** (`make test`, fp64 oracle, one packed batch with
`n_tokens ∈ {1,2,4,7,37,128,300,512}` — covers singleton and long sequences):

| D | rel | max abs | cosine | gate |
|---|---|---|---|---|
| 256  | 1.53e-07 | 1.35e-07 | 1.00000000 | PASS |
| 512  | 1.54e-07 | 9.12e-08 | 1.00000000 | PASS |
| 768  | 1.52e-07 | 8.06e-08 | 1.00000000 | PASS |
| 1024 | 1.46e-07 | 6.84e-08 | 1.00000000 | PASS |

Pure-fp32 kernel vs fp64 oracle: `rel = Σ|got−ref| / Σ|ref| ≈ 1.5e-7`, cosine 1.0
— far inside the repo's `rel < 0.02` gate.

**A/B: fused single-pass vs naive-composed** (HIP-event median, warmup 10 /
iters 50; two back-to-back runs, spread ≤2%). The naive baseline is the same
math composed as two kernels: stage 1 RMS-normalizes every token row into a
`[total_tokens, D]` global temp; stage 2 mean-pools + L2 over the temp. Fusion
removes the temp round-trip (~3× → ~1× token-matrix traffic).

| shape | D | fused | GB/s (read) | naive | speedup | decision |
|---|---|---:|---:|---:|---:|---|
| B2048 × T64  | 256  | 0.0392 ms | 3422 | 0.0978 ms | 2.49× | KEEP |
| B2048 × T64  | 512  | 0.0675 ms | 3974 | 0.2015 ms | 2.98× | KEEP |
| B2048 × T64  | 768  | 0.1086 ms | 3709 | 0.3108 ms | 2.86× | KEEP |
| B2048 × T64  | 1024 | 0.1436 ms | 3739 | 0.4193 ms | 2.92× | KEEP |
| B512 × T256  | 768  | 0.2183 ms | 1844 | 0.3626 ms | 1.66× | KEEP |
| B8192 × T16  | 768  | 0.1145 ms | 3518 | 0.3250 ms | 2.84× | KEEP |

**KEEP** as the (new, only) pooling-head path. The fused kernel sustains
~3.4–4.0 TB/s effective read bandwidth (MI300X HBM3 peak ~5.3 TB/s), so it is
genuinely bandwidth-bound and near-roofline for a reduction of this shape; the
naive two-pass pays ~2.5–3.0× for materializing the normalized token matrix
(1.66× at long T=256, where the extra traffic amortizes). Exact vs the fp64
oracle. No env flag: this is a new op with no slower in-tree variant to gate.

Follow-up: LDS-stage a shared-memory reduction to let >1 wave cooperate on very
long single sequences (the T=256 cell is the only one under ~2 TB/s); a packed
front-end entry to consume the attention output in place.
