# Cross-attention — CDNA3 (gfx942)

Port of the CPU reference `quixicore_cpu::cross_attention`
(`../QuixiCore-CPU/kernels/attention/cross_attention_ref.cpp`). This is the
first kernel of the Metal/CPU parity program tracked in
[`docs/metal-cpu-parity-gaps.md`](../../../../../docs/metal-cpu-parity-gaps.md).

Cross-attention is not self-attention with a rename: the query and key
sequences are **independent**, each batch item carries its own valid key count,
and the score may be biased and softcapped. Neither length is required to be a
multiple of the 16-wide MFMA tile.

## Contract

```
layout   Q [B, Hq, Lq, D]   K,V [B, Hkv, Lk, D]   O [B, Hq, Lq, D]
GQA      kv_head = query_head / (Hq / Hkv),  requires Hq % Hkv == 0
lengths  valid = clamp(key_lengths[b], 0, Lk)     # per batch item, not per head
scale    scale > 0 ? scale : 1/sqrt(D)            # explicit or automatic
bias     optional, bias[((b*Hq + h)*Lq + i)*Lk + j]
softcap  optional, score = softcap * tanh(score / softcap) when softcap > 0
D        64, 128, or 256 only
empty    valid == 0 emits an all-zero row
```

Two ordering rules are contractual and were verified against the reference:

1. The bias is added to the **scaled** score, not the raw dot product.
2. The softcap is applied **after** the bias.

With both present, any other order gives different numbers. `key_lengths` is
indexed per batch item and applies to every head in that item.

The empty case matters: when `valid == 0` the denominator never leaves zero, so
the row must emit zeros rather than dividing by zero. Both kernels here special-
case it, and `empty first batch` covers it in the test set.

## Approach

**Candidate — MFMA-tiled flash attention.** One 64-lane wavefront owns a
`BQ=16` query block for one (head, batch). K/V are consumed in `BK=16` tiles, so
each K/V tile is read once per 16 queries instead of once per query. `QK^T` and
`P@V` both run on `v_mfma_f32_16x16x16{bf16_1k,f16}`; the fragment layout is the
one validated by the landed self-attention kernel
(`kernels/attention/gqa/variants/rocm_cdna3/attn_mfma.cuh`):

```
a[v] = A[m = lane%16   ][k = 4*(lane/16) + v]
b[v] = B[k = 4*(lane/16) + v][n = lane%16   ]
c[v] = C[m = 4*(lane/16) + v][n = lane%16   ]
```

The score tile is staged through LDS so the softmax reduces along a row with
lanes 0..15 each owning one query, rather than shuffling across the MFMA's
distributed accumulator layout.

**Baseline (the A/B comparator) — one wavefront per query row**, head dim split
across lanes, one wavefront dot-product reduction per key. Mathematically
identical and obviously correct, but it re-reads all of K and V once per query.

### What differs from the landed self-attention kernel

That kernel assumes `N % 16 == 0` and one shared sequence length. Cross-attention
cannot, so this kernel masks rather than assumes:

- Query rows past `Lq` load zeroed Q fragments and skip their output writes, so
  a partial query tile never reads or writes out of bounds.
- Key rows past `valid` load zeroed K/V fragments and are forced to `p = 0` in
  the softmax, so a partial key tile contributes nothing.
- The first tile of a row leaves the running max at `-inf`. `exp(-inf - -inf)`
  is `NaN`, so the correction factor is special-cased to `0` on that first tile.
  This is the one place where a naive port of the self-attention kernel produces
  silent NaNs on ragged input.

## Correctness

`make test` — **50/50 PASS on MI300X** against an fp64 host oracle mirroring the
CPU reference. Both the MFMA and the baseline kernel are checked on every case.

Covered: MHA and GQA 4:1; ragged per-batch key lengths; an over-long key length
that must clamp rather than read out of bounds; an empty batch item; bias;
softcap 30; bias and softcap together; explicit scale; `Lq=1` decode; long keys
(`Lk=2048`); `D` 64/128/256; bf16 and fp16 storage.

Observed error is `rel ~2.2e-3, cosine ~0.999997` for bf16 and `rel ~2.6e-4,
cosine ~0.99999996` for fp16 — consistent with the storage dtype, at the
`registry/tolerances.yaml` bounds.

Host inputs are rounded through the storage dtype **before** the oracle runs, so
bf16 rounding is not misattributed to the kernel.

The oracle accumulates output in double while the CPU reference accumulates in
float. That is deliberate: the oracle expresses the operation's mathematics, not
the CPU backend's accumulator width.

## Performance

`make bench` — MFMA candidate vs. the wavefront-per-query baseline. HIP-event
median, warmup 10 / iters 50, bf16, measured on an **idle** MI300X.

| Shape (B,Hq,Hkv,Lq,Lk,D) | Baseline | Candidate | Speedup |
|---|---:|---:|---:|
| prefill 4,32,8,512,512,64 | 2.5900 ms (3.3 TFLOP/s) | 0.2782 ms (30.9 TFLOP/s) | **9.31x** |
| prefill 4,32,8,512,512,128 | 3.3491 ms (5.1 TFLOP/s) | 0.5347 ms (32.1 TFLOP/s) | **6.26x** |
| prefill 2,32,8,512,2048,128 | 7.5734 ms (4.5 TFLOP/s) | 1.1357 ms (30.3 TFLOP/s) | **6.67x** |
| enc-dec 8,16,4,128,1024,128 | 1.8896 ms (4.5 TFLOP/s) | 0.4714 ms (18.2 TFLOP/s) | **4.01x** |
| ragged 4,16,4,517,1031,128 | 0.9731 ms (17.9 TFLOP/s) | 0.3998 ms (43.7 TFLOP/s) | **2.43x** |
| bias 4,16,4,256,1024,128 | 1.8694 ms (4.6 TFLOP/s) | 0.5813 ms (14.8 TFLOP/s) | **3.22x** |

Decision: **KEEP**. The win holds across every measured shape, including the
ragged and biased cases, and is largest exactly where K/V reuse matters most.

**Measurement note worth repeating.** An earlier run of this same A/B was taken
while an unrelated kernel sweep held GPU 0. It reported the ragged shape as a
**0.64x regression** with a 4.12x min/max spread. Re-measured on an idle GPU the
same shape is a 2.43x win with 1.04x spread. Timing spread is the tell: a spread
above roughly 1.2x means the median is measuring contention, not the kernel, and
the run must be repeated before any decision is recorded.

## Non-ports

- **FP32 storage.** Only bf16 and fp16 are implemented, matching the rest of
  this repo's attention family and the MFMA hardware paths. The CPU reference's
  `cross_attention_storage` adapter accepts fp32 as well; an fp32 boundary here
  would need either a converting wrapper or an MFMA-free path, and neither is
  worth adding before there is a caller.
- **Backward pass.** The CPU reference publishes forward only for
  cross-attention; `attention_backward` is a separate operation.
- **Softcap on the LSE output.** This kernel returns `O` only. The landed
  self-attention kernel also emits log-sum-exp; cross-attention has no published
  LSE consumer, so it is not computed.

## Follow-ups

- LDS double-buffering of the K/V tiles. The current loop loads the next tile
  after `__syncthreads()`, leaving the MFMA units idle during the load.
- Larger `BQ` (32 or 64 queries per wavefront) to raise arithmetic intensity;
  this is the same lever noted for the self-attention kernel, which sits at
  0.30x of SDPA flash.
- A packed-QKV entry point, if a caller materializes Q/K/V contiguously.
