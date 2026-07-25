# Biased attention — CDNA3 (gfx942)

Port of the CPU reference `quixicore_cpu::biased_attention`
(`../QuixiCore-CPU/kernels/attention/attention_extended_ref.cpp`, ~L357). Part of
the Metal/CPU parity program in
[`docs/metal-cpu-parity-gaps.md`](../../../../../docs/metal-cpu-parity-gaps.md).

## Contract

```
layout   Q [Hq, Lq, D]   K,V [Hkv, Lk, D]   O [Hq, Lq, D]   -- no batch dim
GQA      kv_head = query_head / (Hq / Hkv),  requires Hq % Hkv == 0
bias     optional, PER HEAD:      bias[(h*Lq + i)*Lk + j]
mask     optional, SHARED across heads: mask[i*Lk + j], 0 == skip the key
scale    scale == 0 ? 1/sqrt(D) : scale
masked   a fully masked query row emits zeros
```

Two details differ from `cross_attention` and are easy to get wrong by analogy:

- **The bias is per head; the mask is not.** The mask is a single `[Lq, Lk]`
  plane reused by every head.
- **The automatic-scale trigger is `scale == 0`, not `scale > 0`.** A negative
  explicit scale is legal here and must be honoured. `cross_attention` really
  does use `scale > 0` — the two references differ, and each kernel follows its
  own. The `negative scale` test case pins this.

A masked key is *skipped*, not driven to `-inf`, so it never participates in the
running maximum. In a tiled softmax that distinction has teeth: a tile where
every key is masked leaves the running max at `-inf`, and `exp(-inf - -inf)` is
`NaN`. The correction factor is special-cased for that tile.

## Approach

MFMA-tiled flash attention, `BQ=BK=16`, one 64-lane wavefront per (query block,
head), `QK^T` and `P@V` on `v_mfma_f32_16x16x16{bf16_1k,f16}`, softmax over an
LDS transpose of S. Fragment layout comes from
[`kernels/common/cdna3_mfma.cuh`](../../../../common/cdna3_mfma.cuh).

The baseline for the A/B is one wavefront per query row with a wavefront
dot-product per key — same math, and it *skips masked keys outright*.

### Masked-tile skipping

The tiled kernel's first version walked every `Lk` tile regardless of the mask.
That is fine for dense attention and catastrophic for sparse masks: on a ±8
banded mask at `Lq=Lk=1024` it did roughly 64x the necessary work and measured
**0.70x — an outright regression** against the baseline, which touches ~17 keys
per query.

The fix is a cheap pre-check: the wavefront reads the 16x16 mask tile (256
bytes), OR-reduces liveness, and skips the whole tile when nothing is live —
avoiding a 16xD K/V load and `D/16` MFMAs. Because the block is exactly one
wavefront the reduced predicate is uniform, so `continue` cannot desynchronize
the `__syncthreads()` calls in the loop body.

That single change took the banded case from **0.70x to 3.62x**.

## Correctness

`make test` — **50/50 PASS on MI300X** against an fp64 host oracle mirroring the
reference. Both kernels are checked on every case.

Covered: MHA and GQA 4:1; per-head bias; banded, causal, and one-fully-masked-row
masks; bias and mask together; ragged `37x101`; negative explicit scale; explicit
positive scale; `Lq=1` decode; `D` 32/64/128/256 (32 is the Swin head dim);
bf16 and fp16.

### A tolerance that could not be met

The causal `512x512` case initially failed with exactly one bad element out of
2,097,152: `got -1.3671875 vs ref -1.37182645`, absolute error `4.6e-3`.

That is not a kernel bug. bf16 keeps 8 mantissa bits, so near magnitude 1.4 its
representable values are spaced `7.8e-3` apart, and the two candidates are
`1.3671875` and `1.375`. The kernel landed one ULP below correctly-rounded.

The registry's bf16 `atol` of `2e-3` is **smaller than a single bf16 ULP** at
that magnitude, so for a bf16-*stored* output the elementwise bound is
unsatisfiable in the worst case and passing it is luck. The harness now offers
`qc::Tol::bf16_output()` / `fp16_output()`, which raise only the elementwise
relative bound to one storage ULP (`2^-8` / `2^-10`) and leave the aggregate
rel-L1 and cosine bounds untouched. Those aggregates are the real correctness
signal: a genuinely wrong kernel moves them, one-ULP storage noise does not.

Observed: `rel ~2.1e-3, cosine ~0.999998` (bf16); `rel ~2.5e-4, cosine
~0.99999996` (fp16).

## Performance

`make bench` — MFMA candidate vs. wavefront-per-query baseline. HIP-event
median, warmup 10 / iters 50, bf16, measured on an **idle** MI300X.

| Shape (Hq,Hkv,Lq,Lk,D) | Baseline | Candidate | Speedup |
|---|---:|---:|---:|
| 32,8,512,512,128 + bias | 1.3981 ms (3.1 TFLOP/s) | 0.2962 ms (14.5 TFLOP/s) | **4.72x** |
| 32,8,512,2048,128 + bias | 5.8754 ms (2.9 TFLOP/s) | 1.1525 ms (14.9 TFLOP/s) | **5.10x** |
| 16,4,1024,1024,128 banded | 0.3549 ms (24.2 TFLOP/s) | 0.0980 ms (87.7 TFLOP/s) | **3.62x** |
| 32,8,512,512,128 causal | 0.8525 ms (5.0 TFLOP/s) | 0.2942 ms (14.6 TFLOP/s) | **2.90x** |
| 32,8,512,512,64 + bias | 0.9856 ms (2.2 TFLOP/s) | 0.2025 ms (10.6 TFLOP/s) | **4.87x** |

Decision: **KEEP**. No shape regresses. The banded case is the most instructive:
it is the only one where the baseline was competitive, and it is now the fastest
absolute result in the table at 87.7 TFLOP/s precisely because the tile skip lets
the kernel do proportionally less work as the mask gets sparser.

## Non-ports

- **FP32 storage.** bf16 and fp16 only, matching this repo's attention family and
  the MFMA hardware paths.
- **Backward pass.** The reference publishes forward only for this operation.
- **Bias with a batch dimension.** The reference has no batch dim here; batched
  biased attention is `cross_attention`'s shape, which is implemented separately.

## Follow-ups

- Reuse the mask-tile liveness check to build a per-query-block key range for
  banded and causal masks, skipping the tile scan entirely when the mask is
  structured rather than arbitrary.
- LDS double-buffering of K/V tiles.
- Larger `BQ` for arithmetic intensity.
