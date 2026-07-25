# Composite attention — CDNA3 (gfx942)

Three composite attention shapes from the Metal/CPU parity program
([`docs/metal-cpu-parity-gaps.md`](../../../../../docs/metal-cpu-parity-gaps.md)),
grouped as one operation directory like `kernels/serving` and
`kernels/linear_attention`. All three produce one output row per (request, head)
with a small head dim, so they share their geometry.

| Operation | CPU reference |
|---|---|
| `swin_attention_d32` | `attention/attention_extended_ref.cpp:602` |
| `decode_cache_attention` | `attention/attention_composites_ref.cpp:332` |
| `cascade_attention_multi` | `attention/attention_composites_ref.cpp:470` |

## Contracts, and where a careless port goes wrong

**`swin_attention_d32`** — QKV is interleaved at the token level, not three
tensors: `qkv[(((w*tokens + pos)*3 + which)*heads + head)*32 + d]` with
`which ∈ {0=Q, 1=K, 2=V}`. Head dim is fixed at 32 and the scale is a hardcoded
`0.1767766952966369`, matched bit-for-bit rather than recomputed as
`1/sqrt(32)`. The relative bias is per head. The optional mask is indexed by
**`window % windows_per_image`** — the shifted-window mask repeats per image, so
using the raw window index silently applies the wrong shift to every window past
the first image. Covered by a two-image test case.

**`decode_cache_attention`** — two dependent stages. Stage 1 RoPEs and
optionally RMS-norms the new K, then writes it and V into the cache at slot
`context_lengths[item]`. Stage 2 has the query attend over slots
`[0, context_lengths[item]]` **inclusive**. The inclusive bound is the point: the
token just appended must be visible to the same call. That forces two kernels
with a launch boundary between them, because stage 2 reads what stage 1 wrote
from other workgroups. Q and K carry independent `do_norm` flags and share the
Gemma weight convention.

**`cascade_attention_multi`** — one online softmax spans several prefix levels
and then a paged cache, consumed in that order. Prefix levels are separate
allocations reached through a pointer array; the paged tail is gathered via
`block_table`. Both feed the same running max and denominator, so they cannot be
computed independently and merged without a separate merge step. `levels == 0`
(paged only) is legal and tested.

## Correctness

`make test` — **24/24 PASS on MI300X** against fp64 oracles. Every operation is
checked in both its baseline and candidate form; `decode_cache_attention` also
verifies the mutated key cache, not just the attention output.

fp32 error is `rel ~1e-7 … 6e-7, cosine 1.000000000`.

Coverage: Swin with and without the shifted-window mask and a multi-image case
that exercises the `window % windows_per_image` rule; decode with norm on/off,
GQA and MHA, Gemma weights, an explicit scale, and a ragged cache length; cascade
with `levels` 0, 1, 2 and 3, ragged prefix lengths, and two page sizes.

## Performance

`make bench`, HIP-event median, measured on an idle MI300X.

| Operation | Baseline | Candidate | Speedup |
|---|---:|---:|---:|
| `swin_attention_d32` (W1024 T49 H8) | 1.1245 ms global K/V | 1.0836 ms LDS K/V | **1.04x** |
| `decode_cache_attention` (B256 Hq32 Hkv8 D128 ctx4096) | 587.62 ms scalar | 13.16 ms wavefront | **44.64x** |
| `cascade_attention_multi` (B128 Hq32 Hkv8 D128 levels=2) | 109.69 ms scalar | 3.43 ms wavefront | **32.02x** |

Decision: **KEEP** all three, with two honest caveats.

**The Swin LDS win is marginal — 3.8%.** The hypothesis was that staging a
window's K and V in LDS would pay off because the global version re-reads them
for every query position. It barely does, and the reason is instructive: a Swin
window's K and V are only `2 × 49 × 32 × 4 B ≈ 12 KB`, which already lives in
cache across the query positions of the same workgroup. LDS saves the L1/L2
round-trip, not a DRAM round-trip. Kept because it is a consistent win at 1.03x
spread and reduces cache pressure for the larger `tokens` configurations, but it
is not the lever it looked like.

**The 44.64x decode number conflates two effects and should not be read as
"44x faster than a reasonable implementation".** The baseline is a direct scalar
transliteration of the CPU reference — one thread per output row — and it
additionally recomputes the query rotation inside the per-key dot product rather
than materializing it, exactly as a naive port would to avoid a scratch buffer.
The candidate's structural advantages are (a) lane-parallel dot products with
coalesced cache reads and (b) holding the rotated query in registers across the
whole cache walk. The cascade comparison has no such confound — both sides do the
same arithmetic — and its 32.02x is a clean scalar-vs-wavefront result.

## Non-ports

- **fp16/bf16 storage.** fp32 only. These are decode-shaped and the CPU
  references are fp32; a narrow-float boundary needs a caller first.
- **Split-K / partitioned decode.** `cascade_attention_multi` walks the whole
  context in one workgroup. For very long contexts the repo's existing
  `paged_attention_partition` + `merge_attn_states` pattern would apply, but it
  needs a merge step and is a separate operation.
- **Position and block-index validation.** The references return
  `kInvalidArgument` for out-of-range positions and block ids. That belongs in
  the dispatch layer, not a per-element device loop.

## Follow-ups

- MFMA the Swin path: `D=32` with `tokens=49` rounds to 4 MFMA tiles, and the
  biased-attention kernel already has the fragment layout.
- Split-K over the context for `decode_cache_attention` at very long contexts,
  reusing `merge_attn_states`.
- Vectorize the cache walk to `float4`; `head_dim` is a multiple of 4 in every
  supported configuration.
