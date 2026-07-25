# Q8_0 paged KV cache — CDNA3 (gfx942)

Four operations from Metal/CPU parity Phase 2
([`docs/metal-cpu-parity-gaps.md`](../../../../../docs/metal-cpu-parity-gaps.md)),
ported from `../QuixiCore-CPU/kernels/attention/attention_q8_kv.cpp`:
`kv_cache_scatter_q8_0` (:102), `kv_cache_gather_q8_0` (:235),
`kv_cache_copy_blocks_q8_0` (:351), `paged_attention_q8_0` (:417).

## Canonical layout — two planes, not GGUF blocks

This is **not** the GGUF interleaved-block Q8_0. Codes and scales are two
independent arrays:

```
codes   int8    [cache_slots, heads, head_dim]        slot = block*page_size + offset
scales  uint16  [cache_slots, heads, head_dim/32]     raw fp16 BITS, group of 32
```

A port that packs a 2-byte scale ahead of each 32-byte code run — the actual
GGUF block — produces plausible numbers and a cache no other backend can read.
`head_dim` must be a multiple of 32.

## Quantization rule

```
scale = amax(group) / 127                        stored as fp16 bits
code  = clamp(copysign(floor(|v/scale| + 0.5), v), -127, 127)
```

`floor(|x| + 0.5)` with the sign reapplied is round-half-away-from-zero, which
differs from `rintf`'s round-half-to-even on exact `.5` ties. Matched exactly —
this is why the code planes compare bit-exact rather than within a tolerance. A
group whose `amax` is 0 stores scale 0 and codes 0; the inverse is forced to 0
rather than dividing by zero, and an all-zero group is in the test set.

## Three behaviours a port gets wrong

**Scatter rewrites the whole cache.** The reference zero-fills every code and
scale before writing the requested slots — it is a full rewrite, not an
incremental append — and slots carrying `-1` are skipped, staying zeroed. The
test pre-dirties the cache with `0x7f` before scattering, so a missing zero-fill
fails rather than passing on the slots it happened to write.

**A block id below zero is a hole, not a zero.** In `gather` it produces a zeroed
output row; in `paged_attention` the position is skipped entirely and contributes
nothing to the softmax — which is different from contributing a zero score.

**The decode softmax is tiled, not per-key.** `paged_attention_q8_0` accumulates
over 32-key tiles: it scores the whole tile, takes that tile's max, rescales the
accumulator **once**, then adds all 32 weighted values. A per-key online softmax
is equally valid numerically but gives different low-order bits, so the tiling is
reproduced to keep the backends comparable. Sliding window is
`first = window > 0 ? max(0, context - window) : 0`.

The query stays fp32; only K and V are quantized. Scores are
`sum_groups(scale_g * sum_lanes(q * code))`.

## Correctness

`make test` — **28/28 PASS on MI300X**.

Scatter code and scale planes and `copy_blocks` compare **bit-exact**; gather is
exact too (it is a pure dequantize). `paged_attention_q8_0` lands at
`rel ~1.5e-7, cosine 1.000000000` against an fp64 oracle that mirrors the
reference's tiling and its `double` denominator.

Coverage: head dims 64/128/256, page sizes 16/32, MQA and GQA, block-table holes,
skipped scatter slots, an all-zero quantization group, a sliding window, and an
explicit score scale.

## Performance

`make bench`, HIP-event median, idle MI300X.

| Shape | Baseline | Candidate | Speedup |
|---|---:|---:|---:|
| B128 Hq32 Hkv8 D128 ctx512 | 7.2420 ms scalar | 1.0480 ms wavefront | **6.91x** |

Baseline is a direct scalar transliteration — one thread per output row. The
candidate gives each row a 32-lane group, holds the query in registers, and
computes each tile's score with a lane-parallel dot product plus a wavefront
reduction. Both sides do identical arithmetic and preserve the 32-key tiling, so
unlike the `decode_cache_attention` A/B this number is not confounded by a second
change.

Decision: **KEEP**.

## Non-ports

- **fp16/bf16 storage boundaries.** The reference exposes `_storage` variants
  accepting f32/f16/bf16 for the unpacked side. Only fp32 is implemented; the
  packed side is byte-identical either way.
- **Ordered-unique fast path.** The reference parallelizes scatter only when
  slots are strictly increasing, falling back to serial otherwise to avoid
  races. The GPU version has no such split: duplicate slots race, and the
  reference's own serial fallback exists precisely because that case is
  ill-defined. Callers must not pass duplicate slots.
- **Non-finite rejection as a status code.** The kernel flags non-finite input
  through a device flag rather than returning `kInvalidArgument`; surfacing it is
  the dispatch layer's job.

## Follow-ups

- `sdot4` (`__builtin_amdgcn_sdot4`) for the score dot product, quantizing the
  query per tile — the repo's int8 GEMM already uses it and measured a win there.
- Split-K over the context with `merge_attn_states` for long contexts.
- `float4`/`int32` vectorized code loads; `head_dim` is always a multiple of 32.
