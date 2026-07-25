# RoPE variants — CDNA3 (gfx942)

The eight rotary-position-embedding operations the Metal and CPU manifests
publish that ROCm lacked. Part of the Metal/CPU parity program in
[`docs/metal-cpu-parity-gaps.md`](../../../../../docs/metal-cpu-parity-gaps.md).

Grouped into one operation directory the way this repo already groups
`kernels/serving` (12 kernels) and `kernels/linear_attention` (3): they share
their row geometry and their optimization lever. Each still carries its own
correctness cases and its own focused A/B.

| Operation | CPU reference |
|---|---|
| `rotary_positioned` | `attention/rotary_positioned_ref.cpp:88` |
| `mrope` | `attention/rotary_positioned_ref.cpp:143` |
| `rope_table` | `attention/attention_extended_ref.cpp:409` |
| `rope_interleaved_to_split` | `attention/attention_ref.cpp:137` |
| `rope_backward` | `utils/tensor_ops_ref.cpp:515` |
| `qk_norm_rope_positioned` | `attention/attention_extended_ref.cpp:453` |
| `qk_norm_rope_split` | `attention/attention_extended_ref.cpp:554` |
| `rope_q_norm` | `attention/attention_serving_ref.cpp:317` |

## Contract

Two layout conventions run through all of them and must not be conflated:

```
split        pair p couples elements (p, half + p)      <- the common case
interleaved  pair p couples elements (2p, 2p + 1)
```

`mrope` and `rope_q_norm` are split-only by contract.
`rope_interleaved_to_split` reads interleaved and writes split — it is a layout
conversion fused into the rotation, not a rotation with a flag.

The rotation, given `cos`/`sin` for pair p:

```
y[first]  = x[first]  * cos - x[second] * sin
y[second] = x[second] * cos + x[first]  * sin
```

`rope_backward` applies the transpose (rotation by −θ), so its signs differ:
`g[first] = a*cos + b*sin`, `g[second] = -a*sin + b*cos`. It is not the forward
with swapped operands.

**A `rotary_dim` smaller than `head_dim` leaves the tail unrotated.** For the
plain variants that tail is copied through; for the `qk_norm` variants it is
still normalized and weighted, just not rotated. Silently dropping it is an
accuracy bug on partial-RoPE models, so every variant that accepts `rotary_dim`
has a partial-rotary test case.

Other contract details pinned by tests:

- `mrope` axis selection: `section_interleaved` gives `axis = pair % 3`;
  otherwise the three `sections` counts partition the pair range into contiguous
  x/y/t spans that must sum to `rotary_dim / 2`.
- `qk_norm_rope_*`: Q and K heads are RMS-normalized over the **full** head_dim,
  not `rotary_dim`; norm and per-dim weight are applied **before** the rotation;
  V heads are copied through untouched.
- `weight_offset` carries Gemma's stored "(1+w)" convention without
  materializing an adjusted weight buffer.
- `rope_q_norm` with `do_norm == false` uses unit weights and no scaling.

## Approach

One row (a head's D-vector) per lane group, lanes striding the pair index,
`kBlock / LANES` rows per 256-thread block. fp32/bf16/fp16 storage via the
shared traits in [`kernels/common/cdna3_harness.cuh`](../../../../common/cdna3_harness.cuh).

## Correctness

`make test` — **78/78 PASS on MI300X** against fp64 oracles mirroring each
reference. Every operation is checked at both lane widths.

fp32 error is `rel ~2.5e-8 … 5.3e-8, cosine 1.000000000` — essentially exact.
bf16 sits at the one-storage-ULP bound.

Coverage includes split and interleaved layouts, partial rotary (`rd = 64`, `96`
against `D = 128`), per-batch position tables, `mrope` with both sectioned and
section-interleaved axis selection, GQA head splits, a no-V-head case, the Gemma
weight offset, `qk_norm_rope` with mrope tables, split vs packed outputs,
`rope_q_norm` with norm on and off, and non-zero `pos0`.

## Performance

Two experiments, `make bench`, HIP-event median, warmup 25 / iters 50, fp32,
measured on an idle MI300X.

### Experiment 1 — pow → exp2 + sincos: **KEEP**

`rope_interleaved_to_split` and `rope_backward` derive angles from `base` and
`pos0` rather than a lookup table, and were running `pow()` per element. They sat
at ~1.2 TB/s while the table-driven variants reached ~4.0 TB/s — transcendental
bound, not memory bound.

`base^e == exp2(e * log2(base))` and `log2(base)` is loop-invariant, so a `pow`
becomes one `exp2`; one `sincos` then replaces a separate `cos` and `sin`. Still
evaluated in double, because the frequency spread across pairs is exactly where
float loses the low-order bits.

| Kernel | Before | After | Speedup |
|---|---:|---:|---:|
| `rope_interleaved_to_split` | 0.4466 ms (1202 GB/s) | 0.2656 ms (2021 GB/s) | **1.68x** |
| `rope_backward` | 0.4416 ms (1216 GB/s) | 0.2809 ms (1912 GB/s) | **1.57x** |

### Experiment 2 — 32 → 64 lanes per row: **REJECT**

Widening to the full 64-lane wavefront is this repo's standard first lever for
row kernels; it measured +30–71% on the norm kernels. Here it **lost on all
eight operations**, and the result reproduced across two independent runs:

| Operation (shape) | 32 lanes/row | 64 lanes/row | Ratio |
|---|---:|---:|---:|
| `rotary_positioned` (B8 H32 T2048 D128) | 0.1327 ms (4045 GB/s) | 0.1538 ms (3492 GB/s) | 0.86x |
| `mrope` (B8 H32 T2048 D128) | 0.1640 ms (3274 GB/s) | 0.1705 ms (3150 GB/s) | 0.96x |
| `rope_table` (T16384 H32 D128) | 0.1375 ms (3905 GB/s) | 0.1544 ms (3476 GB/s) | 0.89x |
| `rope_interleaved_to_split` (T16384 H32 D128) | 0.2656 ms (2021 GB/s) | 0.3194 ms (1681 GB/s) | 0.83x |
| `rope_backward` (T16384 H32 D128) | 0.2809 ms (1912 GB/s) | 0.3165 ms (1696 GB/s) | 0.89x |
| `qk_norm_rope_positioned` (T8192 Hq32 Hk8 Hv8 D128) | 0.1549 ms (2599 GB/s) | 0.1853 ms (2173 GB/s) | 0.84x |
| `qk_norm_rope_split` (T8192 Hq32 Hk8 Hv8 D128) | 0.1533 ms (2626 GB/s) | 0.1864 ms (2160 GB/s) | 0.82x |
| `rope_q_norm` (T16384 H32 D128) | 0.2493 ms (2154 GB/s) | 0.2758 ms (1947 GB/s) | 0.90x |

**Why the norm result did not transfer.** The norm win came from a shape where
half the wavefront sat idle — 32 threads assigned to a row inside a 64-wide
wavefront. These kernels pack `kBlock / LANES` rows per block, so a 32-lane row
already fills the wavefront with two rows; there is no idle half to reclaim.
Halving the lanes per row instead doubles the pairs each lane owns, doubling the
independent loads in flight, and with no cross-lane reduction to amortize that
extra ILP is worth more than the wider row. The two `qk_norm` variants and
`rope_q_norm` *do* have a reduction, and they lose by the most (0.82–0.90x),
which is consistent: a 64-lane reduction costs an extra shuffle step.

Selected configuration is therefore `kRopeLanes = 32`, recorded as a constant in
the source next to this reasoning so the choice is not silently re-litigated.

Shipped bandwidth: 3.3–4.0 TB/s for the table-driven variants, roughly 75% of
MI300X HBM3 peak.

### A measurement caveat

Several timings report a min/max spread above the 1.2x threshold
`perf/harness/run_kernel_bench.sh` warns at (up to 2.1x). Inspecting the samples,
this is a **single slow iteration** — min and median are tight (e.g.
`min 0.1290 / median 0.1327 / max 0.2696`) and the medians reproduce to within
1.7% across independent runs. It reads as a clock or power transient rather than
contention, and every conclusion above holds on the minima as well as the
medians. Reported rather than suppressed.

## Non-ports

- **`qk_norm_rope`** (the non-positioned entry point) is not a separate kernel:
  the CPU reference implements it as `qk_norm_rope_positioned` with
  `rotary_dim = head_dim`, no mrope sections, and zero weight offset. ROCm
  already has a `kernels/norms/qk_norm_rope` operation for that shape.
- **In-place operation.** The CPU reference for `rope_interleaved_to_split`
  handles `x == y` with a scratch copy. These kernels require distinct input and
  output buffers; the layout change makes in-place operation read already-written
  elements.
- **Position validation.** The references return `kInvalidArgument` for a
  position outside `[0, max_position)`. Range checking belongs in the dispatch
  layer, not in a per-element device loop.

## Follow-ups

- Cache the `cos`/`sin` table row in LDS when many heads share one token; at
  `H = 32` the same table row is re-read 32 times from global memory.
- Vectorize to `float4` loads for the split layout, where pairs `p` and
  `half + p` are each contiguous runs.
