# Serving family — CDNA3 (gfx942)

Native CDNA3 port of `../QuixiCore-CUDA/kernels/serving`, the plain-CUDA `tm_*`
inference/serving kernels. QuixiCore-internal port, not a third-party import.
Every harness is self-checking (generates its own inputs, compares against an
in-process fp64 / exact-replay reference) — no golden files.

## Kernels / harnesses

| harness | covers |
|---|---|
| `attn_q` | quantized-KV attention decode (fp8/format KV) |
| `attn_varlen` | ragged/prefix/GQA variable-length prefill attention |
| `kv_cache` | paged attention (dense/alibi/window/blocksparse), GQA-staged, scatter/gather, sliding-window |
| `paged_attn_v2` | partitioned paged attention + cascade prefix/suffix reduce |
| `mla` | Multi-head Latent Attention decode (partitioned, bf16 + reduce) |
| `rope_kv` | fused RoPE on Q and KV-cache rows |
| `sampling` | top-k/top-p/min-p, temperature, bitmask/bad-word masking |
| `logits_proc_test` | logit processors (penalties, softcap, etc.) |
| `beam_xcache` | beam-search KV-cache reindex |
| `spec_beam` | speculative-decode tree build + verify |
| `eagle_test` | EAGLE speculative helpers |
| `sparse_serving_test` | sparse/blocksparse serving helpers |

## Port notes (CUDA -> CDNA3)

- `hipify-perl`: headers, `__nv_bfloat16` -> `__hip_bfloat16`, runtime API.
- `__shfl_*_sync(0xffffffffu, ...)` (xor/up/down/plain) -> mask-free `__shfl_*`
  (HIP 64-bit-mask rule on the 64-wide wavefront; reductions use offsets <=16 in
  32-lane groups and stay correct on warp64).
- `__dp4a` -> `__builtin_amdgcn_sdot4` in the shared `quant_formats.cuh` used by
  the fp8/quant-KV attention path.
- `paged_attn_v2_kernels.cuh`: added `#include <hip/hip_bf16.h>` (HIP's
  `hip_fp16.h` does not transitively pull bf16 the way `cuda_fp16.h` does).
- **`kv_cache` gqa_staged vs v1**: the original test required *bit-for-bit*
  equality between the v1 and staged paged-attention kernels. They do identical
  math but different reduction schedules; on the 64-wide CDNA wavefront the
  cross-warp merge order is not fixed, so results can differ by ~1 fp16 ULP on a
  few elements (and non-deterministically). Both match the fp64 reference to
  <5e-3 (v1 is checked directly). The self-comparison was relaxed from exact to
  a 1 fp16-ULP tolerance; see the comment in `kv_cache.cu`.

## Build / run

```bash
make test     # build + run all 12 harnesses on GPU 0
```

## Result (MI300X, 2026-07-06)

All 12 harnesses pass — 104 pass-lines, 0 failures. Raw:
`perf/results/2026-07-06/serving/`. Attention/MLA decode timing and the
occupancy follow-ups (64-lane wavefront, partition-size sweeps) are tracked in
`perf/optimization_status.md`.

## v2_batch_prep (added 2026-08-01)

`v2_batch_kernels.cuh` + `slot_mapping_kernels.cuh`, harness `v2_batch_test.cu`.
Ported from `SlimServe/csrc/quixicore/serving/` (the vLLM fork serving
GLM-5.2-Vision GGUF), where they replace the Triton batch-preparation kernels of
the V2 GPU worker: slot mapping (single- and multi-group), block-table gather,
prefill and decode input prep, sampled/rejected counts, the post-step request
state update, DSA indexer metadata, uniform-decode expansion, and DFlash
speculative input prep.

**No CDNA3 adaptation was required** — the only change is the leading
`#include "hip/hip_runtime.h"`. These kernels are grid-stride integer index
arithmetic with no warp shuffle, no ballot, no `__shared__`, no inline PTX and
no hardcoded 32, so nothing in them is wave32-dependent. The single
`__syncthreads()` (the DFlash CUDA-graph padding barrier) is width-independent.
That is the whole reason this family ported cleanly while the sampling and
logit-processor kernels next door needed mask-free `__shfl_*` work.

The oracle is a host replay and the bar is **bitwise** equality, not a
tolerance: every output is an exact integer. 18/18 checks pass on MI300X,
covering ragged query lengths, chunked prefill, `cp_world` 1/2 interleave,
`dcp_world` 1/4, a non-trivial indexer query slice, and the CUDA-graph padding
tails.

**Not ported:** nothing from these two headers is omitted. The launch-geometry
sweep and the reject decision on a narrower block are in
`perf/optimization_status.md` (2026-08-01); short version, the family is launch
-bound at ~4.5 us against a 1.56 us empty-kernel floor, and a one-wave block
wins at decode but loses 2.4x at a 16384-token prefill chunk.

```bash
make v2_batch_test.out && HIP_VISIBLE_DEVICES=0 ./v2_batch_test.out
make bench    # launch-geometry sweep
```
