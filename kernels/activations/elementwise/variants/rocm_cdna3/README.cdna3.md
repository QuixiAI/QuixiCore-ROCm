# Elementwise / Norm family — CDNA3 (gfx942)

Native CDNA3 port of `../QuixiCore-CUDA/kernels/elementwise`, the plain-CUDA
`tm_*` elementwise / norm / training family (twin of the QuixiCore Metal W3
kernels). This is a QuixiCore-internal port (same project/owner), not a
third-party import.

## Kernels

`rms_norm` / `layernorm` fwd + bwd (dx-only and fully-fused dW/dB), `add_norm`
(fused residual-add + norm, with static and dynamic per-row fp8-e4m3 epilogues),
`softmax`, `gelu` fwd/bwd (tanh approx), `glu` fwd/bwd (6 modes: reglu, geglu
tanh, swiglu, swiglu-oai, geglu-erf, geglu-quick), inverted `dropout` (seed-
replayed mask), fused `cross_entropy` (+ `_mw` multi-warp) with label smoothing /
z-loss / softcap, `embedding` lookup + atomic and sorted-segment backwards +
multimodal span build/merge, Walsh-`hadamard`, `adamw`, `add`.

## Port notes (CUDA -> CDNA3)

- Translated with `hipify-perl`: headers (`cuda_fp16/bf16.h` -> `hip/hip_*`),
  `__nv_bfloat16` -> `__hip_bfloat16`, runtime API `cuda*` -> `hip*`.
- **Warp width.** The one behavioral adaptation. HIP's `__shfl_*_sync` requires
  a 64-bit lane mask on the 64-wide CDNA wavefront, so the `0xffffffffu`
  (32-bit) NVIDIA masks were dropped in favor of the mask-free `__shfl_*`
  intrinsics. All reduction offsets are `<= 16` and the row kernels launch
  32-thread blocks, so shuffles stay within the intended 32-lane groups on a
  64-wide wavefront. The Hadamard's 32-lane butterfly passing (`0 mismatches`
  is not applicable — err 3.7e-08) confirms the lane mapping.
- Compute is fp32 throughout; I/O templated on `{float, __half, __hip_bfloat16}`.
  The oracle exercises `T=float`.

## Build / run

```bash
make test    # fp64-oracle correctness on GPU 0  -> "ALL PASS (0 failures)"
make bench   # focused perf run (32-lane vs 64-lane wavefront) on GPU 0
```

## Perf

Correctness: 56/56 fp64-oracle checks pass on MI300X (worst rel err 2.9e-06).
A measured wavefront-width A/B (32-lane faithful port vs 64-lane full-wavefront)
shows a consistent +15% to +61% bandwidth win for the forward norm kernels; see
`perf/optimization_status.md` (2026-07-06 entry) for the table and the
keep/land decision. The shipped kernels here are the faithful 32-lane port; the
64-lane widening is prototyped in `bench.cu` and queued for a family-wide pass.
