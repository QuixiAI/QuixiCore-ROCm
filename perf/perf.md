# QuixiCore ROCm Performance Handbook

This is the operating guide for baselining and optimizing QuixiCore ROCm
kernels. It ports the hardware-independent discipline from QuixiCore Metal while
keeping the backend notes specific to AMD ROCm, HIP, and CDNA GPUs.

The goal is not to collect tricks. For each kernel, find references, state a
bottleneck hypothesis, measure a clean baseline, run controlled experiments,
keep only verified wins, and record enough detail that the next pass starts from
evidence instead of memory.

The running notebook is `perf/optimization_status.md`. Baseline snapshots live
in `perf/baseline_status.md`. Raw results belong under `perf/results/`. Existing
ROCm benchmark scripts currently live in `analysis/`, `docs/profiling/`, and
kernel-local directories; migrate them into `perf/harness/` only when doing so
improves reuse.

## Principles

Correctness comes before performance. A change is not a win until it passes the
kernel's correctness tests, improves the target metric on realistic shapes, and
does not regress supported edge shapes or numeric tolerances.

Prefer experiments that attack a named bottleneck:

- Memory-bound: reduce bytes moved, improve coalescing, improve cache/LDS reuse,
  avoid extra global-memory passes, or use narrower formats.
- Compute-bound: raise arithmetic intensity, feed MFMA/WMMA or library matrix
  paths effectively, reduce scalar side work, and fuse epilogues.
- Latency-bound: grow resident work, reduce serial loops, batch tiny launches,
  and avoid unnecessary host/device synchronization.
- Occupancy-bound: tune workgroup size, wavefront count, register pressure, LDS
  use, and grid size so the GPU has enough resident work.
- Synchronization-bound: reduce barriers, LDS traffic, atomics, and cross-kernel
  dependencies.
- Launch-bound: fuse small operations or route them through a framework/library
  path if dispatch overhead dominates custom code.

## ROCm Baseline Assumptions

Do not blindly port Metal, CUDA, XPU, or Gaudi mechanisms.

Metal simdgroup matrix paths, Apple command-buffer timing, CUDA `cp.async`, TMA,
WGMMA, CUDA events, SYCL queues, and Gaudi TPC/graph behavior are not ROCm design
rules. They can suggest experiments, but ROCm kernels should be written and
measured in terms of AMD-native mechanisms:

- HIP kernels and HIP events for custom kernel timing.
- CDNA wavefronts, LDS, vector memory behavior, and MFMA/WMMA paths where the
  target architecture exposes them.
- rocBLAS, hipBLASLt, Composable Kernel, AITER, rocWMMA, and Triton/ROCm as
  production baselines or references.
- rocprof/rocprofiler and ROCm profiling counters for bottleneck evidence.
- PyTorch on ROCm or framework baselines when they represent the user-facing
  route.

When a backend-local layout is needed for speed, preserve QuixiCore contract
names and byte layouts at public API boundaries.

## Repo Facts To Preserve

Current ROCm performance material is split between the target `perf/` layout and
legacy analysis directories:

- `analysis/bf16_gemm/`, `analysis/fp8_gemm/`, and `analysis/fp6_gemm/` contain
  GEMM benchmark scripts, JSON results, plots, and MI-series experiments.
- `analysis/attn/fwd/` and `analysis/attn/bkwd/` contain attention forward and
  backward comparisons.
- `analysis/layernorm/` and `analysis/rotary/` contain row-kernel benchmarks.
- `analysis/baselines/` contains PyTorch/Triton/library baseline scripts.
- `docs/profiling/` contains rocprof trace/counter workflows and helper scripts.
- `kernels/` contains active kernel-local build and test scripts.

The target shared layout is:

```text
perf/
  README.md
  perf.md
  optimization_status.md
  baseline_status.md
  harness/
  configs/
  baselines/
  results/
```

Use `scripts/bench` as the common entrypoint when possible. If a kernel still
uses an `analysis/` or kernel-local script, record the exact command and path in
the status notebook.

## Reference Search Protocol

For each kernel, inspect references before designing or tuning the ROCm path.
Record exact files in `perf/optimization_status.md`.

Reference classes to check:

- Existing ROCm analysis scripts under `analysis/`.
- Baseline implementations under `analysis/baselines/`.
- Kernel-local README/test files under `kernels/`.
- Sibling QuixiCore backend docs for operation contracts, shape sets, and
  measurement discipline.
- External references mirrored under `.reference/` when available.

Useful search patterns:

```bash
rg -n "mfma|wmma|lds|wave|__builtin_amdgcn|hipEvent|hipblas|rocblas|ck_" .
rg -n "layernorm|rms_norm|softmax|gemm|attention|quant|mla|paged" analysis kernels
rg -n "rocprof|pmc|counter|trace|roofline" docs analysis perf
```

Classify reference ideas into:

- Portable algorithm idea: worth considering.
- ROCm/CDNA-specific mechanism: translate only if it matches the target GPU.
- Library/framework baseline: usually adopt as a comparison.
- Benchmark shape or oracle idea: usually adopt.

Do not import implementation code from references into this repository unless a
future license and provenance review explicitly allows it.

## Measurement Harness Requirements

Every benchmark result should include:

- Git commit or working-tree label.
- QuixiCore contract version.
- ROCm version, HIP compiler version, container image, and relevant environment
  variables.
- AMD GPU model, target architecture, HBM size, driver/runtime versions, and
  device count.
- Kernel family, operation, public entry point, dtype, quant format, and shape.
- Warmup count, measured iteration count, median, p20/p80 or min/max, and
  coefficient of variation.
- Correctness tolerance and observed max absolute/relative error.
- Derived throughput: GB/s, GFLOP/s, TOP/s, tokens/s, or elements/s.
- Raw output path under `perf/results/` or the legacy analysis path used for the
  run.

When a shared harness is available, prefer:

```bash
scripts/bench
```

Until each benchmark is migrated, use the existing kernel-local command and copy
stable conclusions into the notebook instead of committing bulky raw traces.

## Timing Rules

Use native device timing for device work.

- Use HIP events around the kernel or library call for custom HIP paths.
- Synchronize outside the measured region unless the experiment includes host
  synchronization cost.
- For PyTorch on ROCm, use the framework synchronization API for the selected
  build; many PyTorch ROCm builds expose device sync through `torch.cuda`.
- Warm up first to avoid module load, autotuning, graph capture, cache, and
  power-state artifacts.
- Do not allocate, initialize, randomize, or copy inputs in the measured region
  unless that is the metric under study.
- For tiny kernels, batch repeated launches per sample and divide, then record
  the batch size.
- Re-run surprising results on an idle machine before trusting an A/B.

Derived metrics:

```text
GEMM FLOPs          = 2 * M * N * K
attention FLOPs     ~= 4 * B * H * N * N * D   (halve for causal)
quant decode GB/s   = packed_weight_bytes_read / seconds / 1e9
row-kernel GB/s     = conservative required reads+writes / seconds / 1e9
```

State when an estimate ignores cache reuse, repeated passes, metadata reads, or
write allocation.

Use rocprof/rocprofiler when timing alone does not explain a result. Record
which counters were collected, the local trace path, and the conclusion. Do not
commit large profiler traces unless explicitly requested.

## Shape Strategy

Do not optimize only square toy shapes. For each family, cover:

- Small edge shapes and non-power-of-two dimensions.
- Tile-aligned fast-path shapes.
- Tile-ragged shapes.
- Real model shapes from Llama/Qwen/DeepSeek-style projections and attention.
- Stress shapes: long context, large K/N, batch sweeps, and many experts.

Starter shapes:

- Norm/softmax/GELU: rows in `{4096, 16384, 65536}`, hidden in
  `{64, 128, 256, 512, 768, 1024, 2048, 4096, 8192}`.
- GEMM: square `{1024, 2048, 4096, 8192, 16384}` and LLM rectangles such as
  `K=4096`, `N=11008`, `N=14336`.
- Quant GEMV/GEMM: `M in {1, 2, 4, 8, 16, 32, 64, 128}`,
  `N/K in {4096, 8192, 16384}`.
- Attention: `D in {64, 128}`, context in `{512, 2048, 8192}`.
- MoE: tokens in `{128, 1024, 4096}`, experts in `{8, 16, 64}`,
  top-k in `{1, 2, 4}`.

Record skipped shapes with the reason.

## Per-Kernel Optimization Loop

1. Inventory the kernel: entry points, dtypes, shape constraints, tests, and
   benchmark coverage.
2. Find references in `analysis/`, `kernels/`, `.reference/`, and sibling
   backend docs.
3. Establish correctness against a deterministic host or framework reference.
4. Measure the baseline against framework, library, and naive decomposed
   baselines where available.
5. Classify the bottleneck with bytes, FLOPs, achieved throughput, variance, and
   profiling data.
6. Define experiments before editing code. Change one meaningful factor at a
   time.
7. Execute focused tests first, then the same benchmark matrix.
8. Decide with recorded numbers and rejected alternatives.
9. Update `perf/optimization_status.md`.

## Experiment Catalogue

Use these as templates. Not every kernel needs every experiment.

### Launch Geometry

- Sweep workgroup size and wavefronts per workgroup.
- Change rows/items per workgroup for row kernels.
- For GEMM/attention, sweep output tile sizes and K/sequence block sizes.
- Split large reductions across more workgroups only when merge overhead is
  measured.
- Watch tail effects when grid size does not fill the device evenly.

### Memory Layout And Coalescing

- Verify adjacent lanes read adjacent addresses on hot global-memory paths.
- Compare direct global loads against explicit LDS staging.
- Test padded or swizzled LDS layouts when bank conflicts or vector access
  patterns suggest it.
- Use vectorized loads/stores where alignment and layout allow.
- Keep scales, metadata, and lookup tables in layouts that favor wavefront
  access patterns.

### Tiling And Reuse

- Sweep `BM`, `BN`, `BK`, sequence block size, rows per block, and experts per
  scheduling unit.
- Compare library paths against custom HIP kernels at each shape class.
- Separate alignment-specialized fast paths from generic edge paths.
- Record compiler flags and target architecture whenever a matrix path is used.

### Matrix And Quant Paths

- Compare MFMA/WMMA/library-backed paths against scalar/vector HIP kernels.
- For quantized kernels, test dequant-direct-to-fragment against materialize-then-
  matmul.
- Hoist scales, zero-points, and table lookups out of inner loops.
- Test format-specialized kernels when runtime branching shows up in profiling.

### Fusion

- Fuse epilogues such as bias, residual, scale, activation, or normalization
  when an intermediate would otherwise round-trip through device memory.
- Fuse dequantization with matmul or attention when the dequantized value is
  used once.
- Split a fused kernel when register pressure, branching, or lower occupancy
  dominates saved memory traffic.

### Branches And Scalar Side Work

- Hoist format, dtype, causal, and dimension decisions out of hot loops through
  templates or separate entry points.
- Precompute base offsets and use simple increments in inner loops.
- Specialize common dimensions such as `D=64`, `D=128`, aligned K tiles, and
  supported quant block sizes.
- Measure decode-only or epilogue-only microkernels when scalar work may hide
  the true bottleneck.

### Reductions And Numerics

- Prefer wavefront reductions before LDS reductions when possible.
- Keep fp32 accumulation for softmax, norms, attention, and long K reductions
  unless a lower-precision variant passes tolerance.
- Use deterministic reduction orders where the contract needs determinism.
- For exact integer or packing kernels, exactness is part of the contract.

### Routing And Shape Specialization

- Find GEMV-to-GEMM and custom-to-library crossovers by sweeping `M`.
- Route tiny elementwise shapes to a framework/library path if launch overhead
  dominates custom code.
- Add fast paths for aligned shapes only when edge handling remains correct.
- Keep generic padding/slicing outside hot kernels when host overhead is smaller
  than in-kernel predicates.

## Kernel-Specific Starting Hypotheses

Use these as first-pass ideas. Replace them with measured facts as the project
progresses.

### BF16/FP8/FP6 GEMM

Measure against hipBLASLt, rocBLAS, Composable Kernel, AITER, Triton, and any
existing analysis baseline. Large square and LLM rectangle shapes should be
compute-bound; small-M decode shapes often become memory- or launch-bound.

Experiments: tile size sweep, MFMA/WMMA path selection, LDS staging, library
handoff thresholds, split-K/stream-K, alignment fast paths, and epilogue fusion.

### Attention Forward/Backward

Attention performance depends on on-chip softmax state, Q/K/V memory traffic,
sequence tiling, and launch geometry. Treat causal, non-causal, GQA, D=64, and
D=128 as separate cases until measurements prove they share a winner.

Experiments: sequence block size, K/V staging versus direct loads, causal branch
placement, GQA reuse layout, dQ versus dKV split geometry, logsumexp storage
format, and recompute-versus-store tradeoffs.

### LayerNorm, RMSNorm, Softmax, GELU, Rotary

These are row reductions or bandwidth-sensitive pointwise transforms. Framework
baselines may already be strong, so report GB/s and shape coverage.

Experiments: rows per workgroup, vectorized loads, wavefront-only reductions for
small hidden sizes, two-pass versus one-pass variance, reciprocal/sqrt
placement, hidden-size specialization, and fusing with neighboring ops.

### Quant GEMV/GEMM And Quantized Attention

Quantized kernels should win by reducing bytes moved. If a lower-bit format is
not faster, investigate dequant branch cost, uncoalesced packed loads, metadata
traffic, low occupancy, and library crossover thresholds.

Experiments: format sweep, packed-load vectorization, scales/zero-point layout,
branchless dequant variants, dequant-only microkernel, split-K, output rows per
workgroup, and dequant-direct-to-matrix path.

### Serving Kernels

Paged attention, MLA, KV-cache updates, MoE routing, sampling, and speculative
decode kernels often mix memory bandwidth, occupancy, and launch overhead.

Experiments: partition size per context length, fp8/cache dequant-on-read cost,
block-size sensitivity, grouped-GEMM versus per-expert dispatch crossover,
scatter/gather vectorization, and routing thresholds for tiny request batches.

## Decision Rules

A change is a candidate winner when:

- Focused correctness tests pass.
- Median performance improves by at least 3% for low-risk local changes, or
  8-10% for changes that add meaningful complexity.
- Required correctness shapes do not regress.
- Secondary performance shapes do not regress beyond an agreed tolerance.
- There is a plausible explanation backed by bytes, FLOPs, profiling data, or a
  clean A/B.

Reject or defer when:

- The win is inside measurement noise.
- The win appears only on toy shapes.
- Complexity rises without a durable real-shape win.
- The optimization depends on unavailable hardware or runtime features.
- The numeric contract changes.

## Recording Format

Each section in `perf/optimization_status.md` should contain:

- Status: not started, baselining, experimenting, candidate, landed, deferred.
- Current implementation and public route.
- References inspected, with exact paths.
- Correctness command and last result.
- Baseline table.
- Experiment table.
- Decision log.
- Open questions.

Raw results should be stored in a stable location once a harness exists, for
example:

```text
perf/results/YYYY-MM-DD/<kernel>/<run-id>.json
perf/results/YYYY-MM-DD/<kernel>/<run-id>.txt
```

Do not commit enormous profiler traces unless explicitly requested. Record their
local path, device, and summary instead.

## Final Verification Before Landing A Win

Before applying an optimization permanently:

```bash
python -m pytest <focused kernel test> -q
scripts/test
scripts/bench
```

For kernel-local Makefile workflows, run the focused `make`/`test_python.py`
commands documented by that kernel and record them in the status log.

When publishing a verified improvement, include the performance table in the PR
or commit notes. Commit messages should be normal descriptive messages with no
generated-by trailer.

## External References

- AMD ROCm documentation: HIP programming, rocBLAS, hipBLASLt, rocWMMA,
  rocprofiler, and profiling tools.
- Composable Kernel and AITER: production ROCm kernel/library references.
- Triton ROCm backend: compiler/runtime reference for generated tensor kernels.
- PyTorch ROCm: framework baseline and user-facing integration behavior.
- Sibling QuixiCore backend performance handbooks for shared contracts, shape
  sets, and optimization-recording discipline.
