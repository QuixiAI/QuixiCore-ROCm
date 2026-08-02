# QuixiCore ROCm Performance Handbook

This is the operating guide for baselining and optimizing QuixiCore ROCm kernels.
It keeps the hardware-independent discipline shared with the Metal handbook, but
every hardware assumption, roofline number, and worked example here is AMD CDNA3.

The goal is not to collect tricks. For each kernel: find references, state a
bottleneck hypothesis, measure a clean baseline, run controlled experiments, keep
only verified wins, and record enough detail that the next pass starts from
evidence instead of memory.

The running notebook is `perf/optimization_status.md` (the authority for what has
been measured — read it before forming a hypothesis). Baseline snapshots live in
`perf/baseline_status.md`. Raw results belong under `perf/results/`.

---

## 1. Target Hardware — Know These Numbers

**Every roofline judgement in this repo is against these.** Do not optimize
without knowing which side of the ridge point you are on.

Development box (verified 2026-07-27 via `rocminfo` / `torch.cuda`):

| property | value |
|---|---|
| GPU | AMD Instinct MI300X, 8x per node |
| Arch target | `gfx942:sramecc+:xnack-` (CDNA3) |
| Compute units | 304 |
| **Wavefront** | **64 lanes** — not 32. See §3. |
| HBM3 per GPU | 192 GiB |
| LDS per workgroup | 64 KiB |
| Peak engine clock | 2100 MHz |
| Host | AMD EPYC 9754, 128 cores |
| ROCm / HIP | 7.2.53211 (`hipcc` 7.2.4) |
| PyTorch | 2.14.0.dev20260723+rocm7.2 |

Vendor-published peak throughput (use as roofline ceilings; measured achievable
peak is always lower — see the flux example in §7):

| path | peak |
|---|---|
| HBM3 bandwidth | **5.3 TB/s** |
| FP16 / BF16 matrix (MFMA) | ~1307 TFLOP/s |
| FP8 / INT8 matrix (MFMA) | ~2615 TFLOP/s |
| FP32 vector | ~81.7 TFLOP/s |
| FP32 / FP64 matrix | ~163.4 TFLOP/s |

**Ridge point** (arithmetic intensity where a kernel stops being memory-bound):

```text
ridge_bf16_matrix = 1307e12 / 5.3e12  ~= 247 FLOP/byte
ridge_fp32_vector =   81.7e12 / 5.3e12 ~=  15 FLOP/byte
```

Compute your kernel's AI and compare. A GEMM at `M=N=K=4096` has
AI ≈ 2·4096³ / (3·4096²·2 bytes) ≈ 683 FLOP/byte → compute-bound. A decode GEMV
with one activation column has AI ≈ 2 FLOP/byte → hard memory-bound, and no
amount of MFMA tuning will help it.

Multi-GPU: 8 GPUs per node over XGMI. Collectives go through RCCL. TP2 uses only
GPUs 0–1, so **GPUs 2–7 are free for concurrent benchmark runs** — use
`HIP_VISIBLE_DEVICES` to run experiments in parallel rather than serializing.

---

## 2. Principles

Correctness comes before performance. A change is not a win until it passes the
kernel's correctness tests, improves the target metric on realistic shapes, and
does not regress supported edge shapes or numeric tolerances.

Attack a *named* bottleneck:

- **Memory-bound** — reduce bytes moved, improve coalescing, improve cache/LDS
  reuse, avoid extra global passes, use narrower formats.
  *Caveat: narrower is not automatically faster — see §7, q8_0→q5_K.*
- **Compute-bound** — raise arithmetic intensity, feed MFMA effectively, reduce
  scalar side work, fuse epilogues.
- **Decoder-ALU-bound** — specific to quantized decode: the kernel is nominally
  memory-bound but the unpack/dequant ALU work is the real limiter. Diagnose by
  measuring achieved fraction of HBM roofline; if it is ~10–20% and narrower
  formats do not help, you are here.
- **Latency-bound** — grow resident work, reduce serial loops, batch tiny
  launches, avoid unnecessary host/device sync.
- **Occupancy-bound** — tune workgroup size, waves/workgroup, VGPR pressure, LDS
  footprint, grid size.
- **Synchronization-bound** — reduce `__syncthreads`, LDS traffic, atomics.
- **Launch-bound** — fuse small ops or route to a library path.
- **Structurally-bound** — the output tile / algorithm shape caps you far below
  peak regardless of micro-optimization. **Check for this first**: it makes every
  load/store experiment a waste of time (§7, flux).

---

## 3. CUDA → CDNA3 Porting Rules

Most kernels here were ported from CUDA. These are the mistakes that porting
makes, in the order they cost the most.

**Wavefront is 64, not 32.** A ported kernel that assumes `warpSize == 32`,
hardcodes `0xffffffff` shuffle masks, or sizes reductions for 32 lanes will be
*correct but half-width*. Measured gain from widening the row-kernel family to
64 lanes (`optimization_status.md`, 2026-07-07):

| shape (rows × hidden) | rms 32-lane | rms 64-lane | ln 32-lane | ln 64-lane |
|---|---|---|---|---|
| 4096 × 768 | 1415 GB/s | **2196 GB/s (+55%)** | 1181 GB/s | **1785 GB/s (+51%)** |
| 4096 × 4096 | 1810 GB/s | **2613 GB/s (+44%)** | 1164 GB/s | **1878 GB/s (+61%)** |
| 16384 × 2048 | 2673 GB/s | 3070 GB/s (+15%) | 1800 GB/s | 2410 GB/s (+34%) |
| 65536 × 4096 | 2120 GB/s | 2791 GB/s (+32%) | 1608 GB/s | 2265 GB/s (+41%) |

Audit any newly ported kernel for this before benchmarking anything else.

**Never read `warpSize` from a device-property struct on the host.** See §4 —
there is a live ABI hazard that makes this return the clock rate.

**FP8 on gfx942 is `fnuz`**, not OCP `e4m3`/`e5m2`. Weights converted assuming
OCP encoding will be silently wrong (bias differs). Convert explicitly and verify
against a host reference.

**`hipify-perl` handles the mechanical layer only** — headers,
`__nv_bfloat16`→`__hip_bfloat16`, runtime API. It does not fix warp width, does
not fix shuffle masks, and does not fix `cp.async`/TMA/WGMMA usage, which have no
CDNA3 equivalent and must be redesigned (LDS staging, or nothing).

**Do not port CUDA occupancy heuristics.** VGPR budgets, LDS sizes, and the
scheduler differ. Re-sweep workgroup size on CDNA3 rather than trusting a tuned
CUDA constant.

---

## 4. Environment Hazards (ROCm-specific landmines)

These have each cost real debugging days. Check them before chasing a "kernel
bug".

**`hipDeviceProp_t` ABI skew — the `warpSize == 2100000` trap.** ROCm ships two
struct layouts: `hipDeviceProp_tR0600` (current, `sizeof` 1472, `warpSize` at
offset 308) and `hipDeviceProp_tR0000` (legacy, `sizeof` 792, `warpSize` at 276).
`hipGetDeviceProperties` is a versioned thunk. If *any* loaded library exports a
stub for `hipGetDevicePropertiesR0600` backed by the legacy implementation, it
interposes process-wide on every PLT-bound HIP call resolved afterwards — and
`warpSize` is then read at the R0600 offset of `clockRate`, i.e. **2100000**
(2100 MHz in kHz). Kernels then launch with `blockDim.x = 2100000` and die.

Observed cause: importing `tilelang` loads its bundled `libhip_stub.so`, which
exports the full public HIP ABI. Symptom is a nonsense `blockDim` or shared-memory
size far from anything in your code. Diagnostic: a 30-second A/B — run the failing
kernel with and without the suspect import.

Note the asymmetry that makes this hard to see: PLT-bound calls obey global symbol
interposition, but `dlsym(handle, ...)` does not. A probe using `dlsym` will
report everything as fine while the real call path is hijacked.

**`fork` poisons a child's HIP runtime** if the parent has already made a HIP
call. Use `spawn` for multi-process workers.

**`pgrep -f "<pattern>"` matches its own wrapper shell.** `until ! pgrep -f
"make"; do sleep 5; done` never exits. Use a PID file, or `pgrep -f pat | grep -v
$$`.

**Repeated re-reads report above-HBM bandwidth.** Batched timing of a small
buffer measures cache, not HBM. A/B comparisons stay valid; absolute GB/s above
~5.3 TB/s means "cache-resident", not "magic".

---

## 5. Timing Rules

Use native device timing for device work.

- **HIP events** around the kernel or library call for custom HIP paths.
- Synchronize *outside* the measured region unless host sync is the metric.
- Warm up first — module load, autotuning, graph capture, and power state all
  perturb the first iterations.
- Do not allocate, initialize, randomize, or copy inputs inside the measured
  region.
- For tiny kernels, batch repeated launches per sample and divide; **record the
  batch size**. One launch per sync measures dispatch latency, not the kernel.
- Re-run surprising results on an idle machine. With 8 GPUs, "idle" means
  *your* GPU is idle — check with `rocm-smi` rather than assuming.
- For PyTorch on ROCm, device sync is exposed through `torch.cuda.synchronize()`.

Standard settings used by existing entries: 10 warmup / 50 timed iterations,
HIP-event median, `HIP_VISIBLE_DEVICES` pinned to one GPU.

Derived metrics:

```text
GEMM FLOPs          = 2 * M * N * K
attention FLOPs    ~= 4 * B * H * N * N * D      (halve for causal)
quant decode GB/s   = packed_weight_bytes_read / seconds / 1e9
row-kernel GB/s     = conservative required reads+writes / seconds / 1e9
arithmetic intensity = FLOPs / bytes_moved       (compare against §1 ridge point)
```

State when an estimate ignores cache reuse, repeated passes, metadata reads, or
write allocation.

### Profiling with rocprofv3

When timing alone does not explain a result:

```bash
# kernel trace: per-dispatch durations, grid/workgroup sizes
rocprofv3 --kernel-trace --output-format csv -d out/ -- ./bench.out

# hardware counters
rocprofv3 --pmc GRBM_GUI_ACTIVE SQ_WAVES SQ_INSTS_VALU SQ_INSTS_MFMA \
          TCC_HIT_sum TCC_MISS_sum --output-format csv -d out/ -- ./bench.out
```

Counters worth knowing:

- `SQ_INSTS_VALU` vs `SQ_INSTS_MFMA` — is the kernel doing matrix work or scalar
  side work? A quant kernel dominated by VALU is decoder-ALU-bound.
- `TCC_HIT_sum` / `TCC_MISS_sum` — L2 behaviour; distinguishes a real HBM-bound
  kernel from a cache-thrashing one.
- `SQ_WAVES` vs CU count — occupancy sanity check.

Record which counters were collected, the local trace path, and the conclusion.
Do not commit large traces.

---

## 6. Reference Search Protocol

Inspect references before designing or tuning. Record exact files in
`perf/optimization_status.md`.

- Existing ROCm analysis scripts under `analysis/` (`bf16_gemm/`, `fp8_gemm/`,
  `fp6_gemm/`, `attn/fwd/`, `attn/bkwd/`, `layernorm/`, `rotary/`,
  `mla_decode/`, `baselines/`).
- Kernel-local README/test files under `kernels/`.
- `docs/profiling/` for rocprof workflows.
- Production ROCm references: **Composable Kernel**, **AITER**, **hipBLASLt**,
  **rocBLAS**, **rocWMMA**, **Triton ROCm backend**, and `llama.cpp`'s
  `ggml-cuda`/HIP path for quantized decode geometry.
- Sibling QuixiCore backend docs for operation contracts and shape sets.

```bash
rg -n "mfma|wmma|lds|wave|__builtin_amdgcn|hipEvent|hipblas|rocblas|ck_" .
rg -n "layernorm|rms_norm|softmax|gemm|attention|quant|mla|paged" analysis kernels
rg -n "rocprof|pmc|counter|trace|roofline" docs analysis perf
```

Classify every idea before acting on it:

- **Portable algorithm idea** — worth considering.
- **CDNA3-specific mechanism** — translate only if the target arch exposes it.
- **CUDA-specific mechanism** (`cp.async`, TMA, WGMMA) — no CDNA3 analogue;
  redesign or drop.
- **Library/framework baseline** — adopt as a comparison.
- **Benchmark shape or oracle idea** — usually adopt.

Do not import implementation code from references without a license and
provenance review.

---

## 7. Established Findings — Do Not Re-Derive

Distilled from `perf/optimization_status.md`. Treat as current truth until
re-measured; each entry names the date so it can be challenged with new data.

### Wins

| finding | effect | date |
|---|---|---|
| Row-kernel 64-lane widening | +15–61% GB/s (table in §3) | 2026-07-07 |
| Attention forward MFMA-tiling | **13–16x** | 2026-07-07 |
| `qgemm` wide N-tile (X-reuse) | +47–54% at M≥256 | 2026-07-07 |
| LDS-staged k-quant tile decode (span fill) | 1.4–2.0x on ctaLDS | 2026-07-26 |
| In-GEMM k-quant decode (dequant4 MFMA fragment) | landed | 2026-07-26 |

Flat elementwise reference points (n = 64 Mi): `gelu_fwd` 2577 GB/s,
`add_ew` 3699 GB/s. Row kernels peak ~3070 GB/s. Against a 5.3 TB/s roofline
these are 49–70% — reasonable for read+write streaming, and the bar a new
elementwise kernel must clear.

### Rejected — with the reason, so they are not retried

**`flux` GEMM LDS-B staging — REJECTED.** Staging a 16×16 B tile through LDS to
fix uncoalesced column-strided loads gave +16% at 2048³ but **−6% at 4096³**
(per-k-step barriers stop amortizing at large K). The decisive observation:
*both* kernels sit at 35–45 TFLOP/s against ~1300 peak, because the structural
limiter is the 16×16 output tile, not the B load. **Lesson: find the structural
limiter before optimizing memory access.** A 3% load-path win is noise when you
are at 3% of peak.

**`qgemm` LDS-staged wide tile — REJECTED.** Bit-identical, but slower than the
existing register-fragment wide kernel. Kept in `qgemm_bench.cu` as a comparator
only.

**Chunked-RCCL compute/comm overlap — REJECTED.** Chunking along N with async
per-chunk collectives measured **0.82–0.96x** vs a single fused collective on 4
and 8 GPUs. Correct, just slower. A hardware finding, not a bug.

**Attention weight downgrade q8_0 → q5_K — REJECTED.** The hypothesis was
textbook-correct and still wrong, which is why it is worth remembering. Decode is
weight-bandwidth bound (AI ≈ 6 FLOP/byte vs a roofline far above it), attention
tensors are ~56% of per-token decode traffic, so ~35% fewer bytes should have
bought ~20% end-to-end. It did not: **the k-quant decoder is ALU-bound, so
narrower formats cost more unpack work than they save in bytes.** Every denser
format tested (q6_K, q5_K, q4_K, q4_0, iq4_nl, mxfp4, nvfp4) was *slower in wall
clock* than q8_0 at attention shapes despite moving fewer bytes; nvfp4 was worst
at ~2.5x slower while moving 47% fewer bytes.

**Generalized rule:** on CDNA3, "fewer bytes" only wins if the kernel is actually
near the bandwidth roofline. Measure achieved fraction of roofline *first*. At
~14% of roofline you are ALU-bound and format changes will backfire.

### The LDS pattern

LDS staging **lost** for dense GEMM B-tiles and for `qgemm` wide tiles, but
**won 1.4–2.0x** for k-quant span-fill decode. The distinction is reuse per
barrier: staging pays only when many lanes read each staged byte multiple times.
Do not assume either direction — A/B it, and report which side of that line the
kernel is on.

---

## 8. Shape Strategy

Do not optimize only square toy shapes. Cover:

- Small edge shapes and non-power-of-two dimensions.
- Tile-aligned fast-path shapes.
- Tile-ragged shapes (padding/boundary predicates can dominate).
- Real model shapes from Llama/Qwen/DeepSeek/GLM projections and attention.
- Stress shapes: long context, large K/N, batch sweeps, many experts.

Starter shapes:

- Norm/softmax/GELU: rows `{4096, 16384, 65536}` × hidden
  `{64, 128, 256, 512, 768, 1024, 2048, 4096, 8192}`.
- GEMM: square `{1024, 2048, 4096, 8192, 16384}` plus LLM rectangles
  (`K=4096`, `N=11008`, `N=14336`).
- Quant GEMV/GEMM: `M ∈ {1,2,4,8,16,32,64,128}`, `N/K ∈ {4096, 8192, 16384}`.
  The GEMV→GEMM crossover is a *measured* threshold, not a guess.
- Attention: `D ∈ {64,128}`, context `{512, 2048, 8192}`; treat causal,
  non-causal, and GQA as separate cases until proven to share a winner.
- MoE: tokens `{128, 1024, 4096}`, experts `{8, 16, 64}`, top-k `{1, 2, 4}`.

**Always include M=1.** Decode is the shape that ships, and it behaves nothing
like M=256.

Record skipped shapes with the reason.

---

## 9. Per-Kernel Optimization Loop

1. **Inventory** — entry points, dtypes, shape constraints, tests, bench coverage.
2. **Find references** — `analysis/`, `kernels/`, `.reference/`, sibling docs.
3. **Establish correctness** against a deterministic host or framework reference.
4. **Measure the baseline** against framework, library, and naive-decomposed
   baselines.
5. **Classify the bottleneck** — bytes, FLOPs, AI vs §1 ridge point, achieved
   fraction of roofline, variance, rocprof counters. **Do this before writing any
   optimization.**
6. **Define experiments before editing code.** One meaningful factor at a time.
7. **Execute** focused correctness test first, then the same benchmark matrix.
8. **Decide** with recorded numbers and rejected alternatives.
9. **Record** in `perf/optimization_status.md`.

---

## 10. Experiment Catalogue

Templates. Not every kernel needs every experiment.

**Launch geometry** — sweep workgroup size and waves/workgroup (remember wave64);
rows/items per workgroup for row kernels; output tile and K/sequence block sizes
for GEMM/attention; split large reductions only when merge overhead is measured;
watch tail effects when grid size does not fill 304 CUs evenly.

**Memory layout and coalescing** — verify adjacent lanes touch adjacent
addresses; direct global loads vs LDS staging (see §7); padded/swizzled LDS for
bank conflicts; vectorized (`dwordx4`) loads where alignment allows; keep scales
and metadata in wavefront-friendly layouts.

**Tiling and reuse** — sweep `BM`/`BN`/`BK`, sequence block, rows per block,
experts per scheduling unit; compare library paths per shape class; separate
alignment-specialized fast paths from generic edge paths; record compiler flags
and `--offload-arch` whenever an MFMA path is used.

**Matrix and quant paths** — MFMA/WMMA/library vs scalar-vector HIP; dequant
direct-to-fragment vs materialize-then-matmul; hoist scales/zero-points out of
inner loops; format-specialized kernels when profiling shows runtime branching.

**Fusion** — fuse epilogues (bias, residual, scale, activation, norm) when the
intermediate would round-trip through HBM; fuse dequant with matmul when the
value is used once; split when register pressure or occupancy loss dominates.

**Branches and scalar side work** — hoist format/dtype/causal/dimension decisions
into templates or separate entry points; precompute base offsets; specialize
`D=64`, `D=128`, aligned K tiles, supported quant block sizes; measure a
decode-only microkernel when scalar work may be hiding the true bottleneck.

**Reductions and numerics** — wavefront (64-lane) reductions before LDS
reductions; keep fp32 accumulation for softmax, norms, attention, and long-K
reductions unless a lower-precision variant passes tolerance; deterministic
reduction order where the contract requires it; exactness is part of the contract
for integer/packing kernels.

**Routing and shape specialization** — find GEMV→GEMM and custom→library
crossovers by sweeping `M`; route tiny shapes to a library path when dispatch
dominates; add aligned fast paths only when edge handling stays correct.

---

## 11. Kernel-Specific Starting Hypotheses

Replace with measured facts as work progresses.

**BF16/FP8/FP6 GEMM** — measure against hipBLASLt, rocBLAS, CK, AITER, Triton.
Large square and LLM rectangles should be compute-bound; small-M decode becomes
memory- or launch-bound. *Known:* the current `flux` path is structurally capped
by its 16×16 output tile at 35–45 TFLOP/s (§7) — enlarging the output tile is the
open lead, not load-path tuning.
Experiments: tile sweep, MFMA path selection, LDS staging, library handoff
threshold, split-K/stream-K, alignment fast paths, epilogue fusion.

**Attention forward/backward** — depends on on-chip softmax state, Q/K/V traffic,
sequence tiling, launch geometry. MFMA tiling already bought 13–16x; treat
causal, non-causal, GQA, D=64, D=128 as separate cases.
Experiments: sequence block size, K/V staging vs direct loads, causal branch
placement, GQA reuse layout, dQ vs dKV split, logsumexp storage, recompute vs
store.

**LayerNorm / RMSNorm / Softmax / GELU / Rotary** — row reductions and
bandwidth-bound pointwise. Already 64-lane widened and at 49–70% of roofline;
further wins are likely small. Report GB/s and justify against the elementwise
reference points in §7.

**Quant GEMV/GEMM and quantized attention** — should win by reducing bytes, *but
verify you are bandwidth-bound first* (§7). If a lower-bit format is not faster,
the decoder is ALU-bound: investigate unpack cost, uncoalesced packed loads,
metadata traffic, occupancy, library crossover.
Experiments: format sweep, packed-load vectorization, scale/zero-point layout,
branchless dequant, dequant-only microkernel, split-K, output rows per workgroup,
dequant-direct-to-MFMA-fragment.

**Serving kernels** (paged attention, MLA, KV-cache, MoE routing, sampling,
speculative decode) — mix bandwidth, occupancy, and launch overhead.
Experiments: partition size per context length, fp8/mxfp8 dequant-on-read cost,
block-size sensitivity, grouped-GEMM vs per-expert dispatch crossover at small E,
scatter/gather vectorization, routing thresholds for tiny batches.

**Collectives** — the fused non-overlapped path is the current champion;
chunked-async overlap lost (§7). Re-test only with a materially different
mechanism (e.g. in-kernel XGMI), not by re-tuning chunk count.

---

## 12. Decision Rules

A change is a candidate winner when:

- Focused correctness tests pass.
- Median improves ≥3% for low-risk local changes, or ≥8–10% for changes adding
  meaningful complexity.
- Required correctness shapes do not regress.
- Secondary shapes do not regress beyond agreed tolerance.
- There is a plausible explanation backed by bytes, FLOPs, counters, or a clean
  A/B. *"It got faster and I don't know why" is a reason to keep measuring, not
  to land.*

Reject or defer when:

- The win is inside measurement noise.
- The win appears only on toy shapes, or only at one shape and regresses another
  (the `flux` LDS-B pattern).
- Complexity rises without a durable real-shape win.
- It depends on unavailable hardware or runtime features.
- The numeric contract changes.

---

## 13. Recording Format

Each section in `perf/optimization_status.md` should contain:

- **Status**: not started | baselining | experimenting | candidate | landed |
  deferred — and put the verdict in the heading (`— LANDED (13-16x)`,
  `— REJECTED`) so the table of contents is scannable.
- Current implementation and public route.
- References inspected, with exact paths.
- Correctness command and last result.
- Baseline table.
- Experiment table with the full environment line (GPU, ROCm/hipcc version,
  `HIP_VISIBLE_DEVICES`, warmup/iters, timing method).
- Decision log — **including why a rejected idea was rejected**, which is the
  highest-value content in the file.
- Open questions.

Every benchmark record should carry: git commit or working-tree label, QuixiCore
contract version, ROCm/HIP/compiler versions, GPU model and arch, kernel family
and entry point, dtype/format/shape, warmup and iteration counts, median and
p20/p80, correctness tolerance and observed max error, derived throughput, and
the raw output path.

```text
perf/results/YYYY-MM-DD/<kernel>/<run-id>.json
perf/results/YYYY-MM-DD/<kernel>/<run-id>.txt
```

Do not commit large profiler traces — record path, device, and summary.

---

## 14. Final Verification Before Landing A Win

```bash
python -m pytest <focused kernel test> -q
scripts/test
scripts/bench
```

For kernel-local Makefile workflows, run that kernel's documented `make` /
`test_python.py` commands and record them in the status log. For shared substrate
changes under `kernels/common/` (`cdna3_harness.cuh`, `cdna3_mfma.cuh`), run the
broader kernel correctness suite — those headers are used across families.

When publishing a verified improvement, include the performance table in the PR
or commit notes. Commit messages are normal descriptive messages with no
generated-by trailer.

---

## 15. External References

- **AMD CDNA3 ISA reference** and MI300 tuning guides — MFMA instruction shapes,
  VGPR/LDS budgets, wavefront scheduling.
- **AMD ROCm documentation** — HIP programming guide, rocBLAS, hipBLASLt,
  rocWMMA, rocprofiler/rocprofv3.
- **Composable Kernel**, **AITER** — production ROCm kernel/library references
  and the baselines to beat.
- **Triton ROCm backend** — compiler/runtime reference for generated kernels.
- **PyTorch ROCm** — framework baseline and user-facing route.
- **`llama.cpp`** HIP path — mature quantized decode geometry (multiple rows per
  wavefront, register-cached activations).
- Sibling QuixiCore backend handbooks — shared contracts, shape sets, and
  recording discipline. Note that Apple-specific conclusions (simdgroup matrix
  behaviour, command-buffer timing, "staging rarely pays") are *suggestions for
  experiments here, never rules*.
