# QuixiCore ROCm Baseline Status

Method and measurement policy live in `perf/perf.md`. The running notebook —
the authority for what has been measured — is `perf/optimization_status.md`;
distilled truths are in `perf/findings.md`; the active experiment queue is
`perf/backlog.md`; raw benchmark output belongs under
`perf/results/YYYY-MM-DD/<label>/`.

## Environment

As of 2026-08-02 (last notebook entry with provenance). This file was
regenerated 2026-08-15 on a documentation host — no new measurements were
taken for it.

| property | value | source |
|---|---|---|
| GPU | AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 304 CUs, 64 KiB LDS/workgroup, 8x per node | notebook provenance blocks |
| ROCm / HIP | 7.2.4 / 7.2.53211-97f5574fe2 (`hipcc` from ROCm 7.2.4) | notebook provenance blocks |
| Container | bare metal (no container) | notebook provenance blocks |
| OS / driver | Ubuntu 22.04.5, driver 6.16.13 | 2026-07-07 qgemm entries |
| PyTorch | repo venv 2.12.1+rocm7.2 (`.venv`); perf.md §1 (2026-07-27) records nightly 2.14.0.dev20260723+rocm7.2 — confirm which is active: TBD (record on next MI300X session) | notebook + perf.md §1 |
| Empty-kernel launch floor | 1.56 us | 2026-08-01 v2_batch_prep |
| Measured achievable HBM ceiling | TBD (record on next MI300X session) | — |

## Build + gate (the standing invariant)

- `scripts/build|test|bench --arch cdna3 kernels` discovers active
  `variants/rocm_cdna3` Makefiles; each kernel also runs via its local
  `make test` / `make bench`.
- `kernels/common/cdna3_harness.cuh`: fp64-oracle comparison with tolerances
  from `../registry/tolerances.yaml` (elementwise rtol/atol AND aggregate
  rel-L1 + cosine; `Tol::bf16_output()`/`fp16_output()` widen only the
  elementwise bound to one storage ULP), deterministic host RNG with an
  adversarial generator, wave64 reductions, and a timing loop reporting
  median/min/max/spread, GB/s, TFLOP/s. `make -C kernels/common test`
  (self-test) runs before the sweep via `scripts/test kernels`.
- `perf/harness/run_kernel_bench.sh`: runs a kernel harness, archives raw
  output under `perf/results/<date>/<label>/` with
  GPU/ROCm/HIP/commit/container provenance, emits a pre-filled notebook
  entry, and warns when any reported spread exceeds 1.2x.
- `perf/configs/shapes.yaml` mirrors `../registry/benchmark-shapes.yaml`.
- Timing convention: HIP events, 10 warmup / 50 timed iterations, median,
  device pinned with `HIP_VISIBLE_DEVICES`; launch-bound kernels batch
  repeated launches per sample and record the batch size.
- The gate (AGENTS.md / CLAUDE.md): no kernel, routing, benchmark, or
  performance-claim commit without at least one focused optimization run
  recorded in `perf/optimization_status.md`; if the runtime is unavailable,
  report the blocker — no claim.

## Kernel roofline snapshot (2026-08-02)

Ceilings as stated in the notebook and handbook: HBM3 ~5.3 TB/s (vendor);
bf16 MFMA ~1300 TFLOP/s (vendor); fp8 MFMA measured at 1.98x fp16
K-throughput (2059 vs 1039 TFLOP/s, 2026-07-26 addendum). Measured achievable
HBM ceiling: TBD (record on next MI300X session).

| kernel / route | shape | measured | roofline position | date |
|---|---|---|---|---|
| rms_norm 64-lane | 16384x2048 f32 | 3109 GB/s | ~59% HBM | 2026-07-07 |
| layernorm 64-lane | 16384x2048 f32 | 2399 GB/s | ~45% HBM; 0.71x `F.layer_norm` | 2026-07-07 |
| gelu_fwd / add_ew | n=64Mi f32 | 2577 / 3699 GB/s | 49–70% HBM | 2026-07-06 |
| RoPE table-driven | T16384 H32 D128 | 3905–4045 GB/s | ~75% HBM | 2026-07-25 |
| GQA fwd MFMA | B4 H32 Hkv8 N2048 D128 | 97.3 / 77.2 TFLOP/s | ~7% bf16 peak; 0.30–0.34x SDPA flash | 2026-07-07 |
| qgemm wide NT=4 | M2048 N=K=4096 fp16 | 55.9 TFLOP/s | 0.12x hipBLASLt dense fp16 (471) | 2026-07-07 |
| qgemm ctaLDS + span fill, q2_K | M1024–4096 GLM shapes | 86–106 TFLOP/s | ~3x faster than fp16 GEMM on the same tile | 2026-07-26 |
| k-quant decode ksplit, q8_0 | o_proj 6144x16384, M=16 | 745.7 GB/s | ~14% HBM (decode/latency-bound) | 2026-07-26 |
| fp8_mqa_logits fp8 MFMA | H32 M512 N2048 | 111.8 TFLOP/s | bitwise Triton parity; fixed-order contract | 2026-08-01 |
| v2_temperature thr=1024 | T=64 V=151552 | 2293.8 GB/s | ~43% HBM | 2026-08-01 |
| mxfp4_gguf GEMV | 4096x4096 | 1840.7 GB/s | ~1/3 HBM (notebook's own estimate) | 2026-08-02 |
| Standalone `kernels/softmax` route | — | TBD (record on next MI300X session) | — | — |

## Per-family status

Updated from the 2026-07-06/07 "Kernel Family Status" table (now inside the
notebook's 2026-07-07 "Metal Gap Kernel Ports" entry) by all later entries.

| family | status | next step |
|---|---|---|
| Elementwise / norms | 64-lane widening LANDED +30–71% (07-07); 45–70% HBM | float4 + multi-row A/B vs `F.layer_norm` (backlog #5) |
| RoPE (rope_variants) | LANDED; pow→exp2+sincos 1.57–1.68x; 32-lane rows pinned (07-25) | LDS cos/sin sharing (parked) |
| Attention forward | MFMA tiling LANDED 13–16x (07-07); cross/biased/composites/swa KEPT (07-25) | K/V LDS + larger BQ + double-buffer (backlog #3) |
| Attention backward | correctness-valid wavefront kernels (07-06) | MFMA tiling (parked) |
| Dense GEMM (flux / bf16 / fp8 / int8 / block-scaled) | structurally capped 35–45 TFLOP/s by 16x16 tile; LDS-B REJECTED (07-07); explicit dequant + fp32 kept for MXFP8/NVFP4 | hipBLASLt is the practical dense path; wide-tile rewrite deferred |
| Quant GEMM (qgemm / qflux) | wide N-tile LANDED; multi-wave ctaLDS CANDIDATE 86–106 TFLOP/s; dequant4 in-GEMM + span fill LANDED (07-26) | format sweep + `qgemm_pick_route` (backlog #1) |
| Quant GEMV / decode | 29 formats bit-exact (07-06); small-M at 200–750 GB/s, decode/latency-bound; q8_0 stays — downgrade REJECTED (07-26); mxfp4_gguf added (08-02) | `dequant_into_shared` staging at small M (backlog #2) |
| Serving | 12 harnesses (07-06); Phase 2 KV codecs + MLA + paged attention (07-26); V2 batch prep / sample / sparse indexer / fp8_mqa_logits (08-01), 143 pass-lines | fuse launch-bound batch prep (backlog #4) |
| MoE | dense + quant MFMA (07-06); Phase 4 backward LANDED (07-26); wave64 grouped GEMM REJECTED — scalar kept | MFMA/LDS grouped-GEMM design (parked) |
| Collectives / distributed | fused collective+GEMM VALIDATED on 4 and 8 GPUs; chunked overlap REJECTED; Iris feasibility only (07-07) | Iris fused-epilogue kernel (parked) |
| Linear attention / SSM | TK tier complete, correctness-valid (07-06); Phases 8/9 LANDED (07-26) | — |
| Sampling / training / tensor-ops / conv / vision | Phases 7, 10, 11, 12, 13 LANDED correctness-first (07-26) | im2col+MFMA convolution when production shapes exist (parked) |
| BaseQ | Phase 5 LANDED, wave64 split-dot routes (07-26) | prefill specialization when a model routes it (parked) |

## Deferred (bigger projects, flagged not faked)

- Full wide-tile, double-buffered, LDS-staged dense GEMM rewrite (the
  HipKittens 256x256 model) — the same rewrite flux and qgemm both point at.
- Iris/XGMI producer-consumer fused GEMM+reduce_scatter (Step 8 follow-up).
- MFMA-tiled attention backward.
- CDNA4 (gfx950) variants — no hardware on this box; unmeasured claims are
  prohibited by the gate.
- `lm_head_topkp` (blocked on MFMA-swapping the consolidated
  `tm_kernels.cuh`); `followups_test` (mamba2 dependency).
- Full-node (8-GPU) collective sweeps; AITER fused-MoE/attention comparators.
- Normalizing the legacy `analysis/` baselines into `perf/results/` —
  partially superseded by the Phase 0 harness (2026-07-25).

## Decision log

| date | decision |
|---|---|
| 2026-07-07 | Fused single collective is the champion; chunked async overlap rejected as a hardware finding (RCCL collectives run on the CUs). |
| 2026-07-07 | `qgemm_pick_nt` occupancy-picks NT: NT=1 decode, NT=4 prefill; K-split owns decode occupancy. |
| 2026-07-25 | `kRopeLanes = 32` pinned; bf16/fp16-stored outputs judged at one-storage-ULP elementwise + registry aggregates. |
| 2026-07-25 | `run_kernel_bench.sh` warns at >1.2x spread after a contention-inverted result; Makefiles no longer hardcode `HIP_VISIBLE_DEVICES=0`. |
| 2026-07-26 | GLM-5.2 serving stays on the antirez routed quant (q8_0 attention + q2_K routed experts); no requantization. |
| 2026-08-01 | SlimServe wrappers adopt thr=256 (batch prep), thr=1024 (`v2_temperature`), thr=64 (`indexer_k_quant`). |

## Superseded (historical)

The section below is the 2026-07-06 version of this file, kept verbatim.
Reason: it was written on a documentation host before the MI300X box entered
the loop and points into the legacy `analysis/` tree; the canonical flow above
replaces it. Its migration tasks are largely delivered by the Phase 0 harness
(2026-07-25) and the `perf/results/` conventions now in force.

### Existing Baseline Index (2026-07-06, superseded)

| Area | Existing source | Current state |
|---|---|---|
| BF16 GEMM | `analysis/bf16_gemm/` | Existing MI-series scripts, JSON, and plots |
| FP8 GEMM | `analysis/fp8_gemm/` | Existing MI-series scripts, CSV/JSON, and plots |
| FP6 GEMM | `analysis/fp6_gemm/` | Existing JSON and plots |
| Attention forward | `analysis/attn/fwd/` | Existing MHA/GQA causal/non-causal plots |
| Attention backward | `analysis/attn/bkwd/` | Existing MHA/GQA causal/non-causal plots |
| LayerNorm | `analysis/layernorm/` | Existing row-kernel script and JSON |
| Rotary | `analysis/rotary/` | Existing row-kernel script and JSON |
| MLA decode | `analysis/mla_decode/` | Existing latency/gain plots |
| Framework/library baselines | `analysis/baselines/` | PyTorch, Triton, HIPBLASLT, AITER, CK-style comparisons |

### Migration Tasks (2026-07-06, superseded)

- Move reusable benchmark runners into `perf/harness/` or wrap them with
  `scripts/bench`.
- Add stable run output under `perf/results/YYYY-MM-DD/<kernel>/<run-id>/`.
- Summarize each accepted baseline in `perf/optimization_status.md`.
- Keep large profiler traces out of git; record trace paths and summaries only.
