# QuixiCore ROCm Optimization Status

This is the running notebook for ROCm kernel implementation and optimization.
Raw output belongs under `perf/results/` or the current legacy analysis path;
stable conclusions belong here.

## Entry Template

Use this structure for every kernel family or optimization pass:

```text
## YYYY-MM-DD: <kernel or pass name>

Status: not started | baselining | experimenting | candidate | landed | deferred.
Current implementation:
Current public route:
References inspected:
Correctness:
Baseline:
Experiments:
Decision:
Open questions:
Raw results:
```

Record enough context to reproduce the run: GPU, ROCm/HIP version, container,
command, git commit or working-tree label, dtype, shape, quant format, warmups,
iterations, median, variance, correctness tolerance, and observed error.

## 2026-07-06: Shared Performance Documentation Port

Status: landed documentation scaffold.

Added the shared QuixiCore performance workflow from the Metal handbook:

- `perf/perf.md` for the optimization loop, measurement rules, shape strategy,
  experiment catalogue, decision rules, and verification checklist.
- `perf/optimization_status.md` as the running optimization notebook.
- `perf/baseline_status.md` as the stable baseline index.
- `perf/results/`, `perf/harness/`, `perf/configs/`, and `perf/baselines/`
  placeholders for the common layout.

Existing ROCm benchmark material remains under `analysis/`, `docs/profiling/`,
and kernel-local scripts until it is worth migrating into `perf/harness/`.

## Current Baseline Sources

Status: baselines exist, not yet normalized into the shared harness.

| Area | Source | Notes |
|---|---|---|
| BF16 GEMM | `analysis/bf16_gemm/` | MI325/MI350/MI355 scripts, JSON, plots |
| FP8/FP6 GEMM | `analysis/fp8_gemm/`, `analysis/fp6_gemm/` | Library and custom comparisons |
| Attention fwd/bwd | `analysis/attn/fwd/`, `analysis/attn/bkwd/` | GQA/MHA, causal/non-causal plots |
| LayerNorm | `analysis/layernorm/` | MI350/MI355 row-kernel benchmark |
| Rotary | `analysis/rotary/` | MI350/MI355 benchmark |
| MLA decode | `analysis/mla_decode/` | Context-length latency/gain plots |
| Framework/library baselines | `analysis/baselines/` | PyTorch, Triton, HIPBLASLT, AITER, CK where available |
| Profiling workflow | `docs/profiling/` | rocprof trace and PMC counter collection |

## Kernel Family Status

| Kernel family | ROCm status | Next optimization step |
|---|---|---|
| BF16 GEMM | baselined in analysis | Normalize benchmark metadata and compare custom/library routes by shape |
| FP8/FP6 GEMM | baselined in analysis | Record format-specific library/custom crossover points |
| Attention forward | baselined in analysis | Capture shape table and profile the top remaining bottleneck |
| Attention backward | baselined in analysis | Capture correctness tolerances and profile D=64/D=128 separately |
| LayerNorm | baselined in analysis | Convert row-kernel numbers to GB/s and add framework comparison table |
| Rotary | baselined in analysis | Record vectorization/layout conclusions |
| Softmax | implemented under `kernels/softmax` | Add baseline and correctness entry |
| Quant GEMV/GEMM | not normalized | Add benchmark matrix and reference comparisons |
| Serving kernels | partial analysis | Tie MLA/paged/KV/MoE/sampling work to shared status entries |

## Open Questions

- Which ROCm target should be the canonical baseline host for each family:
  CDNA3, CDNA4, or both?
- Should migrated benchmark output preserve the current `analysis/` JSON schema
  or convert directly to the shared `perf/results/YYYY-MM-DD/<kernel>/` layout?
- Which library baseline should be primary for each matrix family: hipBLASLt,
  rocBLAS, Composable Kernel, AITER, Triton, or a per-shape best-of table?
