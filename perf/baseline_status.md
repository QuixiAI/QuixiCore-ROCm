# QuixiCore ROCm Baseline Status

Method and measurement policy are described in `perf/perf.md`. Raw benchmark
output should live under `perf/results/`; current historical ROCm results also
live under `analysis/`.

## Environment

Date: 2026-07-06.

The current local host is a documentation/editing machine, not a ROCm validation
host. Runtime baselines should be recorded on an AMD GPU machine with ROCm, HIP,
and the expected benchmark dependencies installed.

Each real baseline entry should record:

- AMD GPU model and target architecture.
- ROCm version, HIP compiler version, driver/runtime versions, and container.
- Git commit or working-tree label.
- Command line and benchmark script path.
- Warmups, measured iterations, median, variance, and raw result path.
- Correctness tolerance and observed error.

## Existing Baseline Index

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

## Migration Tasks

- Move reusable benchmark runners into `perf/harness/` or wrap them with
  `scripts/bench`.
- Add stable run output under `perf/results/YYYY-MM-DD/<kernel>/<run-id>/`.
- Summarize each accepted baseline in `perf/optimization_status.md`.
- Keep large profiler traces out of git; record trace paths and summaries only.
