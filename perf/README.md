# Benchmarking

QuixiCore ROCm benchmarks should use native HIP, ROCm, framework, and library
timing/profiling tools.

The operating guide is [`perf.md`](perf.md). The running status notebook is
[`optimization_status.md`](optimization_status.md), with baseline snapshots in
[`baseline_status.md`](baseline_status.md).

Current benchmark and profiling workflows still live under `analysis/`,
`docs/profiling/`, and kernel-local scripts. Use `scripts/bench` as the common
entrypoint when possible, and record legacy commands exactly when a benchmark has
not moved under `perf/` yet.

Initial reporting should include:

- QuixiCore ROCm commit.
- QuixiCore contract version.
- AMD GPU target and driver/runtime versions.
- ROCm and HIP compiler versions.
- Container image and relevant environment variables.
- Kernel family, operation, dtype, quant format, and shape.
- Warmup iterations and measured iterations.
- Latency summary.
- Throughput summary where applicable.
- Correctness tolerance and observed error.

Raw benchmark output should be written under `perf/results/`, which is ignored by
git. Summaries that matter for future work should be copied into a tracked
status document.
