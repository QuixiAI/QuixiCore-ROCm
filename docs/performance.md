# Performance

ROCm performance work currently lives under `analysis/`, `docs/profiling/`, and
kernel-local benchmark scripts.

The target QuixiCore layout is:

```text
perf/
  harness/
  configs/
  results/
  baselines/
```

Benchmark reports should include GPU, ROCm version, container image, command
line, input shape, dtype, quant format, and relevant kernel variant.
