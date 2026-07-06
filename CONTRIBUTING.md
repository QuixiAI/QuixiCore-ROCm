# Contributing To QuixiCore ROCm

QuixiCore ROCm is one native backend in the QuixiCore family. Contributions
should preserve the shared QuixiCore contract while using ROCm-native
implementation techniques.

## Backend Boundary

- Implementation code belongs in this repository, not in the QuixiCore umbrella
  repository and not in another backend.
- Shared semantics belong in QuixiAI/QuixiCore.
- ROCm-specific tuning, HIP, CK, rocBLAS, rocWMMA, RCCL, profiling, and Docker
  workflows belong in this repository.

## Adding Or Changing A Kernel

1. Put source under `kernels/<family>/<operation>/` when adding new work.
2. If touching legacy ROCm directories, update toward the migration map in
   `docs/repository-structure.md`.
3. Add or update correctness coverage under `tests/correctness/<family>/<operation>/`
   or the current legacy test location.
4. Add or update benchmark coverage under `perf/` or the current `analysis/`
   workflow.
5. Update `.quixicore/kernels.yaml`.
6. Update `.quixicore/quant-formats.yaml` when quant layouts, packing, or
   supported formats change.
7. Document backend-specific behavior in the operation README or relevant docs.

## Required Checks

Use the common entrypoints when possible:

```bash
scripts/configure
scripts/build
scripts/test
scripts/bench
scripts/coverage-report
```

For legacy ROCm paths, per-kernel Makefiles and documented Docker flows are
still valid. Mention exactly which commands, ROCm version, container, and GPU
target you used in the pull request.

## Pull Request Checklist

- Kernel semantics match the QuixiCore contract or document a ROCm-only
  extension.
- Correctness tests cover the changed behavior.
- Benchmarks cover the relevant shapes or explain why benchmark coverage is not
  applicable yet.
- `.quixicore/kernels.yaml` reflects implementation status.
- `.quixicore/quant-formats.yaml` reflects quant format support.
- New source follows `docs/repository-structure.md`.
