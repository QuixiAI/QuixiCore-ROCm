# Development

QuixiCore ROCm uses HIP/ROCm source, per-kernel Makefiles, Docker workflows, and
profiling tools while tracking the shared QuixiCore contract.

Use the common scripts first:

```bash
scripts/configure
scripts/build help
scripts/test help
scripts/bench
```

Most current workflows expect a ROCm container and `source env.src`. New
contract work should follow `docs/repository-structure.md` and update
`.quixicore/kernels.yaml`.
