# Repository Structure

QuixiCore-ROCm should share the same contract-facing structure as the other
QuixiCore backends while keeping ROCm-specific build, profiling, distributed,
and architecture choices below the operation boundary.

The rule is: public taxonomy is common; implementation details are native ROCm.

## Target Layout

```text
QuixiCore-ROCm/
  .quixicore/
    backend.yaml
    kernels.yaml
    quant-formats.yaml

  docs/
    repository-structure.md
    development.md
    kernel-roadmap.md
    performance.md
    backend-notes.md

  include/quixicore/rocm/
    backend.hpp
    runtime.hpp
    ops.hpp

  src/
    runtime/
    dispatch/
    errors/

  kernels/
    common/
    norms/
    activations/
    attention/
    linear_attention/
    ssm/
    matmul/
    quantization/
    moe/
    sampling/
    serving/
    optimizers/
    collectives/
    utils/

  bindings/
    c/
    python/
    pytorch/

  tests/
    correctness/
    integration/
    smoke/
    testdata/

  perf/
    harness/
    configs/
    results/
    baselines/

  examples/
  scripts/
  tools/
  assets/
  analysis/
  training/
```

ROCm-specific additions that may remain at the top level:

```text
Makefile
kernels/common.mk
Doxyfile
env.src
distributed-kernels/
```

`distributed-kernels/` should migrate into `kernels/collectives/` or an
operation-specific `variants/distributed/` directory when the operation becomes
part of contract coverage.

## Manifests

`.quixicore/backend.yaml` identifies this repository as the ROCm backend and
declares supported CDNA targets and contract compatibility.

`.quixicore/kernels.yaml` should be the machine-readable parity source for
implemented operations:

```yaml
operations:
  dense_gemm:
    family: matmul
    status: implemented
    path: kernels/matmul/dense_gemm
    bindings:
      pytorch: bindings/pytorch/dense_gemm.cpp
    tests:
      correctness: tests/correctness/matmul/dense_gemm
    benchmarks:
      default: perf/configs/matmul_dense_gemm.yaml
    variants:
      - name: rocm_cdna3
        status: optimized
      - name: rocm_ck
        status: experimental
```

`.quixicore/quant-formats.yaml` should list supported quant formats, packing
layouts, and ROCm-only layout constraints.

## Kernel Families

The top-level directories under `kernels/` are semantic families, not build or
profiling buckets:

- `norms/`: RMSNorm, LayerNorm, add-norm, norm-to-quant, QK norm.
- `activations/`: GELU, GLU, SiLU/SwiGLU helpers, standalone softmax.
- `attention/`: flash attention, causal/non-causal/varlen attention, backward,
  paged attention, MLA, rotary, quantized-KV attention, state merging.
- `linear_attention/`: Based, Hedgehog, linear attention, causal/decay linear
  attention, GDN, complex linear attention primitives.
- `ssm/`: Mamba, SSD, selective scan, FFT convolution.
- `matmul/`: dense GEMM, staged GEMM, complex matmul, Flux, CK/rocBLAS-backed
  or custom HIP GEMM.
- `quantization/`: act quant, runtime quant, qgemm, qgemv, quantized LM head,
  fp8/int8/fp4 packing, TurboQuant.
- `moe/`: routing, expert alignment, gather/scatter, grouped GEMM, quantized
  MoE GEMM, LoRA alignment, finalize.
- `sampling/`: sampling, logit transforms, penalties, rejection sampling, beam
  search, speculative decode and EAGLE helpers.
- `serving/`: KV cache mutation, block/page tables, indexers, MInference, cache
  copy/gather helpers.
- `optimizers/`: AdamW and other training optimizer kernels.
- `collectives/`: RCCL-style collectives and fused collective kernels.
- `utils/`: bit packing, column permutation, Hadamard/FWHT, small reusable
  user-visible utilities.

## Operation Layout

Prefer one directory per operation:

```text
kernels/<family>/<operation>/
  README.md
  include/
  src/
  variants/
    rocm_cdna2/
    rocm_cdna3/
    rocm_cdna4/
    rocm_ck/
    rocm_rocblas/
  tests/
  bench/
```

For small operations, direct `.hip`, `.cpp`, or `.cu` compatibility files under
the family are acceptable until there is more than one implementation.

## Tests And Benchmarks

Correctness and performance assets should mirror the kernel taxonomy:

```text
tests/correctness/<family>/<operation>/
perf/configs/<family>_<operation>.yaml
perf/baselines/<family>/<operation>/
```

Common developer entrypoints should exist:

```text
scripts/configure
scripts/build
scripts/test
scripts/bench
scripts/coverage-report
scripts/clean
```

For ROCm these scripts should wrap the existing Make/HIP flow, Docker launch
helpers, and profiler tooling as appropriate.

## Current Migration Map

| Current area | Target area |
| --- | --- |
| `kernels/attn` | `kernels/attention/` |
| `kernels/layernorm` | `kernels/norms/layernorm/` |
| `kernels/rotary` | `kernels/attention/rotary/` |
| `kernels/softmax` | `kernels/activations/softmax/` |
| `kernels/gemm` | `kernels/matmul/` |
| `kernels/torch_scaled` | `kernels/matmul/scaled_matmul/` and `bindings/pytorch/` as appropriate |
| `distributed-kernels` | `kernels/collectives/` or operation-specific `variants/distributed/` |
| `training` | `training/` or `examples/training/`, not contract kernel source |
| `analysis` | `analysis/` or `docs/analysis/`, not contract kernel source |

Move files in behavior-preserving steps. Rename APIs only when synchronizing
the CUDA, Metal, ROCm, XPU, and Gaudi bindings deliberately.

## Rules For New Work

- Add new kernels under semantic family directories.
- Keep PyTorch, Python, and extension glue in `bindings/`.
- Keep HIP, CK, rocBLAS, rocWMMA, RCCL, and architecture-specific choices under
  operation variants.
- Use `collectives/` for multi-GPU ROCm/RCCL extensions; mark them capability
  gated in `.quixicore/kernels.yaml`.
- If an operation has no meaningful ROCm implementation, mark it unsupported in
  metadata rather than adding a stub kernel.
