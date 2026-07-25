# Kernel Roadmap

The ROCm backend currently has focused coverage for GEMM, attention, rotary,
layernorm, softmax, and early distributed work. The roadmap is to expand toward
the shared QuixiCore kernel contract.

Parity is tracked against two different sources, because they cover different
surfaces:

- [`docs/cuda-to-cdna3-port-status.md`](cuda-to-cdna3-port-status.md) inventories
  kernel directories in `../QuixiCore-CUDA/kernels` and records which have
  active CDNA3 variants, which are planned, and which are capability-gated by
  CUDA/NVIDIA-specific mechanisms. **This port is complete.**
- [`docs/metal-cpu-parity-gaps.md`](metal-cpu-parity-gaps.md) inventories the
  operations published by the **Metal** and **CPU** manifests that CUDA never
  published, and is therefore the current work list: **178 kernels in 13
  phases**, including two families (vision, convolution/audio) with no ROCm
  directory at all.

The CUDA tracker alone is not sufficient for parity: `QuixiCore-CUDA` publishes
only family-level metadata, while Metal publishes 56 operation-level entries and
CPU publishes ~290 public symbols.

Priorities:

1. Move new work into `kernels/<family>/<operation>/`.
2. Expand `.quixicore/kernels.yaml` from family-level status to operation-level
   status.
3. Close the Metal/CPU gap phase by phase, hot path first: attention and RoPE
   variants, quantized KV-cache codecs, decode-path matmul epilogues, MoE
   backward, BaseQ, then the vision/convolution/audio families.
4. Keep distributed kernels capability-gated until the runtime requirements are
   explicit.

Every kernel carries its own focused performance run recorded in
`perf/optimization_status.md`. Use the shared harness
(`kernels/common/cdna3_harness.cuh`) and `perf/harness/run_kernel_bench.sh`
rather than rebuilding the oracle/timing pattern per kernel.
