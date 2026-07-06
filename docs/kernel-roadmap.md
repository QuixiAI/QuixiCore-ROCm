# Kernel Roadmap

The ROCm backend currently has focused coverage for GEMM, attention, rotary,
layernorm, softmax, and early distributed work. The roadmap is to expand toward
the shared QuixiCore kernel contract.

CUDA-to-CDNA3 parity is tracked in
[`docs/cuda-to-cdna3-port-status.md`](cuda-to-cdna3-port-status.md). That file
is the current inventory for kernel directories in `../QuixiCore-CUDA/kernels`
and records which ones have active CDNA3 variants, which are planned, and which
are capability-gated by CUDA/NVIDIA-specific mechanisms.

Priorities:

1. Move new work into `kernels/<family>/<operation>/`.
2. Expand `.quixicore/kernels.yaml` from family-level status to operation-level
   status.
3. Add quantization, serving, sampling, MoE, and SSM coverage.
4. Keep distributed kernels capability-gated until the runtime requirements are
   explicit.
