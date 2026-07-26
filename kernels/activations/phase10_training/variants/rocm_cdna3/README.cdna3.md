# Phase 10 Training CDNA3 Variant

This standalone HIP harness lands the Phase 10 parity rows:

- `kd_kl_dense_fwd`
- `kd_kl_dense_bwd`
- `kd_kl_topk_fwd`
- `kd_kl_topk_bwd`
- `kd_ce_fused_fwd`
- `kd_ce_fused_bwd`
- `adamw_masked`
- `sgd`
- `softmax_backward`
- `silu_backward`

The KD contracts follow CPU `utils/kd_ref.cpp` and Metal
`utils/kd_kl_dense` / `utils/kd_kl_topk`. `adamw_masked` follows CPU
`optimizers/adamw_ref.cpp` and Metal `optimizers/optim`. `sgd` and
`softmax_backward` follow CPU `utils/tensor_ops_ref.cpp`, and
`silu_backward` follows CPU `activations/activations_ref.cpp`.

The CDNA3 routes use one row block for softmax/KD reductions and elementwise
updates for backward, optimizer, SGD, and SiLU. `adamw_masked` implements both
mask modes from the CPU reference: mode 0 skips inactive segments entirely,
while mode 1 updates moments and parameters without decoupled weight decay for
inactive segments.
