# Phase 11 Tensor Ops CDNA3 Variant

This standalone HIP harness lands the Phase 11 elementwise and tensor-op parity
rows from `docs/metal-cpu-parity-gaps.md`.

The contracts follow CPU `utils/tensor_ops_ref.cpp`,
`activations/activations_ref.cpp`, `sampling/selection_ref.cpp`, and the small
vision tensor helpers in `vision/conv_pool_ref.cpp`.

The CDNA3 implementation uses direct elementwise grids for unary/binary/scatter
ops, one block per row or group for reductions and normalization, and
correctness-first row-owned serial loops for stable `argsort`,
`threshold_topk_indices`, and lower-triangular solve. `set_rows` and
`tensor_set_4d` use deterministic destination-owned scans so duplicate or
overlapping writes follow CPU loop order instead of racing.
