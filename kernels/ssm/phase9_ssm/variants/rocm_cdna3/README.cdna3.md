# Phase 9 State-Space CDNA3 Variant

This standalone HIP harness lands the Phase 9 parity rows:

- `mamba2_backward`
- `ssd_chunked_backward`
- `ssd_decode`
- `dsv4_hc_pre`
- `dsv4_hc_post`
- `dsv4_hc_comb`

The Mamba2/SSD contracts follow CPU `ssm/ssm_extended_ref.cpp`; the decode
layout is cross-checked against Metal `ssm/mamba2/mamba2.metal`. The DSV4
hyper-connection contracts follow CPU
`linear_attention/llama_recurrent_ref.cpp`.

`mamba2_backward` and `ssd_chunked_backward` share the same mathematical
contract in the CPU backend. The CDNA3 path uses separate direct-gradient
kernels for `grad_c`, `grad_b`/`grad_x`, and `grad_cumulative_log`, avoiding
atomics while preserving the triangular source/target dependency. `ssd_decode`
uses one wave64 block per output/state row. DSV4 pre/post are elementwise
parallel reductions over the fixed four-connection width, while `dsv4_hc_comb`
parallelizes independent tokens and keeps the small Sinkhorn loop local.
