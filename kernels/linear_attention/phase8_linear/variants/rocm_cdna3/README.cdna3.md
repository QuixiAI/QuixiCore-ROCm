# Phase 8 Linear Attention CDNA3 Variant

This standalone HIP harness lands the Phase 8 parity rows:

- `gated_linear_attention`
- `rwkv_wkv6`
- `rwkv_wkv7`
- `gdn_short_conv`
- `gdn_qkv_prepare`
- `gdn_gate_beta`
- `gdn_gated_rmsnorm`
- `gdn_recurrence`
- `linear_attention_unnormalized`

The recurrent GLA/RWKV contracts follow CPU
`linear_attention/llama_recurrent_ref.cpp`. The GDN contracts follow CPU
`linear_attention/gdn_ref.cpp`, with Metal `linear_attention/gdn/gdn.metal` used
for the row-owned state layout. `linear_attention_unnormalized` follows CPU
`linear_attention/linear_attention_extended_ref.cpp` and Metal
`linear_attention/linear_attn/linear_attn.metal`.

The CDNA3 kernels keep serial time-order where recurrence makes it contractual,
but parallelize independent state rows, output channels, and normalization rows
with wave64/block reductions. The implementation is fp32-only; storage-wrapper
variants are left as explicit non-ports because the Phase 8 tracker enumerates
operation-level parity rather than per-storage dispatch entries.
