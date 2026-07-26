# Phase 2 Quantized Decode CDNA3 Variant

This standalone HIP harness lands the six remaining Phase 2 parity rows:

- `kv_cache_scatter_bitnet_kv3`
- `kv_cache_gather_bitnet_kv3`
- `paged_attention_bitnet_kv3`
- `paged_attention_turboquant`
- `paged_attention_advanced`
- `quantized_attention`

The contracts follow CPU `attention/attention_bitnet_kv3.cpp`,
`attention/attention_turboquant.cpp`, `attention/attention_serving_ref.cpp`,
and the operation declarations in `include/quixicore_cpu/ops.h`.

The CDNA3 approach is correctness-first. BitNet-KV3 uses a destination-owned
scatter scan so duplicate slots resolve in CPU token order, keeps low-bit-first
3-bit packing, supports signed/unsigned code interpretation, none/integer
zero-point modes, group sizes dividing D, and FP16/FP32 scale storage.
TurboQuant attention preserves packed K, rotated V, per-32 scale metadata,
centroid decode, and the final inverse signed FWHT. Advanced paged attention
preserves optional block masks, ALiBi slopes, sink logits, sliding windows, and
softcap. `quantized_attention` covers the GGUF `q8_0` and `q4_0` formats that
the existing ROCm `attn_q` path already exercises.

Explicit non-ports: this variant does not add MFMA prefill attention, a unified
runtime dispatcher, or broader GGUF format dispatch for `quantized_attention`.
Those are follow-up performance/integration work after the tracked Phase 2
operation parity is closed.
