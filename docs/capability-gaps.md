# ROCm Capability Gaps

Inventory date: 2026-07-27.

This file compares the ROCm backend with the **263-operation semantic
union** and **17 exact quant-format IDs** recorded in the
[QuixiCore umbrella capability map](https://github.com/QuixiAI/QuixiCore/blob/main/matrices/capability-map.md).

Backend snapshot: `main` @ `636ae5ae983f`.

Normalized adapter stubs, including planned practical inference and fused
operations, are indexed in `.quixicore/kernel-stubs.yaml` and declared in
`include/quixicore/rocm/contract_stubs.hpp`. Their counts differ from this
exact-ID evidence comparison because aliases are collapsed and planned
contracts are included.

Union source revisions: CUDA d959679b0163; Metal a6d984377288; ROCm 636ae5ae983f; XPU 67c70fe4dc0c; CPU 0159223979db.

## How gaps are classified

- **family-only metadata**: this backend marks the family implemented but
  does not publish the exact operation ID. This is an enumeration/evidence
  gap, not proof that the semantic kernel is missing.
- **partial-family coverage**: the exact ID is absent and the backend marks
  the family partial.
- **capability-gated**: the family or operation depends on hardware/runtime
  conditions and is not an unconditional capability.
- **planned family**, **no family claim**, **partial operation**, and
  **experimental operation** are implementation or maturity gaps relative
  to a fully evidenced union capability.

Exact accelerator stage/layout aliases remain separate because the umbrella
map preserves published operation IDs. A backend may close a metadata gap
by documenting a proven semantic collapse instead of adding duplicate code.

## Summary

| Measure | Count |
|---|---:|
| Union operation capabilities | 263 |
| Fully implemented or semantically mapped | 215 |
| Operation gaps or enumeration gaps | 48 |
| family-only metadata | 5 |
| partial-family coverage | 43 |
| Union quant-format IDs | 17 |
| Fully declared quant-format IDs | 0 |
| Quant-format gaps or missing declarations | 17 |
| quant: partial operation | 5 |
| quant: no exact format declaration | 12 |

## Operation gap list

| Union family | Capability | Gap class |
|---|---|---|
| Norms | `qk_norm_rope_positioned` | family-only metadata |
| Norms | `rms_norm` | family-only metadata |
| Activations | `gelu` | partial-family coverage |
| Activations | `gelu_backward` | partial-family coverage |
| Activations | `glu` | partial-family coverage |
| Activations | `silu` | partial-family coverage |
| Attention | `attention` | partial-family coverage |
| Attention | `mrope` | partial-family coverage |
| Attention | `paged_attention_q8_0` | partial-family coverage |
| Attention | `rope` | partial-family coverage |
| Attention | `rotary_positioned` | partial-family coverage |
| Linear attention | `gdn_recur` | partial-family coverage |
| Linear attention | `linear_attn` | partial-family coverage |
| State-space models | `selective_scan` | partial-family coverage |
| Dense matmul and projections | `dense_gemm` | family-only metadata |
| Dense matmul and projections | `lora_apply` | family-only metadata |
| Quantization | `act_quant_int8` | partial-family coverage |
| Quantization | `base_q_fused_consumers` | partial-family coverage |
| Quantization | `base_q_qkv` | partial-family coverage |
| Quantization | `base_q_swiglu` | partial-family coverage |
| Quantization | `fp8_gemm` | partial-family coverage |
| Quantization | `gguf_gemv` | partial-family coverage |
| Quantization | `nvfp4_gemv` | partial-family coverage |
| Quantization | `qgemm_int8` | partial-family coverage |
| Quantization | `qgemv_int4` | partial-family coverage |
| Quantization | `quantize_int4_group` | partial-family coverage |
| Mixture of experts | `moe_route_topk` | partial-family coverage |
| Sampling | `argmax` | partial-family coverage |
| Sampling | `sample_categorical` | partial-family coverage |
| Sampling | `top_k_sample` | partial-family coverage |
| Serving and caches | `embedding_lookup` | partial-family coverage |
| Serving and caches | `kv_cache_copy_blocks_q8_0` | partial-family coverage |
| Serving and caches | `kv_cache_gather` | partial-family coverage |
| Serving and caches | `kv_cache_gather_q8_0` | partial-family coverage |
| Serving and caches | `kv_cache_scatter` | partial-family coverage |
| Serving and caches | `kv_cache_scatter_q8_0` | partial-family coverage |
| Serving and caches | `masked_mean_pool_rms_l2` | partial-family coverage |
| Optimizers | `adamw` | partial-family coverage |
| Audio | `audio_conv1d` | partial-family coverage |
| Utilities and training | `cross_entropy` | partial-family coverage |
| Utilities and training | `dropout` | partial-family coverage |
| Utilities and training | `hadamard` | partial-family coverage |
| Attention | `attention_with_lse` | partial-family coverage |
| Utilities and training | `cross_entropy_backward` | partial-family coverage |
| Serving and caches | `embedding_backward` | partial-family coverage |
| Serving and caches | `indexer_k_gather` | partial-family coverage |
| Norms | `rms_norm_backward` | family-only metadata |
| Activations | `swiglu_oai` | partial-family coverage |

## Quant-format gap list

| Format ID | Gap class |
|---|---|
| `awq` | no exact format declaration |
| `base_qn` | no exact format declaration |
| `bitnet` | partial operation |
| `fp4` | partial operation |
| `fp8` | partial operation |
| `gguf` | partial operation |
| `int4_group` | no exact format declaration |
| `int8` | no exact format declaration |
| `marlin_awq_gptq_hqq` | no exact format declaration |
| `mx` | partial operation |
| `mxfp4` | no exact format declaration |
| `mxfp6` | no exact format declaration |
| `mxfp8` | no exact format declaration |
| `nvfp4` | no exact format declaration |
| `q8_0_kv` | no exact format declaration |
| `tq2_0` | no exact format declaration |
| `turboquant` | no exact format declaration |

## Evidence rule

Removing an implementation or maturity gap requires the backend's native
path, correctness coverage, focused performance evidence, and an updated
manifest/status entry. Removing a family-only metadata gap requires an exact
operation entry or a documented semantic alias backed by the existing tests
and performance notebook. Directory presence alone is not sufficient.

Evidence remains backend-owned in `perf/optimization_status.md`,
`perf/baseline_status.md`, `perf/results/`, and the
backend correctness tests.
