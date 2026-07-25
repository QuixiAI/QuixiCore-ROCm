# Metal / CPU To CDNA3 Parity Gaps

This tracker inventories the operations published by the **Metal** and **CPU**
backend manifests that have no ROCm CDNA3 implementation, and records the work
to close them. It is the counterpart to
[`cuda-to-cdna3-port-status.md`](cuda-to-cdna3-port-status.md), which tracks the
already-complete CUDA port.

The CUDA tracker is not sufficient for parity: `QuixiCore-CUDA` publishes only
family-level metadata, while Metal publishes 56 operation-level entries and CPU
publishes ~290 public symbols across `include/quixicore_cpu/`. The gap below is
what the CUDA port never covered because CUDA never published it.

Inventory date: 2026-07-25. Sources compared:

- `../QuixiCore-Metal/kernels/**` (68 kernel directories) and
  `../QuixiCore-Metal/.quixicore/kernels.yaml`
- `../QuixiCore-CPU/include/quixicore_cpu/*.h` and `../QuixiCore-CPU/kernels/**`
- `../QuixiCore-CPU/parity/sibling_operations.tsv` and
  `../QuixiCore-CPU/docs/sibling-port-matrix.md`
- This repo's ~250 `__global__` entry points and `.quixicore/kernels.yaml`

**Total: 178 kernels in 13 phases.** An earlier estimate of ~150 was a
group-level approximation; the per-kernel enumeration below is exact and
supersedes it. The largest single correction is the elementwise/tensor-op
surface, which is 42 operations rather than the ~35 estimated.

## Status Meanings

- `planned`: no CDNA3 implementation exists yet.
- `active`: a CDNA3 variant exists and passes its fp64/reference oracle.
- `landed`: `active`, plus a focused performance run recorded in
  `perf/optimization_status.md` and raw output under `perf/results/`.

A kernel is not `landed` until it carries its own perf run. Per `AGENTS.md`, a
documented rejection is a valid run — an honest "no speedup found" entry lands
the kernel; a missing entry does not.

## Progress

| Phase | Area | Kernels | planned | active | landed |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Harness infrastructure | — | — | — | done |
| 1 | Attention & RoPE variants | 13 | 11 | 0 | 2 |
| 2 | Quantized KV-cache codecs | 17 | 17 | 0 | 0 |
| 3 | Dense matmul epilogues | 10 | 10 | 0 | 0 |
| 4 | MoE completeness | 7 | 7 | 0 | 0 |
| 5 | BaseQ canonical family | 9 | 9 | 0 | 0 |
| 6 | Quant authoring & quantized embedding | 14 | 14 | 0 | 0 |
| 7 | Sampling & embedding stragglers | 6 | 6 | 0 | 0 |
| 8 | Linear attention | 9 | 9 | 0 | 0 |
| 9 | State-space & hyper-connections | 6 | 6 | 0 | 0 |
| 10 | Training & distillation | 10 | 10 | 0 | 0 |
| 11 | Elementwise & tensor-op surface | 42 | 42 | 0 | 0 |
| 12 | Convolution & audio | 16 | 16 | 0 | 0 |
| 13 | Vision | 19 | 19 | 0 | 0 |
| | **Total** | **178** | **176** | **0** | **2** |

## Phase 0 — Harness Infrastructure

Complete. Docs/scaffolding only; makes no performance claim.

| Artifact | Purpose |
| --- | --- |
| `kernels/common/cdna3_harness.cuh` | fp64-oracle comparison, tolerances from `registry/tolerances.yaml`, deterministic RNG, wave64 reductions, timing with median/min/max + GB/s and TFLOP/s |
| `kernels/common/harness_selftest.cu` | Validates the harness itself, including that `compare()` rejects bad data |
| `perf/harness/run_kernel_bench.sh` | Runs a kernel harness, archives raw output with GPU/ROCm/HIP/commit provenance, emits a pre-filled notebook entry |
| `perf/configs/shapes.yaml` | Shape sets mirroring `registry/benchmark-shapes.yaml` plus per-family additions |

## Phase 1 — Attention And RoPE Variants (13)

Build on the landed MFMA flash kernel
(`kernels/attention/gqa/variants/rocm_cdna3/attn_kernel.cuh`). `cross_attention`
and `biased_attention` are that kernel plus per-batch key lengths and a score
bias; `swin_attention_d32` is a packed-D32 adapter over `biased_attention`.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `cross_attention` | `attention/cross_attention_ref.cpp` | — | **landed** |
| `biased_attention` | `attention/attention_extended_ref.cpp` | — | **landed** |
| `swin_attention_d32` | `attention/attention_composites_ref.cpp` | `attention/swin_attn` | planned |
| `decode_cache_attention` | `attention/attention_serving_ref.cpp` | `attention/attn_decode` | planned |
| `cascade_attention_multi` | `attention/attention_composites_ref.cpp` | — | planned |
| `mrope` | `attention/rotary_positioned_ref.cpp` | `attention/rotary` | planned |
| `rotary_positioned` | `attention/rotary_positioned_ref.cpp` | `attention/rotary` | planned |
| `rope_table` | `attention/rotary_positioned_ref.cpp` | `attention/rotary` | planned |
| `rope_interleaved_to_split` | `attention/rotary_positioned_ref.cpp` | `attention/rotary` | planned |
| `rope_backward` | `utils/tensor_ops_ref.cpp` | — | planned |
| `qk_norm_rope_positioned` | `attention/attention_extended_ref.cpp` | `norms/qk_norm_rope` | planned |
| `qk_norm_rope_split` | `attention/attention_extended_ref.cpp` | `norms/qk_norm_rope` | planned |
| `rope_q_norm` | `attention/attention_extended_ref.cpp` | `norms/qk_norm_rope` | planned |

## Phase 2 — Quantized KV-Cache Codecs And Decode Attention (17)

Extends the landed `kv_cache_*` and `paged_attention_*` kernels in
`kernels/serving/variants/rocm_cdna3/`. Canonical layouts are contractual:
Q8_0 keeps **separate** signed-int8 code planes and raw FP16 scale planes
(sparse negative blocks zero-fill or skip); MXFP8 keeps interleaved blocks;
TurboQuant keeps packed K and rotated V, with V staying in the signed-FWHT
domain until one final inverse transform and canonical K **not** transformed;
BitNet-KV3 keeps low-bit-first ordering with explicit signedness, zero-point
mode, group size, and FP16/FP32 scale encoding.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `kv_cache_scatter_q8_0` | `serving/serving_quant_ref.cpp` | `serving/kv_cache` | planned |
| `kv_cache_gather_q8_0` | `serving/serving_quant_ref.cpp` | `serving/kv_cache` | planned |
| `kv_cache_copy_blocks_q8_0` | `serving/serving_quant_ref.cpp` | `serving/kv_cache` | planned |
| `paged_attention_q8_0` | `attention/attention_q8_kv.cpp` | — | planned |
| `kv_cache_scatter_mxfp8` | `serving/serving_quant_ref.cpp` | — | planned |
| `kv_cache_gather_mxfp8` | `serving/serving_quant_ref.cpp` | — | planned |
| `paged_attention_mxfp8` | `attention/attention_mxfp8.cpp` | — | planned |
| `kv_cache_scatter_bitnet_kv3` | `serving/serving_quant_ref.cpp` | — | planned |
| `kv_cache_gather_bitnet_kv3` | `serving/serving_quant_ref.cpp` | — | planned |
| `paged_attention_bitnet_kv3` | `attention/attention_bitnet_kv3.cpp` | — | planned |
| `paged_attention_turboquant` | `attention/attention_turboquant.cpp` | — | planned |
| `turboquant_query_transform` | `quantization/turboquant_ref.cpp` | — | planned |
| `paged_attention_advanced` | `serving/serving_extended_ref.cpp` | — | planned |
| `quantized_mla_decode_absorbed` | `attention/mla_absorb_ref.cpp` | — | planned |
| `quantized_mla_decode_absorbed_sparse` | `attention/mla_absorb_ref.cpp` | — | planned |
| `quantized_attention` | `attention/attention_turboquant.cpp` | — | planned |
| `kv_cache_scale_update` | `serving/kv_cache_extended_ref.cpp` | `serving/kv_cache` | planned |

## Phase 3 — Dense Matmul Epilogues And Decode Path (10)

This phase carries the **LDS-staged double-buffered GEMM** rewrite identified in
`perf/optimization_status.md` as the next lever after register tiling plateaued
at ~56 TFLOP/s (0.12–0.18× hipBLASLt, commit `16c7fd63`). Do it once here and
reuse it for qgemm and flux.

ROCm's `cmplx_matmul` currently exists only inside
`kernels/linear_attention/variants/rocm_cdna3/tm_linattn_kernels.cuh`; this
phase promotes it to a public `kernels/matmul/cmplx_matmul/` operation.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `decode_linear` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | planned |
| `decode_linear_residual` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | planned |
| `decode_linear_q8` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | planned |
| `decode_linear_epilogue_dense` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | planned |
| `decode_linear_epilogue_packed` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | planned |
| `decode_swiglu_dense` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | planned |
| `decode_swiglu_packed` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | planned |
| `gemm_gate_residual` | `matmul/matmul_extended_ref.cpp` | `matmul/gemm_v3` | planned |
| `grouped_gemm` (dense) | `matmul/dense_gemm_ref.cpp` | — | planned |
| `lora_apply_direct_f16` | `matmul/lora_ref.cpp` | — | planned |
| `complex_gemm` (promote to public op) | `matmul/matmul_extended_ref.cpp` | `matmul/cmplx_matmul` | planned |

> The table lists 11 rows because `complex_gemm` is a promotion of existing
> in-repo code rather than a new kernel; it is counted once in the phase total
> of 10 new kernels plus one promotion.

## Phase 4 — MoE Completeness (7)

Extends `kernels/moe/variants/rocm_cdna3/tm_moe_kernels.cuh`. Preserve the
existing 32-row padded expert schedule and row-shaped expert ids. The generic
packed `moe_grouped_qgemm`/`qswiglu` are the format-agnostic counterparts to the
landed named fp8/nvfp4/wna16 grouped GEMMs.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `moe_route_grouped` | `moe/moe_ref.cpp` | `moe/moe` | planned |
| `moe_gather_backward` | `moe/moe_extended_ref.cpp` | — | planned |
| `moe_finalize_backward` | `moe/moe_extended_ref.cpp` | — | planned |
| `moe_grouped_gemm_backward_input` | `moe/moe_extended_ref.cpp` | — | planned |
| `moe_grouped_gemm_backward_weight` | `moe/moe_extended_ref.cpp` | — | planned |
| `moe_grouped_qgemm` | `moe/moe_extended_ref.cpp` | — | planned |
| `moe_grouped_qswiglu` | `moe/moe_extended_ref.cpp` | — | planned |

## Phase 5 — BaseQ Canonical Family (9)

The largest new design in this program. `../QuixiCore-CPU/parity/sibling_operations.tsv`
attributes `base_q_*` to Metal, but **no BaseQ code exists in Metal's working
tree** — `QuixiCore-CPU/include/quixicore_cpu/base_q.h` and
`kernels/quantization/base_q_ref.cpp` are the only live definitions, and are
therefore treated as canonical here.

Port the shared decode core first (little-endian bitstream, symmetric/affine
rule, group sizes 32/64/128, BaseQ2/3/4/5/6/8, BF16/F16/E8M0/E4M3 scale
storage) and validate it byte-exact against the CPU reference before building
the nine consumers on it. `base_q_lm_head_argmax` must round to output storage
**before** the argmax and break ties toward the lower token id.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `base_q_dequant` | `quantization/base_q_ref.cpp` | — | planned |
| `base_q_gemv` | `quantization/base_q_ref.cpp` | — | planned |
| `base_q_gemm` | `quantization/base_q_ref.cpp` | — | planned |
| `base_q_embedding` | `quantization/base_q_ref.cpp` | — | planned |
| `base_q_gemv_qkv` | `quantization/base_q_ref.cpp` | — | planned |
| `base_q_gemv_swiglu` | `quantization/base_q_ref.cpp` | — | planned |
| `base_q_lm_head_argmax` | `quantization/base_q_ref.cpp` | — | planned |
| `base_q_moe_gemm` | `quantization/base_q_ref.cpp` | — | planned |
| `base_q_moe_swiglu` | `quantization/base_q_ref.cpp` | — | planned |

## Phase 6 — Quant Authoring, Fake-Quant, Quantized Embedding (14)

Extends `kernels/quantization/qgemv/variants/rocm_cdna3/quant_rt.cu` and the
landed 29-format dequant table. Also updates `.quixicore/quant-formats.yaml`.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `fake_quant_int8` | `quantization/quantization_ref.cpp` | `quantization/fake_quant` | planned |
| `fake_quant_float8` | `quantization/quantization_ref.cpp` | `quantization/fake_quant_fp8` | planned |
| `tq2_0_pack` | `quantization/quantization_ref.cpp` | `quantization/quantize_tq2_0` | planned |
| `tq2_0_unpack` | `quantization/quantization_ref.cpp` | `quantization/quantize_tq2_0` | planned |
| `ternary_pack` | `quantization/quantization_ref.cpp` | `quantization/weight_quant_ternary` | planned |
| `ternary_unpack` | `quantization/quantization_ref.cpp` | `quantization/weight_quant_ternary` | planned |
| `ternary_stats` | `quantization/quantization_ref.cpp` | `quantization/ternary_stats` | planned |
| `ternary_code_flip_count` | `quantization/quantization_ref.cpp` | `quantization/ternary_stats` | planned |
| `calibration_absmax` | `quantization/activation_quant_ref.cpp` | — | planned |
| `qgemm_backward_input` | `quantization/qgemm_extended_ref.cpp` | `quantization/qgemm_bwd` | planned |
| `dequant_gather` | `quantization/base_q_ref.cpp` | `quantization/dequant_gather` | planned |
| `quantized_embedding` | `quantization/qgemm_extended_ref.cpp` | `quantization/dequant_gather` | planned |
| `quantized_embedding_bag` | `quantization/qgemm_extended_ref.cpp` | `quantization/dequant_gather` | planned |
| `mxfp4_gemv` | `quantization/microscale_ref.cpp` | — | planned |

## Phase 7 — Sampling And Embedding Stragglers (6)

Extends `kernels/serving/variants/rocm_cdna3/logits_proc_test.cu`.
`logits_softcap` is the **final-logit** softcap and is a distinct operation from
attention score capping.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `top_k_renorm` | `sampling/transforms_ref.cpp` | `sampling/sampling` | planned |
| `top_p_renorm` | `sampling/transforms_ref.cpp` | `sampling/sampling` | planned |
| `logits_softcap` | `sampling/transforms_ref.cpp` | `sampling/sampling` | planned |
| `embedding_lookup_types` | `serving/serving_ref.cpp` | `serving/embedding` | planned |
| `broadcast` | `collectives/collectives_ref.cpp` | — | planned |
| `reduce_sum` (collective) | `collectives/collectives_ref.cpp` | — | planned |

## Phase 8 — Linear Attention (9)

RWKV6/7 are new math with no ROCm precedent — derive from the CPU reference and
validate against an fp64 oracle before any tiling. GDN preserves functional FP32
state/history pools, slot mapping, D64/D128 heads, and MQA/GQA mapping; aliased
slots retain request order.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `gated_linear_attention` | `linear_attention/linear_attention_extended_ref.cpp` | — | planned |
| `rwkv_wkv6` | `linear_attention/llama_recurrent_ref.cpp` | — | planned |
| `rwkv_wkv7` | `linear_attention/llama_recurrent_ref.cpp` | — | planned |
| `gdn_short_conv` | `linear_attention/gdn_ref.cpp` | `linear_attention/gdn` | planned |
| `gdn_qkv_prepare` | `linear_attention/gdn_ref.cpp` | `linear_attention/gdn` | planned |
| `gdn_gate_beta` | `linear_attention/gdn_ref.cpp` | `linear_attention/gdn` | planned |
| `gdn_gated_rmsnorm` | `linear_attention/gdn_ref.cpp` | `linear_attention/gdn` | planned |
| `gdn_recurrence` (varlen) | `linear_attention/gdn_ref.cpp` | `linear_attention/gdn` | planned |
| `linear_attention_unnormalized` | `linear_attention/linear_attention_ref.cpp` | `linear_attention/linear_attn` | planned |

## Phase 9 — State-Space And Hyper-Connections (6)

Extends `kernels/ssm/mamba2/variants/rocm_cdna3/mamba2_ssd.cu`. The backward
chunk-scan is the hard one; the landed forward SSD kernel supplies the chunk
decomposition to mirror.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `mamba2_backward` | `ssm/ssm_extended_ref.cpp` | `ssm/mamba2` | planned |
| `ssd_chunked_backward` | `ssm/ssm_extended_ref.cpp` | `ssm/mamba2` | planned |
| `ssd_decode` | `ssm/ssm_extended_ref.cpp` | `ssm/mamba2` | planned |
| `dsv4_hc_pre` | `ssm/ssm_extended_ref.cpp` | — | planned |
| `dsv4_hc_post` | `ssm/ssm_extended_ref.cpp` | — | planned |
| `dsv4_hc_comb` | `ssm/ssm_extended_ref.cpp` | — | planned |

## Phase 10 — Training And Distillation (10)

Lands in `kernels/activations/elementwise/variants/rocm_cdna3/tm_elementwise_kernels.cuh`
alongside the existing `cross_entropy_*`, `dropout_*`, and `adamw_step`. Apply
the proven 64-lane wavefront widening from the outset rather than porting narrow
and re-widening.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `kd_kl_dense_fwd` | `utils/kd_ref.cpp` | `utils/kd_kl_dense` | planned |
| `kd_kl_dense_bwd` | `utils/kd_ref.cpp` | `utils/kd_kl_dense` | planned |
| `kd_kl_topk_fwd` | `utils/kd_ref.cpp` | `utils/kd_kl_topk` | planned |
| `kd_kl_topk_bwd` | `utils/kd_ref.cpp` | `utils/kd_kl_topk` | planned |
| `kd_ce_fused_fwd` | `utils/kd_ref.cpp` | `utils/kd_kl_dense` | planned |
| `kd_ce_fused_bwd` | `utils/kd_ref.cpp` | `utils/kd_kl_dense` | planned |
| `adamw_masked` | `optimizers/adamw_ref.cpp` | `optimizers/optim` | planned |
| `sgd` | `utils/tensor_ops_ref.cpp` | `optimizers/optim` | planned |
| `softmax_backward` | `utils/tensor_ops_ref.cpp` | `activations/softmax` | planned |
| `silu_backward` | `activations/activations_ref.cpp` | `activations/glu` | planned |

## Phase 11 — Elementwise And Tensor-Op Surface (42)

Highest kernel count, lowest per-kernel risk. Nearly all of it comes from
`../QuixiCore-CPU/kernels/utils/tensor_ops_ref.cpp`.

`unary` is one templated dispatch covering all 22 llama selectors; top-level
coverage must not hide a missing activation mode, so its harness enumerates all
22 explicitly. Operations are grouped into a small number of `.cu` files by
shape class, but each carries its own perf run.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `unary` (22 selectors) | `utils/tensor_ops_ref.cpp` | `activations/gelu` | planned |
| `sigmoid_mul` | `activations/activations_ref.cpp` | `activations/glu` | planned |
| `sigmoid_mul_backward` | `activations/activations_ref.cpp` | `activations/glu` | planned |
| `value_clip` | `utils/tensor_ops_ref.cpp` | — | planned |
| `clamp` | `utils/tensor_ops_ref.cpp` | — | planned |
| `leaky_relu` | `utils/tensor_ops_ref.cpp` | — | planned |
| `group_norm` | `utils/tensor_ops_ref.cpp` | — | planned |
| `l2_normalize` | `utils/tensor_ops_ref.cpp` | — | planned |
| `add_id` | `vision/conv_pool_ref.cpp` | — | planned |
| `add_scalar` | `utils/tensor_ops_ref.cpp` | — | planned |
| `multiply` | `utils/tensor_ops_ref.cpp` | — | planned |
| `divide` | `utils/tensor_ops_ref.cpp` | — | planned |
| `subtract` | `utils/tensor_ops_ref.cpp` | — | planned |
| `scale` | `utils/tensor_ops_ref.cpp` | — | planned |
| `square` | `utils/tensor_ops_ref.cpp` | — | planned |
| `square_root` | `utils/tensor_ops_ref.cpp` | — | planned |
| `sine` | `utils/tensor_ops_ref.cpp` | — | planned |
| `cosine` | `utils/tensor_ops_ref.cpp` | — | planned |
| `logarithm` | `utils/tensor_ops_ref.cpp` | — | planned |
| `accumulate` | `utils/tensor_ops_ref.cpp` | — | planned |
| `reduce_mean` | `utils/tensor_ops_ref.cpp` | — | planned |
| `reduce_sum_all` | `utils/tensor_ops_ref.cpp` | — | planned |
| `cumulative_sum` | `utils/tensor_ops_ref.cpp` | — | planned |
| `count_equal` | `utils/tensor_ops_ref.cpp` | — | planned |
| `arange` | `utils/tensor_ops_ref.cpp` | — | planned |
| `fill` | `utils/tensor_ops_ref.cpp` | — | planned |
| `concat` | `utils/tensor_ops_ref.cpp` | — | planned |
| `repeat_2d` | `utils/tensor_ops_ref.cpp` | — | planned |
| `repeat_backward_2d` | `utils/tensor_ops_ref.cpp` | — | planned |
| `pad_2d` | `utils/tensor_ops_ref.cpp` | — | planned |
| `pad_reflect_1d` | `utils/tensor_ops_ref.cpp` | — | planned |
| `roll_2d` | `utils/tensor_ops_ref.cpp` | — | planned |
| `set_rows` | `utils/tensor_ops_ref.cpp` | — | planned |
| `tensor_copy` | `vision/conv_pool_ref.cpp` | — | planned |
| `tensor_set_4d` | `vision/conv_pool_ref.cpp` | — | planned |
| `diag_embed` | `utils/tensor_ops_ref.cpp` | — | planned |
| `diag_mask` | `utils/tensor_ops_ref.cpp` | — | planned |
| `triangular_fill` | `utils/tensor_ops_ref.cpp` | — | planned |
| `argsort` | `utils/tensor_ops_ref.cpp` | — | planned |
| `outer_product` | `utils/tensor_ops_ref.cpp` | — | planned |
| `solve_lower_triangular` | `vision/conv_pool_ref.cpp` | — | planned |
| `threshold_topk_indices` | `utils/utils_extended_ref.cpp` | — | planned |

## Phase 12 — Convolution And Audio (16)

New `kernels/conv/` and `kernels/audio/` families — neither exists in this repo
today. Audio routes preserve `[B,T,C]`, `[O,K,C]`/`[C,K]`, stride/padding/
dilation, optional bias, and explicit symmetric-versus-causal geometry.
`audio_relative_attention` preserves chunk/left/right geometry, learned
per-dimension query scaling, relative shifts, lengths, and optional softcap.

Perf lever for this phase: im2col + MFMA-GEMM versus direct convolution. Measure
both and record the crossover, mirroring the `int8_gemm` scalar-vs-sdot4 A/B
already in the notebook.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `im2col_2d` | `vision/conv_pool_ref.cpp` | — | planned |
| `im2col_3d` | `vision/conv_pool_ref.cpp` | — | planned |
| `col2im_1d` | `vision/conv_pool_ref.cpp` | — | planned |
| `col2im_2d` | `vision/conv_pool_ref.cpp` | — | planned |
| `conv2d` | `vision/conv_pool_ref.cpp` | — | planned |
| `conv3d` | `vision/conv_pool_ref.cpp` | — | planned |
| `depthwise_conv2d` | `vision/conv_pool_ref.cpp` | — | planned |
| `conv_transpose_1d` | `vision/conv_pool_ref.cpp` | — | planned |
| `conv_transpose_2d` | `vision/conv_pool_ref.cpp` | — | planned |
| `pool1d` | `vision/conv_pool_ref.cpp` | — | planned |
| `pool2d` | `vision/conv_pool_ref.cpp` | — | planned |
| `pool2d_backward` | `vision/conv_pool_ref.cpp` | — | planned |
| `audio_conv1d_direct` | `audio/conv1d_ref.cpp` | — | planned |
| `audio_depthwise_conv1d` | `audio/conv1d_ref.cpp` | — | planned |
| `audio_causal_depthwise_conv1d` | `audio/conv1d_ref.cpp` | — | planned |
| `audio_relative_attention` | `audio/relative_attention_ref.cpp` | — | planned |

## Phase 13 — Vision (19)

New `kernels/vision/` family. Preserve padding/stride/temporal patch order,
`[O,KH,KW,C]` / `[O,KT,KH,KW,C]` projection weights, half-pixel versus
aligned-corner interpolation, invalid-token zeroing, coordinate masks,
local-axis/global-split RoPE, and partial ceil-mode pool windows. Depends on
Phase 12's pooling primitives.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `extract_patches_2d` | `vision/patch_ops_ref.cpp` | — | planned |
| `extract_patches_3d` | `vision/patch_ops_ref.cpp` | — | planned |
| `vision_patch_projection` | `vision/patch_ops_ref.cpp` | — | planned |
| `vision_patch_projection_3d` | `vision/patch_ops_ref.cpp` | — | planned |
| `patch_merge_layer_norm` | `vision/vision_ref.cpp` | `vision/patch_merge` | planned |
| `space_to_depth_norm_linear` | `vision/vision_ref.cpp` | `vision/patch_merge` | planned |
| `edge_mlp_256x7` | `vision/vision_ref.cpp` | `vision/edge_mlp` | planned |
| `vision_rope_2d` | `attention/vision_rope_ref.cpp` | — | planned |
| `qwen_vision_rope_2d` | `attention/vision_rope_ref.cpp` | — | planned |
| `interpolate_position_2d` | `vision/patch_ops_ref.cpp` | — | planned |
| `factorized_position_2d` | `vision/patch_ops_ref.cpp` | — | planned |
| `add_relative_position_2d` | `vision/conv_pool_ref.cpp` | — | planned |
| `get_relative_position` | `vision/conv_pool_ref.cpp` | — | planned |
| `window_partition` | `vision/conv_pool_ref.cpp` | — | planned |
| `window_unpartition` | `vision/conv_pool_ref.cpp` | — | planned |
| `avg_pool2d_tokens` | `vision/patch_ops_ref.cpp` | — | planned |
| `pool_tokens_by_position` | `vision/patch_ops_ref.cpp` | — | planned |
| `timestep_embedding` | `vision/conv_pool_ref.cpp` | — | planned |
| `upscale_nearest_2d` | `utils/tensor_ops_ref.cpp` | — | planned |

## Out Of Scope

- **CDNA4 (gfx950) variants.** These 178 kernels land CDNA3-only. MI300X is the
  available hardware and the perf gate forbids committing an unmeasured CDNA4
  claim. The resulting CDNA3/CDNA4 variant gap is tracked separately.
- **A unified public API header.** ROCm keeps the standalone kernel + harness
  convention; it does not gain a `quixicore_rocm/ops.h` mirroring CPU's `ops.h`.
- **Host-side quant import** (`quant_import.h`: AWQ/GPTQ/AutoRound/SmoothQuant/
  BitNet checkpoint readers). These are host format converters, not kernels.

## Already At Parity

For the record, ROCm already covers and does **not** need work in: norms
(fwd/bwd/add/residual-next/norm-quant), core GQA attention forward and backward,
softmax/gelu/glu/dropout/cross-entropy/hadamard/adamw, dense and
FP8/INT8/MXFP8/NVFP4 GEMM, qgemv/qgemm and their int8/azp/w2a8/q4q8/actorder/
blockscale variants, MoE routing and grouped GEMM including fp8/nvfp4/wna16,
paged attention and MLA and the FP8 KV cache, the sampling and logit-processor
surface, beam and speculative decode and EAGLE, sparse serving and MInference
and the indexer, linear attention (based/hedgehog/decay/GDN core), SSM
(selective_scan/SSD/fftconv), turboquant, and collectives.

ROCm is **ahead** of Metal on collectives: Metal's `collectives` family is
`planned` with no paths, while ROCm has RCCL all_reduce/all_gather/all_to_all/
reduce_scatter plus fused GEMM collectives, ring attention, and Ulysses
attention validated across 8 MI300X.
