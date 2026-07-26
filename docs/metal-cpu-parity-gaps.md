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
| 1 | Attention & RoPE variants | 13 | 0 | 0 | **13** |
| 2 | Quantized KV-cache codecs | 17 | 0 | 0 | **17** |
| 3 | Dense matmul epilogues | 10 | 0 | 0 | **10** |
| 4 | MoE completeness | 7 | 0 | 0 | **7** |
| 5 | BaseQ canonical family | 9 | 0 | 0 | **9** |
| 6 | Quant authoring & quantized embedding | 14 | 0 | 0 | **14** |
| 7 | Sampling & embedding stragglers | 6 | 0 | 0 | **6** |
| 8 | Linear attention | 9 | 0 | 0 | **9** |
| 9 | State-space & hyper-connections | 6 | 0 | 0 | **6** |
| 10 | Training & distillation | 10 | 0 | 0 | **10** |
| 11 | Elementwise & tensor-op surface | 42 | 0 | 0 | **42** |
| 12 | Convolution & audio | 16 | 0 | 0 | **16** |
| 13 | Vision | 19 | 0 | 0 | **19** |
| | **Total** | **178** | **0** | **0** | **178** |

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
| `swin_attention_d32` | `attention/attention_composites_ref.cpp` | `attention/swin_attn` | **landed** |
| `decode_cache_attention` | `attention/attention_serving_ref.cpp` | `attention/attn_decode` | **landed** |
| `cascade_attention_multi` | `attention/attention_composites_ref.cpp` | — | **landed** |
| `mrope` | `attention/rotary_positioned_ref.cpp` | `attention/rotary` | **landed** |
| `rotary_positioned` | `attention/rotary_positioned_ref.cpp` | `attention/rotary` | **landed** |
| `rope_table` | `attention/rotary_positioned_ref.cpp` | `attention/rotary` | **landed** |
| `rope_interleaved_to_split` | `attention/rotary_positioned_ref.cpp` | `attention/rotary` | **landed** |
| `rope_backward` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `qk_norm_rope_positioned` | `attention/attention_extended_ref.cpp` | `norms/qk_norm_rope` | **landed** |
| `qk_norm_rope_split` | `attention/attention_extended_ref.cpp` | `norms/qk_norm_rope` | **landed** |
| `rope_q_norm` | `attention/attention_extended_ref.cpp` | `norms/qk_norm_rope` | **landed** |

## Phase 2 — Quantized KV-Cache Codecs And Decode Attention (17)

Extends the landed `kv_cache_*` and `paged_attention_*` kernels in
`kernels/serving/variants/rocm_cdna3/`. Canonical layouts are contractual:
Q8_0 keeps **separate** signed-int8 code planes and raw FP16 scale planes
(sparse negative blocks zero-fill or skip); MXFP8 keeps interleaved blocks;
TurboQuant keeps packed K and rotated V, with V staying in the signed-FWHT
domain until one final inverse transform and canonical K **not** transformed;
BitNet-KV3 keeps low-bit-first ordering with explicit signedness, zero-point
mode, group size, and FP16/FP32 scale encoding.

2026-07-26 update: `kernels/serving/phase2_quant_decode/variants/rocm_cdna3`
landed the six remaining Phase 2 rows. The standalone harness reports
`ALL PASS` for 24 checks covering BitNet-KV3 scatter/gather/decode over
signed-FP16 and unsigned-zero-point-FP32 metadata, TurboQuant paged decode,
advanced paged attention with masks/ALiBI/sinks/softcap, and `quantized_attention`
over `q8_0` and `q4_0` in full and causal modes. The focused benchmark keeps
the direct row/element routes over scalar GPU baselines, with raw output
archived under `perf/results/2026-07-26/phase2-quant-decode-final3/`.

### Reconnaissance notes from the Phase 2 close-out

**MXFP8** (`attention_mxfp8.cpp`). Unlike Q8_0's two planes, MXFP8 is a single
**interleaved 33-byte block**: `[e8m0 scale byte][32 × e4m3 code bytes]`, indexed
`((row*heads + head)*groups + group) * 33`. `head_dim` must be 64 or 128.
Encoding is `scale_code = e8m0_encode_up(amax / 448.0)` then
`code = e4m3_encode(v / e8m0_decode(scale_code))`. A scale byte of 255 is the
**invalid marker** — the reference rejects any cache row containing it, and
`paged_attention_mxfp8` validates the whole reachable cache up front rather than
skipping bad rows. Its score tile is **16**, not Q8_0's 32.

`kernels/quantization/qgemv/variants/rocm_cdna3/quant_formats.cuh` already
provides device `e4m3_encode`/`e4m3_decode`/`e8m0_decode` and documents the same
33-byte block, so most of the codec is reusable. **But its `e8m0_encode` is raw
exponent extraction (truncation), while the reference needs
`e8m0_encode_up` = `clamp(ceil(log2(x)) + 127, 0, 254)`.** Those differ for every
non-power-of-two input. Reusing the existing helper produces a cache that is
subtly wrong and still round-trips through its own decode — write
`e8m0_encode_up` separately.

**BitNet-KV3** (`attention_bitnet_kv3.cpp`, 609 lines) is the largest remaining
codec: low-bit-first ordering with explicit signedness, zero-point mode, group
size, and FP16/FP32 scale encoding all as parameters.

**TurboQuant** (`attention_turboquant.cpp`) keeps K packed and V *rotated*, with
V staying in the signed-FWHT domain until one final inverse transform; canonical
K is **not** transformed. `turboquant_query_transform` is the query-side half.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `kv_cache_scatter_q8_0` | `serving/serving_quant_ref.cpp` | `serving/kv_cache` | **landed** |
| `kv_cache_gather_q8_0` | `serving/serving_quant_ref.cpp` | `serving/kv_cache` | **landed** |
| `kv_cache_copy_blocks_q8_0` | `serving/serving_quant_ref.cpp` | `serving/kv_cache` | **landed** |
| `paged_attention_q8_0` | `attention/attention_q8_kv.cpp` | — | **landed** |
| `kv_cache_scatter_mxfp8` | `attention/attention_mxfp8.cpp` | — | **landed** |
| `kv_cache_gather_mxfp8` | `attention/attention_mxfp8.cpp` | — | **landed** |
| `paged_attention_mxfp8` | `attention/attention_mxfp8.cpp` | — | **landed** |
| `kv_cache_scatter_bitnet_kv3` | `attention/attention_bitnet_kv3.cpp` | — | **landed** |
| `kv_cache_gather_bitnet_kv3` | `attention/attention_bitnet_kv3.cpp` | — | **landed** |
| `paged_attention_bitnet_kv3` | `attention/attention_bitnet_kv3.cpp` | — | **landed** |
| `paged_attention_turboquant` | `attention/attention_turboquant.cpp` | — | **landed** |
| `turboquant_query_transform` | `attention/attention_turboquant.cpp` | — | **landed** |
| `paged_attention_advanced` | `attention/attention_serving_ref.cpp` | — | **landed** |
| `quantized_mla_decode_absorbed` | `attention/mla_absorb_ref.cpp` | — | **landed** |
| `quantized_mla_decode_absorbed_sparse` | `attention/mla_absorb_ref.cpp` | — | **landed** |
| `quantized_attention` | `attention/attention_serving_ref.cpp` | — | **landed** |
| `kv_cache_scale_update` | `serving/kv_cache_extended_ref.cpp` | `serving/kv_cache` | **landed** |

## Phase 3 — Dense Matmul Epilogues And Decode Path (10)

This phase carries the **LDS-staged double-buffered GEMM** rewrite identified in
`perf/optimization_status.md` as the next lever after register tiling plateaued
at ~56 TFLOP/s (0.12–0.18× hipBLASLt, commit `16c7fd63`). Do it once here and
reuse it for qgemm and flux.

ROCm's `cmplx_matmul` currently exists only inside
`kernels/linear_attention/variants/rocm_cdna3/tm_linattn_kernels.cuh`; this
phase promotes it to a public Phase 3 matmul operation.

2026-07-26 update: `kernels/matmul/decode_epilogues/variants/rocm_cdna3`
landed the dense decode/epilogue, Q8_0 decode, packed epilogue coverage for
`q4_0`, `q8_0`, `q6_K`, `mxfp8`, `nvfp4`, and `mxfp4`, SwiGLU dense/packed,
gate/residual, dense grouped GEMM, LoRA-F16, and complex GEMM contracts with a
shared-harness oracle and scalar-vs-wave64 optimization runs.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `decode_linear` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | **landed** |
| `decode_linear_residual` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | **landed** |
| `decode_linear_q8` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | **landed** |
| `decode_linear_epilogue_dense` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | **landed** |
| `decode_linear_epilogue_packed` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | **landed** |
| `decode_swiglu_dense` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | **landed** |
| `decode_swiglu_packed` | `matmul/matmul_extended_ref.cpp` | `matmul/decode_linear` | **landed** |
| `gemm_gate_residual` | `matmul/matmul_extended_ref.cpp` | `matmul/gemm_v3` | **landed** |
| `grouped_gemm` (dense) | `matmul/dense_gemm_ref.cpp` | — | **landed** |
| `lora_apply_direct_f16` | `matmul/lora_ref.cpp` | — | **landed** |
| `complex_gemm` (promote to public op) | `matmul/matmul_extended_ref.cpp` | `matmul/cmplx_matmul` | **landed** |

> The table lists 11 rows because `complex_gemm` is a promotion of existing
> in-repo code rather than a new kernel; it is counted once in the phase total
> of 10 new kernels plus one promotion.

## Phase 4 — MoE Completeness (7)

Extends `kernels/moe/variants/rocm_cdna3/tm_moe_kernels.cuh`. Preserve the
existing 32-row padded expert schedule and row-shaped expert ids. The generic
packed `moe_grouped_qgemm`/`qswiglu` are the format-agnostic counterparts to the
landed named fp8/nvfp4/wna16 grouped GEMMs.

2026-07-26 update: recent MoE commits added all seven Phase 4 rows. The dense
MoE harness reports `ALL PASS (0 failures)` for grouped routing and the four
backward rows, and the quantized MoE harness reports `ALL PASS (0 failures)` for
`moe_grouped_qgemm` and `moe_grouped_qswiglu` over `q2_K`. The same harnesses
now carry focused scalar-vs-parallel Phase 4 timing runs, so the full phase is
landed.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `moe_route_grouped` | `moe/moe_ref.cpp` | `moe/moe` | **landed** |
| `moe_gather_backward` | `moe/moe_extended_ref.cpp` | — | **landed** |
| `moe_finalize_backward` | `moe/moe_extended_ref.cpp` | — | **landed** |
| `moe_grouped_gemm_backward_input` | `moe/moe_extended_ref.cpp` | — | **landed** |
| `moe_grouped_gemm_backward_weight` | `moe/moe_extended_ref.cpp` | — | **landed** |
| `moe_grouped_qgemm` | `moe/moe_extended_ref.cpp` | — | **landed** |
| `moe_grouped_qswiglu` | `moe/moe_extended_ref.cpp` | — | **landed** |

## Phase 5 — BaseQ Canonical Family (9)

The largest new design in this program. `../QuixiCore-CPU/parity/sibling_operations.tsv`
attributes `base_q_*` to Metal, but **no BaseQ code exists in Metal's working
tree** — `QuixiCore-CPU/include/quixicore_cpu/base_q.h` and
`kernels/quantization/base_q_ref.cpp` are the only live definitions, and are
therefore treated as canonical here.

Port the shared decode core first (little-endian bitstream, symmetric/affine
rule, group sizes 32/64/128, BaseQ2/3/4/5/6/8, BF16/F16/E8M0/E4M3 scale
storage) and validate it byte-exact against the CPU reference before building
the nine consumers on it. `base_q_lm_head_argmax` must round scores to the
input storage type **before** the argmax and break ties toward the lower token
id.

2026-07-26 update: `kernels/quantization/base_q/variants/rocm_cdna3` landed the
canonical BaseQ decode core and all nine consumers. The standalone harness
reports `ALL PASS` for 246 fp64-oracle checks covering all supported bit widths,
group sizes, BF16/F16/E8M0/E4M3 scale storage, symmetric/affine modes,
FP32/FP16/BF16 output storage, lower-token LM-head ties, and the 32-row padded
MoE expert schedule. The Phase 5 benchmark keeps wave64 split-dot projection
routes for GEMV/GEMM/QKV/SwiGLU/MoE, keeps current dequant/embedding kernels,
and rejects the streaming LM-head argmax in favor of the materialized
projection-plus-argmax route.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `base_q_dequant` | `quantization/base_q_ref.cpp` | — | **landed** |
| `base_q_gemv` | `quantization/base_q_ref.cpp` | — | **landed** |
| `base_q_gemm` | `quantization/base_q_ref.cpp` | — | **landed** |
| `base_q_embedding` | `quantization/base_q_ref.cpp` | — | **landed** |
| `base_q_gemv_qkv` | `quantization/base_q_ref.cpp` | — | **landed** |
| `base_q_gemv_swiglu` | `quantization/base_q_ref.cpp` | — | **landed** |
| `base_q_lm_head_argmax` | `quantization/base_q_ref.cpp` | — | **landed** |
| `base_q_moe_gemm` | `quantization/base_q_ref.cpp` | — | **landed** |
| `base_q_moe_swiglu` | `quantization/base_q_ref.cpp` | — | **landed** |

## Phase 6 — Quant Authoring, Fake-Quant, Quantized Embedding (14)

Landed in `kernels/quantization/quant_authoring/variants/rocm_cdna3`. It reuses
the landed 29-format dequant table where packed GGUF rows are consumed, and it
updates `.quixicore/quant-formats.yaml`.

2026-07-26 update: the Phase 6 harness reports `ALL PASS` for 25 checks covering
INT8 and FP8 fake quantization, byte-exact TQ2_0 and ternary pack/unpack,
ternary statistics and flip counts, NaN-preserving calibration absmax, BitNet
`qgemm_backward_input`, packed-table gather/embedding over `q4_0`/`q8_0`/`q6_K`,
and MXFP4 GEMV. The focused benchmark keeps wave64/block-parallel routes for
the reduction and packing kernels, with raw output archived under
`perf/results/2026-07-26/quant-authoring-phase6-final3/`.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `fake_quant_int8` | `quantization/quantization_ref.cpp` | `quantization/fake_quant` | **landed** |
| `fake_quant_float8` | `quantization/quantization_ref.cpp` | `quantization/fake_quant_fp8` | **landed** |
| `tq2_0_pack` | `quantization/quantization_ref.cpp` | `quantization/quantize_tq2_0` | **landed** |
| `tq2_0_unpack` | `quantization/quantization_ref.cpp` | `quantization/quantize_tq2_0` | **landed** |
| `ternary_pack` | `quantization/quantization_ref.cpp` | `quantization/weight_quant_ternary` | **landed** |
| `ternary_unpack` | `quantization/quantization_ref.cpp` | `quantization/weight_quant_ternary` | **landed** |
| `ternary_stats` | `quantization/quantization_ref.cpp` | `quantization/ternary_stats` | **landed** |
| `ternary_code_flip_count` | `quantization/quantization_ref.cpp` | `quantization/ternary_stats` | **landed** |
| `calibration_absmax` | `quantization/activation_quant_ref.cpp` | — | **landed** |
| `qgemm_backward_input` | `quantization/qgemm_extended_ref.cpp` | `quantization/qgemm_bwd` | **landed** |
| `dequant_gather` | `quantization/base_q_ref.cpp` | `quantization/dequant_gather` | **landed** |
| `quantized_embedding` | `quantization/qgemm_extended_ref.cpp` | `quantization/dequant_gather` | **landed** |
| `quantized_embedding_bag` | `quantization/qgemm_extended_ref.cpp` | `quantization/dequant_gather` | **landed** |
| `mxfp4_gemv` | `quantization/microscale_ref.cpp` | — | **landed** |

## Phase 7 — Sampling And Embedding Stragglers (6)

Landed in `kernels/sampling/phase7_stragglers/variants/rocm_cdna3`.
`logits_softcap` is the **final-logit** softcap and is a distinct operation from
attention score capping.

2026-07-26 update: the Phase 7 harness reports `ALL PASS` for stable
top-k/top-p probability renormalization, final-logit softcap, FP32/FP16/BF16
token+type embedding lookup, and CPU root-collective `broadcast`/`reduce_sum`.
The collectives here are the CPU tensor contracts; multi-GPU transport remains
covered by the existing RCCL collectives tree. Raw benchmark output is archived
under `perf/results/2026-07-26/phase7-stragglers-final/`.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `top_k_renorm` | `sampling/transforms_ref.cpp` | `sampling/sampling` | **landed** |
| `top_p_renorm` | `sampling/transforms_ref.cpp` | `sampling/sampling` | **landed** |
| `logits_softcap` | `sampling/transforms_ref.cpp` | `sampling/sampling` | **landed** |
| `embedding_lookup_types` | `serving/serving_ref.cpp` | `serving/embedding` | **landed** |
| `broadcast` | `collectives/collectives_ref.cpp` | — | **landed** |
| `reduce_sum` (collective) | `collectives/collectives_ref.cpp` | — | **landed** |

## Phase 8 — Linear Attention (9)

Landed in `kernels/linear_attention/phase8_linear/variants/rocm_cdna3`.
RWKV6/7 are new math with no ROCm precedent; derive from the CPU reference and
validate against an fp64 oracle before any tiling. GDN preserves functional FP32
state/history pools, slot mapping, D64/D128 heads, and MQA/GQA mapping; aliased
slots retain request order.

2026-07-26 update: the Phase 8 harness reports `ALL PASS` for GLA, RWKV6,
RWKV7, unnormalized linear attention, GDN recurrence output/state, GDN short
convolution output/state, QKV preparation, gate/beta, and gated RMSNorm. The
focused benchmark keeps wave64 row/channel routes and the staged KV route for
unnormalized linear attention, with raw output archived under
`perf/results/2026-07-26/phase8-linear-final2/`.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `gated_linear_attention` | `linear_attention/llama_recurrent_ref.cpp` | — | **landed** |
| `rwkv_wkv6` | `linear_attention/llama_recurrent_ref.cpp` | — | **landed** |
| `rwkv_wkv7` | `linear_attention/llama_recurrent_ref.cpp` | — | **landed** |
| `gdn_short_conv` | `linear_attention/gdn_ref.cpp` | `linear_attention/gdn` | **landed** |
| `gdn_qkv_prepare` | `linear_attention/gdn_ref.cpp` | `linear_attention/gdn` | **landed** |
| `gdn_gate_beta` | `linear_attention/gdn_ref.cpp` | `linear_attention/gdn` | **landed** |
| `gdn_gated_rmsnorm` | `linear_attention/gdn_ref.cpp` | `linear_attention/gdn` | **landed** |
| `gdn_recurrence` (varlen) | `linear_attention/gdn_ref.cpp` | `linear_attention/gdn` | **landed** |
| `linear_attention_unnormalized` | `linear_attention/linear_attention_extended_ref.cpp` | `linear_attention/linear_attn` | **landed** |

## Phase 9 — State-Space And Hyper-Connections (6)

Landed in `kernels/ssm/phase9_ssm/variants/rocm_cdna3`. The backward
chunk-scan is the hard one; the landed forward SSD kernel supplies the chunk
decomposition to mirror.

2026-07-26 update: the Phase 9 harness reports `ALL PASS` for Mamba2 backward,
SSD chunked backward, SSD decode output/next-state, and DSV4 hyper-connection
comb/pre/post. The Mamba2 backward kernels compute direct per-output gradients
to avoid unordered atomics; `ssd_decode` uses one wave64 row per state/output
row. Raw benchmark output is archived under
`perf/results/2026-07-26/phase9-ssm-final3/`.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `mamba2_backward` | `ssm/ssm_extended_ref.cpp` | `ssm/mamba2` | **landed** |
| `ssd_chunked_backward` | `ssm/ssm_extended_ref.cpp` | `ssm/mamba2` | **landed** |
| `ssd_decode` | `ssm/ssm_extended_ref.cpp` | `ssm/mamba2` | **landed** |
| `dsv4_hc_pre` | `linear_attention/llama_recurrent_ref.cpp` | — | **landed** |
| `dsv4_hc_post` | `linear_attention/llama_recurrent_ref.cpp` | — | **landed** |
| `dsv4_hc_comb` | `linear_attention/llama_recurrent_ref.cpp` | — | **landed** |

## Phase 10 — Training And Distillation (10)

Landed in `kernels/activations/phase10_training/variants/rocm_cdna3`. The KD
reductions use one row block, while backward, optimizer, SGD, softmax-backward,
and SiLU-backward use direct elementwise routes.

2026-07-26 update: the Phase 10 harness reports `ALL PASS` for 22 checks
covering dense/top-k KL distillation forward and backward, fused CE+KD forward
and backward, both `adamw_masked` modes, SGD, softmax backward, and SiLU
backward. Raw benchmark output is archived under
`perf/results/2026-07-26/phase10-training-final2/`.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `kd_kl_dense_fwd` | `utils/kd_ref.cpp` | `utils/kd_kl_dense` | **landed** |
| `kd_kl_dense_bwd` | `utils/kd_ref.cpp` | `utils/kd_kl_dense` | **landed** |
| `kd_kl_topk_fwd` | `utils/kd_ref.cpp` | `utils/kd_kl_topk` | **landed** |
| `kd_kl_topk_bwd` | `utils/kd_ref.cpp` | `utils/kd_kl_topk` | **landed** |
| `kd_ce_fused_fwd` | `utils/kd_ref.cpp` | `utils/kd_kl_dense` | **landed** |
| `kd_ce_fused_bwd` | `utils/kd_ref.cpp` | `utils/kd_kl_dense` | **landed** |
| `adamw_masked` | `optimizers/adamw_ref.cpp` | `optimizers/optim` | **landed** |
| `sgd` | `utils/tensor_ops_ref.cpp` | `optimizers/optim` | **landed** |
| `softmax_backward` | `utils/tensor_ops_ref.cpp` | `activations/softmax` | **landed** |
| `silu_backward` | `activations/activations_ref.cpp` | `activations/glu` | **landed** |

## Phase 11 — Elementwise And Tensor-Op Surface (42)

Landed in `kernels/utils/phase11_tensor_ops/variants/rocm_cdna3`. The harness
enumerates all 22 `unary` selectors explicitly and covers the tensor utility
surface with direct elementwise grids, row reductions, stable row-owned sorting,
and deterministic destination-owned scans where CPU loop order matters.

2026-07-26 update: the Phase 11 harness reports `ALL PASS` for 66 checks,
including both `diag_mask` fill modes, stable ascending/descending `argsort`,
duplicate-row `set_rows`, and lower-triangular solve. Raw benchmark output is
archived under `perf/results/2026-07-26/phase11-tensor-ops-final4/`.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `unary` (22 selectors) | `utils/tensor_ops_ref.cpp` | `activations/gelu` | **landed** |
| `sigmoid_mul` | `activations/activations_ref.cpp` | `activations/glu` | **landed** |
| `sigmoid_mul_backward` | `activations/activations_ref.cpp` | `activations/glu` | **landed** |
| `value_clip` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `clamp` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `leaky_relu` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `group_norm` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `l2_normalize` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `add_id` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `add_scalar` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `multiply` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `divide` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `subtract` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `scale` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `square` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `square_root` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `sine` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `cosine` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `logarithm` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `accumulate` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `reduce_mean` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `reduce_sum_all` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `cumulative_sum` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `count_equal` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `arange` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `fill` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `concat` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `repeat_2d` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `repeat_backward_2d` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `pad_2d` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `pad_reflect_1d` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `roll_2d` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `set_rows` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `tensor_copy` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `tensor_set_4d` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `diag_embed` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `diag_mask` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `triangular_fill` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `argsort` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `outer_product` | `utils/tensor_ops_ref.cpp` | — | **landed** |
| `solve_lower_triangular` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `threshold_topk_indices` | `utils/utils_extended_ref.cpp` | — | **landed** |

## Phase 12 — Convolution And Audio (16)

Landed in `kernels/conv/phase12_conv_audio/variants/rocm_cdna3`. Audio routes
preserve `[B,T,C]`, `[O,K,C]`/`[C,K]`, stride/padding/dilation, optional bias,
and explicit symmetric-versus-causal geometry. `audio_relative_attention`
preserves chunk/left/right geometry, learned per-dimension query scaling,
relative shifts, lengths, and optional softcap.

2026-07-26 update: the Phase 12 harness reports `ALL PASS` for 19 host-oracle
checks covering im2col/col2im, direct/depthwise/transposed conv, average and max
pooling, pool backward, direct/depthwise/causal audio conv, and audio relative
attention. Raw benchmark output is archived under
`perf/results/2026-07-26/phase12-conv-audio-final3/`.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `im2col_2d` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `im2col_3d` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `col2im_1d` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `col2im_2d` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `conv2d` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `conv3d` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `depthwise_conv2d` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `conv_transpose_1d` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `conv_transpose_2d` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `pool1d` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `pool2d` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `pool2d_backward` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `audio_conv1d_direct` | `audio/conv1d_ref.cpp` | — | **landed** |
| `audio_depthwise_conv1d` | `audio/conv1d_ref.cpp` | — | **landed** |
| `audio_causal_depthwise_conv1d` | `audio/conv1d_ref.cpp` | — | **landed** |
| `audio_relative_attention` | `audio/relative_attention_ref.cpp` | — | **landed** |

## Phase 13 — Vision (19)

New `kernels/vision/` family. Preserve padding/stride/temporal patch order,
`[O,KH,KW,C]` / `[O,KT,KH,KW,C]` projection weights, half-pixel versus
aligned-corner interpolation, invalid-token zeroing, coordinate masks,
local-axis/global-split RoPE, and partial ceil-mode pool windows. Depends on
Phase 12's pooling primitives.

2026-07-26 update: `kernels/vision/phase13_vision/variants/rocm_cdna3`
landed all 19 Phase 13 rows. The standalone harness reports `ALL PASS` for 22
checks covering NHWC/NTHWC patch extraction and projection, patch merge,
space-to-depth projection, edge MLP, both vision RoPE layouts, interpolation
modes, relative-position helpers, window partition/unpartition, token pooling,
timestep embedding, and nearest upscaling.

| Kernel | CPU reference | Metal source | Status |
| --- | --- | --- | --- |
| `extract_patches_2d` | `vision/patch_ops_ref.cpp` | — | **landed** |
| `extract_patches_3d` | `vision/patch_ops_ref.cpp` | — | **landed** |
| `vision_patch_projection` | `vision/patch_ops_ref.cpp` | — | **landed** |
| `vision_patch_projection_3d` | `vision/patch_ops_ref.cpp` | — | **landed** |
| `patch_merge_layer_norm` | `vision/vision_ref.cpp` | `vision/patch_merge` | **landed** |
| `space_to_depth_norm_linear` | `vision/vision_ref.cpp` | `vision/patch_merge` | **landed** |
| `edge_mlp_256x7` | `vision/vision_ref.cpp` | `vision/edge_mlp` | **landed** |
| `vision_rope_2d` | `attention/vision_rope_ref.cpp` | — | **landed** |
| `qwen_vision_rope_2d` | `attention/vision_rope_ref.cpp` | — | **landed** |
| `interpolate_position_2d` | `vision/patch_ops_ref.cpp` | — | **landed** |
| `factorized_position_2d` | `vision/patch_ops_ref.cpp` | — | **landed** |
| `add_relative_position_2d` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `get_relative_position` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `window_partition` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `window_unpartition` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `avg_pool2d_tokens` | `vision/patch_ops_ref.cpp` | — | **landed** |
| `pool_tokens_by_position` | `vision/patch_ops_ref.cpp` | — | **landed** |
| `timestep_embedding` | `vision/conv_pool_ref.cpp` | — | **landed** |
| `upscale_nearest_2d` | `utils/tensor_ops_ref.cpp` | — | **landed** |

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
