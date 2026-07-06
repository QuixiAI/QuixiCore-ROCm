# CUDA To CDNA3 Port Status

This tracker inventories kernel directories under `../QuixiCore-CUDA/kernels`
and maps them to the ROCm CDNA3 surface in this repository.

Status meanings:

- `active`: a native ROCm CDNA3 variant exists in `variants/rocm_cdna3`.
- `build-valid`: the CDNA3 variant compiled locally for `gfx942`.
- `planned`: no active CDNA3 implementation exists yet.
- `capability-gated`: the CUDA route uses NVIDIA-only hardware/library
  mechanisms or a distributed runtime that needs a separate ROCm design.
- `not-portable-as-is`: the CUDA extension shape should be decomposed into
  semantic ROCm operations instead of copied.

Do not mark a CUDA kernel as ported until the ROCm path has correctness results
and a focused performance run recorded in `perf/optimization_status.md`.

## Current CDNA3 Coverage

These native ROCm CDNA3 variants exist today and compile with ROCm 7.2.4 for
`gfx942`:

| Operation | ROCm CDNA3 path | Build status | Runtime status |
| --- | --- | --- | --- |
| Softmax | `kernels/activations/softmax/variants/rocm_cdna3` | build-valid | blocked by non-ROCm PyTorch in this environment |
| Rotary | `kernels/attention/rotary/variants/rocm_cdna3` | build-valid | blocked by non-ROCm PyTorch in this environment |
| LayerNorm | `kernels/norms/layernorm/variants/rocm_cdna3` | build-valid | blocked by non-ROCm PyTorch in this environment |
| RMSNorm | `kernels/norms/rmsnorm/variants/rocm_cdna3` | build-valid | blocked by non-ROCm PyTorch in this environment |
| BF16 FP32 GEMM | `kernels/matmul/bf16fp32/variants/rocm_cdna3/8192_256_256_64_16` | build-valid | blocked by non-ROCm PyTorch in this environment |
| FP8 FP32 GEMM | `kernels/matmul/fp8fp32/variants/rocm_cdna3/8192_256_256_64_32` | build-valid | blocked by non-ROCm PyTorch in this environment |
| Scaled FP8 matmul | `kernels/matmul/scaled_matmul/variants/rocm_cdna3` | build-valid | standalone binary builds; no normalized benchmark run yet |
| Elementwise/Norm family | `kernels/activations/elementwise/variants/rocm_cdna3` | build-valid | 56/56 fp64-oracle checks pass on MI300X; perf A/B recorded in `perf/optimization_status.md` |
| Quant dequant + GEMV | `kernels/quantization/qgemv/variants/rocm_cdna3` | build-valid | 29/29 format dequant EXACT + GEMV PASS, W8A8/W2A8 int8 GEMV PASS, runtime-quant 4/4 PASS on MI300X. Tensor-core qgemm deferred to MFMA pass. |
| Serving family | `kernels/serving/variants/rocm_cdna3` | build-valid | 12/12 self-checking harnesses pass on MI300X (paged attn, MLA, kv_cache, sampling, spec-decode, …). |
| MoE (routing + grouped GEMM) | `kernels/moe/variants/rocm_cdna3` | build-valid | 8/8 checks pass on MI300X (end-to-end MoE MLP vs fp64). |
| Quant tensor-core GEMM (MFMA) | `kernels/quantization/qgemm/variants/rocm_cdna3` | build-valid | PTX mma.sync.m16n8k16 rewritten to v_mfma_f32_16x16x16_f16. qgemm 58/58 + qflux 29/29 PASS on MI300X. |
| Quant MoE GEMMs (MFMA) | `kernels/moe/variants/rocm_cdna3_quant` | build-valid | fp8/nvfp4/wna16 grouped GEMMs on MFMA + plain silu/quant/routing; ALL PASS vs fp64. |
| lm_head + turboquant + mf_primitives | `kernels/quantization/{lm_head,turboquant}/variants/rocm_cdna3` | build-valid | lm_head sampling 20/20; turboquant/mf_primitives ALL PASS on MI300X. |

### Runtime environment update (2026-07-06)

A ROCm build of PyTorch is now installed in the isolated venv
`~/.venvs/rocm-torch` (`torch 2.9.1+rocm6.4`, `torch.cuda.is_available()` True on
the MI300X). The earlier "blocked by non-ROCm PyTorch" note no longer applies:
the existing `test_python.py` pyext harnesses can be run with that interpreter,
and standalone HIP oracle/bench harnesses (used for the elementwise port) run
directly on the gfx942 GPUs. The system CUDA PyTorch was left untouched.

## CUDA Kernel Inventory

| CUDA source directory | Target ROCm area | CDNA3 status | Notes |
| --- | --- | --- | --- |
| `attention/bf16_b300_mha_causal` | `kernels/attention/gqa_causal/variants/rocm_cdna3` | active | Correct CDNA3 causal GQA forward; vs SDPA is_causal ~0.14%. |
| `attention/bf16_b300_mha_noncausal` | `kernels/attention/gqa/variants/rocm_cdna3` | active | Correct CDNA3 GQA forward (online-softmax); vs SDPA ~0.15%. |
| `attention/mha_ampere` | `kernels/attention/gqa` or MHA variant | planned | Ampere-specific implementation is not a CDNA3 port source. |
| `attention/mha_h100` | `kernels/attention/gqa` or MHA variant | planned | H100/TMA-style code is not portable as-is. |
| `attention/mha_h100_lcf` | `kernels/attention/gqa` or MHA variant | planned | H100/LCSF path needs ROCm-specific equivalent. |
| `based` | `kernels/linear_attention/based/variants/rocm_cdna3` | active | Taylor-2 causal linear attention; vs fp64 oracle PASS ~0.14% on MI300X. |
| `elementwise` | `kernels/activations/elementwise` | active, build-valid | Ported to `variants/rocm_cdna3`; 56/56 fp64-oracle checks pass on MI300X, focused perf A/B recorded (64-lane wavefront widening +15-61%). |
| `fftconv` | `kernels/ssm/fftconv/variants/rocm_cdna3` | active | Circular convolution (=ifft(fft(u)fft(k))); vs fp64 PASS (exact). |
| `flux` | `kernels/matmul/flux/variants/rocm_cdna3` | active | flux_gelu + flux_gate (dense bf16 matmul + epilogue) via MFMA; ALL PASS vs fp64 on MI300X. |
| `gemm` | `kernels/matmul/common` | planned | CUDA shared GEMM helpers should be replaced by ROCm-native common code as needed. |
| `gemm/baselines/bf16_cublas` | `perf/baselines/matmul/bf16fp32` | planned | Use rocBLAS, hipBLASLt, CK, AITER, or Triton/ROCm as ROCm baselines. |
| `gemm/baselines/bf16_cublas_lt` | `perf/baselines/matmul/bf16fp32` | planned | Use hipBLASLt or CK instead of cuBLASLt. |
| `gemm/baselines/fp8_cublas_lt` | `perf/baselines/matmul/fp8fp32` | planned | Use hipBLASLt, CK, AITER, or Triton/ROCm for FP8 baselines. |
| `gemm/baselines/int8_cublas_lt` | `perf/baselines/matmul/int8` | planned | Use ROCm library baselines; no active CDNA3 INT8 matmul path yet. |
| `gemm/baselines/mxfp8_cublas_lt` | `perf/baselines/matmul/mxfp8` | capability-gated | cuBLASLt MXFP8 baseline has no direct ROCm equivalent in this repo. |
| `gemm/baselines/nvfp4_cublas_lt` | `perf/baselines/matmul/nvfp4` | capability-gated | NVFP4 cuBLASLt baseline has no direct ROCm equivalent in this repo. |
| `gemm/bf16_b200` | `kernels/matmul/bf16fp32` | partial | Covered only by existing shape-specific CDNA3 BF16 GEMM. |
| `gemm/bf16_h100` | `kernels/matmul/bf16fp32` | partial | Covered only by existing shape-specific CDNA3 BF16 GEMM. |
| `gemm/educational_b200` | `kernels/matmul` | planned | Educational CUDA progression should not be copied into public CDNA3 variants. |
| `gemm/educational_h100` | `kernels/matmul` | planned | Educational CUDA progression should not be copied into public CDNA3 variants. |
| `gemm/fp8_ampere` | `kernels/matmul/fp8fp32` | partial | Covered only by existing shape-specific CDNA3 FP8/scaled matmul. |
| `gemm/fp8_b200` | `kernels/matmul/fp8fp32` | partial | B200-specific route needs CDNA3-specific implementation or library baseline. |
| `gemm/fp8_h100` | `kernels/matmul/fp8fp32` | partial | H100 route needs CDNA3-specific implementation or library baseline. |
| `gemm/fp8_h100_scaled` | `kernels/matmul/scaled_matmul` | partial | Existing CDNA3 scaled matmul builds; needs correctness and perf run. |
| `gemm/int8_ampere` | `kernels/matmul/int8` | planned | No active CDNA3 INT8 matmul path yet. |
| `gemm/int8_b200` | `kernels/matmul/int8` | planned | B200-specific implementation is not portable as-is. |
| `gemm/int8_h100` | `kernels/matmul/int8` | planned | H100-specific implementation is not portable as-is. |
| `gemm/mxfp8_b200` | `kernels/matmul/mxfp8` | capability-gated | CDNA4 scaled-MFMA MXFP8 copy does not map directly to gfx942. |
| `gemm/nvfp4_b200` | `kernels/matmul/nvfp4` | capability-gated | NVFP4 and B200 path need an AMD format and instruction strategy. |
| `hedgehog` | `kernels/linear_attention/hedgehog/variants/rocm_cdna3` | active | Hybrid dual-softmax-feature linear + windowed-exact attention; vs fp64 PASS ~0.16%. |
| `layernorm` | `kernels/norms/layernorm` | active, build-valid | CDNA3 path exists; correctness/perf needs ROCm PyTorch environment. |
| `lin_attn_tm` | `kernels/linear_attention/variants/rocm_cdna3` | active, build-valid | GDN + linear-attention (non-causal/causal/chunk/complex) ALL PASS on MI300X. |
| `linear_attention` | `kernels/linear_attention/variants/rocm_cdna3` | active | TM (lin_attn_tm) + decay linear attention (kittens linear_attention.cu) ported; all PASS vs fp64. |
| `mamba2` | `kernels/ssm/mamba2/variants/rocm_cdna3` | active | selective_scan (+APC) AND the SSD/state-space-duality form (mamba2.cu) ported; all PASS vs fp64. |
| `moe` | `kernels/moe/variants/rocm_cdna3` | active, build-valid | Routing + grouped-GEMM MoE MLP ported; 8/8 checks pass on MI300X (end-to-end MoE MLP vs fp64 5.9e-7). |
| `moe_quant` | `kernels/moe/variants/rocm_cdna3_quant` | active, build-valid | 3 quantized grouped GEMMs (fp8/nvfp4/wna16) rewritten to MFMA + plain silu/quant/routing; ALL PASS vs fp64 on MI300X. |
| `parallel/ag_gemm` | `kernels/collectives/ag_gemm` | capability-gated | Requires distributed ROCm/RCCL design and validation. |
| `parallel/ag_gemm_fp8` | `kernels/collectives/ag_gemm_fp8` | capability-gated | Requires distributed ROCm/RCCL design and validation. |
| `parallel/all_gather` | `kernels/collectives/all_gather` | capability-gated | Requires distributed ROCm/RCCL design and validation. |
| `parallel/all_reduce` | `kernels/collectives/all_reduce` | capability-gated | Existing `distributed-kernels/` work should be reconciled first. |
| `parallel/all_reduce_educational` | `kernels/collectives/all_reduce` | planned | Educational CUDA path should not be imported as production coverage. |
| `parallel/all_to_all` | `kernels/collectives/all_to_all` | capability-gated | Requires distributed ROCm/RCCL design and validation. |
| `parallel/gemm_ar` | `kernels/collectives/gemm_ar` | capability-gated | Requires distributed ROCm/RCCL design and validation. |
| `parallel/gemm_rs` | `kernels/collectives/gemm_rs` | capability-gated | Requires distributed ROCm/RCCL design and validation. |
| `parallel/gemm_rs_fp8` | `kernels/collectives/gemm_rs_fp8` | capability-gated | Requires distributed ROCm/RCCL design and validation. |
| `parallel/moe_dispatch_gemm` | `kernels/collectives/moe_dispatch_gemm` and `kernels/moe` | capability-gated | Needs MoE route plus distributed runtime design. |
| `parallel/reduce_scatter` | `kernels/collectives/reduce_scatter` | capability-gated | Requires distributed ROCm/RCCL design and validation. |
| `parallel/ring_attn` | `kernels/collectives/ring_attn` and `kernels/attention` | capability-gated | Requires distributed attention design and validation. |
| `parallel/ulysses_attn` | `kernels/collectives/ulysses_attn` and `kernels/attention` | capability-gated | Requires distributed attention design and validation. |
| `quant` | `kernels/quantization` | partial | Dequant + fp16/int8 GEMV + runtime-quant ported to `quantization/qgemv/variants/rocm_cdna3`: 29/29 format dequant EXACT + GEMV PASS, W8A8/W2A8 PASS, runtime-quant 4/4 PASS on MI300X (`__dp4a`->`__builtin_amdgcn_sdot4`). Tensor-core qgemm/lm_head/qflux (`tm_qmm.cuh` PTX `mma.sync`) deferred to the MFMA pass. |
| `rotary` | `kernels/attention/rotary` | active, build-valid | CDNA3 path exists; correctness/perf needs ROCm PyTorch environment. |
| `serving` | `kernels/serving/variants/rocm_cdna3` | active, build-valid | All 12 self-checking harnesses pass on MI300X (paged attn v1/v2, GQA-staged, attn_q, attn_varlen, MLA, rope_kv, kv_cache, sampling, logits, beam, spec_beam, eagle, sparse) — 104 pass-lines, 0 fail. |
| `tm_cuda` | operation-specific ROCm directories | not-portable-as-is | Python/CUDA extension aggregator should be decomposed into native ROCm operations. |

## Known Non-Ports

2026-07-06 update: the GQA-forward CDNA3 failure was independently reproduced
with a torch-free standalone harness (`kernels/attention/gqa/variants/rocm_cdna3`,
`make -f Makefile.harness test` → ~101% rel error on gfx942). Root cause is
confirmed: the HipKittens attention kernel uses CDNA4 register-tile geometry and
HipKittens ships no CDNA3 attention kernel, so a real CDNA3 port must be written
around CDNA3 MFMA/LDS geometry. That variant dir now holds the kernel split
(`attn_kernel.cuh` + pybind) and the standalone oracle (`harness.cpp`) as
scaffolding for the rewrite; it is build-valid only, not correctness-valid.
Dense bf16/fp8 GEMM already has working CDNA3 variants in-repo.

The archived CDNA3 attention and MXFP8 attempts are intentionally not active:

- `kernels/attention/gqa/archive/rocm_cdna3_cdna4_shape_attempt`
- `kernels/attention/gqa_causal/archive/rocm_cdna3_cdna4_shape_attempt`
- `kernels/attention/gqa_backward/archive/rocm_cdna3_cdna4_isa_attempt`
- `kernels/attention/gqa_causal_backward/archive/rocm_cdna3_cdna4_isa_attempt`
- `kernels/matmul/mxfp8/archive/rocm_cdna3_cdna4_scaled_mfma_attempt`

Their local `README.cdna3.md` files document correctness or ISA blockers. A
real CDNA3 port should replace them with kernels designed for gfx942 MFMA, LDS,
and scheduling constraints.
