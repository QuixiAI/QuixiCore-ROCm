# Quantized MoE grouped GEMMs — CDNA3 (gfx942) MFMA

Native CDNA3 port of `../QuixiCore-CUDA/kernels/moe_quant`. The 3 tensor-core
quantized grouped GEMMs were rewritten from the NVIDIA PTX `mma.sync.m16n8k16`
path to gfx942 `v_mfma_f32_16x16x16_f16` (via the validated `tm_qmm_mfma.cuh`);
the activation-quant / routing kernels are plain-CUDA ports. Self-checking
harness (fp64 references, no golden files).

## Kernels

- `moe_gemm_fp8` — e4m3 weight-only grouped GEMM, rowwise B-scale.
- `moe_gemm_nvfp4` — dual-operand fp4 e2m1, per-16-block e4m3 scales
  (A swizzled, B plain), per-expert alpha.
- `moe_gemm_wna16<BIT>` — GPTQ/AWQ int4/int8 (uint32-packed, de-interleaved),
  per-group scale + optional zero-point.
- Plain ports: `silu_and_mul_quant_{static,perblock}`, `per_token_group_quant_fp8`,
  `nvfp4_experts_quant`, `moe_route_scored`.

## MFMA port

The CUDA path processed a 32-row expert tile as two 16-row m16n8k16 subtiles
(`aT`/`aB`), two `mma16816` each, and stored with a 32-lane column map. On MFMA
each 16-row subtile is one `v_mfma_f32_16x16x16_f16`, so per k-step: `aT`, `aB`,
one dequantized `b` fragment, two MFMAs. Store uses the MFMA layout — lane l owns
output column n = n0 + l%16 over rows m0 + 4*(l/16) + {0..3}, so the per-column
scale (B-scale / alpha / 1) is a single value per lane. The quant fragment
loaders (`load_afrag_nvfp4`, `load_wfrag_nvfp4`, `load_wfrag_wna16`) return a
`half4` of 4 contiguous K, all inside one 16-wide scale group (k step = 16), so
one scale applies per fragment.

## Build / run

```bash
make test    # -> "ALL PASS (0 failures)"
```

## Result (MI300X, 2026-07-06)

All pass vs fp64: moe_gemm_fp8 (1.9e-4), moe_gemm_nvfp4 (2.1e-8),
moe_gemm_wna16 int4 (2.7e-4) / int8 (2.7e-4); silu/quant/routing kernels within
tolerance. Raw: `perf/results/2026-07-06/moe-quant-mfma/`.
