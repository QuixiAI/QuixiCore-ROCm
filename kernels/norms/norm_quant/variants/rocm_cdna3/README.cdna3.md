# Norm Quant — CDNA3 (gfx942)

Functional ROCm port of the CUDA `elementwise/tm_norm_quant_kernels.cuh`
family:

- `rms_norm_quant<T, FP8, DYN, RESID>` for FP8 or INT8 output, static or
  dynamic per-row scale, with optional residual output.
- `azp_int8_quant<T, DYN>` for asymmetric int8 quantization.
- `per_token_group_int8_quant<T>` for symmetric per-token group int8.

The port uses HIP half/bfloat16 types, the repository FP8 e4m3 encoder, and
32-lane subgroup reductions staged through shared memory for block-wide RMSNorm
rows on gfx942 wavefronts.

```bash
make test
```

The test binary runs fp64/CPU round-trip checks and a focused block-size timing
run for `rms_norm_quant` on MI300X.
