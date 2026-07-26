# qgemv_fused - CDNA3

Port of the current QuixiCore-Metal fused packed-Q4_0 decode GEMV kernels:

- `qgemv_q4_0_f32_up_gate_gelu`
- `qgemv_q4_0_f32_up_gate`
- `qgemv_q4_0_f32_qkv`

Contract:

```text
weights : uint8 bytes (N, K/32, 18), GGUF Q4_0 blocks
          block = fp16 scale + 16 packed nibbles, value = scale * (nibble - 8)
x       : fp32 (K, 1)
outputs : fp32 (N, 1) or split Q/K/V vectors
```

The CDNA3 candidate maps one wave64 to one output row and reduces the dot
product with `qc::wave_reduce_sum`. The up+gate+GELU variant computes both
rows in one wave and applies tanh-GELU before the single output store. The
harness compares all three public variants against an fp64 dequantized oracle
and benchmarks against a scalar one-thread-per-row baseline.
