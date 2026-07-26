# qk_norm_rope - CDNA3

This directory contains CDNA3 standalone harnesses for the QuixiCore QK RMSNorm
+ RoPE contracts.

- `qk_norm_rope.cu` is the original packed-output bf16 variant: packed QKV in,
  packed QKV out, V copied through.
- `qk_norm_rope_kv_f16.cu` ports the newer Metal split-store variant: packed
  bf16 QKV in, Q bf16 out, K/V fp16 out.

The `qk_norm_rope_kv_f16` contract matches Metal:

```text
qkv       : bf16 (T, (Hq + Hk + Hv) * D)
q_weight  : bf16 (D)
k_weight  : bf16 (D)
cos/sin   : bf16 (max_position, D/2)
positions : int32 (T)
q_out     : bf16 (T, Hq * D)
k_out     : fp16 (T, Hk * D)
v_out     : fp16 (T, Hv * D)
D in {64,128,256}; split-half or interleaved RoPE; optional Gemma weight mode
```

One CDNA3 block owns one `(token, head)` row. Q/K rows use a block reduction for
the RMS denominator and write the rotated split destination; V rows copy through
with the bf16-to-fp16 conversion folded into the same pass. The harness compares
Q against bf16-output tolerance and K/V against fp16-output tolerance, and it
benchmarks against a scalar row baseline.
