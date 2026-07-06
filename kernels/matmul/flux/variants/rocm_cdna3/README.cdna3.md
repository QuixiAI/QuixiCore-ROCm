# flux — CDNA3 (gfx942): correctness-valid

Dense bf16 matmul with fused epilogues, ported from QuixiCore-CUDA/kernels/flux
(ThunderKittens). CDNA3 MFMA `v_mfma_f32_16x16x16_f16` on a 64-wide wavefront,
one wavefront per 16x16 output tile (same primitive as the quant qgemm MFMA port).

  flux_gelu : Y = gelu_tanh(A @ B + bias)
  flux_gate : Y = (A @ B) * gate

Standalone fp64 oracle (`flux.cu`): both PASS on MI300X (essentially exact,
fp32 accum). `make test`. Perf follow-up: LDS staging + wider tiles + a contiguous
B layout (B is loaded column-strided here for correctness).
