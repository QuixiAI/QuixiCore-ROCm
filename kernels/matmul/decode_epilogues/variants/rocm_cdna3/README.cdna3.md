# decode epilogues - CDNA3

Phase 3 CPU/Metal parity port for decode-oriented matmul epilogues.

## Source Contracts

- CPU: `../QuixiCore-CPU/kernels/matmul/matmul_extended_ref.cpp`
- CPU: `../QuixiCore-CPU/kernels/matmul/lora_ref.cpp`
- CPU: `../QuixiCore-CPU/kernels/matmul/dense_gemm_ref.cpp`
- Metal: `../QuixiCore-Metal/kernels/matmul/decode_linear/decode_linear.metal`
- Metal: `../QuixiCore-Metal/kernels/matmul/cmplx_matmul/cmplx_matmul.metal`

Implemented in this variant:

- `decode_linear`
- `decode_linear_residual`
- `decode_linear_q8` for canonical GGUF `q8_0` packed weights
- `decode_linear_epilogue_dense`
- `decode_linear_epilogue_packed` for `q4_0`, `q8_0`, `q6_K`, `mxfp8`,
  `nvfp4`, and `mxfp4` packed weights
- `decode_swiglu_dense`
- `decode_swiglu_packed` for `q4_0`, `q8_0`, `q6_K`, `mxfp8`, `nvfp4`, and
  `mxfp4` packed weights
- `gemm_gate_residual`
- `grouped_gemm`
- `lora_apply_direct_f16`
- `complex_gemm`

## CDNA3 Approach

Decode shapes are latency oriented. The main kernels assign one 64-lane CDNA3
wavefront to one output element and reduce the input dimension with
`qc::wave_reduce_sum`. This preserves the CPU references' fp32/fp64 oracle
semantics while giving a direct scalar-vs-wavefront optimization lever.

Packed decode epilogues share the quant-format device decoders from
`kernels/quantization/qgemv/variants/rocm_cdna3`. The harness builds
deterministic canonical packed blocks for `q4_0`, `q8_0`, `q6_K`, `mxfp8`,
`nvfp4`, and `mxfp4`, then compares against a byte-layout-aware host oracle.

`grouped_gemm` keeps a scalar one-thread-per-output implementation. The same
wave64 split-dot lever was measured for grouped tiles and rejected because it
was 0.16x the scalar baseline on the recorded shape.

LoRA follows the CPU storage contract: the low-rank intermediate is rounded to
FP16, then the adapter-B product is rounded to FP16 before adding the optional
base and scale.

`complex_gemm` is promoted as a public planar real/imaginary op. It is a scalar
wavefront dot-product implementation, not an MFMA complex GEMM.

## Non-Ports

MoE-specific grouped GEMM backward and quantized grouped variants live in the
MoE variant, not this decode epilogue harness. This port covers only the public
dense CPU `grouped_gemm` contract:
`A[groups,M,K] @ B[groups,K,N] -> C[groups,M,N]`.
