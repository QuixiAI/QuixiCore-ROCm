# Phase 6 Quant Authoring CDNA3 Variant

This variant ports the Phase 6 CPU/Metal parity surface:

- `fake_quant_int8`, `fake_quant_float8`
- `tq2_0_pack`, `tq2_0_unpack`
- `ternary_pack`, `ternary_unpack`, `ternary_stats`,
  `ternary_code_flip_count`
- `calibration_absmax`
- BitNet `qgemm_backward_input`
- packed-table `dequant_gather`, `quantized_embedding`,
  `quantized_embedding_bag`
- `mxfp4_gemv`

The file is a standalone HIP harness, not a PyTorch extension. It uses the
shared CDNA3 harness for fp64-oracle checks, wave64 reductions, and timing.

## Source Contracts

The CPU contracts are from:

- `../QuixiCore-CPU/kernels/quantization/quantization_ref.cpp`
- `../QuixiCore-CPU/kernels/serving/basert_ref.cpp`
- `../QuixiCore-CPU/kernels/quantization/qgemm_extended_ref.cpp`
- `../QuixiCore-CPU/kernels/serving/serving_quant_ref.cpp`
- `../QuixiCore-CPU/kernels/quantization/microscale_ref.cpp`

The Metal contracts are from:

- `../QuixiCore-Metal/kernels/quantization/dequant_gather/dequant_gather.metal`
- `../QuixiCore-Metal/kernels/quantization/qgemm_bwd/qgemm_bwd.metal`

## CDNA3 Approach

Row/group reductions use one CDNA3 wavefront per row or output element where a
reduction is needed. Packing paths use one workgroup per canonical block/group
and store byte layouts exactly as the CPU contract defines them. Packed-table
embedding uses the existing ROCm quant-format decode structs for `q4_0`,
`q8_0`, and `q6_K`; the harness generates deterministic valid blocks directly
so it does not import sibling reference code.

The BitNet backward-input port follows the Metal route: dense `grad_y` is
contracted against ternary 10-byte weight blocks without materializing dense
weights. The measured comparison keeps the wave64 split-dot candidate over a
scalar baseline.

## Non-Ports

This is correctness-first CDNA3 coverage, not a public C ABI. The embedding
ports validate FP32 output storage in this tranche; the CPU storage wrappers
for FP16/BF16 can be added as thin store-type variants if the ROCm public API
grows a shared storage dispatcher.
