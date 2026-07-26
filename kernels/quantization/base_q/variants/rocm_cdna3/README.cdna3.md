# BaseQ canonical family - CDNA3 (gfx942)

Canonical BaseQ is defined by the CPU backend, not by a live Metal kernel:
`../QuixiCore-CPU/include/quixicore_cpu/base_q.h` and
`../QuixiCore-CPU/kernels/quantization/base_q_ref.cpp`. The ROCm port follows
that contract directly.

## Contract

BaseQ stores row-major packed integer codes with little-endian bit order at
`column * bits`. Supported bit widths are 2, 3, 4, 5, 6, and 8. Supported group
sizes are 32, 64, and 128. Per-group scales, and affine biases when present,
are row-major `[rows, columns / group_size]` in BF16, F16, E8M0, or E4M3
storage. E4M3 is valid only for 8-bit codes.

Decoded values are:

```text
symmetric: (code - 2^(bits - 1)) * scale
affine:    code * scale + bias
```

The harness covers the nine Phase 5 operations:

- `base_q_dequant`
- `base_q_gemv`
- `base_q_gemm`
- `base_q_embedding`
- `base_q_gemv_qkv`
- `base_q_gemv_swiglu`
- `base_q_lm_head_argmax`
- `base_q_moe_gemm`
- `base_q_moe_swiglu`

Invalid embedding ids produce zero rows. LM-head scores are rounded to the input
storage type before argmax, and ties pick the lower token id. The MoE kernels
preserve the canonical 32-row padded expert schedule.

## Approach

All consumers share one device decode core for packed codes and scale storage.
Projection-style rows use a wave64 split dot product: one CDNA3 wavefront owns
one output element and reduces the K dimension with `qc::wave_reduce_sum`.
Scalar one-thread-per-output kernels remain in the harness as the measured
baseline.

`base_q_lm_head_argmax` keeps the materialized route: wave64 BaseQ projection
into score storage matching the input dtype, followed by a block argmax. The
single-kernel streaming argmax was measured and rejected for the tested vocab
shape because it underfills the device.

## Files

- `base_q.cu` - standalone HIP harness, kernels, scalar baselines, fp64 host
  oracles, correctness matrix, and `--bench`.
- `Makefile` - `make test`, `make bench`, and `make clean`.

## Build / run

```bash
make test
make bench
```

## Result (MI300X, gfx942, ROCm/HIP 7.2)

Correctness: `make test` reports `ALL PASS` for 246 checks. Coverage includes
all supported bits, group sizes 32/64/128, BF16/F16/E8M0 scale storage,
E4M3 for 8-bit weights, symmetric and affine modes, FP32/FP16/BF16 outputs,
Q/K/V row-count skew, lower-token LM-head ties, and the 32-row padded MoE
expert schedule.

Focused benchmark: BaseQ4 affine, BF16 scales/biases, group size 64, FP16 input
unless noted, HIP events with 5 warmups, 20 iterations, and 100 repeated
launches per sample. Raw output:
`perf/results/2026-07-26/base-q-phase5-final3/bench.txt`.

| operation | baseline | kept/current | decision |
|---|---:|---:|---|
| `base_q_dequant` | n/a | 0.0150 ms, 1273.1 GB/s | KEEP current |
| `base_q_gemv` | scalar 0.8204 ms | wave64 0.0119 ms, 68.99x | KEEP wave64 |
| `base_q_gemm` | scalar 0.8661 ms | wave64 0.1093 ms, 7.93x | KEEP wave64 |
| `base_q_embedding` | n/a | 0.0298 ms, 1408.8 GB/s | KEEP current |
| `base_q_gemv_qkv` | scalar 0.9173 ms | wave64 0.0126 ms, 72.85x | KEEP wave64 |
| `base_q_gemv_swiglu` | scalar 1.4720 ms | wave64 0.0187 ms, 78.81x | KEEP wave64 |
| `base_q_lm_head_argmax` | materialized 0.0389 ms | streaming 13.7230 ms | REJECT streaming |
| `base_q_moe_gemm` | scalar 0.8871 ms | wave64 0.1132 ms, 7.84x | KEEP wave64 |
| `base_q_moe_swiglu` | scalar 1.6136 ms | wave64 0.1672 ms, 9.65x | KEEP wave64 |

All reported benchmark spreads are <= 1.12x.
