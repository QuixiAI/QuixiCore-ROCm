# MXFP4 (GGML_TYPE_MXFP4) GGUF kernels — CDNA3 (gfx942)

Native HIP for the OCP MXFP4 block layout as stored in GGUF. This is the format
DeepSeek-V4-Flash keeps its routed experts in (129 MXFP4 tensors in the 0731
release), and it is the one quant SlimServe's GGUF stack could not read.

## Layout

```text
struct { uint8_t e; uint8_t qs[16]; }     // 17 bytes, 32 values
  value[j]      = e8m0(e) * e2m1(qs[j] & 0xF)   for j in [0,16)
  value[j + 16] = e8m0(e) * e2m1(qs[j] >> 4)
```

The halves matter: low nibbles supply the **first** 16 values and high nibbles
the second 16 — not an even/odd interleave. Swapping them leaves the value
multiset and the vector norm essentially unchanged, so `test_dequant` pins the
ordering by element position with a hand-built probe block, not by comparing
norms.

Three independent sources agree on this layout: llama.cpp's `block_mxfp4`
(`sizeof == 1 + QK_MXFP4/2`), QuixiCore-ROCm's own `mxfp4` format struct
(`block_k=32, block_bytes=17`), and ds4's GGUF quant table (`{"mxfp4", 32, 17}`).

## Approach

e2m1 is **not** uniform, so unlike q4_0 the nibbles cannot be fed straight to a
dot-product instruction. They index a 16-entry table holding 2x the true values
— `{0,1,2,3,4,6,8,12}` and negatives, all integers — with the factor of 2 folded
into the scale. That keeps the inner loop on integer `v_dot4`
(`__builtin_amdgcn_sdot4`) rather than unpacking to float.

The lookup is four `__builtin_amdgcn_perm` byte-permutes plus a blend on nibble
bit 3, which indexes the 16-byte table without LDS or branches. That is the
CDNA idiom; llama.cpp uses the same instruction for its HIP path.

`e8m0(0)` decodes to the smallest normal, not zero, matching ggml.

## Kernels

- `dequant_mxfp4<dst_t>` — block → fp32/fp16, one thread per block.
- `vec_dot_mxfp4_q8_1` — one block against int8 activations with an fp32 scale.
- `mxfp4_gemv_q8_1<dst_t>` — one wave per output row. Doubles as the MoE form:
  pass `expert_ids` and a row stride and each token indexes its own expert, with
  a negative id zeroing the row (the padded-row convention the GGUF MoE path
  uses).
- `mxfp4_mmq_q8_1<MMQ_X, MMQ_Y, NWARPS, NEED_CHECK>` — the tiled GEMM, in
  `mxfp4_mmq_kernels.cuh`.

## GEMV or tile?

Both, and the caller picks. The GEMV reloads the weight row for every output
column, so its cost is linear in columns while the tile kernel's is flat.
Measured on MI300X at DeepSeek-V4-Flash expert shapes (4096 -> 2048, 32
experts, top-6), tile speedup over GEMV:

| routed tokens | 1 | 8 | 32 | 128 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMQ vs GEMV | 0.17x | 0.42x | 1.52x | 2.18x | 2.43x |

So the crossover is between 8 and 32 tokens, and shipping only one of the two
kernels is wrong at one end or the other. Neither is a fallback for the other.

MXFP4's block constants (QK 32 / QR 2 / QI 4) are identical to q4_0's, so the
tile geometry is the classic ggml q4_0 MMQ layout with no index changes at all
— only the e8m0 scale and the table lookup differ, and dropping q4_0's
activation-sum correction term, which MXFP4 has no offset to need.

The 17-byte block is the hazard worth naming: `qs` is at an odd offset and
blocks are 17 apart, so nothing in the weight stream is 4-byte aligned. Reads
of the quants must be `memcpy`, never a cast — the usual `get_int_from_uint8`
helper assumes 2-byte alignment and is not safe here.

## Correctness

Dequant is checked **bitwise** — a nibble decodes to a table entry times a power
of two, so there is no rounding to tolerate. The dots are checked against an
fp64 accumulation, judged as `|err| / sum|terms|` rather than `|err| / |result|`:
dividing by a cancelling result makes the metric explode on rows that happen to
cancel, which says nothing about the kernel. Measured 2.4e-08 to 5.3e-08 against
an fp32 bound of roughly `sqrt(N) * 2^-24` ≈ 1.3e-06.

The tile kernel is checked the same way and lands at 7.2e-08 to 8.3e-08,
including the two ragged cases (rows not a multiple of `MMQ_Y`, columns not a
multiple of `MMQ_X`) where the tile reads past the live region and has to drop
the result at write-back rather than in the inner loop.

Its activations are quantized on the **host** and handed over already in q8_1
form. Letting a device quantizer intervene would fold that kernel's rounding
into the result and could mask an error here. Two details of q8_1 have to be
mirrored exactly or the reference drifts far enough to hide real bugs: the
quants are formed with the **fp32** scale while the stored and later-applied
scale is **fp16**, and `roundf` breaks ties away from zero where most
round-to-nearest implementations break them to even.

```bash
make test    # dequant (bitwise), ordering probe, gemv, MoE gemv, tiled GEMM
make bench
```
