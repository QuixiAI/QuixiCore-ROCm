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

## Correctness

Dequant is checked **bitwise** — a nibble decodes to a table entry times a power
of two, so there is no rounding to tolerate. The dots are checked against an
fp64 accumulation, judged as `|err| / sum|terms|` rather than `|err| / |result|`:
dividing by a cancelling result makes the metric explode on rows that happen to
cancel, which says nothing about the kernel. Measured 2.4e-08 to 5.3e-08 against
an fp32 bound of roughly `sqrt(N) * 2^-24` ≈ 1.3e-06.

```bash
make test    # dequant (bitwise), ordering probe, gemv, MoE gemv
make bench
```
