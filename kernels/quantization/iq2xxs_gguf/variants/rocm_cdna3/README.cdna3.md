# IQ2_XXS (GGML_TYPE_IQ2_XXS) GGUF kernels — CDNA3 (gfx942)

Native HIP for the E8-lattice codebook quant at 2.0625 bpw, as stored in GGUF.
DeepSeek-V4-Flash ships two files whose routed experts use it, and they are the
two that fit on 2 MI300X rather than 4.

## Layout

```text
struct { half d; uint16_t qs[32]; }        // 66 bytes, 256 weights
```

Each 32-weight group takes four of those uint16: two carry four 8-bit indices
into a 256-entry grid (each entry is 8 packed magnitudes), the other two carry
four 7-bit sign-table indices and, in the top 4 bits, the group's sub-scale.

```text
weight = grid_byte * (sign ? -1 : 1) * d * (0.5 + sub_scale) * 0.25
```

`quant_tables.cuh` holds the grid and sign tables, generated from llama.cpp's
`ggml-common.h`. The ggml-metal dequant and llama.cpp's CUDA vec-dot agree on
this field layout, and the harness re-derives it independently on the host.

## The point: where decode happens

Every non-codebook format in a tiled GEMM stages *packed* quants in LDS and
decodes inside the inner dot, because decode is a shift and a mask. That design
is wrong for IQ2_XXS, and not by a little.

Decode here is four dependent random reads of a 2 KB grid plus a sign-table
read, and the inner dot runs once per (row, column) pair — so staging packed
data multiplies the gather by the tile's column count. Measured in vLLM at
DeepSeek-V4 expert shapes (4096 → 2048, 32 experts, top-6), that design reached
**1.07x** the per-row vector kernel at 512 routed tokens, where q4_0-shaped
formats reach 2.5–4x. The tile amortized the weight load and then paid the
gather `MMQ_X` times over.

Decoding once per weight at **tile-load** time, into signed bytes plus one float
scale per 32 weights, takes the same shape to **3.4x**. The inner loop collapses
to `dp4a` over a q8_0-shaped tile. `quant_tables.cuh` says as much in its own
header note: these grid indices are data-dependent rather than warp-uniform, so
hot kernels should stage them.

The cost is tile width — 512 decoded weights per row is 128 ints against the 32
the packed form needs — so `MMQ_Y` stays small to keep LDS in budget. That is a
real trade, not a free win: the wider tile buys fewer rows per block.

## Kernels

- `dequant_iq2_xxs<dst_t>` — block → fp32/fp16, one thread per superblock.
- `iq2xxs_decode_group` — one 32-weight group to eight ints of signed bytes.
  Signs are applied byte-wise (xor by the mask, then subtract it, which is a
  conditional two's-complement negate), so there is no branch and no
  per-element work.
- `iq2xxs_mmq_q8_1<MMQ_X, MMQ_Y, NWARPS, NEED_CHECK>` — the tiled GEMM.

`__vcmpeq4` and `__vsub4` have no HIP equivalent and are defined here the way
ggml's ROCm path defines them. The per-byte subtract matters: a plain 32-bit
subtract would propagate borrows across byte lanes and corrupt the neighbouring
weights.

## Correctness

Dequant is checked **bitwise** — a weight is a grid byte times a sign times a
scale, all exactly representable, so there is nothing to tolerate. The GEMM is
checked against an fp64 accumulation, judged as `|err| / sum|terms|` rather than
`|err| / |result|`, since dividing by a cancelling result makes the metric
explode on rows that happen to cancel. Measured 2.5e-08 to 3.7e-08, including
the ragged cases where a tile overruns the live region and has to drop the
result at write-back.

Activations are quantized on the **host** and handed over already in q8_1 form,
so this measures the tile kernel alone. Two q8_1 details have to be mirrored or
the reference drifts far enough to hide real errors: the quants are formed with
the **fp32** scale while the stored and later-applied scale is **fp16**, and
`roundf` breaks ties away from zero.

```bash
make test    # dequant (bitwise) + tiled GEMM over five shapes
make bench
```

Bench on MI300X: `2048x4096x512` in 0.54 ms.
