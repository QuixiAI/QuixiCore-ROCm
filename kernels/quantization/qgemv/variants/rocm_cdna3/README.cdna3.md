# Quantized dequant + GEMV — CDNA3 (gfx942)

Native CDNA3 port of the dequant + GEMV slice of
`../QuixiCore-CUDA/kernels/quant` (the plain-CUDA `tm_*` quant format layer;
twin of QuixiCore Metal's `tk.quant`). QuixiCore-internal port, not a
third-party import.

## Coverage

- **Format dequant + fp16 GEMV** for all 29 tranche-1/2 formats: `q8_0 q4_0
  q4_1 q5_0 q5_1 kU4B8 kU4 hqq fp8_e4m3 e5m2 fp8_block fp4_e2m1 mxfp8 mxfp4
  nvfp4 mxfp6_e3m2 mxfp6_e2m3 bitnet q2_K q3_K q4_K q5_K q6_K iq4_nl iq4_xs
  iq2_xxs iq2_xs iq3_xxs iq1_s` (`qgemv.cu` + `quant_formats*.cuh`,
  `quant_tables.cuh`).
- **Integer GEMV**: W8A8 and W2A8/BitNet (`qgemv_int.cu`).
- **Runtime quantize**: per-token / per-tensor, int8 + fp8-e4m3 (`quant_rt.cu`).

The tensor-core `qgemm` / `lm_head` / `qflux` path (PTX `mma.sync` in
`tm_qmm.cuh`) is **not** in this variant — it needs an MFMA rewrite and is
tracked under the HipKittens/MFMA pass.

## Port notes (CUDA -> CDNA3)

- `hipify-perl`: headers, `__nv_bfloat16` -> `__hip_bfloat16`, runtime API.
- **`__dp4a` -> `__builtin_amdgcn_sdot4(a, b, acc, /*clamp=*/false)`** — the
  signed int8x4 dot (`idot4`) maps to the gfx942 `V_DOT4_I32_I8` instruction.
  Validated by the int8 GEMV paths and every dp4a-based integer format.
- **Warp width**: `__shfl_*_sync(0xffffffffu, ...)` -> mask-free `__shfl_*`
  (HIP needs a 64-bit mask on the 64-wide wavefront; the row reductions use
  offsets <=16 in 32-lane groups and are correct on warp64).

## Golden data

Correctness is checked against the CUDA repo's checked-in `quant.py`-derived
golden (`../QuixiCore-CUDA/kernels/quant/golden{,_int}`), which is byte-exact
across Metal/CUDA. That golden omits the `meta.txt` the original harness read,
so the ported `main()` falls back to the directory basename for the format and
the fixed gen_golden dims (N=512, K=4096); pass `[N K]` to override. Golden
binaries are not copied into this repo.

## Build / run

```bash
make test    # dequant+GEMV all formats, int8 GEMV, runtime-quant
```

## Result (MI300X, 2026-07-06)

29/29 formats: dequant **EXACT** (0 mismatching values vs quant.py) and GEMV
PASS (~0.018% rel err). Int8 W8A8 603 GB/s / W2A8 129 GB/s PASS. Runtime
quantize 4/4 PASS. See `perf/optimization_status.md` (2026-07-06 quant entry).
