# GQA forward attention — CDNA3 (gfx942): correctness-valid

Status: **works on CDNA3.** Validated against PyTorch SDPA (mean rel ~0.15% at
B16 H64 H_KV8 N2048 D128) and an fp32 host reference (D=128 GQA and D=64 MHA).
Replaces the archived CDNA4-shape copy, which was numerically wrong on gfx942.

## Approach (why not the HipKittens CDNA4 kernel)

The HipKittens GQA kernel is micro-optimized for CDNA4 (manual LDS swizzles,
`s_waitcnt` scheduling, `mma_AtB`/`col_max` on a transposed layout, and
`reinterpret_cast`s between packed tile shapes `rt_32x32_s`↔`rt_16x32_4_s` that
share register layout on CDNA4 but not CDNA3). Recompiled for gfx942 it builds
but produces ~100% error (confirmed vs a standalone host ref AND live SDPA).

Rather than fight that microarchitecture-specific code, this is a clean, correct
CDNA3 flash-attention forward using the same online-softmax proven in the serving
`paged_attention` port: **one 64-wide wavefront per (query, head, batch)**; the
64 lanes split the head dim (`EPL = D/64` elements per lane); `QK^T` is a
wavefront reduction; the softmax state `(m, l)` and the `O` accumulator are kept
online across the KV sequence. GQA head map `kv = h/(H/H_KV)`, scale `1/sqrt(D)`,
layout `[B,N,H,D]` (Q/O) and `[B,N,H_KV,D]` (K/V). `ATTN_CAUSAL=1` restricts each
query to keys `0..q`.

## Files

- `attn_kernel.cuh` — the kernel + `attn_globals`/`dispatch_micro` (shared by the
  pybind module and the standalone harness).
- `kernel.cpp` — pybind entry (`tk_kernel.dispatch_micro`).
- `harness.cpp` / `Makefile.harness` — torch-free standalone oracle (host fp32
  ref). `verify_sdpa.py` — pybind vs `scaled_dot_product_attention`.

## Build / test

```bash
make -f Makefile.harness test         # standalone host-ref oracle
make GPU_TARGET=CDNA3 PYTHON=python3.10   # pybind .so
LD_PRELOAD="/opt/rocm/lib/libhsa-runtime64.so /opt/rocm/lib/libamdhip64.so" \
  ~/.venvs/rocm-torch/bin/python verify_sdpa.py   # vs PyTorch SDPA
```

## Perf note

Correctness-first: O(N) per query with no K/V reuse across queries — correct but
bandwidth-bound. An MFMA-tiled flash version (reusing `tm_qmm_mfma.cuh` for
QK^T/PV with block K/V reuse) is the performance follow-up; this kernel is the
verified correctness baseline and oracle for it.
