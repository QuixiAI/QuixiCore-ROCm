# GQA causal forward attention — CDNA3 (gfx942): correctness-valid

Causal variant of the CDNA3 GQA forward attention (see
`../../../gqa/variants/rocm_cdna3/README.cdna3.md` for the approach). Same
correct online-softmax kernel with `ATTN_CAUSAL=1` (query q attends to keys
0..q). Validated vs PyTorch SDPA `is_causal=True` (mean rel ~0.14% at
B16 H64 H_KV8 N2048 D128) and an fp32 host reference. Replaces the archived
CDNA4-shape copy which was numerically wrong on gfx942.

    make -f Makefile.harness test    # standalone host-ref oracle (causal)
    LD_PRELOAD="/opt/rocm/lib/libhsa-runtime64.so /opt/rocm/lib/libamdhip64.so" \
      ~/.venvs/rocm-torch/bin/python verify_sdpa.py   # vs SDPA is_causal=True
