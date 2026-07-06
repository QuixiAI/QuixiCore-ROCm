# GQA causal attention backward — CDNA3 (gfx942): correctness-valid

Causal variant of the CDNA3 attention backward (see
`../../../gqa_backward/variants/rocm_cdna3/README.cdna3.md`). Same kernels with
`CAUSAL_=1` (key j receives gradient only from queries i>=j). Validated by the
standalone fp64 analytic oracle: dQ/dK/dV PASS (mean rel <=0.06%, dV exact).

    make test    # causal dQ/dK/dV vs fp64 analytic
