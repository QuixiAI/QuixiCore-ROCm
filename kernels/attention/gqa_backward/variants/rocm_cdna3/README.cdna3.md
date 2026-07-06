# GQA attention backward — CDNA3 (gfx942): correctness-valid

Native CDNA3 attention backward (dQ, dK, dV), the counterpart to the forward at
`../../../gqa/variants/rocm_cdna3`. Same correctness-first wavefront/online style
(one 64-wide wavefront per row, lanes split the head dim). Two kernels: `bwd_dq`
(one wavefront per query) and `bwd_dkv` (one wavefront per key, looping the GQA
query-head group + query rows). Uses O and L from the forward; recomputes
S/P/dP/D_i online. Replaces the archived CDNA4-ISA copy.

Validated by a standalone fp64 analytic oracle (`attn_bwd.cu`): dQ/dK/dV PASS at
D=128 GQA and D=64 MHA, non-causal and causal (`make CAUSAL=1 test`) — mean rel
<=0.06%, dV exact.

    make test              # non-causal dQ/dK/dV vs fp64 analytic
    make CAUSAL=1 test     # causal
