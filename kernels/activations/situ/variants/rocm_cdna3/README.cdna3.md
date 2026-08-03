# SITU (SituGLU) — CDNA3

The gated activation used by Kimi K3, in both its dense MLP and its 896-expert
MoE.

```text
gate_out = beta * tanh(gate / beta) * sigmoid(gate)
up_out   = linear_beta > 0 ? linear_beta * tanh(up / linear_beta) : up
out      = gate_out * up_out
```

Gate and up occupy **separate halves** of the input (`[..., 2d]` → `[..., d]`),
not alternating lanes. `linear_beta <= 0` is the "unset" sentinel and passes the
up half through unchanged. Kimi K3 ships `beta = 4.0`, `linear_beta = 25.0`.

Two entry points:

- `situ_and_mul` — dense, one block per row.
- `masked_situ_and_mul` — MoE, `[E, T, 2d]` plus a per-expert token count.
  Experts with zero tokens exit immediately and rows past an expert's count are
  never written, so padding costs nothing and cannot leak garbage downstream.

## Correctness

`make test` — fp64 oracle on the host, over the same input bytes the GPU reads.

```text
dense fp32   beta=1 lb=-1    worst_rel 2.27e-07   ok
dense fp32   beta=4 lb=25    worst_rel 2.92e-07   ok
dense bf16   beta=4 lb=25    worst_rel 3.87e-03   ok
dense fp16   beta=4 lb=25    worst_rel 4.85e-04   ok
masked fp32                  worst_rel 3.44e-07   dead-rows-written 0   ok
masked bf16                  worst_rel 3.85e-03   dead-rows-written 0   ok
```

Errors land at roughly one ulp of the output dtype (2⁻⁸ for bf16, 2⁻¹¹ for
fp16), which is the single rounding on store — compute is fp32 throughout.
Tolerances are relative on purpose: at `beta=4 / linear_beta=25` the output
reaches ~16, where one bf16 ulp is already ~3e-2 absolutely, so an absolute
bound would either wave through a broken kernel at small beta or fail a correct
one at large beta.

## Performance

`make bench` — MI300X (gfx942), bf16, peak 5.3 TB/s. Memory bound by
construction: 2d in, d out, a few transcendentals between.

```text
rows=1      d=33792       48.6 us      4.2 GB/s    0.1% of peak
rows=512    d=33792       69.5 us   1494.5 GB/s   28.2% of peak
rows=4096   d=3072        26.4 us   2857.2 GB/s   53.9% of peak
rows=16384  d=3072       113.1 us   2670.5 GB/s   50.4% of peak
```

The MoE widths — the ones that matter, since the routed experts dominate the
call count — reach about half of peak bandwidth. Two known gaps, neither
addressed here:

- **Single row is latency bound, not bandwidth bound.** One block occupies one
  CU of 304, so the 0.1% figure is occupancy, not a kernel defect. It is the
  decode-time shape, though, so it is worth revisiting if SITU ever shows up in
  a decode profile.
- **Scalar loads.** The kernel reads element-wise. Vectorising to `float4`-width
  accesses is the obvious next step for the ~50% cases and is where the
  remaining headroom is.
