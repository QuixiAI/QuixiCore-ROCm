This directory is a preserved CDNA3 port attempt copied from the CDNA4 MXFP8
variants.

It is not an active `variants/rocm_cdna3` implementation because the copied
kernels depend on CDNA4 scaled-MFMA MXFP8 paths such as
`mma_ABt_base_scaled` / `mfma_scale_f32_16x16x128_f8f6f4`. gfx942 does not have
the same native MXFP8 scaled-MFMA path exposed by the CDNA4 kernels.

A real CDNA3 port needs an emulated scale path or a CDNA3-specific MXFP8
algorithm rather than the CDNA4 scaled-MFMA implementation.
