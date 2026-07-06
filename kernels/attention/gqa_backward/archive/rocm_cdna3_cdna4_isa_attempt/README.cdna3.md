This directory is a preserved CDNA3 port attempt copied from the CDNA4 GQA
backward variant.

It is not an active `variants/rocm_cdna3` implementation because the copied
kernel uses CDNA4 assembly-register paths and instructions that do not assemble
for gfx942, including `v_mfma_f32_16x16x32_bf16` and `ds_read_b64_tr_b16`.

A real CDNA3 port needs a backward kernel written around the supported CDNA3
MFMA and LDS load primitives instead of the CDNA4 art assembly path.
