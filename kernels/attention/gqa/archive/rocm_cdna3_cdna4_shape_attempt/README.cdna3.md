This directory is a preserved CDNA3 port attempt copied from the CDNA4 GQA
forward variant.

It is not an active `variants/rocm_cdna3` implementation. The copied kernel can
be made to compile with CDNA3 compatibility shape tags, but it fails correctness
against a PyTorch SDPA reference. The kernel relies on CDNA4 register tile
geometry such as `rt_32x16`, `rt_16x32`, and `rt_32x32`; on CDNA3 those tags are
only source-compatibility metadata unless a kernel is rewritten around CDNA3
tile geometry.

A real CDNA3 port needs a CDNA3-specific GQA kernel instead of this CDNA4 shape
copy.
