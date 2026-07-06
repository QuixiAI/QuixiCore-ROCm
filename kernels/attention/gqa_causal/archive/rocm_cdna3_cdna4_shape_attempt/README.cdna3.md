This directory is a preserved CDNA3 port attempt copied from the CDNA4 causal
GQA forward variant.

It is not an active `variants/rocm_cdna3` implementation. It shares the same
CDNA4 register tile geometry assumptions as the non-causal GQA forward copy, so
it should not be advertised as a CDNA3 kernel until rewritten around CDNA3 tile
geometry and validated against a reference.

A real CDNA3 port needs a CDNA3-specific causal GQA kernel instead of this CDNA4
shape copy.
