# Phase 12 Convolution And Audio CDNA3 Variant

This standalone HIP harness lands the Phase 12 parity rows:

- `im2col_2d`, `im2col_3d`, `col2im_1d`, `col2im_2d`
- `conv2d`, `conv3d`, `depthwise_conv2d`
- `conv_transpose_1d`, `conv_transpose_2d`
- `pool1d`, `pool2d`, `pool2d_backward`
- `audio_conv1d_direct`, `audio_depthwise_conv1d`,
  `audio_causal_depthwise_conv1d`, `audio_relative_attention`

The contracts follow CPU `vision/conv_pool_ref.cpp`,
`audio/conv1d_ref.cpp`, and `audio/relative_attention_ref.cpp`.

The CDNA3 approach is correctness-first direct mapping. Forward conv/pool/audio
routes own one output element or row. Scatter-style contracts (`col2im`,
transposed convolution, and pool backward) are implemented as destination-owned
scans so they need no unordered atomics. `audio_relative_attention` is a
row-owned serial softmax over the chunk context, preserving valid-length,
relative-position, per-dimension query-scale, key-scale, and softcap semantics.
