# Phase 13 Vision CDNA3 Variant

This standalone HIP harness lands the Phase 13 vision parity rows:

- `extract_patches_2d`, `extract_patches_3d`
- `vision_patch_projection`, `vision_patch_projection_3d`
- `patch_merge_layer_norm`, `space_to_depth_norm_linear`
- `edge_mlp_256x7`
- `vision_rope_2d`, `qwen_vision_rope_2d`
- `interpolate_position_2d`, `factorized_position_2d`
- `add_relative_position_2d`, `get_relative_position`
- `window_partition`, `window_unpartition`
- `avg_pool2d_tokens`, `pool_tokens_by_position`
- `timestep_embedding`, `upscale_nearest_2d`

The contracts follow CPU `vision/patch_ops_ref.cpp`,
`vision/vision_ref.cpp`, `vision/conv_pool_ref.cpp`,
`attention/vision_rope_ref.cpp`, and `utils/tensor_ops_ref.cpp`. Metal
coverage exists for the patch-merge and edge-MLP families.

The CDNA3 approach is correctness-first direct mapping over the CPU layouts:
NHWC/NTHWC patch extraction and projection, destination-owned padding/window
transforms, row-owned normalization, one output-owned scan for
`pool_tokens_by_position` to preserve CPU accumulation order, and a two-stage
`edge_mlp_256x7` route matching the CPU left/right intermediate contract.

Explicit non-ports: this variant does not introduce a production MFMA patch
projection, storage-type dispatch wrappers, or fused vision-transformer blocks.
Those are follow-up optimization work after operation-level CDNA3 parity.
