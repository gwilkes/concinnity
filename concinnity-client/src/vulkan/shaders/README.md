# Vulkan engine-internal pass shaders

GLSL sources for the Vulkan backend's fixed pipeline passes — the main /
bindless / instanced / shadow / skinned scene passes, the text overlay,
composite, bloom, the TAA velocity pre-pass + resolve, the GPU-cull compute
kernel, and the Hi-Z occlusion build kernels (`hiz_init.comp` reduces the main
depth into mip 0, `hiz_downsample.comp` builds the rest of the depth pyramid the
cull kernel tests AABBs against).

- `include_str!`'d into `src/vulkan/pipeline.rs` and compiled from source to
  SPIR-V at **runtime** (`shaderc`).
- Engine code, not assets: not packed into the asset blob, not swappable
  per-world. `main.vert` / `main.frag` / `shadow.vert` are the built-in
  fallbacks used when a world ships no precompiled SPIR-V.
- `main_bindless.frag` carries a `{POOL_SIZE}` token substituted at compile
  time with the bindless texture-pool length.

The user-facing builtin shader *assets* (`default`, `outdoor`, `shadow_map`)
live in `src/build/shaders/` instead and are compiled at world-build time.
