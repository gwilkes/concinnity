# Metal engine-internal pass shaders

MSL sources for the Metal backend's fixed pipeline passes — post-process, TAA,
velocity pre-pass, SSAO, SSR, bloom, GPU-cull kernel, the shadow-map depth pass
(`shadow_map.metal`, static + skinned), and the text overlay.

- `include_str!`'d into `src/metal/pipeline.rs` and compiled from source at
  **runtime** (`newLibraryWithSource`).
- Engine code, not assets: not packed into the asset blob, not swappable
  per-world.

The user-facing builtin shader *assets* (`default`) live in `src/build/shaders/`
instead and are compiled at world-build time.
