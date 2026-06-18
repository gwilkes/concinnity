# DirectX engine-internal pass shaders

HLSL sources for the D3D12 backend's fixed pipeline passes — the bindless and
skinned scene passes, the text overlay, composite, bloom, the TAA velocity
pre-pass + resolve, and the GPU-cull compute kernel.

- `include_str!`'d into `src/directx/pipeline.rs` and compiled from source to
  DXBC at **runtime** (`D3DCompile`).
- Engine code, not assets: not packed into the asset blob, not swappable
  per-world.

The user-facing builtin shader *assets* (`default`, `outdoor`, `shadow_map`)
live in `src/build/shaders/` and are compiled at world-build time. The runtime
fallback used when a world ships no precompiled DXBC for the main / instanced /
shadow pass `include_str!`s those same `build/shaders/*.hlsl` files directly, so
the fallback and the shipped asset can never drift apart.
