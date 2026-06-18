// src/metal/resources/mod.rs
//
// Runtime GPU resource management for `MtlContext`. The methods are split
// across sibling files by resource family:
//
//   textures.rs   albedo + normal-map pool slot updates, IBL envmap +
//                 colour-grading LUT hot-swap
//   geometry.rs   `rebuild_static_geometry` -- hot-reload rebuild of the
//                 shared static vertex + index buffers
//   streaming.rs  per-mesh upload / eviction via the sub-allocators, plus
//                 in-place per-slot `update_mesh_geometry` for hot-reload
//   skinning.rs   skinned pipelines + buffer setup, per-frame pose updates,
//                 and skinned hot-reload paths
//
// Each file is a single `impl MtlContext { pub fn ... }` block; nothing is
// re-exported here -- callers reach the methods directly through `MtlContext`.

mod geometry;
pub(crate) mod skinning;
mod streaming;
mod textures;
