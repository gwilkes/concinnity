// src/gfx/mod.rs
//
// Backend-agnostic GPU data layouts and CPU-side render-prep math shared by the
// client renderer and the build pipeline: vertex / mesh payload formats, LOD
// decimation, skinning, camera math, frustum, the chunk-streaming coordinate /
// allocator helpers, and the post-process setting structs. No backend handles
// and no rendering logic: the render graph, draw lists, and per-backend
// executors stay in the client crate's own `gfx` module.
pub mod auto_exposure;
pub mod camera;
pub mod chunk_coord;
pub mod frustum;
pub mod lod;
pub mod mesh_payload;
pub mod mesh_seed;
pub mod overlay;
pub mod profile;
pub mod range_alloc;
pub mod render_types;
pub mod rt_reflections;
pub mod scroll_layout;
pub mod skinning;
pub mod ssao;
pub mod ssgi;
pub mod ssr;
