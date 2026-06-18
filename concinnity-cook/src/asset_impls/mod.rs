// src/asset_impls/mod.rs
//
// The `BuildAsset` trait implementations for each compiled asset type. The asset
// data types, their `Component` and `SourceBacked` impls, and their runtime
// helpers stay in concinnity-core; only the build-time `compile_payload` /
// `source_files` impls live here, calling the compile pipeline in this crate.
// These are trait impls only, so the modules need no re-exports.

mod audio_clip;
mod color_lut;
mod cubemap_texture;
mod environment_map;
mod file;
mod font;
mod mesh;
mod procedural_mesh;
mod room;
mod sdf_volume;
mod shader_stage;
mod skinned_mesh;
mod texture;
mod voxel_chunk;
