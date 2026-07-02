// src/lib.rs
//
// concinnity-cook: the asset compile pipeline, extracted from concinnity-core
// so the runtime foundation no longer carries the build-only dependencies
// (fbxcel, fontdue, shaderc, sha2, kira). This crate turns world.jsonl + source
// files into the binary blobs the runtime reads; it depends on concinnity-core
// and core has no edge back into it.
//
// The staying-in-core modules are re-exported so code moved here keeps resolving
// its `crate::{assets,ecs,gfx,geometry,result}` paths. The payload *decoders*
// and shared payload types live in `concinnity_core::build`; this crate's
// modules call back into them.
pub use concinnity_core::{assets, ecs, geometry, gfx, paths, result};

pub mod asset;
pub mod asset_impls;
pub mod audio_clip;
pub mod blob;
pub mod cache;
pub mod check;
pub mod color_lut;
pub mod cubemap;
pub mod environment_map;
pub mod fbx;
pub mod file;
pub mod font;
pub mod glb;
pub mod gltf;
pub mod import;
pub mod mesh_compile;
pub mod mesh_reimport;
pub mod pipeline;
pub mod shader;
pub mod texture;
pub mod wavefront;
pub mod world;

// Public build API: the entry points the CLI, the editor FFI, and the infra
// server call. The runtime-side decode + world parse API stays in
// concinnity-core.
pub use pipeline::{
    PipelineResult, build_compiled, build_from_path, build_pipeline_from_str, validate_asset,
    validate_world_jsonl, write_build_outputs,
};
pub use world::prepare_world;
