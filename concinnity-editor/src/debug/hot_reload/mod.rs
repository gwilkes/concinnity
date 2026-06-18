// src/debug/hot_reload/mod.rs
//
// Asset / shader / world.jsonl hot-reload machinery (`cn debug` only). Moved
// out of the library into the binary tree: the watcher, off-thread decode, and
// the reload passes run only under `cn debug`, driven once per frame from
// `DebugHook::tick` (see `crate::debug::DebugServer::drive_hot_reload`). The
// passive source catalogues these consume are captured at
// `GraphicsSystem::init` and live in the library
// (`crate::gfx::graphics_system::hot_reload_sources`); the per-frame backend
// + Prop-tracking handle comes from `GraphicsSystem::hot_reload_apply_parts`.
//
// Split by responsibility:
//   state    `AssetHotReloadState` + decode result types + `run_frame` entry
//   watcher  the `notify` filesystem watcher
//   decode   off-thread payload decode + poll/apply (textures, meshes, IBL)
//   passes   world.jsonl / ProceduralMesh / VolumetricFog / ShaderStage reload
//   pending  process-wide world.jsonl / ShaderStage "changed" flags

mod decode;
mod passes;
mod pending;
mod state;
mod watcher;

#[cfg(test)]
mod tests;

pub(crate) use pending::{set_pending_shader_stages, set_pending_world};
pub(crate) use state::{AssetHotReloadState, run_frame};
