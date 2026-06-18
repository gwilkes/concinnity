// src/app/mod.rs
// Client application runtime: the world loop and the runtime systems that drive
// a compiled world. The build / edit / debug / preview paths live in the
// editor crate.
// `pub` so the editor crate (which drives a live App via the runtime API) can
// reach these runtime app items through `concinnity_client::app::*`.
pub mod anim_runtime;

pub mod dev_flags;
pub mod run;
pub mod state;

// Async asset-streaming drivers. Currently driven only by the Metal
// backend (Vulkan and DirectX catch-up is a separate follow-up), so on
// non-macOS builds these modules are compiled but unreferenced.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) mod chunk_stream;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) mod mesh_stream;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) mod texture_stream;

// Run the app from compiled binary data: the `cn run` production path. `pub`
// so the editor crate's CLI can dispatch to it.
pub use run::run;
