// src/lib.rs
//
// The runtime crate. Holds the world loop, the ECS, the three renderer
// backends, audio, physics, and the lean runtime FFI. Depends on
// concinnity-core alone (no concinnity-cook, no image decoders). The editor
// crate (concinnity-editor) drives this crate's App / renderer through the
// public API widened here; the modules the editor reaches into are `pub` so it
// can name their paths, but individual internals stay `pub(crate)` unless the
// editor specifically needs them.
pub mod assets;
pub mod blob;
#[cfg(backend_dx)]
pub mod directx;
pub mod ecs;
#[cfg(backend_metal)]
pub mod metal;
#[cfg(backend_vk)]
pub mod vulkan;

// Renderer-free foundation shared with the build/validate pipeline lives in
// concinnity-core. Re-export its modules under the historical crate::* paths so
// the rest of the client keeps resolving (crate::result / crate::gfx are the
// pre-existing slices; build / check / geometry / world join them here).
pub(crate) use concinnity_core::result;
pub(crate) use concinnity_core::{build, geometry, world};

pub mod app;
pub(crate) mod audio;
pub mod config;
pub mod gfx;
pub(crate) mod hud;
pub mod jobs;
pub(crate) mod physics;
pub(crate) mod ui;

// Asset API
pub use assets::*;
