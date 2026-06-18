// src/lib.rs
//
// The editor crate: the asset build/edit pipeline driven on top of the runtime
// crate. Holds cn build / cn add / cn rm / cn check / cn debug, the in-tab
// preview, the full C-ABI surface the Swift app links, and the dev CLI. Depends
// on both concinnity-client (the runtime) and concinnity-cook (the compiler);
// the runtime crate itself links neither, so a shipped runtime sheds the build
// dependencies.

// Bridge: the runtime crate (concinnity-client) owns the renderer, ECS, asset
// types, audio, physics, and the world loop. Re-export its modules under crate::*
// so the editor code moved out of the runtime crate keeps its historical
// `crate::<module>` import paths. (Some are only used by the binary-side cli /
// debug modules, hence the blanket allow.)
#[cfg(backend_dx)]
#[allow(unused_imports)]
pub(crate) use concinnity_client::directx;
#[cfg(backend_metal)]
#[allow(unused_imports)]
pub(crate) use concinnity_client::metal;
#[cfg(backend_vk)]
#[allow(unused_imports)]
pub(crate) use concinnity_client::vulkan;
#[allow(unused_imports)]
pub(crate) use concinnity_client::{assets, blob, config, ecs, gfx, jobs};
// Decode/world/geometry helpers + the crate-wide result type come from core.
#[allow(unused_imports)]
pub(crate) use concinnity_core::{build, geometry, result, world};

// Editor-owned modules (moved out of the runtime crate).
pub(crate) mod app;
pub(crate) mod ffi;
#[cfg(backend_metal)]
mod shader_reflect;

// Build API (the published FFI/Swift build + validate surface).
pub use concinnity_cook::{build_pipeline_from_str, validate_asset, validate_world_jsonl};
pub use concinnity_core::world::{parse_world_jsonl, write_world_jsonl};
