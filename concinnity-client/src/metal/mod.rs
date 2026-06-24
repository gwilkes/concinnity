// Metal rendering backend. Gated by #[cfg(backend_metal)] on the mod
// declaration in lib.rs; compiled on macOS only.

mod auto_exposure;
mod backend;
mod context;
mod cull;
mod decal;
mod draw;
mod fog;
mod frame_pacing;
mod glass;
mod graph_exec;
mod hiz;
mod hot_reload;
mod init;
mod input;
mod instanced;
mod math;
mod parallel_encoder;
mod particle;
mod pass_timing;
mod pipeline;
mod planar;
mod post;
mod probe;
mod quality;
mod raymarch;
mod raytrace;
mod resources;
mod scoped_encoder;
mod screenshot;
// The Metal-free engine-binding contract + comparison. `pub` so the editor
// crate's shader-reflection adapter (moved out of the runtime crate) can drive
// `validate_stage` against the engine layouts.
pub mod shader_layout;
mod streaming;
mod texture;
mod transient;
mod transient_pool;
mod transparent;
mod uniforms;
mod water;
mod window_delegate;

pub use context::{MtlContext, set_embedded_pump_events, set_preview_view};
