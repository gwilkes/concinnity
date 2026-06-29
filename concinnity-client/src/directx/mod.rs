// D3D12 rendering backend. Gated by #[cfg(backend_dx)] on the mod declaration
// in lib.rs; compiled on Windows unless the `vulkan` feature is enabled.

mod auto_exposure;
mod backend;
mod barrier_translate;
mod context;
mod cull;
mod decal;
mod draw;
mod draw_iter;
mod dxc;
mod fog;
mod geometry_rebuild;
mod glass;
mod gpu_profile;
mod graph_exec;
mod hiz;
mod hot_reload;
mod init;
mod input;
mod math;
mod parallel_encoder;
mod particle;
mod pass_timing;
mod pipeline;
mod planar;
mod post;
mod probe;
mod probe_uniforms;
mod quality;
mod raymarch;
mod raytrace;
mod resize;
mod resources;
mod screenshot;
mod texture;
mod transient_pool;
mod window;

pub use context::DxContext;
pub(crate) use gpu_profile::probe_gpu_profile;
