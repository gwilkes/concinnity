// Vulkan rendering backend. Gated by #[cfg(backend_vk)] on the mod declaration
// in lib.rs; compiled on Linux always and on Windows with the `vulkan` feature.

mod auto_exposure;
mod backend;
mod barrier_translate;
mod composite;
mod context;
mod cull;
mod decal;
mod descriptor_layout;
mod device;
mod draw;
mod fog;
mod glass;
mod graph_exec;
mod hiz;
mod hot_reload;
mod init;
mod input;
mod main;
mod math;
mod parallel_encoder;
mod particle;
mod pass_timing;
mod pipeline;
mod post;
mod quality;
mod raymarch;
mod raytrace;
mod render_pass;
mod resources;
mod screenshot;
mod shadow;
mod swapchain;
mod texture;
mod transient_pool;
pub(crate) mod window;

pub use context::VkContext;
