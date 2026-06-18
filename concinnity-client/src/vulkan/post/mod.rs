// src/vulkan/post/mod.rs
//
// Screen-space and post-process passes for the Vulkan frame encoder. Each
// effect lives in its own file with its pipeline builder(s), target
// allocator(s), and per-frame encoder(s) co-located; mirrors the Metal
// `metal/post/` shape:
//
//   bloom.rs    prefilter + downsample/upsample mip chain
//   ssao.rs     GTAO depth+normal pre-pass + horizon-search kernel + blur
//   ssgi.rs     hemisphere gather + depth-aware blur over the SSR G-buffer
//   ssr.rs      depth+normal+roughness pre-pass + fullscreen ray-march resolve
//   taa.rs      velocity (motion-vector) pre-pass + TAA resolve
//   upscale/    temporal upscaling (FSR / DLSS / XeSS) behind VkUpscaleBackend

pub(in crate::vulkan) mod bloom;
pub(in crate::vulkan) mod fullscreen;
pub(in crate::vulkan) mod gbuffer;
pub(in crate::vulkan) mod rt_reflections;
pub(in crate::vulkan) mod ssao;
pub(in crate::vulkan) mod ssgi;
pub(in crate::vulkan) mod ssr;
pub(in crate::vulkan) mod taa;
pub(in crate::vulkan) mod upscale;

pub(in crate::vulkan) use gbuffer::GbufferResources;
pub(in crate::vulkan) use rt_reflections::RtReflectionsResources;
pub(in crate::vulkan) use ssao::SsaoResources;
pub(in crate::vulkan) use ssgi::SsgiResources;
pub(in crate::vulkan) use ssr::SsrResources;
pub(in crate::vulkan) use taa::TaaResources;
pub(in crate::vulkan) use upscale::{
    ResolvedBackend, UpscaleSdk, VkUpscaleBackend, build_upscaler,
};
