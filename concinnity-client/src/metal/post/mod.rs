// src/metal/post/mod.rs
//
// Screen-space and post-process passes for the Metal frame encoder. Each
// effect lives in its own file with its pipeline builder(s), target
// allocator(s), and per-frame encoder(s) co-located so it forms a single
// unit Vulkan / DirectX can mirror:
//
//   gbuffer.rs unified normal+depth / roughness / velocity G-buffer pre-pass
//   ssao.rs   GTAO depth+normal pre-pass + horizon-search kernel + blur
//   ssr.rs    SSR depth+normal+roughness pre-pass + ray-march resolve
//   ssgi.rs   SSGI hemisphere gather + depth-aware blur composite
//   taa.rs    velocity (motion-vector) pre-pass + TAA resolve
//   bloom.rs  prefilter + downsample/upsample mip chain
//
// Pipeline builders / targets that any other module reaches are re-exported
// here so call sites have a single `crate::metal::post::*` import.

pub(super) mod bloom;
pub(super) mod fullscreen;
pub(super) mod gbuffer;
pub(super) mod rt_reflections;
pub(super) mod ssao;
pub(super) mod ssgi;
pub(super) mod ssr;
pub(super) mod taa;
pub(super) mod upscale;

pub(super) use bloom::{BloomPipelines, BloomTargets, build_bloom_pipelines, create_bloom_targets};
pub(super) use gbuffer::{
    GBufferState, build_gbuffer_bindless_pipeline, build_gbuffer_prepass_pipeline,
    create_gbuffer_targets,
};
pub(super) use rt_reflections::build_rt_reflection_pipeline;
pub(super) use ssao::{SsaoState, build_ssao_pipeline, create_ssao_targets};
pub(super) use ssgi::{
    SsgiState, build_ssgi_composite_pipeline, build_ssgi_gather_pipeline, create_ssgi_targets,
};
pub(super) use ssr::{SsrState, build_ssr_pipeline, create_ssr_targets};
pub(super) use taa::{TaaState, build_taa_pipeline, create_taa_targets};
pub(super) use upscale::{MetalFXUpscaler, UpscaleState, temporal_scaler_supported};
