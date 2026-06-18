// src/directx/post/
//
// Post-process effects for the D3D12 backend, each owning pipeline + targets
// + per-frame encoder co-located in one file:
//
//   bloom.rs    prefilter + downsample + additive upsample
//   gbuffer.rs  unified normal+depth / roughness / velocity MRT pre-pass
//   taa.rs    velocity pre-pass + history-resolve
//   ssao.rs   GTAO depth+normal pre-pass + horizon kernel + depth-aware blur
//   ssr.rs    depth+normal+roughness pre-pass + fullscreen ray-march resolve
//   ssgi.rs   hemisphere-gather + depth-aware blur over the SSR pre-pass G-buffer
//
// Mirrors src/metal/post/ (same per-effect file shape).

pub(in crate::directx) mod bloom;
pub(in crate::directx) mod fullscreen;
pub(in crate::directx) mod gbuffer;
pub(in crate::directx) mod rt_reflections;
pub(in crate::directx) mod ssao;
pub(in crate::directx) mod ssgi;
pub(in crate::directx) mod ssr;
pub(in crate::directx) mod taa;
pub(in crate::directx) mod upscale;
