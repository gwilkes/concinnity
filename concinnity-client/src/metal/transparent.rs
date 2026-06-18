// src/metal/transparent.rs
//
// The engine's `PassId::Transparent` slot: one shared pass that draws every
// translucent surface in the world (water, glass, and future ice / holograms /
// force fields). It runs after `SsrResolve` (so translucents see the resolved
// opaque scene + SSR reflections) and before `TaaResolve` / `Upscale` (so they
// pick up temporal accumulation). Output blends over `scene_pre_taa` with
// SRC_ALPHA / ONE_MINUS_SRC_ALPHA.
//
// The pass owns no pipeline of its own. Each translucent subsystem contributes
// a list of [`TransparentDraw`]s (a bound pipeline + buffers + per-draw
// uniforms + texture bindings + a camera distance); `encode_transparent`
// aggregates them, sorts back-to-front, and issues them into a single render
// encoder. This is a fixed sorted draw list, not order-independent
// transparency.
//
// Refraction read-back: at the head of the pass a blit snapshots the current
// `scene_pre_taa` into `hdr_targets.transparent_scene_copy`, which the draws
// sample. This makes refraction work whether or not SSR produced a distinct
// `scene_pre_taa` (with SSR off it aliases `hdr_resolve`, so sampling the
// destination directly would be reading the attachment being written).

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder as _, MTLBuffer, MTLCommandBuffer as _, MTLCommandEncoder as _,
    MTLIndexType, MTLLoadAction, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineState, MTLSamplerState, MTLStoreAction, MTLTexture,
};

use super::context::MtlContext;
use super::scoped_encoder::ScopedEncoder;
use super::uniforms::TransparentView;

// One translucent draw recorded for the transparent pass. Self-contained
// except for the shared [`TransparentView`], which `encode_transparent` binds
// once at buffer(5). The vertex buffer binds at buffer(1) and the per-draw
// `params` blob at buffer(6) (both stages), matching the transparent shaders'
// argument layout.
pub(in crate::metal) struct TransparentDraw {
    pub(in crate::metal) pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub(in crate::metal) vertex_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(in crate::metal) index_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(in crate::metal) index_count: u32,
    // Per-draw uniform blob, bound at vertex + fragment buffer(6). Built from
    // a `#[repr(C)]` params struct via [`bytes_of`].
    pub(in crate::metal) params: Vec<u8>,
    // Fragment textures: `(slot, texture)`. Bound before the draw.
    pub(in crate::metal) fragment_textures: Vec<(usize, Retained<ProtocolObject<dyn MTLTexture>>)>,
    // Fragment samplers: `(slot, sampler)`.
    pub(in crate::metal) fragment_samplers:
        Vec<(usize, Retained<ProtocolObject<dyn MTLSamplerState>>)>,
    // World-space distance from camera to the draw's centre, used for the
    // back-to-front sort. Larger = farther = drawn first.
    pub(in crate::metal) sort_distance: f32,
}

// Copy a `#[repr(C)] Copy` uniform struct into an owned byte buffer for a
// [`TransparentDraw::params`] blob. The bytes are consumed immediately by
// `setVertexBytes` / `setFragmentBytes` (which copy into the command buffer),
// so the buffer's 1-byte alignment is irrelevant to Metal.
pub(in crate::metal) fn bytes_of<T: Copy>(value: &T) -> Vec<u8> {
    let ptr = value as *const T as *const u8;
    // Safety: `T: Copy` is plain old data; we read exactly its size in bytes.
    unsafe { std::slice::from_raw_parts(ptr, std::mem::size_of::<T>()) }.to_vec()
}

impl MtlContext {
    // Encode the transparent pass: snapshot the scene for refraction, then
    // draw every contributed translucent surface back-to-front into
    // `scene_pre_taa`. Returns the number of draws issued (0 short-circuits
    // before allocating the encoder).
    pub(in crate::metal) fn encode_transparent(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        view: &TransparentView,
        scene_pre_taa: &Retained<ProtocolObject<dyn objc2_metal::MTLTexture>>,
        draws: &[TransparentDraw],
    ) -> Result<u32, String> {
        if draws.is_empty() {
            return Ok(0);
        }

        // Snapshot the pre-transparent scene so refraction taps read a stable
        // copy instead of the attachment being written.
        let blit = cmd_buf
            .blitCommandEncoder()
            .ok_or("failed to get transparent scene-copy blit encoder")?;
        blit.pushDebugGroup(&NSString::from_str("transparent_scene_copy"));
        unsafe {
            blit.copyFromTexture_toTexture(
                scene_pre_taa.as_ref(),
                self.hdr_targets.transparent_scene_copy.as_ref(),
            );
        }
        blit.popDebugGroup();
        blit.endEncoding();

        let pass_desc = MTLRenderPassDescriptor::new();
        unsafe {
            let ca = pass_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(scene_pre_taa.as_ref()));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = &self.pass_timing {
            t.attach_render(&pass_desc, super::pass_timing::PassId::Transparent);
        }

        // The blit above is ended explicitly (it must close before this render
        // encoder opens). This render pass spans to the end of the function and
        // has a `?` mid-encode (the per-draw params blob below), so the guard
        // ensures it can't leak an open encoder on an early return.
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or("failed to get transparent render encoder")?,
            "transparent",
        );

        // Shared per-frame view at buffer(5) for both stages. The pass has no
        // depth attachment (translucents are not hardware depth-tested;
        // depth-aware effects sample `depth_resolve` instead), so no
        // depth-stencil state is bound.
        unsafe {
            enc.setVertexBytes_length_atIndex(
                std::ptr::NonNull::from(view).cast(),
                std::mem::size_of::<TransparentView>(),
                5,
            );
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(view).cast(),
                std::mem::size_of::<TransparentView>(),
                5,
            );
        }

        let distances: Vec<f32> = draws.iter().map(|d| d.sort_distance).collect();
        let order = crate::gfx::transparent::back_to_front_order(&distances);

        for &i in &order {
            let d = &draws[i];
            enc.setRenderPipelineState(&d.pipeline);
            unsafe {
                enc.setVertexBuffer_offset_atIndex(Some(&d.vertex_buffer), 0, 1);
                let params_ptr = std::ptr::NonNull::new(d.params.as_ptr() as *mut std::ffi::c_void)
                    .ok_or("transparent draw params blob is null")?;
                enc.setVertexBytes_length_atIndex(params_ptr, d.params.len(), 6);
                enc.setFragmentBytes_length_atIndex(params_ptr, d.params.len(), 6);
                for (slot, tex) in &d.fragment_textures {
                    enc.setFragmentTexture_atIndex(Some(tex.as_ref()), *slot);
                }
                for (slot, samp) in &d.fragment_samplers {
                    enc.setFragmentSamplerState_atIndex(Some(samp.as_ref()), *slot);
                }
                enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                    MTLPrimitiveType::Triangle,
                    d.index_count as usize,
                    MTLIndexType::UInt16,
                    &d.index_buffer,
                    0,
                );
            }
        }

        Ok(order.len() as u32)
    }
}
