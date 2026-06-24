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
    // Index element width. `UInt16` for the per-asset glass/water buffers; `UInt32`
    // for a transparent mesh drawing from the shared scene index buffer.
    pub(in crate::metal) index_type: MTLIndexType,
    // Byte offset of the first index into `index_buffer`. 0 for the per-asset
    // glass/water buffers; `DrawObject.index_offset * index_stride` for a mesh
    // sharing the scene index buffer.
    pub(in crate::metal) index_offset_bytes: usize,
    // Value added to every fetched index (`baseVertex`). 0 for world-space
    // glass/water and static meshes; non-zero for mesh-relative chunk indices.
    pub(in crate::metal) base_vertex: i32,
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
        rt_params: Option<&crate::gfx::render_types::RtParams>,
        bindless_tex_args: Option<&Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
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

        // Reflection sources shared by every transparent shader that samples
        // them (glass + water): the sky prefilter cube at texture(2), the local
        // reflection-probe cubes at texture(3..3+MAX_PROBES), the cube sampler
        // at sampler(1), and the probe set (parallax boxes + count) at fragment
        // buffer(7). Frame-constant, so bound once before the draw loop; a probe
        // count of 0 keeps the sky-only fallback. `probe_cube_or_sky` returns the
        // sky for unbaked slots, so binding all MAX_PROBES is always valid. The
        // per-draw bindings below never touch these slots, so the state persists.
        unsafe {
            enc.setFragmentTexture_atIndex(Some(self.env_map.prefilter.as_ref()), 2);
            for i in 0..super::uniforms::MAX_PROBES {
                enc.setFragmentTexture_atIndex(Some(self.probe_cube_or_sky(i)), 3 + i);
            }
            enc.setFragmentSamplerState_atIndex(Some(self.cube_sampler.as_ref()), 1);
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&self.probe_set).cast(),
                std::mem::size_of::<super::uniforms::ProbeSet>(),
                7,
            );
            // A planar reflection resolve at texture(11), the default for every
            // transparent draw so the slot is always bound (validation-safe) even
            // for slotless / probe-path draws. water.metal + glass.metal sample it
            // when their `planar.x` flag is set; a planar draw overrides this with
            // ITS plane's resolve per-draw (see the collect paths). The first
            // slot's resolve is a valid stand-in for draws that do not sample it.
            if let Some(planar) = self.planar_reflection.as_ref()
                && let Some(first) = planar.targets.first()
            {
                enc.setFragmentTexture_atIndex(Some(first.resolve.as_ref()), 11);
            }
        }

        // Ray-traced glass + water inputs. When the acceleration structure is
        // live and an RT transparent pipeline exists,
        // `collect_glass_transparent_draws` / `collect_water_transparent_draws`
        // select `glass_fragment_rt`(`_textured`) / `water_fragment_rt`(`_textured`),
        // which trace a reflection ray. These share one argument layout, so bind
        // the inputs once here (the non-RT glass / water pipelines ignore these
        // otherwise-free fragment slots): RT params @0, the shared scene geometry
        // @1..3, the TLAS @4, the skinned deformed-vertex / u16 index buffers
        // @8..9, and -- in a bindless world -- the bindless texture pool @10 for
        // the textured variants (the main pass's pool index 7 is the ProbeSet
        // here). The TLAS references each BLAS indirectly, so the BLASes are not
        // auto-tracked -- declare them resident or the trace reads garbage.
        if let (Some(accel), Some(rt_params)) = (
            self.rt.accel.as_ref().filter(|_| {
                self.glass_pipeline_rt.is_some()
                    || self.water_pipeline_rt.is_some()
                    || self.glass_mesh_pipeline_rt.is_some()
            }),
            rt_params,
        ) {
            unsafe {
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(rt_params).cast(),
                    std::mem::size_of::<crate::gfx::render_types::RtParams>(),
                    0,
                );
                enc.setFragmentBuffer_offset_atIndex(Some(self.vertex_buffer.as_ref()), 0, 1);
                enc.setFragmentBuffer_offset_atIndex(Some(self.index_buffer.as_ref()), 0, 2);
                enc.setFragmentBuffer_offset_atIndex(Some(accel.geom_table.as_ref()), 0, 3);
                enc.setFragmentAccelerationStructure_atBufferIndex(Some(accel.tlas.as_ref()), 4);
                enc.setFragmentBuffer_offset_atIndex(Some(accel.deformed_verts.as_ref()), 0, 8);
                enc.setFragmentBuffer_offset_atIndex(Some(accel.skinned_indices.as_ref()), 0, 9);
                super::raytrace::use_blas_resident_fragment(&enc, &accel.blas);
                // Textured variants (bindless world): the albedo / normal /
                // emissive pool at buffer(10) + its textures declared resident.
                if let Some(tex_args) = bindless_tex_args.filter(|_| {
                    self.glass_pipeline_rt_textured.is_some()
                        || self.water_pipeline_rt_textured.is_some()
                        || self.glass_mesh_pipeline_rt_textured.is_some()
                }) {
                    enc.setFragmentBuffer_offset_atIndex(Some(tex_args.as_ref()), 0, 10);
                    self.use_bindless_textures(&enc);
                }
            }
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
                enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                    MTLPrimitiveType::Triangle,
                    d.index_count as usize,
                    d.index_type,
                    &d.index_buffer,
                    d.index_offset_bytes,
                    1,
                    d.base_vertex as isize,
                    0,
                );
            }
        }

        Ok(order.len() as u32)
    }
}
