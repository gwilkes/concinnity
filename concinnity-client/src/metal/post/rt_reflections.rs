// src/metal/post/rt_reflections.rs
//
// Hardware ray-traced reflection pass. A fullscreen-triangle fragment shader
// that, per glossy pixel, rebuilds a world-space surface point + normal from the
// SSR pre-pass G-buffer, traces a reflection ray against the scene acceleration
// structure ([`crate::metal::raytrace`]), shades the hit (sun Lambert + IBL
// ambient * material tint) or the IBL prefilter cube on a miss, and composites
// the result over the scene with the same Fresnel/gloss weighting SSR uses.
//
// It occupies the SSR-resolve slot in the frame graph (reads hdr_resolve, writes
// scene_pre_taa, which aliases `ssr_targets.output`) and is mutually exclusive
// with SSR resolve. Like SSGI it relies on the SSR pre-pass G-buffer, so the
// pre-pass is forced on whenever RT reflections are enabled.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer as _, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder as _, MTLRenderPassDescriptor, MTLRenderPipelineState, MTLStoreAction,
};

use crate::metal::context::MtlContext;
use crate::metal::pipeline::shader_source;
use crate::metal::post::fullscreen::{FullscreenBlend, build_fullscreen_pipeline, compile_library};
use crate::metal::scoped_encoder::ScopedEncoder;

// Build one RT-reflection pipeline for the given fragment entry: a
// fullscreen-triangle pass that traces a reflection ray and composites it over
// the scene, writing a single-sample `RGBA16Float` target (the same
// `ssr_targets.output` SSR resolve would write). Two entries exist:
// `rt_reflections_fragment` (flat tint) and `rt_reflections_fragment_textured`
// (samples the bindless albedo pool). Compiled only when RT reflections are
// enabled and the GPU supports ray tracing (the shader uses `metal_raytracing`,
// so it must not be compiled on a non-RT device).
pub(crate) fn build_rt_reflection_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    fragment_entry: &str,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "rt_reflections.metal");
    let library = compile_library(device, msl.as_ref(), "RT reflections")?;
    build_fullscreen_pipeline(
        device,
        &library,
        "rt_fullscreen_vertex",
        fragment_entry,
        MTLPixelFormat::RGBA16Float,
        FullscreenBlend::Replace,
    )
}

impl MtlContext {
    // Encode the RT-reflection resolve: a fullscreen pass that traces each
    // glossy pixel's reflection ray against the scene BVH and writes the
    // reflected radiance + composite weight into the reflection target, then
    // runs the shared roughness-aware blur + composite into `ssr_targets.output`.
    // Runs after the main pass in the SSR-resolve slot; only called when RT
    // reflections are on and the acceleration structure + pipeline are live.
    // Returns `Ok(0)` (a no-op) when any required resource is missing: the engine
    // then leaves the base scene untouched, the same defensive pattern
    // `encode_ssgi` uses.
    pub(in crate::metal) fn encode_rt_reflections(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        rt_params: &crate::gfx::render_types::RtParams,
        bindless_tex_args: Option<&Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    ) -> Result<u32, String> {
        let (targets, accel, gbuf) =
            match (&self.ssr.targets, &self.rt.accel, &self.gbuffer.targets) {
                (Some(t), Some(a), Some(g)) => (t, a, g),
                // No G-buffer or acceleration structure (unsupported GPU / empty
                // scene): skip, leaving the base scene.
                _ => return Ok(0),
            };

        // The BVH this pass reads (TLAS, skinned BLAS, deformed-vertex buffer)
        // was built earlier this frame in `raytrace::rebuild_skinned`, on command
        // buffers committed (in `rt_dynamic_update`) before this trace's command
        // buffer on the shared queue. That rebuild is fully async (no
        // `waitUntilCompleted`), so same-queue FIFO commit order runs skin → build →
        // trace, the same ordering every cross-pass read relies on. No explicit
        // GPU-side wait is encoded here.
        // Sample the hit's albedo texture from the bindless pool when it is
        // available (the standard GPU-cull path); otherwise fall back to the
        // flat-tint pipeline so non-bindless worlds still get RT reflections.
        let textured = bindless_tex_args.is_some() && self.rt.pipeline_textured.is_some();
        let pipeline = match (textured, &self.rt.pipeline_textured, &self.rt.pipeline) {
            (true, Some(p), _) => p,
            (_, _, Some(p)) => p,
            _ => return Ok(0),
        };

        let desc = MTLRenderPassDescriptor::new();
        unsafe {
            let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(targets.reflection.as_ref()));
            ca.setLoadAction(MTLLoadAction::DontCare);
            ca.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = &self.pass_timing {
            t.attach_render(&desc, crate::metal::pass_timing::PassId::RtReflections);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&desc)
                .ok_or("failed to get RT reflections encoder")?,
            "rt reflections",
        );
        enc.setRenderPipelineState(pipeline);
        unsafe {
            // Textures + samplers mirror the SSR resolve.
            enc.setFragmentTexture_atIndex(Some(self.hdr_targets.hdr_resolve.as_ref()), 0);
            enc.setFragmentTexture_atIndex(Some(gbuf.normal_depth.as_ref()), 1);
            enc.setFragmentTexture_atIndex(Some(gbuf.roughness.as_ref()), 2);
            enc.setFragmentTexture_atIndex(Some(self.env_map.prefilter.as_ref()), 3);
            // Local reflection-probe cubes at texture(4..4+MAX_PROBES): a missed
            // reflection ray reflects the box-projected scene capture instead of
            // the foreign sky HDR (the source the forward IBL specular term uses).
            // probe_cube_or_sky returns the sky for unbaked slots, so binding all
            // MAX_PROBES is always valid; the ProbeSet's count gates use.
            for i in 0..crate::metal::uniforms::MAX_PROBES {
                enc.setFragmentTexture_atIndex(Some(self.probe_cube_or_sky(i)), 4 + i);
            }
            enc.setFragmentSamplerState_atIndex(Some(&self.post_sampler), 0);
            enc.setFragmentSamplerState_atIndex(Some(self.cube_sampler.as_ref()), 1);
            // buffer(0) params; buffers 1..3 the shared geometry the kernel
            // fetches the hit triangle from; the TLAS at buffer(4).
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(rt_params).cast(),
                std::mem::size_of::<crate::gfx::render_types::RtParams>(),
                0,
            );
            enc.setFragmentBuffer_offset_atIndex(Some(self.vertex_buffer.as_ref()), 0, 1);
            enc.setFragmentBuffer_offset_atIndex(Some(self.index_buffer.as_ref()), 0, 2);
            enc.setFragmentBuffer_offset_atIndex(Some(accel.geom_table.as_ref()), 0, 3);
            enc.setFragmentAccelerationStructure_atBufferIndex(Some(accel.tlas.as_ref()), 4);
            // Deformed (posed) skinned vertices + the u16 skinned index buffer
            // the trace fetches for skinned hits. Always bound (1-element
            // dummies when the scene has no skinned geometry) so the shader's
            // bindings are satisfied even though the skinned branch is never
            // taken then. Direct buffer bindings, so Metal makes them resident.
            enc.setFragmentBuffer_offset_atIndex(Some(accel.deformed_verts.as_ref()), 0, 5);
            enc.setFragmentBuffer_offset_atIndex(Some(accel.skinned_indices.as_ref()), 0, 6);
            // Reflection-probe set (count + per-probe parallax boxes) at buffer(8);
            // count == 0 keeps the sky miss fallback. (buffer(7) is the bindless
            // texture pool, bound only on the textured path below.)
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&self.probe_set).cast(),
                std::mem::size_of::<crate::metal::uniforms::ProbeSet>(),
                8,
            );
            // Textured path: bind the bindless albedo pool at buffer(7) (the
            // same index the main pass uses) and declare its textures resident.
            if textured && let Some(tex_args) = bindless_tex_args {
                enc.setFragmentBuffer_offset_atIndex(
                    Some(tex_args.as_ref()),
                    0,
                    crate::metal::context::BINDLESS_TEXTURE_ARG_BUFFER_INDEX,
                );
                self.use_bindless_textures(&enc);
            }
            // The TLAS references each BLAS indirectly, so the BLASes are not
            // auto-tracked: declare them resident or the trace reads garbage.
            crate::metal::raytrace::use_blas_resident_fragment(&enc, &accel.blas);
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        // End the trace encoder before opening the composite render pass: only
        // one command encoder may be live on a command buffer at a time.
        drop(enc);
        self.encode_reflection_composite(cmd_buf)?;
        Ok(0)
    }
}
