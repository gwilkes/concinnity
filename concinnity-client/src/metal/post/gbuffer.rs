// src/metal/post/gbuffer.rs
//
// The unified geometry G-buffer pre-pass. One jittered traversal of the visible
// set (static + instanced + skinned) writes the view-space normal + linear
// depth, perceptual roughness, and screen-space motion vector that SSR, SSAO,
// SSGI, RT reflections, TAA, and the MetalFX upscaler all consume, replacing
// the three separate SSR / SSAO / velocity pre-passes that each re-rasterized
// the same geometry. Pipelines, targets, and the encoder live together so the
// effect is a single unit the other backends can mirror.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer as _, MTLDevice as _, MTLLibrary as _, MTLLoadAction,
    MTLPixelFormat, MTLRenderCommandEncoder as _, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStoreAction, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage, MTLVertexDescriptor, MTLVertexFormat,
    MTLVertexStepFunction,
};

use crate::gfx::mesh_payload::Vertex;

use crate::metal::context::MtlContext;
use crate::metal::pipeline::{ns_str, shader_source};
use crate::metal::scoped_encoder::ScopedEncoder;
use crate::metal::uniforms::{GBufferView, SsrPrepassMat, VelocityModelUniforms};

// All unified-G-buffer pre-pass state grouped into one unit: the shared
// targets (normal+depth / roughness / velocity / sampleable depth) plus the
// per-geometry-kind pipelines. `targets`/`prepass_pipeline` are `Some` when
// any consumer (SSR / SSGI / RT / SSAO / TAA / upscaler) is on;
// `instanced_pipeline` only when the world has GPU-instanced clusters; and
// `skinned_pipeline` is filled later by `upload_skinned` (80-byte layout).
pub(crate) struct GBufferState {
    pub targets: Option<GBufferTargets>,
    pub prepass_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub instanced_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub skinned_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // GPU-driven bindless variant: one unified pipeline that draws the
    // SAME per-frame indirect command set the bindless main pass executes, so the
    // G-buffer feeder goes fully GPU-driven for static / instanced / chunk /
    // skinned geometry (no CPU draw loop). `Some` only when the world is bindless
    // and a G-buffer consumer is active (`bindless && targets.is_some()`);
    // non-bindless / custom-shader worlds leave it `None` and keep the legacy
    // per-geometry-kind CPU loops above. Rebuilt by `reload_shaders`.
    pub bindless_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
}

// Targets

// Off-screen targets for the unified G-buffer pre-pass, shared by every
// consumer of view-space normal/depth, roughness, or motion. All single-sample
// and render resolution; created when any consumer (SSR, SSGI, RT, SSAO, TAA,
// or the upscaler) is active and rebuilt with the HDR targets on resize (sizing
// is keyed off `HdrTargets`, so no dimensions are stored here).
pub(crate) struct GBufferTargets {
    // `RGBA16Float`: view-space normal in rgb, positive linear view depth in a.
    // Read by SSR resolve, the SSAO kernel/blur, SSGI, and RT reflections.
    // Cleared alpha 0 marks "no geometry" (background).
    pub normal_depth: Retained<ProtocolObject<dyn MTLTexture>>,
    // `R8Unorm`: per-pixel perceptual roughness. Read by SSR resolve and RT
    // reflections; cleared 1.0 so the background is treated as non-reflective.
    pub roughness: Retained<ProtocolObject<dyn MTLTexture>>,
    // `RG16Float`: screen-space motion `(prev_uv - cur_uv)`. Read by TAA resolve
    // and the MetalFX upscaler's motion input; cleared 0 (no motion).
    pub velocity: Retained<ProtocolObject<dyn MTLTexture>>,
    // `Depth32Float`, single-sample: the pre-pass z-buffer. Unlike the old
    // per-pass prepass depths this is `ShaderRead | RenderTarget` and stored,
    // because the MetalFX upscaler samples it (`setDepthTexture`). The main pass
    // keeps its own MSAA depth; Hi-Z still reduces that, not this.
    pub depth: Retained<ProtocolObject<dyn MTLTexture>>,
}

// Create or recreate the G-buffer targets at `width`x`height`.
pub(crate) fn create_gbuffer_targets(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
) -> Result<GBufferTargets, String> {
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;

    let sampled = MTLTextureUsage(MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0);
    let make = |fmt: MTLPixelFormat,
                label: &str|
     -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
        let desc = MTLTextureDescriptor::new();
        unsafe {
            desc.setTextureType(MTLTextureType::Type2D);
            desc.setPixelFormat(fmt);
            desc.setWidth(w);
            desc.setHeight(h);
            desc.setUsage(sampled);
            desc.setStorageMode(objc2_metal::MTLStorageMode::Private);
        }
        device
            .newTextureWithDescriptor(&desc)
            .ok_or_else(|| format!("failed to create G-buffer {} texture", label))
    };

    // The depth is sampleable (MetalFX reads it), unlike the old prepass depths.
    let normal_depth = make(MTLPixelFormat::RGBA16Float, "normal+depth")?;
    let roughness = make(MTLPixelFormat::R8Unorm, "roughness")?;
    let velocity = make(MTLPixelFormat::RG16Float, "velocity")?;
    let depth = make(MTLPixelFormat::Depth32Float, "depth")?;

    Ok(GBufferTargets {
        normal_depth,
        roughness,
        velocity,
        depth,
    })
}

// Pipeline

// Build one G-buffer pre-pass pipeline for the given vertex entry
// (`gbuffer_prepass_vertex{,_instanced,_skinned}`). All three share
// `gbuffer_prepass_fragment` and render to three single-sample MRT targets
// (`RGBA16Float` normal+depth, `R8Unorm` roughness, `RG16Float` velocity) plus
// a `Depth32Float` z-buffer. `vert_desc` selects the static (56-byte) or
// skinned (80-byte) vertex layout.
pub(crate) fn build_gbuffer_prepass_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    vertex_entry: &str,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "gbuffer_prepass.metal");
    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&ns_str(msl.as_ref()), Some(&options))
        .map_err(|e| format!("G-buffer pre-pass shader compile error: {:?}", e))?;
    let vert_fn = library
        .newFunctionWithName(&ns_str(vertex_entry))
        .ok_or_else(|| format!("{} not found in G-buffer pre-pass metallib", vertex_entry))?;
    let frag_fn = library
        .newFunctionWithName(&ns_str("gbuffer_prepass_fragment"))
        .ok_or("gbuffer_prepass_fragment not found")?;

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(vert_desc));
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(1);
    unsafe {
        let ca0 = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca0.setPixelFormat(MTLPixelFormat::RGBA16Float);
        ca0.setBlendingEnabled(false);
        let ca1 = desc.colorAttachments().objectAtIndexedSubscript(1);
        ca1.setPixelFormat(MTLPixelFormat::R8Unorm);
        ca1.setBlendingEnabled(false);
        let ca2 = desc.colorAttachments().objectAtIndexedSubscript(2);
        ca2.setPixelFormat(MTLPixelFormat::RG16Float);
        ca2.setBlendingEnabled(false);
    }
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| format!("failed to create G-buffer pre-pass pipeline: {:?}", e))
}

// Two-stream vertex descriptor for the GPU-driven bindless G-buffer pipeline.
// Stream 0 (buffer 1) is the standard 56-byte `Vertex` (pos / normal / tangent /
// colour / uv) the cull-baked indirect commands draw; stream 1 (buffer 2) is the
// PREVIOUS vertex position (attribute 5), read from a second buffer the encoder
// binds (the same static VB for the prefix -> zero per-vertex motion, the
// previous-frame deformed buffer for the skinned tail -> per-vertex skin motion).
// Stream 1 reads only position at offset 0; its stride is the full 56-byte
// `Vertex` so the cull-baked `base_vertex` indexes it identically to stream 0.
pub(crate) fn gbuffer_bindless_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
    let vert_desc = MTLVertexDescriptor::new();
    unsafe {
        // Stream 0 (buffer 1): the attributes the bindless VS reads (pos, normal,
        // colour for the skybox sentinel). Tangent/uv are unused by the G-buffer.
        let set0 = |idx: usize, fmt: MTLVertexFormat, offset: usize| {
            let attr = vert_desc.attributes().objectAtIndexedSubscript(idx);
            attr.setFormat(fmt);
            attr.setOffset(offset);
            attr.setBufferIndex(1);
        };
        set0(0, MTLVertexFormat::Float3, 0); // pos
        set0(1, MTLVertexFormat::Float3, 12); // normal
        set0(3, MTLVertexFormat::Float3, 36); // color
        let layout1 = vert_desc.layouts().objectAtIndexedSubscript(1);
        layout1.setStride(std::mem::size_of::<Vertex>());
        layout1.setStepFunction(MTLVertexStepFunction::PerVertex);

        // Stream 1 (buffer 2): previous vertex position only.
        let attr5 = vert_desc.attributes().objectAtIndexedSubscript(5);
        attr5.setFormat(MTLVertexFormat::Float3);
        attr5.setOffset(0);
        attr5.setBufferIndex(2);
        let layout2 = vert_desc.layouts().objectAtIndexedSubscript(2);
        layout2.setStride(std::mem::size_of::<Vertex>());
        layout2.setStepFunction(MTLVertexStepFunction::PerVertex);
    }
    vert_desc
}

// Build the GPU-driven bindless G-buffer pre-pass pipeline:
// `gbuffer_prepass_vertex_bindless` + `gbuffer_prepass_fragment_bindless`, the
// same 3-MRT + Depth32 targets as the legacy pipeline, but with the two-stream
// vertex descriptor and `supportIndirectCommandBuffers` so it can execute the
// shared cull-produced indirect command buffer. Reads each record's model +
// roughness from the GpuObjectData buffer by `[[base_instance]]`.
pub(crate) fn build_gbuffer_bindless_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "gbuffer_prepass.metal");
    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&ns_str(msl.as_ref()), Some(&options))
        .map_err(|e| format!("G-buffer bindless shader compile error: {:?}", e))?;
    let vert_fn = library
        .newFunctionWithName(&ns_str("gbuffer_prepass_vertex_bindless"))
        .ok_or("gbuffer_prepass_vertex_bindless not found in G-buffer pre-pass metallib")?;
    let frag_fn = library
        .newFunctionWithName(&ns_str("gbuffer_prepass_fragment_bindless"))
        .ok_or("gbuffer_prepass_fragment_bindless not found")?;

    let vert_desc = gbuffer_bindless_vertex_descriptor();
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(&vert_desc));
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(1);
    unsafe {
        let ca0 = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca0.setPixelFormat(MTLPixelFormat::RGBA16Float);
        ca0.setBlendingEnabled(false);
        let ca1 = desc.colorAttachments().objectAtIndexedSubscript(1);
        ca1.setPixelFormat(MTLPixelFormat::R8Unorm);
        ca1.setBlendingEnabled(false);
        let ca2 = desc.colorAttachments().objectAtIndexedSubscript(2);
        ca2.setPixelFormat(MTLPixelFormat::RG16Float);
        ca2.setBlendingEnabled(false);
    }
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    desc.setSupportIndirectCommandBuffers(true);

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| format!("failed to create G-buffer bindless pipeline: {:?}", e))
}

// Encoder

impl MtlContext {
    // Encode the unified G-buffer pre-pass: one jittered traversal of the
    // visible set (static, GPU-instanced, then skinned) writing view-space
    // normal + linear depth at color(0), perceptual roughness at color(1), and
    // screen-space motion at color(2), with a sampleable `Depth32Float`
    // z-buffer. Replaces the separate SSR / SSAO / velocity pre-passes; runs
    // before the main pass so the SSAO kernel and main pass can read its output.
    //
    // Always writes all three color targets (the geometry traversal dominates,
    // so the extra R8 + RG16 stores are negligible). `velocity_active` selects
    // whether the static prev-model + skinned prev-pose come from last frame
    // (true) or collapse to the current frame (false): when false the motion
    // channel is a harmless zero that no consumer reads.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::metal) fn encode_gbuffer_prepass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        view: &GBufferView,
        visible: &[u32],
        cam_pos: [f32; 3],
        prepared_instances: &super::super::instanced::PreparedInstances,
        cur_joint_bufs: &[Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>],
        prev_joint_bufs: &[Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>],
        velocity_active: bool,
        object_buffer: Option<&Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
        prev_model_buffer: Option<&Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
        deformed_current: Option<&Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
        deformed_prev: Option<&Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    ) -> Result<u32, String> {
        let (targets, static_ps) = match (&self.gbuffer.targets, &self.gbuffer.prepass_pipeline) {
            (Some(t), Some(p)) => (t, p),
            _ => return Ok(0),
        };

        let desc = MTLRenderPassDescriptor::new();
        unsafe {
            let ca0 = desc.colorAttachments().objectAtIndexedSubscript(0);
            ca0.setTexture(Some(targets.normal_depth.as_ref()));
            ca0.setLoadAction(MTLLoadAction::Clear);
            ca0.setStoreAction(MTLStoreAction::Store);
            // Cleared alpha 0 marks "no geometry" for the SSR/SSAO/RT consumers.
            ca0.setClearColor(MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            });
            let ca1 = desc.colorAttachments().objectAtIndexedSubscript(1);
            ca1.setTexture(Some(targets.roughness.as_ref()));
            ca1.setLoadAction(MTLLoadAction::Clear);
            ca1.setStoreAction(MTLStoreAction::Store);
            // Background roughness 1.0 -> non-reflective, so the border emits no SSR.
            ca1.setClearColor(MTLClearColor {
                red: 1.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            });
            let ca2 = desc.colorAttachments().objectAtIndexedSubscript(2);
            ca2.setTexture(Some(targets.velocity.as_ref()));
            ca2.setLoadAction(MTLLoadAction::Clear);
            ca2.setStoreAction(MTLStoreAction::Store);
            // Zero motion for the cleared background.
            ca2.setClearColor(MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            });
            let da = desc.depthAttachment();
            da.setTexture(Some(targets.depth.as_ref()));
            da.setLoadAction(MTLLoadAction::Clear);
            da.setClearDepth(1.0);
            // Stored (not DontCare): the MetalFX upscaler samples this depth.
            da.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = &self.pass_timing {
            t.attach_render(&desc, crate::metal::pass_timing::PassId::GBufferPrepass);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&desc)
                .ok_or("failed to get G-buffer pre-pass encoder")?,
            "g-buffer prepass",
        );

        // GPU-driven path: when the world is bindless (the cull
        // produced an object buffer) and the bindless G-buffer pipeline exists,
        // draw the SAME per-frame indirect command set the main pass executes,
        // with no CPU draw loop. Mirrors the main pass's two-range ICB split.
        if self.gbuffer.bindless_pipeline.is_some()
            && let Some(object_buffer) = object_buffer
        {
            let draws = self.encode_gbuffer_prepass_gpu_driven(
                &enc,
                view,
                object_buffer,
                prev_model_buffer,
                deformed_current,
                deformed_prev,
                velocity_active,
            );
            return Ok(draws);
        }

        enc.setRenderPipelineState(static_ps);
        enc.setDepthStencilState(Some(&self.depth_state));
        unsafe {
            enc.setVertexBytes_length_atIndex(
                std::ptr::NonNull::from(view).cast(),
                std::mem::size_of::<GBufferView>(),
                0,
            );
            enc.setVertexBuffer_offset_atIndex(Some(&self.vertex_buffer), 0, 1);
        }

        // Static geometry: model (cur + prev for motion) at vertex(2), roughness
        // at fragment(0). prev_model collapses to cur when velocity is inactive.
        let mut draws = self.draw_static_objects(&enc, visible, cam_pos, |enc, obj, idx| {
            let model = VelocityModelUniforms {
                cur_model: obj.model,
                prev_model: if velocity_active {
                    self.prev_draw_models[idx]
                } else {
                    obj.model
                },
            };
            let mat = SsrPrepassMat {
                roughness: obj.material.roughness,
                _pad: [0.0; 3],
            };
            unsafe {
                enc.setVertexBytes_length_atIndex(
                    std::ptr::NonNull::from(&model).cast(),
                    std::mem::size_of::<VelocityModelUniforms>(),
                    2,
                );
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(&mat).cast(),
                    std::mem::size_of::<SsrPrepassMat>(),
                    0,
                );
            }
        });

        // GPU-instanced clusters: instance transforms are immutable, so binding
        // the same buffer as cur + prev (`bind_prev`) yields zero motion;
        // roughness is cluster-wide.
        if let Some(inst_ps) = &self.gbuffer.instanced_pipeline
            && !prepared_instances.clusters.is_empty()
        {
            enc.setRenderPipelineState(inst_ps);
            draws +=
                self.draw_prepared_instances(&enc, prepared_instances, true, |enc, cluster| {
                    let mat = SsrPrepassMat {
                        roughness: cluster.material.roughness,
                        _pad: [0.0; 3],
                    };
                    unsafe {
                        enc.setFragmentBytes_length_atIndex(
                            std::ptr::NonNull::from(&mat).cast(),
                            std::mem::size_of::<SsrPrepassMat>(),
                            0,
                        );
                    }
                });
        }

        // Skinned meshes: current pose at buffer(8), previous at buffer(9) (falls
        // back to the current pose when no previous buffer exists -> zero motion).
        // The model matrix is static, so cur == prev model.
        if let (Some(skinned_ps), Some(svb), Some(sib)) = (
            &self.gbuffer.skinned_pipeline,
            &self.skinned.vertex_buffer,
            &self.skinned.index_buffer,
        ) && !self.skinned.draw_objects.is_empty()
        {
            enc.setRenderPipelineState(skinned_ps);
            unsafe {
                enc.setVertexBuffer_offset_atIndex(Some(svb), 0, 1);
            }
            draws += self.draw_skinned_objects(&enc, sib, cam_pos, |enc, obj, i| {
                let model = VelocityModelUniforms {
                    cur_model: obj.model,
                    prev_model: obj.model,
                };
                let mat = SsrPrepassMat {
                    roughness: obj.material.roughness,
                    _pad: [0.0; 3],
                };
                let prev = prev_joint_bufs.get(i).unwrap_or(&cur_joint_bufs[i]);
                unsafe {
                    enc.setVertexBytes_length_atIndex(
                        std::ptr::NonNull::from(&model).cast(),
                        std::mem::size_of::<VelocityModelUniforms>(),
                        2,
                    );
                    enc.setFragmentBytes_length_atIndex(
                        std::ptr::NonNull::from(&mat).cast(),
                        std::mem::size_of::<SsrPrepassMat>(),
                        0,
                    );
                    enc.setVertexBuffer_offset_atIndex(Some(&cur_joint_bufs[i]), 0, 8);
                    enc.setVertexBuffer_offset_atIndex(Some(prev), 0, 9);
                }
            });
        }
        Ok(draws)
    }

    // GPU-driven G-buffer pre-pass: draw the SAME per-frame indirect
    // command set the bindless main pass executes, with the unified bindless
    // G-buffer pipeline. Mirrors `execute_bindless_static_icb`'s two-range split
    // -- the static + instance + chunk prefix `[0, skinned_record_base())` over
    // the static VB, then the folded skinned tail `[skinned_record_base(),
    // cull_count())` over the deformed VB + skinned u16 IB -- but reuses the
    // PHASE-1 `cull.icb` (the pre-pass runs before Cull2/Main2, so phase-1
    // coverage is the natural source; the camera frustum is identical to the main
    // pass, so no extra cull dispatch is needed). The previous vertex position
    // rides a second vertex stream (binding 2): the static VB for the prefix
    // (prev_pos == cur_pos -> model-delta motion), the previous-frame deformed
    // buffer for the skinned tail (per-vertex skin motion). Returns the indirect
    // draw count (0-2).
    #[allow(clippy::too_many_arguments)]
    fn encode_gbuffer_prepass_gpu_driven(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        view: &GBufferView,
        object_buffer: &Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        prev_model_buffer: Option<&Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
        deformed_current: Option<&Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
        deformed_prev: Option<&Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
        velocity_active: bool,
    ) -> u32 {
        use objc2_metal::{MTLRenderStages, MTLResourceUsage};
        use std::sync::atomic::Ordering;
        let (Some(pipeline), Some(icb), Some(prev_models)) = (
            self.gbuffer.bindless_pipeline.as_ref(),
            self.cull.icb.as_ref(),
            prev_model_buffer,
        ) else {
            return 0;
        };
        enc.setRenderPipelineState(pipeline);
        enc.setDepthStencilState(Some(&self.depth_state));
        unsafe {
            // GBufferView (vbuf 0), current vertex stream (vbuf 1), previous
            // vertex stream (vbuf 2), object records (vbuf 9), prev_model parallel
            // buffer (vbuf 10). The ICB commands inherit these bindings; the cull
            // baked base_instance = record id, so the VS reads objects[id].model
            // + prev_models[id]. The prefix binds the static VB to BOTH streams
            // (prev_pos == cur_pos), so its motion is purely the model delta.
            enc.setVertexBytes_length_atIndex(
                std::ptr::NonNull::from(view).cast(),
                std::mem::size_of::<GBufferView>(),
                0,
            );
            enc.setVertexBuffer_offset_atIndex(Some(object_buffer), 0, 9);
            enc.setVertexBuffer_offset_atIndex(Some(prev_models), 0, 10);
            enc.setVertexBuffer_offset_atIndex(Some(&self.vertex_buffer), 0, 1);
            enc.setVertexBuffer_offset_atIndex(Some(&self.vertex_buffer), 0, 2);
        }

        let base = self.skinned_record_base();
        let total = self.cull_count();
        let mut draw_calls = 0u32;

        // Static + instance + chunk prefix [0, base): static u32 IB resident.
        if base > 0 {
            enc.useResource_usage_stages(
                ProtocolObject::from_ref(&*self.index_buffer),
                MTLResourceUsage::Read,
                MTLRenderStages::Vertex,
            );
            let range = objc2_foundation::NSRange {
                location: 0,
                length: base,
            };
            // SAFETY: [0, base) spans the static + instance + chunk command slots;
            // the reused main ICB is sized for cull_count() >= base.
            unsafe {
                enc.executeCommandsInBuffer_withRange(icb.as_ref(), range);
            }
            draw_calls += 1;
        }

        // Folded skinned tail [base, total): current deformed at stream 0,
        // previous-frame deformed at stream 1. Until the deformed ring is primed
        // (frame 0 / after a rebuild), or with velocity inactive / a single frame
        // in flight, bind the CURRENT buffer as the previous one -> zero skinned
        // motion (no garbage motion vector from an unposed prior slot).
        if let Some(deformed) = deformed_current
            && total > base
        {
            let prev = if velocity_active
                && self.frames_in_flight >= 2
                && self.skinned.deformed_primed.load(Ordering::Relaxed)
            {
                deformed_prev.unwrap_or(deformed)
            } else {
                deformed
            };
            unsafe {
                enc.setVertexBuffer_offset_atIndex(Some(deformed), 0, 1);
                enc.setVertexBuffer_offset_atIndex(Some(prev), 0, 2);
            }
            if let Some(skinned_ib) = self.skinned.index_buffer.as_ref() {
                enc.useResource_usage_stages(
                    ProtocolObject::from_ref(&**skinned_ib),
                    MTLResourceUsage::Read,
                    MTLRenderStages::Vertex,
                );
            }
            let range = objc2_foundation::NSRange {
                location: base,
                length: total - base,
            };
            // SAFETY: [base, total) spans the folded skinned command slots.
            unsafe {
                enc.executeCommandsInBuffer_withRange(icb.as_ref(), range);
            }
            draw_calls += 1;
            // The current deformed slot now holds a valid pose, so next frame's
            // previous-frame read is well-defined. Relaxed: the only other access
            // is the next frame's same-pass load, ordered by the render-graph
            // scope join between frames; no other pass touches this flag.
            self.skinned.deformed_primed.store(true, Ordering::Relaxed);
        }
        draw_calls
    }

    // Build the per-frame `prev_model` buffer for the GPU-driven G-buffer pass:
    // one column-major `float4x4` per cull record, indexed
    // identically to `build_object_buffer` (static + chunks + clones, then
    // instances, then skinned). The G-buffer VS reads it at `[[base_instance]]`
    // to derive per-object motion. Returns `None` when there is no static
    // geometry. Rebuilt every frame: the static + chunk region follows last
    // frame's model (or the current model when velocity is inactive), the
    // instance region is the immutable instance transforms (camera-only motion),
    // and the skinned region is the current model (per-vertex skin motion comes
    // from the previous-frame deformed buffer, not the model matrix).
    pub(in crate::metal) fn build_gbuffer_prev_models(
        &mut self,
        ring_slot: usize,
        velocity_active: bool,
    ) -> Result<Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>, String> {
        if self.draw_objects.is_empty() {
            return Ok(None);
        }
        let mut models = std::mem::take(&mut self.prev_model_scratch);
        models.clear();
        // Static + chunks + clones: index-parallel to build_object_buffer's
        // draw_objects loop. `velocity_active` gates last-frame vs current model
        // (current -> zero model-delta motion, a harmless zero no consumer reads).
        for (i, obj) in self.draw_objects.iter().enumerate() {
            models.push(if velocity_active {
                self.prev_draw_models[i]
            } else {
                obj.model
            });
        }
        // Instances: transforms are immutable, so cur == prev (camera-only motion).
        if self.n_instances > 0 {
            models.extend(self.instance_records.iter().map(|r| r.model));
        }
        // Skinned: the model matrix is static (cur == prev); per-vertex motion
        // comes from the previous-frame deformed buffer, not the model.
        if self.n_skinned > 0 {
            models.extend(self.skinned.draw_objects.iter().map(|o| o.model));
        }
        let result = self.prev_model_ring.write(
            &self.device,
            ring_slot,
            crate::metal::context::bytes_of_slice(&models),
        );
        self.prev_model_scratch = models;
        result.map(Some)
    }
}
