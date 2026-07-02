// src/metal/draw/main.rs
//
// Main pass (off-screen HDR + 4x MSAA). Renders the visible scene into the
// HDR colour + depth attachments, multisample-resolving to `hdr_resolve`.
// Has three geometry paths:
//
//   * Bindless static path: GPU-driven, issued through `cull_icb` in a single
//     executeCommandsInBuffer. Used when the world's fragment shader provides
//     `fragment_main_bindless` (default.metal) and the static draw list is
//     non-empty -- `object_buffer` / `bindless_tex_args` are `Some` in that
//     case.
//   * Legacy static path: per-draw bindings, walks the `visible` list. Used
//     by shaders without a bindless entry point (custom shaders).
//   * Instanced clusters + skinned meshes: drawn after the static path, with
//     their own pipelines rebound.
//
// The three sub-paths are encoded in fixed (static, instanced, skinned)
// order on a single `MTLRenderCommandEncoder`. Two `MTLParallelRenderCommandEncoder`
// attempts (one full-fat with parallel encoders on every shadow cascade,
// one scoped to just this main pass with 3 sub-encoders + no pass-timing)
// both tripped G14X (M2/M3 Pro/Max class) into an abort inside
// `IOGPUMetalCommandBufferStorageAllocResourceAtIndex`: the crash window
// scaled with parallel-encoder usage rate (~20 s with shadow split, ~90 s
// with main-only) but never went away. The mechanism appears fundamentally
// incompatible with our usage on this hardware / macOS 26.4 combo.
// The per-path helpers below stay split out so a future CPU-parallel
// strategy (per-pass command buffers committed through events, or
// data-parallel draw-record prep) can plug in without re-deriving the
// dispatch shape.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLClearColor, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLLoadAction,
    MTLRenderCommandEncoder as _, MTLRenderPassDescriptor, MTLStoreAction,
};

use crate::gfx::render_types::ShadowUniforms;
use crate::metal::context::{BINDLESS_TEXTURE_ARG_BUFFER_INDEX, MtlContext};
use crate::metal::scoped_encoder::ScopedEncoder;
use crate::metal::uniforms::{ModelUniforms, ViewUniforms};

impl MtlContext {
    // 1.0 when a reflection-resolve pass (SSR resolve or RT reflections) will
    // composite over this frame's HDR target, else 0.0. Mirrors the
    // `scene_input` gate in draw/mod.rs (both resolves write `ssr.targets.output`
    // and the graph picks RT over SSR). The forward shader reads it from
    // `ViewUniforms.reflections_enabled` to hand glossy specular to that resolve.
    fn reflection_resolve_active(&self) -> f32 {
        if self.ssr.settings.is_some() || self.rt.accel.is_some() {
            1.0
        } else {
            0.0
        }
    }

    // pub(in crate::metal) so the render-graph executor in
    // metal/graph_exec.rs can dispatch this pass from a CompiledGraph.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::metal) fn encode_main_pass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        elapsed: f32,
        vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        visible: &[u32],
        prepared_instances: &super::super::instanced::PreparedInstances,
        skinned_joint_bufs: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        object_buffer: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        bindless_tex_args: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        deformed_skinned: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        world_hidden: bool,
    ) -> Result<u32, String> {
        // Build the HDR render pass descriptor. Colour writes into the MSAA
        // attachment and resolves into the single-sample target at end-of-pass;
        // depth lives entirely on the MSAA attachment and is discarded unless
        // a later pass (projected decals, volumetric fog) needs to sample it.
        // Under two-pass occlusion the phase-2 main pass (`Main2`) loads this
        // pass's MSAA colour to draw the disoccluded geometry on top, so the
        // MSAA samples must be stored, not just resolved away. `Main2`
        // performs the final resolve. Without two-pass we resolve-and-discard
        // the MSAA samples as before. The decision mirrors the graph's two-pass
        // gating (`two_pass_occlusion` config AND the bindless cull path active
        // this frame, i.e. an `object_buffer` exists).
        let store_msaa_color = self.cull.two_pass_occlusion
            && object_buffer.is_some()
            && self.cull.pipeline_phase2.is_some();
        let main_pass_desc = MTLRenderPassDescriptor::new();
        let [r, g, b, a] = self.clear_color;
        unsafe {
            let ca = main_pass_desc
                .colorAttachments()
                .objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.hdr_targets.hdr_color.as_ref()));
            ca.setResolveTexture(Some(self.hdr_targets.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Clear);
            ca.setStoreAction(if store_msaa_color {
                MTLStoreAction::StoreAndMultisampleResolve
            } else {
                MTLStoreAction::MultisampleResolve
            });
            ca.setClearColor(MTLClearColor {
                red: r as f64,
                green: g as f64,
                blue: b as f64,
                alpha: a as f64,
            });

            let da = main_pass_desc.depthAttachment();
            da.setTexture(Some(self.hdr_targets.depth.as_ref()));
            da.setLoadAction(MTLLoadAction::Clear);
            da.setClearDepth(1.0);
            // Always resolve depth into the single-sample
            // `hdr_targets.depth_resolve` sibling. This is the canonical
            // post-rasterise scene depth that the post chain consumes:
            // raymarch writes hit depth into it; water / decal / fog
            // sample it (single-sample is enough since they only ever
            // read sample 0 anyway). `Sample0` filter matches the
            // existing MSAA-sample-0 read pattern bit-for-bit. The MSAA
            // attachment also stays alive (`StoreAndMultisampleResolve`):
            // the raymarch fragment shader samples it as a read-only
            // snapshot to drive the cone-march early-out without
            // aliasing the writable depth target.
            da.setResolveTexture(Some(self.hdr_targets.depth_resolve.as_ref()));
            da.setDepthResolveFilter(objc2_metal::MTLMultisampleDepthResolveFilter::Sample0);
            da.setStoreAction(MTLStoreAction::StoreAndMultisampleResolve);
        }

        if let Some(t) = &self.pass_timing {
            t.attach_render(&main_pass_desc, super::super::pass_timing::PassId::Main);
        }
        // ScopedEncoder ends the main HDR pass (MSAA resolves into hdr_resolve)
        // and pops the debug group when it drops at end of scope.
        let encoder = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&main_pass_desc)
                .ok_or("failed to get render encoder")?,
            "main pass",
        );

        let view_uniforms = ViewUniforms {
            vp,
            view: self.view_matrix,
            elapsed,
            reflections_enabled: self.reflection_resolve_active(),
            cam_pos,
            prefilter_mip_count: self.env_map.prefilter_mip_count as f32,
            _end_pad: [0.0; 2],
        };

        // While the world is hidden behind an opaque menu, the pass stops at the
        // descriptor's Clear load action: skip every geometry sub-path so even a
        // non-bindless skinned world (whose draw does not consult the now-empty
        // visible / instance / object inputs) renders nothing behind the menu.
        // A scene-less world (no main pipeline) takes the same bare-clear shape
        // every frame.
        if world_hidden || self.pipeline_state.is_none() {
            return Ok(0);
        }

        let count_static = self.encode_main_static_into(
            &encoder,
            &view_uniforms,
            cam_pos,
            visible,
            object_buffer,
            bindless_tex_args,
            deformed_skinned,
            // Main pass: the main cull ICB (no override).
            None,
        );
        let count_instanced =
            self.encode_main_instanced_into(&encoder, &view_uniforms, prepared_instances);
        let count_skinned =
            self.encode_main_skinned_into(&encoder, &view_uniforms, cam_pos, skinned_joint_bufs);

        Ok(count_static + count_instanced + count_skinned)
    }

    // Render the main pass into one reflection-probe cube face instead of the
    // HDR targets. A thin sibling of `encode_main_pass`: same three geometry
    // sub-paths and shared bindings, but the render-pass descriptor points at a
    // square MSAA colour + depth (resolving colour into `face_resolve`), and the
    // view + view-projection are the caller's face matrices (not `self.*`), so
    // the capture never disturbs the frame's camera state. Depth is cleared and
    // discarded -- the probe consumes only the resolved colour. Driven by
    // `capture_reflection_probe` (metal/probe.rs), once at first frame.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::metal) fn encode_main_into_face(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        face_color_msaa: &ProtocolObject<dyn objc2_metal::MTLTexture>,
        face_depth_msaa: &ProtocolObject<dyn objc2_metal::MTLTexture>,
        face_resolve: &ProtocolObject<dyn objc2_metal::MTLTexture>,
        view: [[f32; 4]; 4],
        vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        elapsed: f32,
        visible: &[u32],
        prepared_instances: &super::super::instanced::PreparedInstances,
        skinned_joint_bufs: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        object_buffer: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        bindless_tex_args: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        deformed_skinned: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        // Bindless ICB to execute instead of the main cull's. The planar mirror
        // render passes its slot's mirror ICB (culled against the reflected
        // frustum); the probe capture passes `None` (reuses the main cull ICB).
        // Only consulted on the bindless static path; the legacy fallback uses
        // `visible` regardless.
        icb_override: Option<&ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>>,
    ) -> Result<u32, String> {
        let desc = MTLRenderPassDescriptor::new();
        let [r, g, b, a] = self.clear_color;
        unsafe {
            let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(face_color_msaa));
            ca.setResolveTexture(Some(face_resolve));
            ca.setLoadAction(MTLLoadAction::Clear);
            ca.setStoreAction(MTLStoreAction::MultisampleResolve);
            ca.setClearColor(MTLClearColor {
                red: r as f64,
                green: g as f64,
                blue: b as f64,
                alpha: a as f64,
            });

            let da = desc.depthAttachment();
            da.setTexture(Some(face_depth_msaa));
            da.setLoadAction(MTLLoadAction::Clear);
            da.setClearDepth(1.0);
            da.setStoreAction(MTLStoreAction::DontCare);
        }

        let encoder = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&desc)
                .ok_or("failed to get probe render encoder")?,
            "probe face",
        );

        let view_uniforms = ViewUniforms {
            vp,
            view,
            elapsed,
            // Probe-face bake: no reflection resolve runs over the probe cube, so
            // the forward probe specular is the only reflection source here. Keep
            // it (0.0) so captured glossy surfaces are not flattened.
            reflections_enabled: 0.0,
            cam_pos,
            prefilter_mip_count: self.env_map.prefilter_mip_count as f32,
            _end_pad: [0.0; 2],
        };

        let count_static = self.encode_main_static_into(
            &encoder,
            &view_uniforms,
            cam_pos,
            visible,
            object_buffer,
            bindless_tex_args,
            deformed_skinned,
            icb_override,
        );
        let count_instanced =
            self.encode_main_instanced_into(&encoder, &view_uniforms, prepared_instances);
        let count_skinned =
            self.encode_main_skinned_into(&encoder, &view_uniforms, cam_pos, skinned_joint_bufs);

        Ok(count_static + count_instanced + count_skinned)
    }

    // Phase-2 main pass for two-pass occlusion (`Main2`). Loads (does not
    // clear) the HDR colour + depth that `encode_main_pass` (phase 1) wrote
    // and re-runs the bindless indirect draw through `cull_icb_2` (the phase-2
    // cull's output), depth-compositing the disoccluded geometry with phase 1.
    // Folded instances AND folded skinned objects ride the unified cull buffers
    // as cullable records, so both are Hi-Z-culled through both phases: drawn in
    // phase 1 when visible, and redrawn here only when phase 1 Hi-Z-occluded
    // them and the rebuilt pyramid (Cull2) disoccludes them. The shared
    // `execute_bindless_static_icb` issues the static+instance range then the
    // skinned tail of `cull_icb_2` (the phase-2 cull resets every already-drawn
    // slot, so nothing double-draws). Resolves colour + depth at end-of-pass so
    // the post-decoration stack reads the combined result. A no-op (returns 0)
    // when there is nothing to redraw: two-pass off, no bindless geometry, or
    // the phase-2 ICB was not built.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::metal) fn encode_main_pass_phase2(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        elapsed: f32,
        vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        object_buffer: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        bindless_tex_args: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        deformed_skinned: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
    ) -> Result<u32, String> {
        let (Some(obj_buf), Some(tex_args), Some(icb)) =
            (object_buffer, bindless_tex_args, &self.cull.icb_2)
        else {
            return Ok(0);
        };

        // Load the phase-1 MSAA colour + depth (phase 1 stored them under
        // two-pass), draw the disoccluded geometry on top, and resolve both at
        // end-of-pass: this resolve is the one the post stack consumes.
        let main_pass_desc = MTLRenderPassDescriptor::new();
        unsafe {
            let ca = main_pass_desc
                .colorAttachments()
                .objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.hdr_targets.hdr_color.as_ref()));
            ca.setResolveTexture(Some(self.hdr_targets.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::StoreAndMultisampleResolve);

            let da = main_pass_desc.depthAttachment();
            da.setTexture(Some(self.hdr_targets.depth.as_ref()));
            da.setLoadAction(MTLLoadAction::Load);
            da.setResolveTexture(Some(self.hdr_targets.depth_resolve.as_ref()));
            da.setDepthResolveFilter(objc2_metal::MTLMultisampleDepthResolveFilter::Sample0);
            da.setStoreAction(MTLStoreAction::StoreAndMultisampleResolve);
        }

        if let Some(t) = &self.pass_timing {
            t.attach_render(&main_pass_desc, super::super::pass_timing::PassId::Main2);
        }
        let encoder = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&main_pass_desc)
                .ok_or("failed to get render encoder")?,
            "main2 pass",
        );

        let view_uniforms = ViewUniforms {
            vp,
            view: self.view_matrix,
            elapsed,
            reflections_enabled: self.reflection_resolve_active(),
            cam_pos,
            prefilter_mip_count: self.env_map.prefilter_mip_count as f32,
            _end_pad: [0.0; 2],
        };
        self.bind_main_pass_shared(&encoder, &view_uniforms);
        let draw_calls = self.execute_bindless_static_icb(
            &encoder,
            obj_buf,
            tex_args,
            icb.as_ref(),
            deformed_skinned,
        );

        Ok(draw_calls)
    }

    // Issue the GPU-driven bindless pass: bind the per-object data +
    // bindless-texture argument buffer, declare the index buffer + sampled
    // textures resident (they are reached only through indirect commands /
    // the argument buffer, never bound on the encoder), then execute the `icb`.
    // Shared by the phase-1 main pass (`icb = cull_icb`) and the phase-2 main
    // pass (`icb = cull_icb_2`) so both issue the bindless geometry identically.
    //
    // The ICB is split into two `executeCommandsInBuffer` ranges because Metal
    // bakes the index buffer into each indirect command: records [0, base) are
    // static + instances (static u32 index buffer, static vertex buffer already
    // bound at binding 1 by `bind_main_pass_shared`); records [base, count) are
    // the folded skinned tail, which the cull kernel encoded against the skinned
    // u16 index buffer and which read the compute-deformed vertices, so this
    // rebinds the deformed buffer at binding 1 for that range (inherited by the
    // ICB commands). Returns the number of indirect draws issued (1 or 2).
    fn execute_bindless_static_icb(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        obj_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
        tex_args: &Retained<ProtocolObject<dyn MTLBuffer>>,
        icb: &ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>,
        deformed_skinned: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
    ) -> u32 {
        use objc2_metal::{MTLRenderStages, MTLResourceUsage};
        // Object records (binding 9) + bindless textures (binding 7) are shared
        // by both ranges: the skinned records live in the same object buffer and
        // sample the same flat pool. The object id reaches the shader via each
        // command's [[base_instance]].
        unsafe {
            enc.setVertexBuffer_offset_atIndex(Some(obj_buf), 0, 9);
            enc.setFragmentBuffer_offset_atIndex(Some(obj_buf), 0, 9);
            enc.setFragmentBuffer_offset_atIndex(
                Some(tex_args),
                0,
                BINDLESS_TEXTURE_ARG_BUFFER_INDEX,
            );
        }
        self.use_bindless_textures(enc);

        let base = self.skinned_record_base();
        let total = self.cull_count();
        let mut draw_calls = 0u32;

        // Static + instance range [0, base). The static u32 index buffer is
        // referenced only inside the indirect commands (not bound), so make it
        // resident; the static vertex buffer is already bound at binding 1.
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
            // SAFETY: [0, base) spans the static + instance command slots
            // (`ensure_icb_capacity` sized the ICB for `cull_count()` >= base).
            unsafe {
                enc.executeCommandsInBuffer_withRange(icb, range);
            }
            draw_calls += 1;
        }

        // Skinned tail [base, total): bind the deformed vertex buffer (inherited
        // by the ICB commands) and make the skinned u16 index buffer resident
        // (the cull kernel baked it into these commands). `deformed_skinned` is
        // `Some` exactly when the fold is active (n_skinned > 0 => total > base).
        if let Some(deformed) = deformed_skinned
            && total > base
        {
            unsafe {
                enc.setVertexBuffer_offset_atIndex(Some(deformed), 0, 1);
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
                enc.executeCommandsInBuffer_withRange(icb, range);
            }
            draw_calls += 1;
        }
        draw_calls
    }

    // Apply the bindings every main-pass sub-path needs (view uniforms,
    // lights, shadow + IBL + SSAO bindings, the shared vertex buffer at
    // binding 1). With the serial encoder this is re-applied on each
    // sub-path entry (duplicating some state writes) so the per-path
    // helpers stay shape-compatible with a future parallel-encoder retry.
    // Returns false without touching the encoder when the world has no main
    // pipeline (no 3D scene content); the caller then skips its draws.
    fn bind_main_pass_shared(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        view_uniforms: &ViewUniforms,
    ) -> bool {
        let Some(pipeline_state) = &self.pipeline_state else {
            return false;
        };
        enc.setRenderPipelineState(pipeline_state);
        enc.setDepthStencilState(Some(&self.depth_state));

        unsafe {
            enc.setVertexBytes_length_atIndex(
                std::ptr::NonNull::from(view_uniforms).cast(),
                std::mem::size_of::<ViewUniforms>(),
                0,
            );
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(view_uniforms).cast(),
                std::mem::size_of::<ViewUniforms>(),
                0,
            );
            enc.setVertexBuffer_offset_atIndex(Some(&self.vertex_buffer), 0, 1);
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&self.light_uniforms).cast(),
                std::mem::size_of::<crate::gfx::render_types::LightUniforms>(),
                4,
            );
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&self.shadow_uniforms).cast(),
                std::mem::size_of::<ShadowUniforms>(),
                5,
            );
            enc.setFragmentTexture_atIndex(Some(self.shadow_map.as_ref()), 2);
            enc.setFragmentSamplerState_atIndex(Some(self.shadow_sampler.as_ref()), 1);
            // IBL bindings: irradiance + prefilter cubes at texture(3) / texture(4)
            // and a shared linear-clamp sampler at sampler(2). Always bound; the
            // shader uses prefilter_mip_count == 0 to detect the fallback case.
            enc.setFragmentTexture_atIndex(Some(self.env_map.irradiance.as_ref()), 3);
            enc.setFragmentTexture_atIndex(Some(self.env_map.prefilter.as_ref()), 4);
            enc.setFragmentSamplerState_atIndex(Some(self.cube_sampler.as_ref()), 2);
            // SSAO occlusion at texture(5): the blurred AO when SSAO is on,
            // else a 1x1 white texture so shade_surface samples a constant 1.0
            // and the ambient term is left untouched. (The bindless static
            // pass instead reaches it through the BindlessTextures argument
            // buffer; see build_bindless_texture_args.)
            enc.setFragmentTexture_atIndex(Some(self.ao_output_texture()), 5);
            // Reflection-probe cube array at texture(6 .. 6+MAX_PROBES): the legacy
            // path now selects + blends per-surface from the same probe set as the
            // bindless path (which reaches the cubes through the BindlessTextures arg
            // buffer instead, ICB-incompatible discrete binds being the reason).
            // probe_cube_or_sky returns the sky prefilter for unbaked slots, so all
            // MAX_PROBES are always valid. Skybox + diffuse keep texture 3/4.
            for i in 0..crate::metal::uniforms::MAX_PROBES {
                enc.setFragmentTexture_atIndex(Some(self.probe_cube_or_sky(i)), 6 + i);
            }
            // Reflection-probe set (count + per-probe parallax boxes) at fragment
            // buffer(6) (a buffer slot, distinct from the texture(6) array). `EMPTY`
            // until a bake; the shader weights every box covering the surface.
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&self.probe_set).cast(),
                std::mem::size_of::<crate::metal::uniforms::ProbeSet>(),
                6,
            );
        }
        true
    }

    // Encode the static-geometry sub-path: either the bindless GPU-driven
    // ICB execution or the legacy per-draw loop, depending on which path
    // the world's pipeline opted into.
    #[allow(clippy::too_many_arguments)]
    fn encode_main_static_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        view_uniforms: &ViewUniforms,
        cam_pos: [f32; 3],
        visible: &[u32],
        object_buffer: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        bindless_tex_args: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        deformed_skinned: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        // ICB to execute instead of the main cull's `self.cull.icb`: the planar
        // mirror render passes its slot's mirror ICB (culled against the
        // reflected frustum) here; the main + probe paths pass `None` to use the
        // main ICB.
        icb_override: Option<&ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>>,
    ) -> u32 {
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str("main static"));
        if !self.bind_main_pass_shared(enc, view_uniforms) {
            enc.popDebugGroup();
            return 0;
        }

        let last_tex = self.textures.len().saturating_sub(1);
        let last_nm = self.normal_map_textures.len().saturating_sub(1);
        let mut draw_calls: u32 = 0;

        let icb_ref = icb_override.or(self.cull.icb.as_deref());
        if let (Some(obj_buf), Some(tex_args), Some(icb)) =
            (object_buffer, bindless_tex_args, icb_ref)
        {
            // Bindless static pass, GPU-driven. Shared with the phase-2 main
            // pass under two-pass occlusion: see `execute_bindless_static_icb`.
            // Returns 1 (static+instances) or 2 (+ skinned tail) draw calls.
            draw_calls +=
                self.execute_bindless_static_icb(enc, obj_buf, tex_args, icb, deformed_skinned);
        } else {
            // Legacy static pass: rebind model/material/textures per draw.
            // Used by shaders without a `fragment_main_bindless` entry point
            // (custom shaders). The shared helper owns the
            // visible/resident filter, the camera-distance LOD pick (matching the
            // bindless path's GpuDrawArgs selection), and the indexed draw.
            draw_calls += self.draw_static_objects(enc, visible, cam_pos, |enc, obj, _| {
                let model_uniforms = ModelUniforms { model: obj.model };
                let slot = obj.texture_slot.min(last_tex);
                let nm_slot = obj.normal_map_slot.min(last_nm);
                unsafe {
                    // model matrix at vertex buffer(2)
                    enc.setVertexBytes_length_atIndex(
                        std::ptr::NonNull::from(&model_uniforms).cast(),
                        std::mem::size_of::<ModelUniforms>(),
                        2,
                    );
                    // material at fragment buffer(3)
                    enc.setFragmentBytes_length_atIndex(
                        std::ptr::NonNull::from(&obj.material).cast(),
                        std::mem::size_of::<crate::gfx::render_types::MaterialUniforms>(),
                        3,
                    );
                    // albedo at texture(0), normal map at texture(1)
                    enc.setFragmentTexture_atIndex(Some(self.textures[slot].as_ref()), 0);
                    enc.setFragmentTexture_atIndex(
                        Some(self.normal_map_textures[nm_slot].as_ref()),
                        1,
                    );
                    enc.setFragmentSamplerState_atIndex(Some(&self.sampler), 0);
                }
            });
        }
        enc.popDebugGroup();
        draw_calls
    }

    // Encode the instanced-cluster sub-path. One drawIndexedInstanced per
    // cluster*LOD bucket, after a cluster-wide frustum/distance cull.
    fn encode_main_instanced_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        view_uniforms: &ViewUniforms,
        prepared: &super::super::instanced::PreparedInstances,
    ) -> u32 {
        let Some(inst_ps) = self.instanced_pipeline_state.clone() else {
            return 0;
        };
        if prepared.clusters.is_empty() {
            return 0;
        }
        // When the bindless static pass is active with build-time geometry, the
        // instances were folded into its cull buffers and drawn by the static
        // ICB (`execute_bindless_static_icb` over `cull_count()`), so the legacy
        // per-cluster main draw would double-draw them. This condition equals
        // `object_buffer.is_some()` (bindless && static geometry present),
        // mirroring DX/VK's `&& !use_bindless`. Gate only the MAIN draw: the
        // SSR / SSAO / velocity pre-passes + shadow still consume `prepared`
        // through the legacy instanced path. Instance-
        // only worlds (no static geometry) keep the legacy draw here.
        if self.bindless && !self.draw_objects.is_empty() {
            return 0;
        }
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str("main instanced"));
        // Share the same view / lights / shadow / IBL / SSAO bindings as the
        // static path. The pipeline override below swaps to the instanced PSO.
        if !self.bind_main_pass_shared(enc, view_uniforms) {
            enc.popDebugGroup();
            return 0;
        }
        enc.setRenderPipelineState(&inst_ps);

        let last_tex = self.textures.len().saturating_sub(1);
        let last_nm = self.normal_map_textures.len().saturating_sub(1);

        // Per cluster: bind material (fragment buffer(3)) + albedo / normal
        // textures, shared across the cluster's LOD buckets. The shared helper
        // owns the cull / LOD-bucket / instance-buffer / draw loop.
        let draw_calls = self.draw_prepared_instances(enc, prepared, false, |enc, cluster| {
            unsafe {
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(&cluster.material).cast(),
                    std::mem::size_of::<crate::gfx::render_types::MaterialUniforms>(),
                    3,
                );
            }
            let slot = cluster.texture_slot.min(last_tex);
            let nm_slot = cluster.normal_map_slot.min(last_nm);
            unsafe {
                enc.setFragmentTexture_atIndex(Some(self.textures[slot].as_ref()), 0);
                enc.setFragmentTexture_atIndex(Some(self.normal_map_textures[nm_slot].as_ref()), 1);
                enc.setFragmentSamplerState_atIndex(Some(&self.sampler), 0);
            }
        });

        // Restore the regular pipeline state so a future addition to the
        // main pass starts from the same shape the static path left it in.
        if let Some(pipeline_state) = &self.pipeline_state {
            enc.setRenderPipelineState(pipeline_state);
        }
        enc.popDebugGroup();
        draw_calls
    }

    // Encode the skinned-mesh sub-path. Linear-blend-skinned geometry,
    // drawn last in the main pass.
    fn encode_main_skinned_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        view_uniforms: &ViewUniforms,
        cam_pos: [f32; 3],
        skinned_joint_bufs: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    ) -> u32 {
        let (Some(skinned_ps), Some(svb), Some(sib)) = (
            self.skinned.pipeline_state.clone(),
            self.skinned.vertex_buffer.clone(),
            self.skinned.index_buffer.clone(),
        ) else {
            return 0;
        };
        if self.skinned.draw_objects.is_empty() {
            return 0;
        }
        // When the GPU-driven skinned fold is active (bindless + static
        // geometry), skinned objects were pre-skinned into the deformed buffer
        // and drawn by the bindless ICB's skinned tail, so this legacy VS-skinned
        // draw would double-draw them. `n_skinned > 0` is set in `upload_skinned`
        // under exactly that condition, mirroring the instanced gate. A pure-
        // skinned or non-bindless world keeps `n_skinned == 0` and draws here.
        if self.n_skinned > 0 {
            return 0;
        }
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str("main skinned"));
        if !self.bind_main_pass_shared(enc, view_uniforms) {
            enc.popDebugGroup();
            return 0;
        }
        enc.setRenderPipelineState(&skinned_ps);
        unsafe {
            enc.setVertexBuffer_offset_atIndex(Some(&svb), 0, 1);
        }

        let last_tex = self.textures.len().saturating_sub(1);
        let last_nm = self.normal_map_textures.len().saturating_sub(1);

        // The shared helper owns the visible filter, the skinned-camera-distance
        // LOD pick, and the u16 indexed draw; the closure binds this mesh's model,
        // joint palette, material, and textures.
        let draw_calls = self.draw_skinned_objects(enc, &sib, cam_pos, |enc, obj, i| {
            let model_uniforms = ModelUniforms { model: obj.model };
            let slot = obj.texture_slot.min(last_tex);
            let nm_slot = obj.normal_map_slot.min(last_nm);
            unsafe {
                enc.setVertexBytes_length_atIndex(
                    std::ptr::NonNull::from(&model_uniforms).cast(),
                    std::mem::size_of::<ModelUniforms>(),
                    2,
                );
                enc.setVertexBuffer_offset_atIndex(Some(&skinned_joint_bufs[i]), 0, 8);
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(&obj.material).cast(),
                    std::mem::size_of::<crate::gfx::render_types::MaterialUniforms>(),
                    3,
                );
                enc.setFragmentTexture_atIndex(Some(self.textures[slot].as_ref()), 0);
                enc.setFragmentTexture_atIndex(Some(self.normal_map_textures[nm_slot].as_ref()), 1);
                enc.setFragmentSamplerState_atIndex(Some(&self.sampler), 0);
            }
        });
        enc.popDebugGroup();
        draw_calls
    }
}
