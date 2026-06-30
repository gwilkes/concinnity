// src/directx/draw/mod.rs
//
// `DxContext::record_frame` -- the per-frame orchestration. The per-pass GPU
// encoders live in sibling files (shadow / main / composite) plus the
// post-process effects in `directx/post/` (bloom / TAA / SSAO):
//
//   shadow.rs              cascaded shadow map (depth-only, per cascade)
//   main.rs                SSAO pre-pass + main HDR pass (bindless / legacy /
//                          instanced / skinned) + HDR resolve barriers
//   composite.rs           ACES tonemap + composite + text overlay
//   ../post/{bloom,taa,ssao}.rs    pipeline + targets + encoder, co-located

use windows::Win32::Graphics::Direct3D12::*;

use crate::gfx::render_graph::{FrameGraphInputs, build_frame_graph};
use crate::gfx::render_types::{LightUniforms, ShadowUniforms, TextDrawCall};

use super::context::DxContext;
use super::graph_exec::GraphFrameParams;
use super::math::{mat4_mul, perspective};
use super::texture::transition_barrier;

mod composite;
mod main;
mod shadow;
mod text_upload;

pub(super) use text_upload::TextUploadRing;

// ViewUniforms layout (160 bytes) must match the HLSL ViewBlock at b1.
// view_mat is the camera view matrix used to compute view-space depth in the
// vertex shader for shadow cascade selection. cam_pos stored as three
// individual floats to avoid HLSL packing surprises.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct ViewUniforms {
    pub vp: [[f32; 4]; 4],
    pub view_mat: [[f32; 4]; 4],
    pub elapsed: f32,
    // 1.0 when a reflection resolve (SSR resolve or RT reflections) composites
    // over this frame's HDR scene. Below the reflection roughness cut the
    // forward probe specular yields to that resolve (whose miss-fallback samples
    // the same probe set), so a glossy dielectric does not show both the
    // parallax-approximate forward probe reflection and the exact resolved one.
    // 0.0 keeps the full forward probe specular (no resolve: probe-face / planar
    // mirror bakes, reflections off, non-RT/SSR worlds).
    pub reflections_enabled: f32,
    pub cam_x: f32,
    pub cam_y: f32,
    pub cam_z: f32,
    // Number of mip levels in the bound IBL prefilter cubemap. 0 = IBL off.
    pub prefilter_mip_count: f32,
    pub _ep0: f32,
    pub _ep1: f32,
}

// One term of the Halton low-discrepancy sequence; drives the sub-pixel
// projection jitter so successive TAA frames sample slightly different
// positions. Mirrors `halton` in vulkan/draw.rs and metal/draw.rs.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0_f32;
    let mut f = 1.0_f32;
    while index > 0 {
        f /= base as f32;
        result += f * (index % base) as f32;
        index /= base;
    }
    result
}

impl DxContext {
    // Drive a single frame through the render graph. `end_cmd` is the
    // outer "end" cmd list (composite + final timestamp + resolve +
    // per-frame restore barriers); the executor encodes the Composite
    // pass onto it inline, and the post-graph restore barriers below
    // also append onto it. Returns the per-pass cmd lists the executor
    // recorded (in topological pass order) so the caller can submit
    // them between the "start" outer cmd list (timestamp pre-init,
    // closed by the caller before record_frame) and the "end" outer
    // cmd list.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_frame(
        &self,
        end_cmd: &ID3D12GraphicsCommandList,
        back_buffer: &ID3D12Resource,
        back_buffer_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        elapsed: f32,
        fov_y_radians: f32,
        near: f32,
        far: f32,
        cam_pos: [f32; 3],
        text_calls: &[TextDrawCall],
        frame_idx: usize,
        // Off-screen scene render resolution (drives every scene pass + the
        // sub-pixel jitter). Equals the output dims when upscaling is off.
        width: u32,
        height: u32,
        // Drawable (swapchain) resolution; only the Composite pass uses it.
        output_width: u32,
        output_height: u32,
        world_hidden: bool,
    ) -> Result<Vec<ID3D12GraphicsCommandList>, String> {
        // Render-target aspect, shared by the main projection below and the
        // cull frustum.
        let aspect = if height == 0 {
            1.0
        } else {
            width as f32 / height as f32
        };

        // Cascaded-shadow UBO upload. `draw_frame` already advanced the cascade
        // schedule and merged this frame's cascades into `self.shadow.uniforms`
        // (skipped cascades keep the VP their slice was last rendered with, so
        // the Main pass samples each cascade consistently); upload the carried
        // set to this frame's shadow UBO (persistent mapping). It is the empty
        // (identity VP / infinite split) set when shadows are disabled.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &self.shadow.uniforms as *const ShadowUniforms as *const u8,
                self.uniforms.shadow_ubo_ptrs[frame_idx],
                std::mem::size_of::<ShadowUniforms>(),
            );
        }
        let shadow_ubo_gva =
            unsafe { self.uniforms.shadow_ubo_resources[frame_idx].GetGPUVirtualAddress() };

        // Reflection-probe set (parallax boxes + live count) into this frame's ring
        // CBV; the bindless main pass binds it at root param [11]. A ring (one CBV per
        // frame) so this write never races a prior frame's in-flight GPU read.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &self.probe_set as *const super::probe_uniforms::ProbeSet as *const u8,
                self.probe_set_cbv_ptrs[frame_idx],
                std::mem::size_of::<super::probe_uniforms::ProbeSet>(),
            );
        }

        // Push this frame's skinning matrices into the per-frame joint buffers
        // before the skinned shadow + main passes read them. No-op when no
        // SkinnedMesh is declared.
        self.upload_joint_matrices(frame_idx);

        // GPU-driven cull gating; matches the inner check in
        // encode_main_pass's bindless branch. When on, the host-side
        // per-frame object buffer rebuild runs inline here (mapped-memory
        // CPU work, mirrors Vulkan's pattern) and the pre-graph picks up
        // a `PassId::Cull` node that writes the indirect command buffer
        // ahead of Main.
        let bindless_cull_enabled = self.cull.main_bindless_pso.is_some() && self.cull_count() > 0;
        // Skipped while the world is hidden behind an opaque menu: the masked
        // graph drops the Cull pass and Main runs as a bare clear, so this
        // per-object buffer rebuild would feed nothing.
        if !world_hidden && bindless_cull_enabled {
            self.build_object_buffer(frame_idx);
        }

        // Per-cluster LOD bucketing + instance-buffer upload. Has to
        // happen BEFORE `execute_graph` because SSAO / SSR / TAA-velocity
        // pre-passes (which run earlier than main in the graph) read the
        // same per-frame upload buffer; with LOD bucketing the byte
        // layout depends on `cam_pos`, so they need the **current**
        // frame's data, not previous-frame leftovers. No-op when no
        // instanced cluster declared LOD alternates (every cluster
        // collapses to a single LOD0 bucket containing all instances,
        // same byte order as the legacy single-draw path).
        if !world_hidden && !self.instanced.clusters.is_empty() {
            self.build_instance_upload(frame_idx, cam_pos);
        }

        // Per-frame seed inputs for the shared backend-agnostic frame builder
        // ([gfx/render_graph/frame.rs](../../gfx/render_graph/frame.rs)).
        // Every backend (Metal / Vulkan / DirectX) now drives the same builder.
        // FSR3 owns its own temporal accumulation, so when the upscaler
        // is built the engine bypasses the TAA pass entirely. The G-buffer
        // pre-pass still runs; FSR consumes its motion vectors.
        let upscale_on = self.upscale.backend.is_some();
        let taa_on = self.taa.is_some() && !upscale_on;
        let seed_inputs = FrameGraphInputs {
            shadow_enabled: !self.shadow.dsvs.is_empty(),
            shadow_map_size: self.shadow.map_size,
            hdr_width: width,
            hdr_height: height,
            hdr_sample_count: self.hdr.msaa_samples,
            bindless_cull_enabled,
            bloom_enabled: self.post_process.bloom_intensity > 0.0,
            velocity_enabled: taa_on || upscale_on,
            taa_enabled: taa_on,
            // Only the SSR *resolve* is gated here; `self.ssr` is also `Some`
            // for a SSGI-only world (which reuses the pre-pass G-buffer), so
            // key off the resolve half rather than the bundle's presence.
            ssr_enabled: self.ssr.as_ref().is_some_and(|s| s.resolve.is_some()),
            // The SSR depth + normal pre-pass feeds SSR resolve *and* SSGI, so
            // `SsrResources` (and thus this flag) is on whenever either is.
            ssr_prepass_enabled: self.ssr.is_some(),
            auto_exposure_enabled: self.auto_exposure.resources.is_some(),
            particles_enabled: self.particle.resources.is_some()
                && !self.particle.records.is_empty(),
            // Gated on both the resources (built at init when the world declared
            // a VolumetricFog) and the live settings, so runtime
            // `update_fog_settings(None)` drops the FogFroxel + Fog passes from
            // the graph entirely. Mirrors Vulkan + Metal's `pipeline && settings`
            // gate; without the settings half a settings-None frame would still
            // emit the (bailing) Fog pass, and the graph-driven froxel-volume
            // consumer barrier would transition the volume with no encoder to
            // reset it.
            fog_enabled: self.fog.resources.is_some() && self.fog.settings.is_some(),
            decals_enabled: self.decal.state.is_some(),
            ssao_enabled: self.ssao.resources.is_some(),
            // FSR3 upscaling (runs at native resolution as a TAA
            // replacement). `Some` only when the FFX DLL loaded;
            // otherwise the renderer silently falls back to TAA-or-none.
            // When on, the engine sets `taa_enabled = false` below
            // (FSR's temporal accumulation supersedes TAA) and
            // `velocity_enabled = true` (FSR needs motion vectors).
            upscale_enabled: self.upscale.backend.is_some(),
            // Generic translucent pass: on when the world declared visible
            // `GlassPanel`s. The shared builder then seeds the Transparent
            // node and the executor draws the glass over the post-SSR scene.
            transparent_enabled: self.transparent_enabled(),
            // Raymarched SDF volumes.
            // Gated on whether any `.hlsl`-payload `SdfVolume` survived
            // the init filter and is currently visible. Metal-only
            // (`.metal`) volumes degrade with a logged warning at init
            // and never flip this flag on the DX backend.
            raymarch_enabled: self.raymarch_enabled(),
            // Two-pass Hi-Z occlusion: inserts HizBuild / Cull2 / Main2 after
            // Main when the world requested `occlusion_two_pass` and the bindless
            // GPU-cull path + phase-2 pipeline are live. `two_pass_occlusion_active`
            // is the single gate the executor's phase-2 arms + the Main resolve
            // skip share, so the graph shape matches what the executor dispatches.
            two_pass_occlusion_enabled: self.two_pass_occlusion_active(),
            // Screen-space global illumination: inserts the `Ssgi` RMW node
            // after `Raymarch` and before `Decals`. On when the world selected
            // `indirect_lighting: ssgi` (which also forces the SSR pre-pass on
            // above so the gather has a G-buffer).
            ssgi_enabled: self.ssgi.is_some(),
            // Hardware ray-traced reflections (DXR inline `RayQuery`). On when
            // the world authored `ray_traced_reflections`, the GPU supports the
            // DXR tier, and the DXC compile + acceleration-structure build
            // succeeded (`rt_reflections` + `rt_accel` both live). The shared
            // builder then seeds `RtReflections` in the SsrResolve slot and omits
            // `SsrResolve`; otherwise it falls back to SSR.
            rt_reflections_enabled: self.rt_reflections_active(),
            // One jittered traversal writes normal+depth, roughness, and motion
            // for every screen-space consumer, replacing the separate SSR /
            // SSAO / velocity geometry pre-passes. On whenever the G-buffer
            // resources exist (any of SSR / SSGI / SSAO / TAA / FSR enabled).
            unified_gbuffer_prepass: self.gbuffer.is_some(),
            // An opaque menu backdrop hides the scene: the shared builder masks
            // every world pass off, collapsing to Main (a bare clear, fed the
            // empty scene below) -> Composite (presents the overlay).
            world_hidden,
        };

        // Compute the camera VPs the main + velocity passes consume.
        let proj = perspective(fov_y_radians, aspect, near, far);
        // Un-jittered camera VP, fed to the velocity pre-pass so the stored
        // motion vector is free of the sub-pixel projection jitter.
        let cur_vp = mat4_mul(proj, self.view_matrix);
        // When TAA is on, offset the projection by a sub-pixel Halton jitter so
        // the accumulation has fresh sample positions each frame. The jitter is
        // applied to the z-coefficients of clip x/y, so subtracting it shifts
        // post-divide NDC by exactly the jitter amount (clip.w == -view_z) and
        // leaves depth untouched. Mirrors the jitter in vulkan/draw.rs.
        //
        // When FSR3 upscale is on instead, the sub-pixel offset comes
        // from FFX's prescribed phase sequence (tuned to FSR's temporal
        // kernel), not Halton-2/3; Halton phases would mis-align with
        // FSR's accumulation and produce blur or ghosting. The offset
        // is queried from the upscaler once per frame and stashed in
        // `upscale_jitter` so the Upscale arm of the executor sees the
        // same value the projection was jittered with.
        let render_proj = match (&self.upscale.backend, &self.taa) {
            (Some(up), _) => {
                // FFX returns jitter in input-pixel coordinates
                // (each axis roughly [-0.5, 0.5]). The projection
                // offset is `(2 * jitter / extent)` in NDC, same
                // conversion as the Halton path below.
                let frame_idx = self.taa.as_ref().map(|t| t.frame.get()).unwrap_or(0);
                let [jx_px, jy_px] = up.jitter_offset(frame_idx);
                self.upscale.jitter.set([jx_px, jy_px]);
                let jx = jx_px * 2.0 / width.max(1) as f32;
                let jy = jy_px * 2.0 / height.max(1) as f32;
                let mut p = proj;
                p[2][0] -= jx;
                p[2][1] -= jy;
                p
            }
            (None, Some(taa)) => {
                let idx = taa.frame.get() % 8 + 1;
                let jx = (halton(idx, 2) - 0.5) * 2.0 / width.max(1) as f32;
                let jy = (halton(idx, 3) - 0.5) * 2.0 / height.max(1) as f32;
                let mut p = proj;
                p[2][0] -= jx;
                p[2][1] -= jy;
                p
            }
            (None, None) => proj,
        };
        let vp_mat = mat4_mul(render_proj, self.view_matrix);

        // Upload this frame's view UBO.
        // Fade the forward probe specular only when a resolve will actually
        // composite its reflection over this scene. `reflection_resolve_active`
        // alone matches the resolve gating, but require the composite target too
        // so the fade can never zero a reflection that was never re-added.
        let reflections_enabled =
            if self.reflection_composite.is_some() && self.reflection_resolve_active() {
                1.0
            } else {
                0.0
            };
        let view_uni = ViewUniforms {
            vp: vp_mat,
            view_mat: self.view_matrix,
            elapsed,
            reflections_enabled,
            cam_x: cam_pos[0],
            cam_y: cam_pos[1],
            cam_z: cam_pos[2],
            prefilter_mip_count: self.env_map.prefilter_mip_count as f32,
            _ep0: 0.0,
            _ep1: 0.0,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &view_uni as *const ViewUniforms as *const u8,
                self.uniforms.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<ViewUniforms>(),
            );
        }

        // BVH-based frustum cull. RefCell::replace swaps out the persistent
        // scratch buffer so its heap allocation is reused across frames; it's
        // put back below before we return Ok (error path loses capacity, fine
        // since record_frame errors are exceptional). RefCell because
        // record_frame is &self (matches the deferred_buffers pattern).
        let frustum = crate::gfx::frustum::Frustum::from_view_projection(vp_mat);
        let mut visible = self.visible_scratch.replace(Vec::new());
        visible.clear();
        // Left empty while the world is hidden behind an opaque menu so the
        // Main pass draws nothing behind it.
        if !world_hidden {
            self.cull_bvh
                .query(&frustum, cam_pos, |idx| visible.push(idx));
            visible.sort_unstable();
            visible.extend_from_slice(&self.always_draw);
        }

        let (view_gva, light_gva) = unsafe {
            (
                self.uniforms.view_ubo_resources[frame_idx].GetGPUVirtualAddress(),
                self.uniforms.light_ubo.GetGPUVirtualAddress(),
            )
        };

        // Scene source for the bloom prefilter + the composite. Priority:
        //   1. FSR3 upscaler output (when temporal upscaling is on; the
        //      graph excludes TaaResolve in that case so the TAA history
        //      slots are never written; sampling them would yield black).
        //   2. TAA history output (when TAA is on and upscale is off).
        //   3. SSR resolve / raw HDR fallback via `scene_srv_for_post`.
        // The handle value is stable across the TAA dispatch:
        // `taa.output_index()` is `frame % 2`, and the frame counter is
        // only bumped *after* Composite, so reading the index before the
        // executor runs GBufferPrepass + TaaResolve gives the same pointer the
        // encoders write into / sample from.
        let scene_srv = if self.upscale.backend.is_some() {
            self.scene_srv_for_post()
        } else {
            match &self.taa {
                Some(taa) => taa.history_srv_gpu[taa.output_index()],
                None => self.scene_srv_for_post(),
            }
        };

        // Single graph dispatch: every render-stack pass plus the Composite
        // presenter routed through one `execute_graph` call. The graph
        // shape lives in the shared
        // [gfx/render_graph/frame.rs::build_frame_graph](../../gfx/render_graph/frame.rs);
        // see [directx/graph_exec.rs](../graph_exec.rs) for the DirectX
        // executor that routes each `PassId` to its `encode_*` method.
        let frame_graph =
            build_frame_graph(&seed_inputs).map_err(|e| format!("frame-graph compile: {e}"))?;
        let frame_params = GraphFrameParams {
            cmd: end_cmd,
            frame_idx,
            back_buffer,
            back_buffer_rtv,
            text_calls,
            world_hidden,
            scene_srv,
            width,
            height,
            output_width,
            output_height,
            cam_pos,
            shadow_ubo_gva,
            view_gva,
            light_gva,
            vp_mat,
            cur_vp,
            frustum: &frustum,
            fov_y_radians,
            aspect,
            elapsed,
            near,
            far,
            visible: &visible,
        };
        let pass_cmd_lists = self.execute_graph(&frame_graph, &frame_params)?;

        // Hi-Z pyramid: read this frame's main depth buffer, write the
        // mip chain that *next* frame's cull dispatch consults. Runs inline
        // on the outer "end" cmd list, after Composite (encoded by
        // `execute_graph`) but before the per-frame restore barriers below.
        // That placement keeps the build off the worker fan-out (so it
        // does not need its own per-pass cmd-list slot) while still
        // executing after the main pass has written depth. A no-op when no
        // Hi-Z resource was built (bindless cull pipeline not active).
        // The encoder owns its own DEPTH_WRITE ↔ NPSR transitions.
        if self.cull.hiz.is_some() {
            self.encode_hiz_build(end_cmd);
            self.cull.hiz_valid.set(true);
        }
        // Capture the un-jittered view-projection for the next frame's cull
        // dispatch. Stored regardless of whether Hi-Z is on so the matrix
        // is always current when it later gets switched on by a hot-reload
        // or a re-init.
        self.cull.prev_view_proj.set(cur_vp);

        // Restore the HDR target → RENDER_TARGET for the next frame. The MSAA
        // path already did this after the resolve; only the MSAA-off path,
        // which left it in PIXEL_SHADER_RESOURCE, still needs it. Lands on
        // the "end" cmd list (after Composite) so it executes after every
        // per-pass cmd list above.
        if self.hdr.resolve.is_none() {
            let hdr_to_rt = transition_barrier(
                &self.hdr.color,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            unsafe { end_cmd.ResourceBarrier(&[hdr_to_rt]) };
        }

        // The shadow map rests sampled between frames; next frame's Shadow
        // producer barrier (graph-driven) performs the PIXEL_SHADER_RESOURCE ->
        // DEPTH_WRITE reset, so no inline end-of-frame restore is needed.

        // Advance the TAA jitter sequence (which also validates history for the
        // next frame). TAA-specific, so gated on `self.taa`.
        if let Some(taa) = &self.taa {
            taa.frame.set(taa.frame.get().wrapping_add(1));
        }
        // Snapshot this frame's un-jittered VP + per-draw transforms so next
        // frame's G-buffer pre-pass can derive motion vectors. Owned by the
        // G-buffer now (decoupled from TAA, so FSR-without-engine-TAA also gets
        // correct motion).
        if let Some(gb) = &self.gbuffer {
            *gb.prev_view_proj.borrow_mut() = cur_vp;
            let mut prev_models = gb.prev_models.borrow_mut();
            prev_models.clear();
            prev_models.extend(self.draw_objects.iter().map(|o| o.model));
        }

        self.visible_scratch.replace(visible);
        Ok(pass_cmd_lists)
    }
}

// Upload LightUniforms to the shared light constant buffer.
pub(super) fn upload_light_uniforms(
    light_ubo: &ID3D12Resource,
    lu: &LightUniforms,
) -> Result<(), String> {
    let size = std::mem::size_of::<LightUniforms>();
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    unsafe { light_ubo.Map(0, None, Some(&mut ptr)) }.map_err(|e| format!("map light ubo: {e}"))?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            lu as *const LightUniforms as *const u8,
            ptr as *mut u8,
            size,
        );
        light_ubo.Unmap(0, None);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ViewUniforms must match the `ViewBlock` cbuffer (b1) in the main-pass
    // shaders: two column-major float4x4 then elapsed/pad and the camera
    // position as three scalars, prefilter mip count, and two end pads
    // (160 B total).
    #[test]
    fn view_uniforms_layout_matches_hlsl() {
        assert_eq!(std::mem::size_of::<ViewUniforms>(), 160);
        assert_eq!(std::mem::offset_of!(ViewUniforms, vp), 0);
        assert_eq!(std::mem::offset_of!(ViewUniforms, view_mat), 64);
        assert_eq!(std::mem::offset_of!(ViewUniforms, elapsed), 128);
        assert_eq!(std::mem::offset_of!(ViewUniforms, reflections_enabled), 132);
        assert_eq!(std::mem::offset_of!(ViewUniforms, cam_x), 136);
        assert_eq!(std::mem::offset_of!(ViewUniforms, cam_y), 140);
        assert_eq!(std::mem::offset_of!(ViewUniforms, cam_z), 144);
        assert_eq!(std::mem::offset_of!(ViewUniforms, prefilter_mip_count), 148);
        assert_eq!(std::mem::offset_of!(ViewUniforms, _ep0), 152);
        assert_eq!(std::mem::offset_of!(ViewUniforms, _ep1), 156);
    }
}
