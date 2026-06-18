// src/gfx/render_graph/frame.rs
//
// Per-frame graph builder. On Metal, every pass that ran inline
// through `draw_frame` now dispatches through a single
// `build_frame_graph()` → `execute_graph()` pair.
//
// The frame builder declares conditional passes based on the
// `FrameGraphInputs` struct (one bool per gated pass). Read / write
// declarations on each pass let the compile pass derive:
//
//   * Execution order (toposort over RAW / WAW / WAR edges, ties broken
//     by declaration order).
//   * Per-pass barriers (`pass.barriers_before` per resource state
//     transition). Metal mostly ignores these (Apple GPUs handle most
//     hazards implicitly); the Vulkan / DirectX executors emit
//     `vkCmdPipelineBarrier` / `D3D12_RESOURCE_BARRIER` from them.
//   * Transient resource lifetimes (`PassRange` per resource), the
//     aliasing input.
//
// Resources split into two origins. `import_texture` = engine-owned: the
// resource outlives the frame (the cross-frame shadow map, the `scene_pre_taa`
// / `scene_color` aliases read at v0, the froxel volume, the cross-frame Hi-Z
// pyramid) and the backend always owns its GPU object. `create_texture` =
// transient: single-frame intermediates (hdr intermediates excepted for now)
// the aliasing planner ([`super::alias`]) may pack into shared physical memory,
// since their `[first, last]` lifetimes are disjoint. In practice only
// `ao_output` and `bloom_top` are independently poolable today; the other
// `create_texture` intermediates fold into the long-lived gbuffer MRT
// (`velocity`, `ssr_gbuffer`) or are themselves long-lived (`gbuffer`), so a
// backend pool leaves them backend-owned. The descs are no longer
// documentation-only: the planner sizes each transient from its desc. Until a
// backend realises the plan, every resource is still backend-owned and bound
// from context fields exactly as before; the origin only marks aliasing
// candidacy and has no effect on pass order or barriers.

use crate::gfx::render_types::NUM_SHADOW_CASCADES;

use super::{
    BufferDesc, BufferUsage, CompiledGraph, GraphBuilder, GraphError, PassId, PassKind,
    PixelFormat, TextureDesc, TextureHandle, TextureSize, TextureUsage,
};

// Per-frame inputs that gate conditional passes. Built by `draw_frame`
// from the live `MtlContext` state and consumed by `build_frame_graph`
// so the conditional-inclusion decisions made here match what the
// executor will dispatch.
#[derive(Copy, Clone, Debug)]
pub struct FrameGraphInputs {
    // `true` when a `ShadowStage` is in the world (i.e. the backend's
    // shadow pipeline + cascade uniforms are live). Skips the Shadow
    // pass when false rather than relying on the encoder's early
    // return, so the compiled graph reflects what actually runs.
    pub shadow_enabled: bool,
    // Per-cascade slice dimensions of the shadow-map array texture.
    // Carried so the imported `shadow_map` resource carries its real
    // shape for aliasing; ignored by the executor.
    pub shadow_map_size: u32,
    // Pixel dimensions of the HDR off-screen targets the Main pass
    // writes (and the post stack consumes). Carried for aliasing;
    // ignored by the executor.
    pub hdr_width: u32,
    pub hdr_height: u32,
    // MSAA sample count of the HDR colour + depth attachments, typically
    // 4. The resolve target is single-sample regardless.
    pub hdr_sample_count: u32,
    // `true` when GPU-driven cull is going to run this frame, i.e. the
    // bindless static path is configured AND there is geometry to cull
    // AND the per-frame `object_buffer` / `draw_args` buffers built. The
    // graph adds the Cull compute pass and the Main read-edge from
    // `draw_args` only when this is on; otherwise Main draws via the
    // legacy per-draw path with no graph dependency on the cull output.
    pub bindless_cull_enabled: bool,
    // `true` when the auto-exposure compute pipelines are built (i.e.
    // the world declared `PostProcessConfig.auto_exposure`). The graph
    // appends an `AutoExposure` compute pass that reads the Main pass's
    // `hdr_resolve_v1` (pre-decoration) and writes the histogram +
    // readback buffer. The compile pass's WAR step pins AutoExposure
    // before the first hdr_resolve post-Main writer (Decals or Fog or
    // ParticlesDraw) so AutoExposure samples the un-decorated scene.
    pub auto_exposure_enabled: bool,
    // `true` when `PostProcessConfig.bloom_intensity > 0.0`. The graph
    // adds a `Bloom` pass that thresholds / downsamples / upsamples the
    // post-TAA scene into the bloom mip chain; Composite reads the
    // bloom output so the toposort orders Bloom before Composite.
    pub bloom_enabled: bool,
    // `true` when TAA is on (the velocity pre-pass only runs as part of
    // the TAA stack). The graph adds a `Velocity` render pass that
    // writes the per-pixel motion-vector buffer TaaResolve consumes;
    // TaaResolve declares the read so Velocity → TaaResolve is explicit.
    pub velocity_enabled: bool,
    // `true` when TAA is on. The graph adds a `TaaResolve` render pass
    // that reads the pre-TAA scene (SSR resolve output or hdr_resolve)
    // and writes the imported `scene_color` Bloom + Composite consume.
    pub taa_enabled: bool,
    // `true` when SSR is on. The graph adds an `SsrResolve` render pass
    // that reads the post-decoration `hdr_resolve` and writes the
    // imported `scene_pre_taa` texture. When TAA is also on, TaaResolve
    // reads the post-SsrResolve `scene_pre_taa` version. When TAA is
    // off, scene_pre_taa is aliased to scene_color (same GPU object) by
    // the engine binding, so Bloom + Composite see the SsrResolve
    // output via declaration-order tie-breaking.
    pub ssr_enabled: bool,
    // `true` when the particle system is going to run this frame:
    // `particle_pipelines` built AND at least one live emitter. The
    // graph adds a `ParticlesDraw` render pass that blend-writes
    // `hdr_resolve`. The bundled ParticlesSim compute sub-pass runs
    // inside the same `encode_particles` call so it keeps its per-pass
    // timing slot without needing its own graph node.
    pub particles_enabled: bool,
    // `true` when a `VolumetricFog` is in the world. The graph adds a
    // `Fog` render pass between Decals and ParticlesDraw on the
    // hdr_resolve RMW chain.
    pub fog_enabled: bool,
    // `true` when at least one `Decal` is in the world AND the decal
    // pipeline is built. The graph adds a `Decals` render pass at the
    // head of the hdr_resolve post-Main RMW chain.
    pub decals_enabled: bool,
    // `true` when the SSR pre-pass should run; matches
    // `self.ssr_settings.is_some()`. The graph adds an `SsrPrepass`
    // render pass that writes the imported `ssr_gbuffer` texture;
    // SsaoBlur reads it when SSAO is also on (G-buffer sharing).
    pub ssr_prepass_enabled: bool,
    // `true` when SSAO should run; matches
    // `self.ssao_settings.is_some()`. The graph adds an `SsaoBlur`
    // render pass that dispatches the bundled `encode_ssao` (which
    // internally encodes SsaoPrepass + SsaoKernel + SsaoBlur). SsaoBlur
    // writes `ao_output`; Main reads it. SsaoPrepass + SsaoKernel
    // stay as timing-only PassIds (same pattern as ParticlesSim).
    pub ssao_enabled: bool,
    // `true` when temporal upscaling is on (e.g. MetalFX on Metal). The
    // graph adds an `Upscale` pass between the post-SSR scene and the
    // Bloom + Composite stack that reads `scene_pre_taa` + `velocity`
    // and writes the imported `scene_color` at output resolution. When
    // this is on, `TaaResolve` is *not* added: the upscaler does
    // temporal accumulation itself, so adding TAA on top would
    // double-temporal. `velocity_enabled` should still be on (the
    // scaler consumes motion vectors); the engine layer is responsible
    // for keeping the two flags in sync.
    pub upscale_enabled: bool,
    // `true` when at least one transparent / translucent draw is in the
    // world (water, glass, ...). The graph adds a `Transparent` render
    // pass after `SsrResolve` and before `TaaResolve` / `Upscale` that
    // reads the latest scene-pre-taa colour + main depth and
    // alpha-blends translucent geometry back-to-front into the same
    // target. The pass aggregates N draws, each owns its own
    // pipeline + descriptor set, the executor receives the sorted list
    // at encode time.
    pub transparent_enabled: bool,
    // `true` when at least one visible `SdfVolume` is in the world AND
    // the backend's raymarch pipeline is live. The graph adds a
    // `Raymarch` render pass between `AutoExposure` and `Decals` on the
    // hdr_resolve RMW chain: it reads the head of the chain (so
    // AutoExposure samples the pre-raymarch scene) and writes the next
    // version that Decals then bumps further. The pass also RMWs the
    // main depth attachment so subsequent passes see raymarched
    // surfaces' depth, but depth is not declared as a graph edge here
    // (same rationale as Transparent / Decals / Fog: the executor
    // binds the live depth attachment at encode time).
    pub raymarch_enabled: bool,
    // `true` when two-pass Hi-Z occlusion culling is requested
    // (`PostProcessConfig.occlusion_two_pass`) AND the bindless GPU-cull
    // path is active this frame. Only meaningful alongside
    // `bindless_cull_enabled`; the builder ANDs the two so a world that
    // asks for two-pass without a bindless shader simply gets the
    // single-pass path. When on, the graph inserts `HizBuild` → `Cull2`
    // → `Main2` between `Main` and the post-decoration chain: `HizBuild`
    // rebuilds the Hi-Z pyramid from phase-1 depth, `Cull2` re-tests the
    // objects phase-1 cull marked occluded, and `Main2` redraws the
    // disoccluded survivors. `Main2`'s hdr_resolve write becomes the head
    // of the post chain so AutoExposure / Decals / Fog / SSR see the
    // combined two-pass result. Metal only today; the other backends keep
    // this false.
    pub two_pass_occlusion_enabled: bool,
    // `true` when screen-space global illumination is on
    // (`PostProcessConfig.indirect_lighting == "ssgi"`); matches
    // `self.ssgi_settings.is_some()`. The graph inserts an `Ssgi` render pass
    // on the hdr_resolve RMW chain right after `Raymarch` and before `Decals`:
    // it reads the head of the chain (the lit scene, its bounce-radiance
    // source) and writes the next version with the gathered indirect term
    // additively composited in. SSGI reuses the SSR pre-pass G-buffer for
    // normals + depth, so `ssr_prepass_enabled` is forced on whenever this is
    // set. Metal only today; the other backends keep this false.
    pub ssgi_enabled: bool,
    // `true` when hardware ray-traced reflections are live (RT requested + GPU
    // supports it + the scene acceleration structure built); matches
    // `self.rt_accel.is_some()`. The graph adds an `RtReflections` render pass
    // in the *same slot* as `SsrResolve` (reads the post-decoration
    // `hdr_resolve`, writes `scene_pre_taa`). RT *takes precedence* over SSR: a
    // world may enable both, and where this is set the builder inserts
    // `RtReflections` and omits `SsrResolve`, so at most one of them is in the
    // graph. Like SSGI it reuses the SSR depth + normal + roughness pre-pass,
    // so `ssr_prepass_enabled` is forced on whenever this is set. Metal only
    // today; the other backends keep this false.
    pub rt_reflections_enabled: bool,
    // `true` to collapse the SSR / SSAO / velocity geometry pre-passes into a
    // single `GBufferPrepass` node that writes view-space normal+depth,
    // roughness, and motion in one traversal: every consumer reads that one
    // output. When set, the builder emits `GBufferPrepass` (gated on any of
    // `ssr_prepass_enabled || ssao_enabled || velocity_enabled`) instead of the
    // separate `SsrPrepass` + `Velocity` nodes. Metal only today; the other
    // backends keep this false and emit their separate prepasses.
    pub unified_gbuffer_prepass: bool,
}

impl FrameGraphInputs {
    // Every gated pass off, at a representative resolution. A neutral base a
    // caller can flip individual flags on, e.g. to plan a worst-case graph for
    // transient-memory allocation (where the allocation must cover every
    // per-frame graph, not just the current frame's active passes).
    pub fn all_off() -> Self {
        FrameGraphInputs {
            shadow_enabled: false,
            shadow_map_size: 2048,
            hdr_width: 1280,
            hdr_height: 720,
            hdr_sample_count: 1,
            bindless_cull_enabled: false,
            auto_exposure_enabled: false,
            bloom_enabled: false,
            velocity_enabled: false,
            taa_enabled: false,
            ssr_enabled: false,
            particles_enabled: false,
            fog_enabled: false,
            decals_enabled: false,
            ssr_prepass_enabled: false,
            ssao_enabled: false,
            upscale_enabled: false,
            transparent_enabled: false,
            raymarch_enabled: false,
            two_pass_occlusion_enabled: false,
            ssgi_enabled: false,
            rt_reflections_enabled: false,
            unified_gbuffer_prepass: false,
        }
    }
}

// Build the full per-frame render graph. Conditional passes are
// included based on the `inputs` flags. The compile pass derives
// execution order, per-pass barriers, and resource lifetimes via
// RAW + WAW + WAR edges over the version-chained read / write
// declarations.
//
// Order (with all flags on):
//
// ```text
// Cull → SsrPrepass → SsaoBlur → Shadow → Main → AutoExposure
//   → Raymarch → Velocity → Decals → Fog → ParticlesDraw → SsrResolve
//   → Transparent → TaaResolve → Bloom → Composite
// ```
//
// The hdr_resolve version chain (Main writes v1, AutoExposure reads
// v1 (WAR-pinned before subsequent writers), Decals → v2, Fog → v3,
// ParticlesDraw → v4, SsrResolve reads v4) is the spine that
// orders the bulk of the post stack. scene_pre_taa / scene_color /
// bloom_top each have their own short version chains that branch off
// the spine. Transparent extends the scene_pre_taa chain by one
// version when enabled (RMW after SsrResolve), so TaaResolve / Upscale
// pick up translucent geometry as part of temporal accumulation.
//
// When `two_pass_occlusion_enabled` is on the spine gains a phase-2
// prefix: `Cull → Main → HizBuild → Cull2 → Main2 → AutoExposure →
// …`. `Main` writes hdr_resolve v1 / hdr_depth v1; `HizBuild` reads
// the depth and writes the Hi-Z pyramid; `Cull2` reads the pyramid +
// the phase-1 status buffer and writes `draw_args2`; `Main2` RMWs
// hdr_color / hdr_depth / hdr_resolve → v2, and that v2 (not v1)
// becomes the head AutoExposure reads and the RMW chain extends.
pub fn build_frame_graph(inputs: &FrameGraphInputs) -> Result<CompiledGraph, GraphError> {
    let mut b = GraphBuilder::new();

    // Engine-owned imports the Main pass writes into. hdr_resolve is
    // also written by Decals / Fog / ParticlesDraw and read by
    // AutoExposure / SsrResolve, so its version chain is the longest.
    let hdr_color = b.import_texture("hdr_color", hdr_color_desc(inputs));
    let hdr_depth = b.import_texture("hdr_depth", hdr_depth_desc(inputs));
    let hdr_resolve = b.import_texture("hdr_resolve", hdr_resolve_desc(inputs));

    // Two-pass occlusion only applies when the bindless GPU-cull path is
    // active: Hi-Z occlusion rides that path. ANDing here means a world
    // that requests two-pass without a bindless shader falls back to the
    // single-pass path with no orphaned phase-2 nodes.
    let two_pass = inputs.bindless_cull_enabled && inputs.two_pass_occlusion_enabled;

    // Cull (compute) writes the indirect-draw args buffer Main consumes
    // through executeCommandsInBuffer. Under two-pass occlusion it also
    // writes a per-object status buffer (drawn / hi-z-candidate / culled)
    // that `Cull2` reads to decide which phase-1-occluded objects to
    // re-test against the rebuilt pyramid.
    let (draw_args_v1, cull_status_v1) = if inputs.bindless_cull_enabled {
        // Import both buffers up front: a live `PassBuilder` holds `&mut b`,
        // so the resource declarations have to happen before `add_pass`.
        let draw_args = b.import_buffer("draw_args", draw_args_desc());
        let cull_status = if two_pass {
            Some(b.import_buffer("cull_status", cull_status_desc()))
        } else {
            None
        };
        let mut cull = b.add_pass(PassId::Cull, PassKind::Compute);
        let da = cull.write_buffer(draw_args);
        let cs = cull_status.map(|h| cull.write_buffer(h));
        (Some(da), cs)
    } else {
        (None, None)
    };

    // Unified G-buffer pre-pass (Metal): one node writes the view-space
    // normal+depth / roughness / velocity that SSR, SSAO, SSGI, RT, TAA, and the
    // upscaler all read, replacing the separate SsrPrepass + Velocity nodes. Runs
    // when any of those consumers is on. The other backends keep
    // `unified_gbuffer_prepass` false and emit the two separate nodes below.
    let gbuffer_v1 = if inputs.unified_gbuffer_prepass
        && (inputs.ssr_prepass_enabled || inputs.ssao_enabled || inputs.velocity_enabled)
    {
        let gbuffer = b.create_texture("gbuffer", gbuffer_desc(inputs));
        let mut gb = b.add_pass(PassId::GBufferPrepass, PassKind::Render);
        // When the GPU-driven cull path is active the pre-pass reuses the main
        // pass's per-frame indirect command buffer (camera frustum, same cull
        // output), so it must run after Cull. Reading the cull-produced draw_args
        // buffer pins that ordering in the toposort (a no-op when bindless cull is
        // off, where draw_args_v1 is None). Mirrors the Main pass's edge.
        if let Some(h) = draw_args_v1 {
            gb.read_buffer(h);
        }
        Some(gb.write_texture(gbuffer))
    } else {
        None
    };

    // SSR pre-pass writes the SSR G-buffer; SSAO reads it when both are on (the
    // shared-G-buffer fast path). Under the unified path the merged node above
    // supplies the same normal+depth handle, so this separate node is skipped.
    let ssr_gbuffer_v1 = if let Some(g) = gbuffer_v1 {
        Some(g)
    } else if inputs.ssr_prepass_enabled {
        let ssr_gbuffer = b.create_texture("ssr_gbuffer", ssr_gbuffer_desc(inputs));
        Some(
            b.add_pass(PassId::SsrPrepass, PassKind::Render)
                .write_texture(ssr_gbuffer),
        )
    } else {
        None
    };

    // SSAO bundle writes ao_output. PassId::SsaoBlur is the single
    // graph node for the entire encode_ssao bundle; SsaoPrepass +
    // SsaoKernel keep their per-pass timing slots via inline
    // `pass_timing.attach_render` calls inside encode_ssao but they're
    // not graph nodes (the executor rejects them if mis-added).
    let ao_output_v1 = if inputs.ssao_enabled {
        let ao_output = b.create_texture("ao_output", ao_output_desc(inputs));
        let mut ssao = b.add_pass(PassId::SsaoBlur, PassKind::Render);
        if let Some(h) = ssr_gbuffer_v1 {
            ssao.read_texture(h);
        }
        Some(ssao.write_texture(ao_output))
    } else {
        None
    };

    // Shadow optionally precedes Main and produces the shadow_map
    // handle Main samples. When off, Main does not declare a shadow_map
    // read, mirroring the encoder's `enable_shadows` shader path.
    let shadow_v1 = if inputs.shadow_enabled {
        let shadow_map = b.import_texture("shadow_map", shadow_map_desc(inputs.shadow_map_size));
        Some(
            b.add_pass(PassId::Shadow, PassKind::Render)
                .write_texture(shadow_map),
        )
    } else {
        None
    };

    // Main pass: reads optional shadow_map / draw_args / ao_output; writes
    // the three HDR targets. Captures hdr_resolve_v1 (head of the
    // hdr_resolve RMW chain, the version AutoExposure reads when two-pass
    // is off) and hdr_depth_v1 (the depth HizBuild reduces under two-pass).
    let (hdr_resolve_v1, hdr_depth_v1) = {
        let mut main = b.add_pass(PassId::Main, PassKind::Render);
        if let Some(h) = shadow_v1 {
            main.read_texture(h);
        }
        if let Some(h) = draw_args_v1 {
            main.read_buffer(h);
        }
        if let Some(h) = ao_output_v1 {
            main.read_texture(h);
        }
        let _ = main.write_texture(hdr_color);
        let depth_v1 = main.write_texture(hdr_depth);
        let resolve_v1 = main.write_texture(hdr_resolve);
        (resolve_v1, depth_v1)
    };

    // Two-pass occlusion phase 2: rebuild the Hi-Z pyramid from phase-1
    // depth (HizBuild), re-test the objects phase-1 cull marked occluded
    // (Cull2), and redraw the disoccluded survivors (Main2). Main2 RMWs
    // hdr_color / hdr_depth / hdr_resolve, so its hdr_resolve write becomes
    // the head of the post-decoration chain: AutoExposure and every later
    // RMW pass see the combined phase-1 + phase-2 scene. Without two-pass
    // the head stays at Main's hdr_resolve_v1.
    let hdr_resolve_head = if two_pass {
        // HizBuild (compute): read phase-1 depth, write the Hi-Z pyramid.
        // The depth RAW edge pins it after Main. Imported, not transient: the
        // pyramid is cross-frame persistent (written this frame, sampled by next
        // frame's Cull before it is rewritten) and lives as a multi-mip image the
        // executor owns, so its memory is never reusable by another transient and
        // the aliasing planner must not place it.
        let hiz = b.import_texture("hiz_pyramid", hiz_pyramid_desc(inputs));
        let mut hizb = b.add_pass(PassId::HizBuild, PassKind::Compute);
        hizb.read_texture(hdr_depth_v1);
        let hiz_v1 = hizb.write_texture(hiz);

        // Cull2 (compute): read the rebuilt pyramid + the phase-1 status
        // buffer, write a second indirect-draw-args buffer Main2 consumes.
        let draw_args2 = b.import_buffer("draw_args2", draw_args_desc());
        let mut cull2 = b.add_pass(PassId::Cull2, PassKind::Compute);
        cull2.read_texture(hiz_v1);
        if let Some(cs) = cull_status_v1 {
            cull2.read_buffer(cs);
        }
        let draw_args2_v1 = cull2.write_buffer(draw_args2);

        // Main2 (render): read the phase-2 draw args; RMW hdr_color /
        // hdr_depth / hdr_resolve. The draw_args2 RAW edge pins it after
        // Cull2; the hdr_depth write (WAR vs HizBuild's read) pins it after
        // HizBuild; the hdr_color / hdr_resolve WAW edges pin it after Main.
        let mut main2 = b.add_pass(PassId::Main2, PassKind::Render);
        main2.read_buffer(draw_args2_v1);
        let _ = main2.write_texture(hdr_depth_v1);
        let _ = main2.write_texture(hdr_color);
        main2.write_texture(hdr_resolve_v1)
    } else {
        hdr_resolve_v1
    };

    // AutoExposure (compute) reads the post-main scene (hdr_resolve_head:
    // Main2's output under two-pass, Main's otherwise). The compile pass's
    // WAR step pins it before the first hdr_resolve writer that bumps the
    // next version (Raymarch / Decals / Fog / ParticlesDraw), so
    // AutoExposure samples the un-decorated scene even though the GPU
    // texture object is the same one those passes later blend-write.
    if inputs.auto_exposure_enabled {
        b.add_pass(PassId::AutoExposure, PassKind::Compute)
            .read_texture(hdr_resolve_head);
    }

    // Velocity (render) writes the per-pixel motion-vector buffer TaaResolve /
    // Upscale consume. The read edge from those passes pins it ahead of them in
    // the toposort. Under the unified path the merged G-buffer node already
    // carries velocity, so TAA / Upscale read that handle and this separate node
    // is skipped.
    let velocity_v1 = if let Some(g) = gbuffer_v1 {
        Some(g)
    } else if inputs.velocity_enabled {
        let velocity = b.create_texture("velocity", velocity_desc(inputs));
        Some(
            b.add_pass(PassId::Velocity, PassKind::Render)
                .write_texture(velocity),
        )
    } else {
        None
    };

    // hdr_resolve post-Main RMW chain: Raymarch → Decals → Fog →
    // ParticlesDraw, each blend- or opaque-writing on top of the
    // previous version. The handle walks forward through `h` so each
    // write picks up the latest version, giving the compile pass clean
    // WAW edges to derive the chain order. Raymarch slots first so its
    // depth+colour write is visible to every later post-decoration
    // pass; AutoExposure's WAR-read on hdr_resolve_head pins it before
    // Raymarch (so SDF brightness doesn't skew exposure for the same
    // frame), matching the doc's chosen one-frame-lag trade-off.
    let mut h = hdr_resolve_head;
    if inputs.raymarch_enabled {
        let mut rm = b.add_pass(PassId::Raymarch, PassKind::Render);
        rm.read_texture(h);
        h = rm.write_texture(h);
    }
    // SSGI reads the lit scene (its bounce-radiance source) and RMWs the
    // gathered + denoised indirect term back in. Slots right after Raymarch so
    // it can bounce raymarched surfaces too, and before Decals / Fog /
    // Particles so those decorations layer on top of the indirect light.
    // AutoExposure's WAR-read on hdr_resolve_head pins it ahead of SSGI, so the
    // added bounce doesn't skew the same frame's exposure (the same one-frame
    // trade-off Raymarch documents).
    if inputs.ssgi_enabled {
        let mut ssgi = b.add_pass(PassId::Ssgi, PassKind::Render);
        ssgi.read_texture(h);
        h = ssgi.write_texture(h);
    }
    if inputs.decals_enabled {
        h = b
            .add_pass(PassId::Decals, PassKind::Render)
            .write_texture(h);
    }
    if inputs.fog_enabled {
        // FogFroxel (compute) populates the 3D scatter/transmittance
        // volume the Fog fragment shader samples. The post-write handle
        // (`froxel_v1`) is what Fog reads: that gives the compile pass
        // a clean RAW edge so FogFroxel runs before Fog in the toposort.
        // All three backends implement the froxel path; the Fog render
        // pass trilinear-samples the volume by (screen_uv, view_z).
        let froxel_v0 = b.import_texture("fog_froxel_volume", froxel_volume_desc(inputs));
        let froxel_v1 = b
            .add_pass(PassId::FogFroxel, PassKind::Compute)
            .write_texture(froxel_v0);
        let mut fog_pass = b.add_pass(PassId::Fog, PassKind::Render);
        fog_pass.read_texture(froxel_v1);
        h = fog_pass.write_texture(h);
    }
    if inputs.particles_enabled {
        h = b
            .add_pass(PassId::ParticlesDraw, PassKind::Render)
            .write_texture(h);
    }
    let hdr_resolve_cur = h;

    // scene_pre_taa is imported when SsrResolve writes to it,
    // Transparent reads + writes it, or TaaResolve / Upscale reads
    // from it. SsrResolve reads the post-Particles hdr_resolve when
    // both are on. Transparent RMWs whatever the latest scene-pre-taa
    // version is (SsrResolve's output when SSR is on, the imported v0,
    // engine-aliased to hdr_resolve, when SSR is off). Without SSR
    // and without transparency, scene_pre_taa stays at v0 and
    // TaaResolve / Upscale read the imported handle (covered by the
    // imported-v0 producer rule).
    let need_scene_pre_taa = inputs.ssr_enabled
        || inputs.rt_reflections_enabled
        || inputs.transparent_enabled
        || inputs.taa_enabled
        || inputs.upscale_enabled;
    let scene_pre_taa_cur = if need_scene_pre_taa {
        let scene_pre_taa = b.import_texture("scene_pre_taa", scene_color_desc(inputs));
        // SsrResolve and RtReflections occupy the same slot: both read the
        // post-decoration hdr_resolve and write scene_pre_taa. Hardware RT
        // *takes precedence* over SSR: a world can enable both (RT on the
        // backend / GPU that supports it, SSR as the cross-backend fallback),
        // and where RT is live the builder picks it and omits SsrResolve. Only
        // one of the two is ever inserted.
        let mut current = if inputs.rt_reflections_enabled {
            let mut rt = b.add_pass(PassId::RtReflections, PassKind::Render);
            rt.read_texture(hdr_resolve_cur);
            rt.write_texture(scene_pre_taa)
        } else if inputs.ssr_enabled {
            let mut ssr = b.add_pass(PassId::SsrResolve, PassKind::Render);
            ssr.read_texture(hdr_resolve_cur);
            ssr.write_texture(scene_pre_taa)
        } else {
            scene_pre_taa
        };
        if inputs.transparent_enabled {
            let mut trans = b.add_pass(PassId::Transparent, PassKind::Render);
            // Read hdr_resolve_cur to pin Transparent after the entire
            // post-decoration hdr_resolve chain (Main → Decals → Fog →
            // ParticlesDraw). When SSR is on this is redundant with the
            // scene_pre_taa edge below; when SSR is off it's the only
            // edge ordering Transparent after Main. Depth is *not*
            // declared here; decals / fog don't declare it either,
            // and reading the imported (v0) handle would create a WAR
            // edge pinning Transparent before Main (cycle). The
            // executor binds the live depth attachment at encode time.
            trans.read_texture(hdr_resolve_cur);
            // RMW the latest scene-pre-taa version. The read declares
            // the sample dependency (translucents sample the resolved
            // scene for refraction); the write produces the new
            // blended version downstream passes consume.
            trans.read_texture(current);
            current = trans.write_texture(current);
        }
        current
    } else {
        TextureHandle::INVALID
    };

    // scene_color is the engine-owned output the post-TAA composite
    // stack consumes. TaaResolve or Upscale writes it; otherwise it's
    // aliased by engine binding to the latest pre-TAA scene texture.
    // The two are mutually exclusive: the upscaler does its own
    // temporal accumulation, so layering TaaResolve on top would
    // double-temporal the scene.
    let scene_color = b.import_texture("scene_color", scene_color_desc(inputs));
    let scene_color_cur = if inputs.upscale_enabled {
        let mut up = b.add_pass(PassId::Upscale, PassKind::Render);
        up.read_texture(scene_pre_taa_cur);
        // Explicit velocity read so the toposort pins Velocity →
        // Upscale. The scaler consumes motion vectors directly.
        if let Some(v) = velocity_v1 {
            up.read_texture(v);
        }
        up.write_texture(scene_color)
    } else if inputs.taa_enabled {
        let mut taa = b.add_pass(PassId::TaaResolve, PassKind::Render);
        taa.read_texture(scene_pre_taa_cur);
        // Explicit velocity read so the toposort pins Velocity →
        // TaaResolve. Without this the order rests on declaration order
        // alone.
        if let Some(v) = velocity_v1 {
            taa.read_texture(v);
        }
        taa.write_texture(scene_color)
    } else {
        scene_color
    };

    let bloom_top_v1 = if inputs.bloom_enabled {
        let bloom_top = b.create_texture("bloom_top", bloom_top_desc(inputs));
        Some(
            b.add_pass(PassId::Bloom, PassKind::Render)
                .read_texture(scene_color_cur)
                .write_texture(bloom_top),
        )
    } else {
        None
    };

    // Composite (the presenter) reads scene_color + optional bloom_top,
    // and writes the swapchain via `presents()`.
    {
        let mut composite = b.add_pass(PassId::Composite, PassKind::Render);
        composite.read_texture(scene_color_cur);
        if let Some(h) = bloom_top_v1 {
            composite.read_texture(h);
        }
        composite.presents();
    }

    b.compile()
}

fn froxel_volume_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // Identity stub only: the graph does not allocate resources, so the
    // backend owns the actual 3D-texture creation. The desc carries the X/Y
    // extent for documentation; the Z extent is communicated separately via
    // the FogFroxelParams uniform. `array_layers` is left at 1 since
    // TextureDesc does not model 3D textures explicitly.
    let _ = inputs;
    TextureDesc {
        width: TextureSize::Absolute(FOG_FROXEL_X),
        height: TextureSize::Absolute(FOG_FROXEL_Y),
        format: PixelFormat::Rgba16Float,
        sample_count: 1,
        array_layers: 1,
        usage: TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
    }
}

// X/Y/Z dimensions of the volumetric-fog froxel volume. Sized to keep the
// per-frame compute cost modest (~230 k threads per dispatch) while
// preserving enough screen-space detail for shaft-of-light shadowing.
// Backends that implement the froxel path read these constants directly;
// the values also ride in `FogFroxelParams.froxel_dims` so shaders can map
// between absolute indices and normalised volume UVs without recompiling.
pub const FOG_FROXEL_X: u32 = 80;
pub const FOG_FROXEL_Y: u32 = 45;
pub const FOG_FROXEL_Z: u32 = 64;

fn shadow_map_desc(size: u32) -> TextureDesc {
    TextureDesc {
        width: TextureSize::Absolute(size.max(1)),
        height: TextureSize::Absolute(size.max(1)),
        format: PixelFormat::Depth32Float,
        sample_count: 1,
        array_layers: NUM_SHADOW_CASCADES as u32,
        usage: TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ),
    }
}

fn hdr_color_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    TextureDesc {
        width: TextureSize::Absolute(inputs.hdr_width.max(1)),
        height: TextureSize::Absolute(inputs.hdr_height.max(1)),
        format: PixelFormat::Rgba16Float,
        sample_count: inputs.hdr_sample_count.max(1),
        array_layers: 1,
        usage: TextureUsage::RENDER_TARGET,
    }
}

fn hdr_depth_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    TextureDesc {
        width: TextureSize::Absolute(inputs.hdr_width.max(1)),
        height: TextureSize::Absolute(inputs.hdr_height.max(1)),
        format: PixelFormat::Depth32Float,
        sample_count: inputs.hdr_sample_count.max(1),
        array_layers: 1,
        usage: TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ),
    }
}

fn draw_args_desc() -> BufferDesc {
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE.union(BufferUsage::INDIRECT),
    }
}

fn cull_status_desc() -> BufferDesc {
    // One u32 per draw object: phase-1 cull writes drawn / hi-z-candidate /
    // culled, Cull2 reads it. Plain storage; the executor owns the
    // allocation (sized to the live draw-object count).
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE,
    }
}

fn hiz_pyramid_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // R32Float depth-mip pyramid rebuilt mid-frame from phase-1 depth.
    // Identity stub: the executor owns the real mip-chain texture (the desc
    // carries render dims + format for graph edges; the mip count is derived
    // backend-side). Imported, so the desc is not an aliasing input. Standard
    // depth, MAX reduction.
    TextureDesc {
        width: TextureSize::Absolute(inputs.hdr_width.max(1)),
        height: TextureSize::Absolute(inputs.hdr_height.max(1)),
        format: PixelFormat::R32Float,
        sample_count: 1,
        array_layers: 1,
        usage: TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
    }
}

fn gbuffer_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // Documentation-only handle for the unified G-buffer pre-pass (the executor
    // owns the real normal+depth / roughness / velocity / depth textures). The
    // RGBA16F normal+depth is the representative format used for graph edges.
    TextureDesc {
        width: TextureSize::Absolute(inputs.hdr_width.max(1)),
        height: TextureSize::Absolute(inputs.hdr_height.max(1)),
        format: PixelFormat::Rgba16Float,
        sample_count: 1,
        array_layers: 1,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    }
}

fn ssr_gbuffer_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RGBA16F view-space normal + linear depth at HDR dims; shared with
    // SSAO when both passes are on.
    TextureDesc {
        width: TextureSize::Absolute(inputs.hdr_width.max(1)),
        height: TextureSize::Absolute(inputs.hdr_height.max(1)),
        format: PixelFormat::Rgba16Float,
        sample_count: 1,
        array_layers: 1,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    }
}

fn ao_output_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // R8 occlusion at HDR dims; sampled by Main's ambient term.
    TextureDesc {
        width: TextureSize::Absolute(inputs.hdr_width.max(1)),
        height: TextureSize::Absolute(inputs.hdr_height.max(1)),
        format: PixelFormat::R8Unorm,
        sample_count: 1,
        array_layers: 1,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    }
}

fn velocity_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RG16F motion-vector buffer at HDR dims, sampled by TaaResolve.
    TextureDesc {
        width: TextureSize::Absolute(inputs.hdr_width.max(1)),
        height: TextureSize::Absolute(inputs.hdr_height.max(1)),
        format: PixelFormat::Rg16Float,
        sample_count: 1,
        array_layers: 1,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    }
}

fn bloom_top_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // bloom_top is `bloom_targets.mips[0]`, the bloom chain's half-resolution
    // top octave (every backend sizes it `hdr_dims >> 1`); the prefilter pass
    // writes into it and the upsample chain accumulates back into it for
    // Composite to sample. Modelling it half-res keeps the aliasing planner's
    // footprint honest -- a full-res desc over-reports `bloom_top` 4x.
    TextureDesc {
        width: TextureSize::Absolute((inputs.hdr_width.max(1) >> 1).max(1)),
        height: TextureSize::Absolute((inputs.hdr_height.max(1) >> 1).max(1)),
        format: PixelFormat::Rgba16Float,
        sample_count: 1,
        array_layers: 1,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    }
}

fn hdr_resolve_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    TextureDesc {
        width: TextureSize::Absolute(inputs.hdr_width.max(1)),
        height: TextureSize::Absolute(inputs.hdr_height.max(1)),
        format: PixelFormat::Rgba16Float,
        sample_count: 1,
        array_layers: 1,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    }
}

fn scene_color_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // The engine-owned scene_color texture the post stack consumes is
    // single-sample at HDR dims regardless of whether the per-frame
    // resolution lands on taa_targets / ssr_targets.output / hdr_resolve.
    TextureDesc {
        width: TextureSize::Absolute(inputs.hdr_width.max(1)),
        height: TextureSize::Absolute(inputs.hdr_height.max(1)),
        format: PixelFormat::Rgba16Float,
        sample_count: 1,
        array_layers: 1,
        usage: TextureUsage::SHADER_READ,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_off() -> FrameGraphInputs {
        FrameGraphInputs {
            shadow_enabled: false,
            shadow_map_size: 2048,
            hdr_width: 1280,
            hdr_height: 720,
            hdr_sample_count: 4,
            bindless_cull_enabled: false,
            auto_exposure_enabled: false,
            bloom_enabled: false,
            velocity_enabled: false,
            taa_enabled: false,
            ssr_enabled: false,
            particles_enabled: false,
            fog_enabled: false,
            decals_enabled: false,
            ssr_prepass_enabled: false,
            ssao_enabled: false,
            upscale_enabled: false,
            transparent_enabled: false,
            raymarch_enabled: false,
            two_pass_occlusion_enabled: false,
            ssgi_enabled: false,
            rt_reflections_enabled: false,
            // Default off: the existing tests exercise the separate-node path
            // (the DX / Vulkan backends). Unified-path tests set this true.
            unified_gbuffer_prepass: false,
        }
    }

    #[test]
    fn minimum_graph_is_main_then_composite() {
        let g = build_frame_graph(&all_off()).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Main, PassId::Composite]);
        assert!(g.passes[1].presents);
    }

    #[test]
    fn shadow_orders_before_main() {
        let mut i = all_off();
        i.shadow_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Shadow, PassId::Main, PassId::Composite]);
    }

    #[test]
    fn cull_orders_before_main_via_draw_args() {
        let mut i = all_off();
        i.bindless_cull_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Cull, PassId::Main, PassId::Composite]);
        assert_eq!(g.passes[0].kind, PassKind::Compute);
    }

    #[test]
    fn two_pass_inserts_phase2_chain_after_main() {
        // With bindless cull + two-pass on, the graph gains the phase-2
        // prefix Cull → Main → HizBuild → Cull2 → Main2, strictly ordered.
        let mut i = all_off();
        i.bindless_cull_enabled = true;
        i.two_pass_occlusion_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::Cull,
                PassId::Main,
                PassId::HizBuild,
                PassId::Cull2,
                PassId::Main2,
                PassId::Composite,
            ]
        );
        assert_eq!(g.passes[2].kind, PassKind::Compute); // HizBuild
        assert_eq!(g.passes[3].kind, PassKind::Compute); // Cull2
        assert_eq!(g.passes[4].kind, PassKind::Render); // Main2
    }

    #[test]
    fn two_pass_without_bindless_cull_is_noop() {
        // Two-pass rides the bindless GPU-cull path; requesting it without
        // a bindless shader must not insert any phase-2 nodes.
        let mut i = all_off();
        i.two_pass_occlusion_enabled = true; // bindless_cull_enabled left false
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Main, PassId::Composite]);
        assert!(!order.contains(&PassId::HizBuild));
        assert!(!order.contains(&PassId::Cull2));
        assert!(!order.contains(&PassId::Main2));
    }

    #[test]
    fn two_pass_shifts_post_chain_head_to_main2() {
        // AutoExposure + the RMW chain must read Main2's hdr_resolve (v2),
        // not Main's (v1), so the post stack sees the combined two-pass
        // scene. Main writes v1, Main2 writes v2, AutoExposure reads v2,
        // Decals bumps to v3.
        let mut i = all_off();
        i.bindless_cull_enabled = true;
        i.two_pass_occlusion_enabled = true;
        i.auto_exposure_enabled = true;
        i.decals_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::Main2) < pos(PassId::AutoExposure));
        assert!(pos(PassId::AutoExposure) < pos(PassId::Decals));
        // Version walk: Main2 RMWs hdr_resolve to v2, Decals to v3.
        let main2 = &g.passes[pos(PassId::Main2)];
        // hdr_resolve is the last write Main2 declares (depth, color, resolve).
        assert_eq!(main2.writes.last().unwrap().version(), 2);
        let decals = &g.passes[pos(PassId::Decals)];
        assert_eq!(decals.writes[0].version(), 3);
    }

    #[test]
    fn ssao_orders_before_main_via_ao_output() {
        let mut i = all_off();
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![PassId::SsaoBlur, PassId::Main, PassId::Composite]
        );
    }

    #[test]
    fn ao_output_barriers_are_graph_driven() {
        // The DirectX + Vulkan executors emit `ao_output`'s transitions from
        // these barriers (resolving them to RENDER_TARGET / COLOR_ATTACHMENT on
        // SsaoBlur and back to the sampled state on Main). Pin the exact pair
        // so the executor's stripped inline / render-pass-baked transitions
        // stay matched to what the graph derives.
        use super::super::ResourceState;
        let mut i = all_off();
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let pass = |id: PassId| g.passes.iter().find(|p| p.id == id).expect("present");

        let ssao = g.pass_barriers_for(pass(PassId::SsaoBlur), &["ao_output"]);
        assert_eq!(ssao.len(), 1, "SsaoBlur has exactly one ao_output barrier");
        assert_eq!(ssao[0].1.from_state(), ResourceState::Undefined);
        assert_eq!(ssao[0].1.to_state(), ResourceState::Write);

        let main = g.pass_barriers_for(pass(PassId::Main), &["ao_output"]);
        assert_eq!(main.len(), 1, "Main has exactly one ao_output barrier");
        assert_eq!(main[0].1.from_state(), ResourceState::Write);
        assert_eq!(main[0].1.to_state(), ResourceState::Read);
    }

    #[test]
    fn shadow_map_barriers_are_graph_driven() {
        // The executors emit `shadow_map`'s transitions from these barriers. The
        // graph derives the producer (Undefined -> Write) + the Main consumer
        // (Write -> Read); each backend resolves the producer against the
        // resource's resting state. DirectX rests it sampled, so the producer is
        // the real PIXEL_SHADER_RESOURCE -> DEPTH_WRITE cross-frame reset (folded
        // off the old inline restore); Main's consumer replaces the encoder's
        // stripped sampled transition. With SSAO also on, Main carries both
        // shadow_map and ao_output barriers, exercising multi-resource emission
        // in one pass.
        use super::super::ResourceState;
        let mut i = all_off();
        i.shadow_enabled = true;
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let pass = |id: PassId| g.passes.iter().find(|p| p.id == id).expect("present");

        let shadow = g.pass_barriers_for(pass(PassId::Shadow), &["shadow_map"]);
        assert_eq!(shadow.len(), 1, "Shadow has exactly one shadow_map barrier");
        assert_eq!(shadow[0].1.from_state(), ResourceState::Undefined);
        assert_eq!(shadow[0].1.to_state(), ResourceState::Write);

        let main = g.pass_barriers_for(pass(PassId::Main), &["shadow_map"]);
        assert_eq!(main.len(), 1, "Main has exactly one shadow_map barrier");
        assert_eq!(main[0].1.from_state(), ResourceState::Write);
        assert_eq!(main[0].1.to_state(), ResourceState::Read);

        // Main carries both migrated resources' barriers in one pass.
        let both = g.pass_barriers_for(pass(PassId::Main), &["shadow_map", "ao_output"]);
        assert_eq!(
            both.len(),
            2,
            "Main carries shadow_map + ao_output barriers"
        );
    }

    #[test]
    fn fog_froxel_volume_barriers_are_graph_driven() {
        // The executors emit `fog_froxel_volume`'s transitions from these
        // barriers. FogFroxel's producer (Undefined -> Write) is the compute
        // write, a real sampled -> storage open on both backends now: DirectX
        // resolves it to PIXEL_SHADER_RESOURCE -> UNORDERED_ACCESS, Vulkan to
        // SHADER_READ_ONLY -> GENERAL (both rest the volume sampled, with no
        // inline reset). Fog's consumer (Write -> Read) is the storage-write ->
        // sampled close the fragment reads through.
        use super::super::ResourceState;
        let mut i = all_off();
        i.fog_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let pass = |id: PassId| g.passes.iter().find(|p| p.id == id).expect("present");

        let froxel = g.pass_barriers_for(pass(PassId::FogFroxel), &["fog_froxel_volume"]);
        assert_eq!(
            froxel.len(),
            1,
            "FogFroxel has exactly one fog_froxel_volume barrier"
        );
        assert_eq!(froxel[0].1.from_state(), ResourceState::Undefined);
        assert_eq!(froxel[0].1.to_state(), ResourceState::Write);

        let fog = g.pass_barriers_for(pass(PassId::Fog), &["fog_froxel_volume"]);
        assert_eq!(
            fog.len(),
            1,
            "Fog has exactly one fog_froxel_volume barrier"
        );
        assert_eq!(fog[0].1.from_state(), ResourceState::Write);
        assert_eq!(fog[0].1.to_state(), ResourceState::Read);
    }

    #[test]
    fn ssr_prepass_and_ssao_share_gbuffer_pinning_order() {
        let mut i = all_off();
        i.ssr_prepass_enabled = true;
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::SsrPrepass,
                PassId::SsaoBlur,
                PassId::Main,
                PassId::Composite,
            ]
        );
    }

    #[test]
    fn unified_gbuffer_prepass_replaces_ssr_and_velocity() {
        // With the unified flag on, one GBufferPrepass node stands in for the
        // separate SsrPrepass + Velocity nodes; SSAO reads its output and TAA
        // reads its motion. Neither old node appears.
        let mut i = all_off();
        i.unified_gbuffer_prepass = true;
        i.ssr_prepass_enabled = true;
        i.ssao_enabled = true;
        i.velocity_enabled = true;
        i.taa_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::GBufferPrepass));
        assert!(!order.contains(&PassId::SsrPrepass));
        assert!(!order.contains(&PassId::Velocity));
        let gb = order
            .iter()
            .position(|p| *p == PassId::GBufferPrepass)
            .unwrap();
        let ssao = order.iter().position(|p| *p == PassId::SsaoBlur).unwrap();
        let main = order.iter().position(|p| *p == PassId::Main).unwrap();
        let taa = order.iter().position(|p| *p == PassId::TaaResolve).unwrap();
        assert!(
            gb < ssao && ssao < main,
            "GBufferPrepass before SsaoBlur+Main"
        );
        assert!(gb < taa, "GBufferPrepass before TaaResolve");
    }

    #[test]
    fn unified_gbuffer_prepass_runs_for_ssao_only() {
        // SSAO alone (no SSR / velocity) still triggers the merged node.
        let mut i = all_off();
        i.unified_gbuffer_prepass = true;
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::GBufferPrepass,
                PassId::SsaoBlur,
                PassId::Main,
                PassId::Composite,
            ]
        );
    }

    #[test]
    fn unified_gbuffer_prepass_runs_for_velocity_only() {
        // Velocity alone (TAA, no SSR/SSAO) still triggers the merged node, and
        // the standalone Velocity node is not emitted.
        let mut i = all_off();
        i.unified_gbuffer_prepass = true;
        i.velocity_enabled = true;
        i.taa_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::GBufferPrepass));
        assert!(!order.contains(&PassId::Velocity));
    }

    #[test]
    fn gbuffer_prepass_orders_after_cull_via_draw_args() {
        // The GPU-driven G-buffer pre-pass reuses the main pass's per-frame
        // indirect command buffer, so it must run after Cull. With bindless cull
        // on and a G-buffer consumer active, the draw_args read edge pins
        // Cull -> GBufferPrepass (-> Main).
        let mut i = all_off();
        i.bindless_cull_enabled = true;
        i.unified_gbuffer_prepass = true;
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let cull = order.iter().position(|p| *p == PassId::Cull).unwrap();
        let gb = order
            .iter()
            .position(|p| *p == PassId::GBufferPrepass)
            .unwrap();
        let main = order.iter().position(|p| *p == PassId::Main).unwrap();
        assert!(cull < gb, "Cull before GBufferPrepass");
        assert!(gb < main, "GBufferPrepass before Main");
    }

    #[test]
    fn unified_gbuffer_prepass_omitted_when_no_consumers() {
        // The flag on but no consumer active: no pre-pass node at all.
        let mut i = all_off();
        i.unified_gbuffer_prepass = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Main, PassId::Composite]);
    }

    #[test]
    fn auto_exposure_war_pinned_before_first_hdr_writer() {
        // AutoExposure reads hdr_resolve_v1. Decals writes v2 (when
        // enabled). The WAR edge from AutoExposure to Decals pins
        // AutoExposure before Decals; without it, the toposort could
        // place them in either order.
        let mut i = all_off();
        i.auto_exposure_enabled = true;
        i.decals_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::Main,
                PassId::AutoExposure,
                PassId::Decals,
                PassId::Composite,
            ]
        );
    }

    #[test]
    fn full_hdr_chain_orders_decals_fog_particles_then_ssr() {
        let mut i = all_off();
        i.decals_enabled = true;
        i.fog_enabled = true;
        i.particles_enabled = true;
        i.ssr_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::Main,
                PassId::Decals,
                PassId::FogFroxel,
                PassId::Fog,
                PassId::ParticlesDraw,
                PassId::SsrResolve,
                PassId::Composite,
            ]
        );
        // Version chain on hdr_resolve walks 1 → 2 → 3 → 4 with
        // SsrResolve reading v4. FogFroxel slots between Decals and Fog
        // (writing the froxel volume to v1) but doesn't touch hdr_resolve,
        // so the version walk skips it.
        let decals = &g.passes[1];
        assert_eq!(decals.writes[0].version(), 2);
        let fog = &g.passes[3];
        assert_eq!(fog.writes[0].version(), 3);
        let particles = &g.passes[4];
        assert_eq!(particles.writes[0].version(), 4);
        let ssr = &g.passes[5];
        assert_eq!(ssr.reads[0].version(), 4);
    }

    #[test]
    fn upscale_replaces_taa_and_pins_after_velocity() {
        // Upscale takes TaaResolve's slot when temporal upscaling is on.
        // TaaResolve must not appear in the compiled graph (the scaler
        // does temporal accumulation itself), and Velocity must precede
        // Upscale via the explicit motion-vector read.
        let mut i = all_off();
        i.velocity_enabled = true;
        i.upscale_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::Upscale));
        assert!(!order.contains(&PassId::TaaResolve));
        assert!(
            order.iter().position(|p| *p == PassId::Velocity).unwrap()
                < order.iter().position(|p| *p == PassId::Upscale).unwrap()
        );
    }

    #[test]
    fn upscale_takes_precedence_when_both_taa_and_upscale_requested() {
        // If both flags somehow arrive set (the engine layer should
        // forbid this, but the graph is the safety net), Upscale wins
        // and TaaResolve is omitted.
        let mut i = all_off();
        i.velocity_enabled = true;
        i.taa_enabled = true;
        i.upscale_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::Upscale));
        assert!(!order.contains(&PassId::TaaResolve));
    }

    #[test]
    fn velocity_taa_pinned_via_explicit_read() {
        // TaaResolve reads the velocity buffer explicitly so the
        // toposort orders Velocity before TaaResolve via RAW (not
        // declaration order).
        let mut i = all_off();
        i.velocity_enabled = true;
        i.taa_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        // Main runs first, then Velocity + TaaResolve in compile-pass
        // order. TaaResolve reads scene_color v0 (imported v0 rule) +
        // velocity v1.
        assert!(
            order.iter().position(|p| *p == PassId::Velocity).unwrap()
                < order.iter().position(|p| *p == PassId::TaaResolve).unwrap()
        );
    }

    #[test]
    fn transparent_pinned_between_ssr_resolve_and_taa() {
        // Transparent extends the scene_pre_taa chain by one version
        // after SsrResolve, so the toposort orders SsrResolve →
        // Transparent → TaaResolve via RAW + WAW edges on the same
        // texture.
        let mut i = all_off();
        i.ssr_enabled = true;
        i.taa_enabled = true;
        i.velocity_enabled = true;
        i.transparent_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::SsrResolve) < pos(PassId::Transparent));
        assert!(pos(PassId::Transparent) < pos(PassId::TaaResolve));
    }

    #[test]
    fn transparent_works_without_ssr() {
        // Without SSR, scene_pre_taa stays at v0 (imported alias of
        // hdr_resolve). Transparent still RMWs it; the imported-v0
        // producer rule treats hdr_resolve_cur as scene_pre_taa's v0.
        let mut i = all_off();
        i.taa_enabled = true;
        i.velocity_enabled = true;
        i.transparent_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::Transparent));
        assert!(!order.contains(&PassId::SsrResolve));
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::Main) < pos(PassId::Transparent));
        assert!(pos(PassId::Transparent) < pos(PassId::TaaResolve));
    }

    #[test]
    fn transparent_off_means_no_slot() {
        // The pass is omitted when nothing in the world is transparent:
        // no orphan slot, no executor stub triggered.
        let mut i = all_off();
        i.ssr_enabled = true;
        i.taa_enabled = true;
        i.velocity_enabled = true;
        // transparent_enabled left at false.
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(!order.contains(&PassId::Transparent));
    }

    #[test]
    fn ssgi_off_means_no_slot() {
        // IBL-only indirect lighting: the pass is omitted entirely.
        let g = build_frame_graph(&all_off()).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(!order.contains(&PassId::Ssgi));
    }

    #[test]
    fn rt_reflections_off_means_no_slot() {
        // No ray tracing requested: the pass is omitted entirely.
        let g = build_frame_graph(&all_off()).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(!order.contains(&PassId::RtReflections));
    }

    #[test]
    fn rt_reflections_occupy_the_ssr_resolve_slot() {
        // RtReflections reads the post-decoration hdr_resolve and writes
        // scene_pre_taa, exactly where SsrResolve would, so it orders after
        // ParticlesDraw and before TaaResolve.
        let mut i = all_off();
        i.rt_reflections_enabled = true;
        i.particles_enabled = true;
        i.taa_enabled = true;
        i.velocity_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(order.contains(&PassId::RtReflections));
        assert!(pos(PassId::ParticlesDraw) < pos(PassId::RtReflections));
        assert!(pos(PassId::RtReflections) < pos(PassId::TaaResolve));
    }

    #[test]
    fn rt_reflections_take_precedence_over_ssr_resolve() {
        // RT alone inserts RtReflections, not SsrResolve.
        let mut i = all_off();
        i.rt_reflections_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::RtReflections));
        assert!(!order.contains(&PassId::SsrResolve));

        // With both flags set (RT available + SSR fallback authored), hardware
        // RT wins and SsrResolve is omitted; never two in the same slot.
        let mut both = all_off();
        both.ssr_enabled = true;
        both.rt_reflections_enabled = true;
        let g = build_frame_graph(&both).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::RtReflections));
        assert!(!order.contains(&PassId::SsrResolve));
    }

    #[test]
    fn ssgi_pinned_between_auto_exposure_and_decals() {
        // AutoExposure reads hdr_resolve_v1 (WAR); SSGI RMWs to v2; Decals
        // RMWs to v3. The toposort orders the three through the version chain.
        let mut i = all_off();
        i.auto_exposure_enabled = true;
        i.ssgi_enabled = true;
        i.decals_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::Main,
                PassId::AutoExposure,
                PassId::Ssgi,
                PassId::Decals,
                PassId::Composite,
            ]
        );
        // Version chain on hdr_resolve walks 1 → 2 → 3.
        let ssgi = &g.passes[2];
        assert_eq!(ssgi.writes[0].version(), 2);
        let decals = &g.passes[3];
        assert_eq!(decals.writes[0].version(), 3);
    }

    #[test]
    fn ssgi_after_raymarch_on_the_chain() {
        // With both on, SSGI reads the post-raymarch scene: Raymarch v1→v2,
        // SSGI v2→v3, SsrResolve reads v3.
        let mut i = all_off();
        i.raymarch_enabled = true;
        i.ssgi_enabled = true;
        i.ssr_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::Raymarch) < pos(PassId::Ssgi));
        assert!(pos(PassId::Ssgi) < pos(PassId::SsrResolve));
        let ssgi = &g.passes[pos(PassId::Ssgi)];
        assert_eq!(ssgi.writes[0].version(), 3);
    }

    #[test]
    fn raymarch_off_means_no_slot() {
        // No `SdfVolume` in the world: pass is omitted, no executor stub
        // ever fires.
        let g = build_frame_graph(&all_off()).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(!order.contains(&PassId::Raymarch));
    }

    #[test]
    fn raymarch_pinned_between_auto_exposure_and_decals() {
        // AutoExposure reads hdr_resolve_v1 (WAR); Raymarch RMWs to v2;
        // Decals RMWs to v3. The toposort orders the three through the
        // version chain without needing declaration-order tie-breaks.
        let mut i = all_off();
        i.auto_exposure_enabled = true;
        i.raymarch_enabled = true;
        i.decals_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::Main,
                PassId::AutoExposure,
                PassId::Raymarch,
                PassId::Decals,
                PassId::Composite,
            ]
        );
        // Version chain on hdr_resolve walks 1 → 2 → 3.
        let raymarch = &g.passes[2];
        assert_eq!(raymarch.writes[0].version(), 2);
        let decals = &g.passes[3];
        assert_eq!(decals.writes[0].version(), 3);
    }

    #[test]
    fn raymarch_works_without_auto_exposure_or_decals() {
        // Standalone Raymarch RMWs hdr_resolve_v1 → v2; SsrResolve reads
        // v2 instead of v1. Nothing else in the post chain.
        let mut i = all_off();
        i.raymarch_enabled = true;
        i.ssr_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::Main,
                PassId::Raymarch,
                PassId::SsrResolve,
                PassId::Composite,
            ]
        );
        let raymarch = &g.passes[1];
        assert_eq!(raymarch.writes[0].version(), 2);
        let ssr = &g.passes[2];
        assert_eq!(ssr.reads[0].version(), 2);
    }

    #[test]
    fn full_graph_orders_all_passes_correctly() {
        // Everything on: every pass shows up in the expected order.
        // This is the showcase configuration.
        let mut i = all_off();
        i.shadow_enabled = true;
        i.bindless_cull_enabled = true;
        i.auto_exposure_enabled = true;
        i.bloom_enabled = true;
        i.velocity_enabled = true;
        i.taa_enabled = true;
        i.ssr_enabled = true;
        i.particles_enabled = true;
        i.fog_enabled = true;
        i.decals_enabled = true;
        i.ssr_prepass_enabled = true;
        i.ssao_enabled = true;
        i.transparent_enabled = true;
        i.raymarch_enabled = true;

        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();

        // Spot-check relative ordering rather than the exact list: with
        // many independent passes the toposort has flexibility on
        // tie-breaks.
        fn idx(order: &[PassId], p: PassId) -> usize {
            order.iter().position(|x| *x == p).expect("pass present")
        }
        // Cull / SsrPrepass / SsaoBlur / Shadow / SSAO all precede Main.
        assert!(idx(&order, PassId::Cull) < idx(&order, PassId::Main));
        assert!(idx(&order, PassId::SsrPrepass) < idx(&order, PassId::Main));
        assert!(idx(&order, PassId::SsaoBlur) < idx(&order, PassId::Main));
        assert!(idx(&order, PassId::Shadow) < idx(&order, PassId::Main));
        // SsrPrepass precedes SsaoBlur (G-buffer share).
        assert!(idx(&order, PassId::SsrPrepass) < idx(&order, PassId::SsaoBlur));
        // AutoExposure post-Main, pre-Raymarch (WAR-pinned on hdr_resolve_v1).
        assert!(idx(&order, PassId::Main) < idx(&order, PassId::AutoExposure));
        assert!(idx(&order, PassId::AutoExposure) < idx(&order, PassId::Raymarch));
        // Raymarch leads the hdr_resolve RMW chain so Decals / Fog /
        // ParticlesDraw blend on top of the raymarched colour.
        assert!(idx(&order, PassId::Raymarch) < idx(&order, PassId::Decals));
        // hdr_resolve chain.
        assert!(idx(&order, PassId::Decals) < idx(&order, PassId::Fog));
        assert!(idx(&order, PassId::Fog) < idx(&order, PassId::ParticlesDraw));
        assert!(idx(&order, PassId::ParticlesDraw) < idx(&order, PassId::SsrResolve));
        // FogFroxel populates the volume Fog samples, so it must precede Fog.
        assert!(idx(&order, PassId::FogFroxel) < idx(&order, PassId::Fog));
        // Velocity precedes TaaResolve.
        assert!(idx(&order, PassId::Velocity) < idx(&order, PassId::TaaResolve));
        // Post-TAA chain. Transparent slots between SsrResolve and TaaResolve.
        assert!(idx(&order, PassId::SsrResolve) < idx(&order, PassId::Transparent));
        assert!(idx(&order, PassId::Transparent) < idx(&order, PassId::TaaResolve));
        assert!(idx(&order, PassId::TaaResolve) < idx(&order, PassId::Bloom));
        assert!(idx(&order, PassId::Bloom) < idx(&order, PassId::Composite));
        // Composite is the presenter and runs last.
        assert_eq!(order.last(), Some(&PassId::Composite));
        assert!(g.passes.last().unwrap().presents);
    }
}
