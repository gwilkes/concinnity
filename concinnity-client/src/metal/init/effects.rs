// src/metal/init/effects.rs
//
// Post-process pipeline + target construction extracted from MtlContext::new:
// bloom, TAA + velocity pre-pass, SSAO, SSR, projected decals, volumetric fog,
// and auto-exposure. Each block is gated on the relevant world setting so a
// world that disables an effect pays zero construction cost.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::too_many_arguments)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLRenderPipelineState, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState, MTLTexture, MTLVertexDescriptor,
};

use crate::gfx::auto_exposure::{AutoExposureSettings, AutoExposureState};
use crate::gfx::decal::DecalRecord;
use crate::gfx::particles::ParticleEmitterRecord;
use crate::gfx::rt_reflections::RtReflectionSettings;
use crate::gfx::ssao::SsaoSettings;
use crate::gfx::ssgi::SsgiSettings;
use crate::gfx::ssr::SsrSettings;
use crate::gfx::volumetric_fog::FogSettings;
use crate::metal::auto_exposure::{AutoExposurePipelines, build_auto_exposure_pipelines};
use crate::metal::decal::build_decal_pipeline;
use crate::metal::fog::build_fog_pipeline;
use crate::metal::particle::{
    ParticleEmitterGpuState, ParticlePipelines, build_emitter_gpu_state, build_particle_pipelines,
};
use crate::metal::post::{
    BloomPipelines, BloomTargets, GBufferState, SsaoState, SsgiState, SsrState,
    build_bloom_pipelines, build_gbuffer_bindless_pipeline, build_gbuffer_prepass_pipeline,
    build_reflection_blur_pipeline, build_reflection_composite_pipeline,
    build_rt_reflection_pipeline, build_ssao_pipeline, build_ssgi_composite_pipeline,
    build_ssgi_gather_pipeline, build_ssr_pipeline, build_taa_pipeline, create_bloom_targets,
    create_gbuffer_targets, create_ssao_targets, create_ssgi_targets, create_ssr_targets,
    create_taa_targets,
};
use crate::metal::texture::create_fallback_texture;
use crate::metal::transient_pool::{TransientTexturePool, transient_specs};

pub(crate) struct EffectsBundle {
    // Bloom is always built.
    pub bloom_targets: BloomTargets,
    pub bloom_pipelines: BloomPipelines,

    // TAA: built only when taa_enabled.
    pub taa_pipeline_state: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub taa_targets: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,

    // SSAO: pipelines + targets built only when ssao_settings is Some (the
    // kernel reads the unified G-buffer pre-pass output, so there is no
    // SSAO-owned pre-pass); the white fallback is always present.
    pub ssao: SsaoState,

    // Render-graph transient texture pool (`gfx::render_graph::alias`). Owns
    // `ao_output` when SSAO is on; empty otherwise.
    pub transient_pool: TransientTexturePool,

    // SSR resolve + output target: built when SSR / SSGI / RT is on (RT reuses
    // `ssr.targets.output`). The resolve pipeline is built only when SSR is on.
    pub ssr: SsrState,

    // Unified G-buffer pre-pass: built when any consumer (SSR / SSGI / RT / SSAO
    // / velocity) is on. The skinned variant is built later by `upload_skinned`.
    pub gbuffer: GBufferState,

    // SSGI: built only when ssgi_settings is Some.
    pub ssgi: SsgiState,

    // RT reflections: built only when rt_reflection_settings is Some (the GPU
    // supports ray tracing). Reuses the SSR pre-pass G-buffer + `ssr_targets`.
    // The flat variant is the non-bindless fallback; the textured variant
    // samples the bindless albedo pool.
    pub rt_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub rt_pipeline_textured: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub rt_skin_pipeline:
        Option<Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>>,

    // Decals: built only when at least one decal is declared.
    pub decal_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub decal_cube_vertex_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub decal_cube_index_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub decal_sampler: Option<Retained<ProtocolObject<dyn MTLSamplerState>>>,

    // Volumetric fog: built only when fog_settings is Some.
    pub fog_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub fog_froxel_pipeline:
        Option<Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>>,
    pub fog_froxel_volume: Option<Retained<ProtocolObject<dyn objc2_metal::MTLTexture>>>,

    // Particles: built only when at least one emitter is declared. The
    // per-emitter GPU state vec is parallel to the `particles` records and
    // empty when none are declared.
    pub particle_pipelines: Option<ParticlePipelines>,
    pub particle_emitter_state: Vec<ParticleEmitterGpuState>,

    // Auto-exposure: built only when auto_exposure_settings is Some.
    pub auto_exposure_pipelines: Option<AutoExposurePipelines>,
    pub auto_exposure_histogram: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub auto_exposure_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub auto_exposure_state: Option<AutoExposureState>,
    pub auto_exposure_bias_ev: f32,
}

// The toggle-controlled subset of the effects stack: the features the Quality
// settings group switches on and off at runtime (TAA, SSAO, SSR, SSGI, RT
// reflection pipelines, auto-exposure) plus the resources they share (the
// G-buffer pre-pass + the SSAO transient pool). Built by [`build_quality_effects`]
// from the per-feature settings, so both `MtlContext::new` (init) and the
// runtime rebuild ([`crate::metal::MtlContext::apply_quality_settings`]) produce
// byte-identical resources from the same inputs. Bloom, decals, fog, and
// particles are NOT here: bloom is always on (only its uniforms change, live),
// and decals/fog/particles are world-content effects a quality toggle never
// affects (and rebuilding particles would reset their live GPU pools).
pub(crate) struct QualityEffectsBundle {
    pub taa_pipeline_state: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub taa_targets: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    pub ssao: SsaoState,
    pub transient_pool: TransientTexturePool,
    pub ssr: SsrState,
    pub gbuffer: GBufferState,
    pub ssgi: SsgiState,
    pub rt_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub rt_pipeline_textured: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub rt_skin_pipeline:
        Option<Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>>,
    pub auto_exposure_pipelines: Option<AutoExposurePipelines>,
    pub auto_exposure_histogram: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub auto_exposure_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub auto_exposure_state: Option<AutoExposureState>,
    pub auto_exposure_bias_ev: f32,
}

// Build the toggle-controlled effects subset (see [`QualityEffectsBundle`]). The
// `gbuffer.skinned_pipeline` is left `None` here (as in the init path); the
// caller builds the 80-byte skinned variant separately when the world has
// skinned meshes (init via `upload_skinned`, the runtime rebuild re-attaches it).
// The RT acceleration structure is also the caller's responsibility (it needs the
// resident geometry buffers); this builds only the RT resolve pipelines.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_quality_effects(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    render_w: u32,
    render_h: u32,
    taa_enabled: bool,
    needs_velocity: bool,
    has_instanced: bool,
    ssao_settings: &Option<SsaoSettings>,
    ssr_settings: &Option<SsrSettings>,
    ssgi_settings: &Option<SsgiSettings>,
    rt_reflection_settings: &Option<RtReflectionSettings>,
    // Per-axis divisor for the roughness-aware reflection blur target, resolved
    // from the world's `reflection_blur_resolution`. Sizes the blur target at
    // render / this; stored on `SsrState` so resize reuses it.
    reflection_blur_scale: u32,
    auto_exposure_settings: &Option<AutoExposureSettings>,
    auto_exposure_bias_ev: f32,
    hot_reload: bool,
) -> Result<QualityEffectsBundle, String> {
    // TAA pipeline + ping-pong history buffers. Built only when TAA is on;
    // upscaling-on worlds skip the TAA pass entirely (the MetalFX scaler does
    // temporal accumulation itself). The TAA targets are sized at
    // render-resolution to match the scene texture they sample.
    let (taa_pipeline_state, taa_targets) = if taa_enabled {
        (
            Some(build_taa_pipeline(device, hot_reload)?),
            create_taa_targets(device, render_w, render_h)?.to_vec(),
        )
    } else {
        (None, Vec::new())
    };

    // SSAO (GTAO): the horizon-search kernel, the depth-aware blur, and their
    // occlusion targets. The depth + normal the kernel reads come from the
    // unified G-buffer pre-pass (below), so SSAO builds no pre-pass of its own.
    let (ssao_targets, ssao_kernel_pipeline, ssao_blur_pipeline) = if ssao_settings.is_some() {
        (
            Some(create_ssao_targets(device, render_w, render_h)?),
            Some(build_ssao_pipeline(device, "ssao_fragment", hot_reload)?),
            Some(build_ssao_pipeline(
                device,
                "ssao_blur_fragment",
                hot_reload,
            )?),
        )
    } else {
        (None, None, None)
    };
    let ssao = SsaoState {
        settings: *ssao_settings,
        targets: ssao_targets,
        kernel_pipeline: ssao_kernel_pipeline,
        blur_pipeline: ssao_blur_pipeline,
        white: create_fallback_texture(device)?,
    };

    // Render-graph transient texture pool. Stage 1 manages only `ao_output`
    // (SSAO's blurred occlusion, render-resolution), relocated off SSAO so a
    // later stage can place it and `bloom_top` on one aliased heap slot.
    let transient_pool = TransientTexturePool::build(
        device,
        &transient_specs(ssao_settings.is_some(), render_w, render_h),
    )?;

    // SSR resolve: the ray-march resolve pipeline + its output target, built when
    // SSR *or* SSGI *or* RT reflections is on (all three need the G-buffer the
    // unified pre-pass below produces; RT reuses `ssr_targets.output`). The
    // fullscreen resolve pipeline is built only when SSR itself is on.
    let needs_ssr_prepass =
        ssr_settings.is_some() || ssgi_settings.is_some() || rt_reflection_settings.is_some();
    let (ssr_targets, ssr_resolve_pipeline, ssr_composite_pipeline, ssr_blur_pipeline) =
        if needs_ssr_prepass {
            let ssr_resolve = if ssr_settings.is_some() {
                Some(build_ssr_pipeline(device, hot_reload)?)
            } else {
                None
            };
            // The reflection composite (roughness blur + blend over the scene)
            // runs for both SSR and RT reflections; both write the reflection
            // target it reads. SSGI alone needs the G-buffer but no composite.
            // The blur is its reduced-resolution first pass.
            let (composite, blur) = if ssr_settings.is_some() || rt_reflection_settings.is_some() {
                (
                    Some(build_reflection_composite_pipeline(device, hot_reload)?),
                    Some(build_reflection_blur_pipeline(device, hot_reload)?),
                )
            } else {
                (None, None)
            };
            (
                Some(create_ssr_targets(
                    device,
                    render_w,
                    render_h,
                    reflection_blur_scale,
                )?),
                ssr_resolve,
                composite,
                blur,
            )
        } else {
            (None, None, None, None)
        };

    // Unified G-buffer pre-pass (Metal): one pipeline per geometry kind + the
    // shared targets, built when any consumer (SSR / SSGI / RT / SSAO / velocity)
    // is on. The skinned variant is built by the caller.
    let needs_gbuffer = needs_ssr_prepass || ssao_settings.is_some() || needs_velocity;
    let (gbuffer_targets, gbuffer_prepass_pipeline, gbuffer_instanced_pipeline) = if needs_gbuffer {
        let inst = if has_instanced {
            Some(build_gbuffer_prepass_pipeline(
                device,
                vert_desc,
                "gbuffer_prepass_vertex_instanced",
                hot_reload,
            )?)
        } else {
            None
        };
        (
            Some(create_gbuffer_targets(device, render_w, render_h)?),
            Some(build_gbuffer_prepass_pipeline(
                device,
                vert_desc,
                "gbuffer_prepass_vertex",
                hot_reload,
            )?),
            inst,
        )
    } else {
        (None, None, None)
    };

    // GPU-driven bindless G-buffer pipeline: built whenever the
    // G-buffer is, so the GPU-driven pre-pass engages on bindless worlds. It is
    // one engine-internal shader (`gbuffer_prepass.metal`), independent of the
    // world's fragment, so it builds the same in init and the runtime quality
    // rebuild; the encode gates on the cull-produced object buffer, so a
    // non-bindless / custom-shader world never reaches it.
    let gbuffer_bindless_pipeline = if needs_gbuffer {
        Some(build_gbuffer_bindless_pipeline(device, hot_reload)?)
    } else {
        None
    };

    // SSGI: the hemisphere-gather + depth-aware-blur composite pipelines and
    // the intermediate `gi` target. Built only when SSGI is on; the gather
    // reads the SSR pre-pass G-buffer built above.
    let (ssgi_targets, ssgi_gather_pipeline, ssgi_composite_pipeline) =
        if let Some(s) = ssgi_settings {
            // The gather runs at `gi_scale`-reduced resolution; the composite
            // bilateral-upsamples it back to full resolution.
            let (gw, gh) = s.gi_dimensions(render_w, render_h);
            (
                Some(create_ssgi_targets(device, gw, gh)?),
                Some(build_ssgi_gather_pipeline(device, hot_reload)?),
                Some(build_ssgi_composite_pipeline(device, hot_reload)?),
            )
        } else {
            (None, None, None)
        };

    let ssr = SsrState {
        settings: *ssr_settings,
        targets: ssr_targets,
        resolve_pipeline: ssr_resolve_pipeline,
        composite_pipeline: ssr_composite_pipeline,
        blur_pipeline: ssr_blur_pipeline,
        blur_scale: reflection_blur_scale.max(1),
    };
    let gbuffer = GBufferState {
        targets: gbuffer_targets,
        prepass_pipeline: gbuffer_prepass_pipeline,
        instanced_pipeline: gbuffer_instanced_pipeline,
        // The skinned variant is built by the caller (80-byte layout).
        skinned_pipeline: None,
        bindless_pipeline: gbuffer_bindless_pipeline,
    };
    let ssgi = SsgiState {
        settings: *ssgi_settings,
        targets: ssgi_targets,
        gather_pipeline: ssgi_gather_pipeline,
        composite_pipeline: ssgi_composite_pipeline,
    };

    // RT reflections: the inline ray-trace resolve pipelines. Built only when RT
    // reflections are on; the caller has already confirmed the GPU supports ray
    // tracing. Writes into `ssr_targets.output`, reusing the SSR pre-pass
    // G-buffer built above.
    let (rt_pipeline, rt_pipeline_textured) = if rt_reflection_settings.is_some() {
        (
            Some(build_rt_reflection_pipeline(
                device,
                "rt_reflections_fragment",
                hot_reload,
            )?),
            Some(build_rt_reflection_pipeline(
                device,
                "rt_reflections_fragment_textured",
                hot_reload,
            )?),
        )
    } else {
        (None, None)
    };
    // Compute-skinning pipeline for ray tracing: deforms skinned vertices into a
    // buffer the BVH can trace. Built alongside the reflection pipelines (same
    // RT gate); unused when the world has no SkinnedMesh.
    let rt_skin_pipeline = if rt_reflection_settings.is_some()
        && crate::metal::raytrace::raytracing_supported(device)
    {
        Some(crate::metal::raytrace::build_rt_skin_pipeline(
            device, hot_reload,
        )?)
    } else {
        None
    };

    // Auto-exposure pipelines + persistent compute buffers. Both buffers are
    // zero-initialised so the build kernel's first dispatch sees an empty
    // histogram.
    let (
        auto_exposure_pipelines,
        auto_exposure_histogram,
        auto_exposure_output,
        auto_exposure_state,
        auto_exposure_bias,
    ) = if let Some(settings) = auto_exposure_settings.as_ref() {
        let pipelines = build_auto_exposure_pipelines(device, hot_reload)?;
        let hist = make_auto_exposure_histogram(device)?;
        let out = make_auto_exposure_output(device)?;
        let state = AutoExposureState::new(settings);
        (
            Some(pipelines),
            Some(hist),
            Some(out),
            Some(state),
            auto_exposure_bias_ev,
        )
    } else {
        (None, None, None, None, 0.0)
    };

    Ok(QualityEffectsBundle {
        taa_pipeline_state,
        taa_targets,
        ssao,
        transient_pool,
        ssr,
        gbuffer,
        ssgi,
        rt_pipeline,
        rt_pipeline_textured,
        rt_skin_pipeline,
        auto_exposure_pipelines,
        auto_exposure_histogram,
        auto_exposure_output,
        auto_exposure_state,
        auto_exposure_bias_ev: auto_exposure_bias,
    })
}

pub(crate) fn build_effects(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    // Render-resolution dimensions: where the 3D scene + most post passes
    // (HDR, TAA, velocity, SSAO, SSR) draw. Equal to `output_w/h` when no
    // upscaler is active; smaller when MetalFX upscaling is on.
    render_w: u32,
    render_h: u32,
    // Output-resolution dimensions: where bloom + composite operate so the
    // upscaled / native scene reads back cleanly into the drawable.
    output_w: u32,
    output_h: u32,
    taa_enabled: bool,
    // Whether the velocity pre-pass + targets should be built. True when
    // TAA is on *or* when temporal upscaling is on (the MetalFX scaler
    // consumes motion vectors).
    needs_velocity: bool,
    has_instanced: bool,
    ssao_settings: &Option<SsaoSettings>,
    ssr_settings: &Option<SsrSettings>,
    ssgi_settings: &Option<SsgiSettings>,
    rt_reflection_settings: &Option<RtReflectionSettings>,
    // Per-axis divisor for the roughness-aware reflection blur target, resolved
    // from the world's `reflection_blur_resolution` (forwarded to
    // `build_quality_effects`).
    reflection_blur_scale: u32,
    decals: &[DecalRecord],
    particles: &[ParticleEmitterRecord],
    fog_settings: &Option<FogSettings>,
    auto_exposure_settings: &Option<AutoExposureSettings>,
    auto_exposure_bias_ev: f32,
    hot_reload: bool,
) -> Result<EffectsBundle, String> {
    // Bloom chain + pipelines. Bloom samples whatever scene_color the post
    // stack hands it: that's at output (drawable) resolution when MetalFX
    // upscaling is on, native resolution otherwise. Sized off `output_w/h`
    // so bloom stays crisp at the panel's pixel grid. Always built (only its
    // uniforms vary), so it is not part of the toggle-controlled subset.
    let bloom_targets = create_bloom_targets(device, output_w, output_h)?;
    let bloom_pipelines = build_bloom_pipelines(device, hot_reload)?;

    // The toggle-controlled subset (TAA, SSAO, SSR, SSGI, RT resolve pipelines,
    // auto-exposure, + the shared G-buffer pre-pass and SSAO transient pool).
    // Shared with the runtime rebuild (`apply_quality_settings`) so init and a
    // live toggle produce byte-identical resources. The skinned G-buffer
    // pipeline + the RT acceleration structure are built below.
    let QualityEffectsBundle {
        taa_pipeline_state,
        taa_targets,
        ssao,
        transient_pool,
        ssr,
        gbuffer,
        ssgi,
        rt_pipeline,
        rt_pipeline_textured,
        rt_skin_pipeline,
        auto_exposure_pipelines,
        auto_exposure_histogram,
        auto_exposure_output,
        auto_exposure_state,
        auto_exposure_bias_ev: auto_exposure_bias,
    } = build_quality_effects(
        device,
        vert_desc,
        render_w,
        render_h,
        taa_enabled,
        needs_velocity,
        has_instanced,
        ssao_settings,
        ssr_settings,
        ssgi_settings,
        rt_reflection_settings,
        reflection_blur_scale,
        auto_exposure_settings,
        auto_exposure_bias_ev,
        hot_reload,
    )?;

    // Projected-decal pass. Built only when the world declares at least one
    // decal; with none, all four resources stay `None` and the pass is
    // skipped by `draw_frame`. The unit cube spans `[-0.5, 0.5]^3` -- the
    // same local space the decal `inv_model` maps a reconstructed world
    // point into. 36 indices form 12 triangles wound CCW outward. The first
    // runtime [`MtlContext::add_decal`] for a world that started with no
    // decals will rebuild the same four resources on demand.
    let (decal_pipeline, decal_cube_vertex_buffer, decal_cube_index_buffer, decal_sampler) =
        if !decals.is_empty() {
            let (ps, vbuf, ibuf, samp) = build_decal_resources_for_runtime(device, hot_reload)?;
            (Some(ps), Some(vbuf), Some(ibuf), Some(samp))
        } else {
            (None, None, None, None)
        };

    // Volumetric-fog pipeline. Built only when the world declares a
    // `VolumetricFog`; with none, the pipeline stays `None` and the fog
    // pass is skipped by `draw_frame`.
    let (fog_pipeline, fog_froxel_pipeline, fog_froxel_volume) = if fog_settings.is_some() {
        let render_ps = build_fog_pipeline(device, hot_reload)?;
        let compute_ps = super::super::fog::build_fog_froxel_pipeline(device, hot_reload)?;
        let volume = super::super::fog::build_fog_froxel_volume(device)?;
        (Some(render_ps), Some(compute_ps), Some(volume))
    } else {
        (None, None, None)
    };

    // Particle compute + render pipelines, plus one persistent GPU pool per
    // emitter. Pools are zero-initialised so every slot starts dead; the
    // compute kernel spawns into them on its first dispatch.
    let (particle_pipelines, particle_emitter_state) = if !particles.is_empty() {
        let pipelines = build_particle_pipelines(device, hot_reload)?;
        let mut states = Vec::with_capacity(particles.len());
        for rec in particles {
            states.push(build_emitter_gpu_state(device, rec)?);
        }
        (Some(pipelines), states)
    } else {
        (None, Vec::new())
    };

    Ok(EffectsBundle {
        bloom_targets,
        bloom_pipelines,
        taa_pipeline_state,
        taa_targets,
        ssao,
        transient_pool,
        ssr,
        gbuffer,
        ssgi,
        rt_pipeline,
        rt_pipeline_textured,
        rt_skin_pipeline,
        decal_pipeline,
        decal_cube_vertex_buffer,
        decal_cube_index_buffer,
        decal_sampler,
        fog_pipeline,
        fog_froxel_pipeline,
        fog_froxel_volume,
        particle_pipelines,
        particle_emitter_state,
        auto_exposure_pipelines,
        auto_exposure_histogram,
        auto_exposure_output,
        auto_exposure_state,
        auto_exposure_bias_ev: auto_exposure_bias,
    })
}

type DecalResources = (
    Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    Retained<ProtocolObject<dyn MTLSamplerState>>,
);

// Build the projected-decal pipeline + the unit-cube vertex / index buffers
// + the shared sampler. Called either at init when the world declared ≥1
// decal, or lazily by [`crate::metal::MtlContext::add_decal`] on the first
// runtime add for a world that started with none.
pub(crate) fn build_decal_resources_for_runtime(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<DecalResources, String> {
    let ps = build_decal_pipeline(device, hot_reload)?;
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct CubeVtx {
        p: [f32; 3],
    }
    const CUBE_VERTS: [CubeVtx; 8] = [
        CubeVtx {
            p: [-0.5, -0.5, -0.5],
        },
        CubeVtx {
            p: [0.5, -0.5, -0.5],
        },
        CubeVtx {
            p: [0.5, 0.5, -0.5],
        },
        CubeVtx {
            p: [-0.5, 0.5, -0.5],
        },
        CubeVtx {
            p: [-0.5, -0.5, 0.5],
        },
        CubeVtx {
            p: [0.5, -0.5, 0.5],
        },
        CubeVtx { p: [0.5, 0.5, 0.5] },
        CubeVtx {
            p: [-0.5, 0.5, 0.5],
        },
    ];
    const CUBE_INDICES: [u16; 36] = [
        // -Z face                    +Z face
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, // -Y                         +Y
        0, 1, 5, 0, 5, 4, 3, 6, 2, 3, 7, 6, // -X                         +X
        0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
    ];
    let vbuf = unsafe {
        let ptr = std::ptr::NonNull::new(CUBE_VERTS.as_ptr() as *mut _)
            .ok_or("decal cube vertex slice is null")?;
        device
            .newBufferWithBytes_length_options(
                ptr,
                std::mem::size_of_val(&CUBE_VERTS),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or("failed to create decal cube vertex buffer")?
    };
    let ibuf = unsafe {
        let ptr = std::ptr::NonNull::new(CUBE_INDICES.as_ptr() as *mut _)
            .ok_or("decal cube index slice is null")?;
        device
            .newBufferWithBytes_length_options(
                ptr,
                std::mem::size_of_val(&CUBE_INDICES),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or("failed to create decal cube index buffer")?
    };
    let samp = {
        let desc = MTLSamplerDescriptor::new();
        desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
        desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
        desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        device
            .newSamplerStateWithDescriptor(&desc)
            .ok_or("failed to create decal sampler state")?
    };
    Ok((ps, vbuf, ibuf, samp))
}

// Shared storage so the average kernel's writes are visible to the CPU
// readback at the top of the next frame without an explicit GPU<->CPU sync.
fn make_auto_exposure_histogram(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
    let hist_bytes =
        vec![0u8; std::mem::size_of::<u32>() * crate::gfx::auto_exposure::HISTOGRAM_BINS];
    unsafe {
        let ptr = std::ptr::NonNull::new(hist_bytes.as_ptr() as *mut _)
            .ok_or("auto-exposure histogram allocation failed")?;
        device
            .newBufferWithBytes_length_options(
                ptr,
                hist_bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| "failed to create auto-exposure histogram buffer".to_string())
    }
}

fn make_auto_exposure_output(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
    let out_bytes = vec![0u8; std::mem::size_of::<f32>()];
    unsafe {
        let ptr = std::ptr::NonNull::new(out_bytes.as_ptr() as *mut _)
            .ok_or("auto-exposure output allocation failed")?;
        device
            .newBufferWithBytes_length_options(
                ptr,
                out_bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| "failed to create auto-exposure output buffer".to_string())
    }
}
