// src/metal/init/mod.rs
//
// MtlContext construction. The constructor is intentionally a flat top-to-
// bottom sequence so the order of dependencies stays obvious; helpers for
// self-contained sub-phases live in sibling modules:
//
//   window.rs    NSWindow + MTKView setup + initial HDR target sizing
//   pipelines.rs Vertex descriptor, main pipeline (+cull/bindless), instanced
//                pipeline, depth-stencil state
//   effects.rs   Bloom, TAA, velocity, SSAO, SSR, decal, volumetric fog,
//                auto-exposure (everything gated on per-world settings)
//
// What still lives inline here:
//   * Device + command queue creation
//   * Geometry, texture, sampler, IBL and LUT uploads (they share local state
//     with shadow + text + post-pipeline setup)
//   * Shadow pipeline + shadow map (depends on the shared vertex descriptor)
//   * Text + post-process pipelines + their samplers
//   * BVH partition + previous-model snapshot + hot-reload watcher
//   * The final `Self { ... }` literal
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

pub(super) mod effects;
pub(crate) mod pipelines;
mod window;
// Runtime vsync toggle reaches the backing CAMetalLayer through this helper.
pub(crate) use window::set_display_sync;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCompareFunction, MTLCreateSystemDefaultDevice, MTLDevice as _, MTLResourceOptions,
    MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLTexture,
};

use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::NUM_SHADOW_CASCADES;

use super::context::*;
use super::input::KeyState;
use super::math::IDENTITY4;
use super::pipeline::{build_post_pipeline, build_text_pipeline};
use super::texture::{
    EnvironmentMapTextures, create_fallback_color_lut, create_fallback_cubemap,
    create_fallback_texture, create_hdr_targets, create_shadow_map_array,
    create_shadow_map_fallback, upload_color_lut, upload_environment_map, upload_texture,
};

impl MtlContext {
    // Create a window and Metal render pipeline from the assembled backend
    // inputs (see `crate::gfx::backend_init::BackendInit` for per-field docs).
    //
    // Main-shader entry points must be named "vertex_main" and
    // "fragment_main"; the shadow pass is engine-internal (compiled from
    // `shadow_map.metal`) and enabled whenever `shadows.map_size > 0`.
    pub fn new(init: crate::gfx::backend_init::BackendInit<'_>) -> Result<Self, String> {
        use crate::gfx::backend_init::{
            BackendInit, MediaPayloads, PostSettings, SceneData, ShaderBytes, ShadowParams, WorldFx,
        };
        let BackendInit {
            window,
            // The Metal validation layer is enabled by the CLI re-execing with
            // MTL_DEBUG_LAYER, not through this flag.
            validation: _,
            frames_in_flight,
            vsync,
            clear_color,
            hot_reload,
            scene:
                SceneData {
                    vertices,
                    indices,
                    draw_objects,
                    instanced_clusters,
                    // Unused on Metal: the object / draw-args transient rings and
                    // the cull ICB auto-grow to `cull_count()` each frame (the
                    // skinned count is set later in `upload_skinned`, and resident
                    // chunks fold into the per-frame rebuild), so no init-time
                    // sizing is needed. DX/VK pre-size fixed buffers from these.
                    n_skinned: _,
                    n_chunk_max: _,
                },
            shaders:
                ShaderBytes {
                    vert: vert_lib_bytes,
                    frag: frag_lib_bytes,
                    // The Metal shadow shader is engine-internal.
                    shadow: _,
                    vert_instanced: vert_instanced_lib_bytes,
                },
            media:
                MediaPayloads {
                    textures,
                    normal_maps,
                    text_atlases,
                    env_map_bytes,
                    color_lut_bytes,
                },
            light_uniforms,
            shadows:
                ShadowParams {
                    map_size: shadow_map_size,
                    update: shadow_update,
                    distance: shadow_distance,
                    cascades: shadow_cascades,
                },
            anisotropy,
            planar_planes,
            post:
                PostSettings {
                    // The backend overwrites `hdr_output` after EDR negotiation,
                    // so this is rebound mutably below.
                    mut post_process,
                    taa_enabled,
                    ssao: ssao_settings,
                    ssr: ssr_settings,
                    ssgi: ssgi_settings,
                    rt_reflections: rt_reflection_settings,
                    reflection_blur_scale,
                    auto_exposure: auto_exposure_settings,
                    auto_exposure_bias_ev,
                    hdr_display: hdr_display_requested,
                    hdr_pq: hdr_pq_requested,
                    temporal_upscaling: temporal_upscaling_requested,
                    upscale_scale: upscale_scale_requested,
                    // Metal always uses MetalFX for temporal upscaling.
                    upscale_backend: _,
                    occlusion_two_pass: occlusion_two_pass_requested,
                },
            fx:
                WorldFx {
                    decals,
                    particles,
                    fog: fog_settings,
                    water_surfaces,
                    glass_panels,
                    sdf_volumes,
                },
            requirements,
        } = init;
        let (title, width, height) = (window.title.as_str(), window.width, window.height);
        // all Metal and AppKit calls must happen on the main thread
        let mtm = objc2::MainThreadMarker::new()
            .ok_or("MtlContext::new must be called from the main thread")?;

        let device = MTLCreateSystemDefaultDevice().ok_or("no default Metal device")?;

        let command_queue = device
            .newCommandQueue()
            .ok_or("failed to create Metal command queue")?;

        // Main + cull + bindless argument encoder. The bundle's `bindless`
        // flag drives the texture-pool overflow warning below. A world with no
        // 3D scene content skips the main PBR pipeline and the whole GPU-cull
        // path: the Main pass then survives as a bare clear the composite pass
        // samples (the same shape a world_hidden frame takes).
        let vert_desc = pipelines::make_vertex_descriptor();
        let (
            pipeline_state,
            bindless,
            cull_pipeline,
            cull_icb_arg_encoder,
            cull_pipeline_phase2,
            cull_icb2_arg_encoder,
            bindless_tex_arg_encoder,
        ) = if requirements.scene {
            let pipelines::MainPipelineBundle {
                pipeline_state,
                bindless,
                cull_pipeline,
                cull_icb_arg_encoder,
                cull_pipeline_phase2,
                cull_icb2_arg_encoder,
                bindless_tex_arg_encoder,
            } = pipelines::build_main_pipeline(
                &device,
                &vert_desc,
                vert_lib_bytes,
                frag_lib_bytes,
                hot_reload,
            )?;
            (
                Some(pipeline_state),
                bindless,
                cull_pipeline,
                cull_icb_arg_encoder,
                cull_pipeline_phase2,
                cull_icb2_arg_encoder,
                bindless_tex_arg_encoder,
            )
        } else {
            (None, false, None, None, None, None, None)
        };

        // Two-pass occlusion is only usable on the bindless cull path (the
        // phase-2 pipeline exists exactly then). Gate the request here so the
        // runtime flag is true only when the feature can actually run.
        let two_pass_occlusion = occlusion_two_pass_requested && cull_pipeline_phase2.is_some();

        let instanced_pipeline_state = pipelines::build_instanced_pipeline(
            &device,
            &vert_desc,
            vert_instanced_lib_bytes,
            frag_lib_bytes,
            !instanced_clusters.is_empty(),
        )?;

        let depth_state = pipelines::make_depth_state(&device)?;
        let depth_state_read_only = pipelines::make_depth_state_read_only(&device)?;

        // upload vertex and index data into GPU-accessible buffers. A
        // geometry-less world (text-only) has empty slices; Metal rejects a
        // zero-length buffer, so a minimal placeholder is allocated instead --
        // the draw list is empty so the placeholder is never read.
        let vertex_buffer = if vertices.is_empty() {
            device
                .newBufferWithLength_options(
                    std::mem::size_of::<Vertex>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or("failed to create placeholder vertex buffer")?
        } else {
            unsafe {
                let ptr = std::ptr::NonNull::new(vertices.as_ptr() as *mut _)
                    .ok_or("vertex slice pointer is null")?;
                device
                    .newBufferWithBytes_length_options(
                        ptr,
                        std::mem::size_of_val(vertices),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or("failed to create vertex buffer")?
            }
        };

        let index_buffer = if indices.is_empty() {
            device
                .newBufferWithLength_options(
                    std::mem::size_of::<u32>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or("failed to create placeholder index buffer")?
        } else {
            unsafe {
                let ptr = std::ptr::NonNull::new(indices.as_ptr() as *mut _)
                    .ok_or("index slice pointer is null")?;
                device
                    .newBufferWithBytes_length_options(
                        ptr,
                        std::mem::size_of_val(indices),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or("failed to create index buffer")?
            }
        };

        // upload textures; fall back to a 1x1 opaque white texture when none provided
        let gpu_textures = if textures.is_empty() {
            vec![create_fallback_texture(&device)?]
        } else {
            textures
                .iter()
                .enumerate()
                .map(|(i, (w, h, pixels))| {
                    upload_texture(&device, *w, *h, pixels)
                        .map_err(|e| format!("texture[{}]: {}", i, e))
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        // normal-map texture pool: slot 0 = 1x1 flat-normal fallback (tangent-space (0,0,1)),
        // followed by caller-supplied normal map textures.
        let flat_normal = upload_texture(&device, 1, 1, &[128u8, 128, 255, 255])
            .map_err(|e| format!("flat normal fallback: {}", e))?;
        let mut gpu_normal_maps: Vec<Retained<ProtocolObject<dyn MTLTexture>>> = vec![flat_normal];
        for (i, (w, h, pixels)) in normal_maps.iter().enumerate() {
            let tex = upload_texture(&device, *w, *h, pixels)
                .map_err(|e| format!("normal_map[{}]: {}", i, e))?;
            gpu_normal_maps.push(tex);
        }

        // The bindless static pass binds the albedo + normal textures into one
        // capped pool. A world that exceeds the cap still renders, but objects
        // whose pool index would overflow get clamped to the last slot.
        if bindless && gpu_textures.len() + gpu_normal_maps.len() > BINDLESS_TEXTURE_COUNT {
            tracing::warn!(
                "Metal: texture pool ({} albedo + {} normal) exceeds bindless capacity {}; \
                 some objects will sample a clamped texture",
                gpu_textures.len(),
                gpu_normal_maps.len(),
                BINDLESS_TEXTURE_COUNT,
            );
        }

        // linear filter, repeat wrap -- matches the room shader expectations.
        // Mipmap linear + anisotropy let minified scene textures trilinear-select
        // down the mip chain now that uploads carry one, instead of aliasing from
        // mip 0. The degree comes from GraphicsConfig.anisotropy (default 8),
        // clamped to Metal's guaranteed 1..16 range.
        let sampler = {
            let desc = MTLSamplerDescriptor::new();
            desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
            desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
            desc.setMipFilter(objc2_metal::MTLSamplerMipFilter::Linear);
            desc.setSAddressMode(MTLSamplerAddressMode::Repeat);
            desc.setTAddressMode(MTLSamplerAddressMode::Repeat);
            desc.setMaxAnisotropy(anisotropy.clamp(1, 16) as usize);
            device
                .newSamplerStateWithDescriptor(&desc)
                .ok_or("failed to create sampler state")?
        };

        // compare sampler for PCF: always created so texture(2) / sampler(1) are
        // always bound; LessEqual returns 1.0 (lit) when reference <= stored depth.
        let shadow_sampler = {
            let desc = MTLSamplerDescriptor::new();
            desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
            desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
            desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
            desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
            desc.setCompareFunction(MTLCompareFunction::LessEqual);
            device
                .newSamplerStateWithDescriptor(&desc)
                .ok_or("failed to create shadow sampler state")?
        };

        // Cube sampler: linear filter + clamp-to-edge + mipmap linear for prefilter
        // roughness lookups. Bound at sampler(2) and shared by both IBL cubes.
        let cube_sampler = {
            let desc = MTLSamplerDescriptor::new();
            desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
            desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
            desc.setMipFilter(objc2_metal::MTLSamplerMipFilter::Linear);
            desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
            desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
            desc.setRAddressMode(MTLSamplerAddressMode::ClampToEdge);
            device
                .newSamplerStateWithDescriptor(&desc)
                .ok_or("failed to create cube sampler state")?
        };

        // IBL: either upload the supplied EnvironmentMap payload or build a
        // 1x1 grey fallback cube pair so texture(3) / texture(4) are always
        // bound. The fragment shader uses `prefilter_mip_count == 0` to
        // detect the fallback and skip IBL math.
        let env_map = if let Some(bytes) = env_map_bytes {
            let view = crate::build::environment_map::deserialise(bytes)
                .map_err(|e| format!("EnvironmentMap payload malformed: {}", e))?;
            upload_environment_map(
                &device,
                view.irradiance_face,
                view.irradiance_bytes,
                view.prefilter_face,
                &view.prefilter_mip_bytes,
            )?
        } else {
            EnvironmentMapTextures {
                irradiance: create_fallback_cubemap(&device, [0.05, 0.05, 0.05, 1.0])?,
                prefilter: create_fallback_cubemap(&device, [0.05, 0.05, 0.05, 1.0])?,
                prefilter_mip_count: 0,
            }
        };

        // Colour-grading LUT: upload the declared ColorLut payload, or build a
        // 2x2x2 identity LUT so the composite pass always binds a valid 3D
        // texture. With the identity LUT the grade is a no-op at any strength.
        let color_lut = if let Some(bytes) = color_lut_bytes {
            let (size, data) = crate::build::color_lut::deserialise(bytes)
                .map_err(|e| format!("ColorLut payload malformed: {}", e))?;
            upload_color_lut(&device, size, data)?
        } else {
            create_fallback_color_lut(&device)?
        };

        // shadow pipeline + array map: created only when shadow_map_size > 0.
        // The fallback 1x1 shadow map (all depth = 1.0 = max = lit) is always
        // bound so fragment shaders can safely sample texture(2) as a depth array.
        let (shadow_pipeline_state, shadow_map, shadow_uniforms_init, effective_shadow_size) =
            if shadow_map_size > 0 {
                let shadow_ps = pipelines::build_shadow_pipeline(&device, &vert_desc, hot_reload)?;
                // Depth32Float 2D array, NUM_SHADOW_CASCADES layers, GPU-private.
                let shadow_tex =
                    create_shadow_map_array(&device, shadow_map_size, NUM_SHADOW_CASCADES as u32)?;
                (
                    Some(shadow_ps),
                    shadow_tex,
                    crate::gfx::csm::empty_shadow_uniforms(),
                    shadow_map_size,
                )
            } else {
                // 1x1 fallback depth array (value 1.0 = fully lit).
                let shadow_tex = create_shadow_map_fallback(&device)?;
                (
                    None,
                    shadow_tex,
                    crate::gfx::csm::empty_shadow_uniforms(),
                    1,
                )
            };

        // GPU-driven cascaded-shadow resources: the frustum-only
        // shadow cull kernel + its ICB argument encoder, and the depth-only
        // bindless shadow render pipeline. Built only on the bindless path with
        // shadows enabled; non-bindless / no-shadow worlds keep the legacy
        // per-cascade CPU shadow loop and leave these `None`. The shadow ICB +
        // its argument buffer are allocated lazily by `ensure_shadow_icb_capacity`
        // (sized to NUM_SHADOW_CASCADES * cull_count once geometry is known).
        let (shadow_cull_pipeline, shadow_bindless_pipeline, shadow_icb_arg_encoder) =
            if shadow_pipeline_state.is_some() && bindless {
                let (sc, sc_enc) = super::cull::build_shadow_cull_pipeline(&device, hot_reload)?;
                let sb =
                    pipelines::build_shadow_bindless_pipeline(&device, &vert_desc, hot_reload)?;
                (Some(sc), Some(sb), Some(sc_enc))
            } else {
                (None, None, None)
            };

        // Cache the first directional light's direction; per-frame CSM updates
        // use it (the light direction is treated as static at init -- if you
        // want a moving sun, re-cache on light change).
        let shadow_light_dir = if light_uniforms.num_directional > 0 {
            light_uniforms.directional[0].direction
        } else {
            // Match LightUniforms::DEFAULT.
            [-0.3, 0.85, 0.4]
        };

        // Window + MTKView + initial drawable sizing. A geometry-less world
        // is clamped to 1x1 HDR/bloom/effect targets so the composite pass
        // alone runs at the full drawable size (it samples the 1x1 uniformly).
        // Window setup also resolves the swapchain colour-output mode
        // (`HdrOutputMode::Sdr` vs `Hdr`); the post + text pipelines that
        // target the drawable need to know that mode to pick BGRA8Unorm vs
        // RGBA16Float, so this hop happens before pipeline construction.
        // Scene-less (UI / text only) worlds clamp the HDR / bloom / effect
        // targets to 1x1; keyed off the derived requirements rather than raw
        // vertex presence so a vertex-less world that still renders 3D content
        // (SDF volumes, water, glass) keeps full-size targets.
        let geometry_less = !requirements.scene;
        let window::WindowSetup {
            window,
            mtk_view,
            pump_events,
            initial_w,
            initial_h,
            fullscreen,
            window_delegate,
            hdr_mode,
        } = window::setup_window_and_view(
            mtm,
            &device,
            title,
            width,
            height,
            geometry_less,
            hdr_display_requested,
            hdr_pq_requested,
            hot_reload,
        )?;
        // Honor the requested vsync on the backing CAMetalLayer (default
        // CAMetalLayer presentation is display-synced).
        window::set_display_sync(&mtk_view, vsync);
        let swap_pixel_format = window::swap_pixel_format(hdr_mode);
        // Resolved EDR encoding, kept for the headless `screenshot` decode (it
        // must know scRGB-linear vs PQ to turn the captured `RGBA16Float`
        // drawable into a display-correct PNG). `None` on the SDR path.
        let hdr_encoding = match hdr_mode {
            crate::gfx::hdr_output::HdrOutputMode::Hdr { encoding, .. } => Some(encoding),
            crate::gfx::hdr_output::HdrOutputMode::Sdr => None,
        };
        // Surface the resolved mode to the composite shader via the post
        // uniform. On the SDR path both flags stay 0.0 and the shader runs
        // the full ACES + gamma + FXAA + LUT chain unchanged. On the HDR
        // path `hdr_output` lights up; `pq_output` further picks PQ-encode
        // vs scRGB-linear passthrough inside that branch.
        post_process.hdr_output = hdr_mode.shader_flag();
        post_process.pq_output = hdr_mode.pq_flag();

        // text rendering resources
        let (text_pipeline_state, gpu_text_atlases) = if text_atlases.is_empty() {
            (None, Vec::new())
        } else {
            let text_ps = build_text_pipeline(&device, swap_pixel_format, hot_reload)?;
            let mut gpu_atlases = Vec::with_capacity(text_atlases.len());
            for (i, (aw, ah, pixels)) in text_atlases.iter().enumerate() {
                let tex = upload_texture(&device, *aw, *ah, pixels)
                    .map_err(|e| format!("text_atlas[{}]: {}", i, e))?;
                gpu_atlases.push(tex);
            }
            (Some(text_ps), gpu_atlases)
        };

        let text_sampler = {
            let desc = MTLSamplerDescriptor::new();
            desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
            desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
            desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
            desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
            device
                .newSamplerStateWithDescriptor(&desc)
                .ok_or("failed to create text sampler state")?
        };

        // Post-process pipeline + sampler. The composite pass samples the
        // resolved HDR target with a linear-clamp filter and writes either
        // ACES-tonemapped + gamma + FXAA-filtered output (SDR drawable) or
        // linear extended-range values (HDR drawable) into the swapchain.
        let post_pipeline_state = build_post_pipeline(&device, swap_pixel_format, hot_reload)?;
        let post_sampler = {
            let desc = MTLSamplerDescriptor::new();
            desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
            desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
            desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
            desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
            // Clamp the R axis too -- the same sampler trilinearly filters the
            // 3D colour LUT in the composite pass.
            desc.setRAddressMode(MTLSamplerAddressMode::ClampToEdge);
            device
                .newSamplerStateWithDescriptor(&desc)
                .ok_or("failed to create post sampler state")?
        };

        // MetalFX temporal upscaler. Built ahead of the HDR + post targets
        // because the resolved input size (clamped to the device's supported
        // scale range) determines the render resolution every other 3D-scene
        // target uses; bloom + composite stay at the drawable (output)
        // resolution. Failure or unsupported hardware falls back silently to
        // native-resolution rendering: the entire `temporal_upscaling`
        // feature is asset-driven so a world that doesn't author it pays no
        // construction cost either way.
        let upscaler = if temporal_upscaling_requested {
            if super::post::temporal_scaler_supported(&device) {
                match super::post::MetalFXUpscaler::new(
                    &device,
                    initial_w,
                    initial_h,
                    upscale_scale_requested,
                ) {
                    Ok(u) => {
                        tracing::info!(
                            "MetalFX: temporal upscaling on: render {}x{} → present {}x{} ({}x scale)",
                            u.input_width,
                            u.input_height,
                            u.output_width,
                            u.output_height,
                            (u.input_width as f32) / (u.output_width.max(1) as f32),
                        );
                        Some(u)
                    }
                    Err(e) => {
                        tracing::warn!(
                            "MetalFX: temporal scaler creation failed ({}); falling back to native resolution",
                            e
                        );
                        None
                    }
                }
            } else {
                tracing::warn!(
                    "MetalFX: temporal scaler not supported on this GPU; falling back to native resolution"
                );
                None
            }
        } else {
            None
        };
        let (render_w, render_h, upscale_scale) = match &upscaler {
            Some(u) => (
                u.input_width,
                u.input_height,
                (u.input_width as f32) / (u.output_width.max(1) as f32),
            ),
            None => (initial_w, initial_h, 1.0),
        };
        // With the MetalFX scaler doing temporal accumulation, the TAA pass
        // is bypassed but the velocity pre-pass and projection jitter stay
        // on (the scaler consumes both). `effective_taa_enabled` is what
        // the engine carries downstream; the asset `taa` flag is ignored
        // when upscaling is on.
        let upscaling_active = upscaler.is_some();
        let effective_taa_enabled = taa_enabled && !upscaling_active;
        let velocity_needed = effective_taa_enabled || upscaling_active;

        let hdr_targets = create_hdr_targets(&device, render_w, render_h, HDR_SAMPLE_COUNT)?;

        // Hi-Z depth pyramid for GPU-driven occlusion culling. Built exactly
        // when the bindless cull pipeline is active and sized to the render
        // (depth) resolution; `resize_targets_if_needed` rebuilds it on a
        // window resize. The cull kernel projects each AABB through the
        // previous frame's depth pyramid and culls fully-occluded objects.
        let hiz = if cull_pipeline.is_some() {
            Some(super::hiz::HiZResources::new(
                &device, render_w, render_h, hot_reload,
            )?)
        } else {
            None
        };

        // Post-process effect chain: bloom is built for any world with a 3D
        // scene; TAA / velocity / SSAO / SSR / decal / fog / auto-exposure are
        // gated on their own settings so a world that disables them pays zero
        // construction cost.
        let effects::EffectsBundle {
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
        } = effects::build_effects(
            &device,
            &vert_desc,
            requirements.scene,
            render_w,
            render_h,
            initial_w,
            initial_h,
            effective_taa_enabled,
            velocity_needed,
            instanced_pipeline_state.is_some(),
            &ssao_settings,
            &ssr_settings,
            &ssgi_settings,
            &rt_reflection_settings,
            reflection_blur_scale,
            &decals,
            &particles,
            &fog_settings,
            &auto_exposure_settings,
            auto_exposure_bias_ev,
            hot_reload,
        )?;

        // Transparent water surfaces. Built only when the world declared
        // ≥1 `WaterSurface`; the transparent-pass executor stays a no-op
        // otherwise. Per-surface tessellated grids upload once at init.
        let (water_pipeline, water_pipeline_rt, water_pipeline_rt_textured, mut water_records) =
            if water_surfaces.is_empty() {
                (None, None, None, Vec::new())
            } else {
                let ps = super::water::build_water_pipeline(&device, hot_reload)?;
                // The ray-traced variants are built whenever the device can ray
                // trace (regardless of whether RT is on at launch), so a live RT
                // toggle can select them without a pipeline rebuild. The shader
                // uses `metal_raytracing`, so it must not be compiled on a non-RT
                // device. The textured variant additionally needs a bindless world
                // at draw time; it is selected over the flat variant then.
                let (ps_rt, ps_rt_tex) = if super::raytrace::raytracing_supported(&device) {
                    (
                        Some(super::water::build_water_pipeline_rt(&device, hot_reload)?),
                        Some(super::water::build_water_pipeline_rt_textured(
                            &device, hot_reload,
                        )?),
                    )
                } else {
                    (None, None)
                };
                let mut records = Vec::with_capacity(water_surfaces.len());
                for s in &water_surfaces {
                    records.push(super::water::build_water_surface_record(&device, s)?);
                }
                (Some(ps), ps_rt, ps_rt_tex, records)
            };

        // Translucent glass panels. Built only when the world declared ≥1
        // `GlassPanel`; rides the same transparent pass as water. Per-panel
        // world-space quads upload once at init.
        let (glass_pipeline, glass_pipeline_rt, glass_pipeline_rt_textured, mut glass_records) =
            if glass_panels.is_empty() {
                (None, None, None, Vec::new())
            } else {
                let ps = super::glass::build_glass_pipeline(&device, hot_reload)?;
                // The ray-traced variants are built whenever the device can ray
                // trace (regardless of whether RT is on at launch), so a live RT
                // toggle can select them without a pipeline rebuild. The shader
                // uses `metal_raytracing`, so it must not be compiled on a non-RT
                // device. The textured variant additionally needs a bindless world
                // at draw time; it is selected over the flat variant then.
                let (ps_rt, ps_rt_tex) = if super::raytrace::raytracing_supported(&device) {
                    (
                        Some(super::glass::build_glass_pipeline_rt(&device, hot_reload)?),
                        Some(super::glass::build_glass_pipeline_rt_textured(
                            &device, hot_reload,
                        )?),
                    )
                } else {
                    (None, None)
                };
                let mut records = Vec::with_capacity(glass_panels.len());
                for g in &glass_panels {
                    records.push(super::glass::build_glass_panel_record(&device, g)?);
                }
                (Some(ps), ps_rt, ps_rt_tex, records)
            };

        // Transparent glass MESH pipelines (Layer 2): built whenever the device can
        // ray trace, INDEPENDENT of any `GlassPanel` -- the transparent material
        // lives on imported meshes, not panels, and a live RT toggle then has them
        // ready. `glass_mesh_pipeline_rt.is_some()` gates the whole transparent-mesh
        // reroute; `seethrough_mesh_indices` marks which `draw_objects` carry it.
        let (glass_mesh_pipeline_rt, glass_mesh_pipeline_rt_textured) =
            if super::raytrace::raytracing_supported(&device) {
                (
                    Some(super::glass::build_glass_mesh_pipeline_rt(
                        &device, hot_reload,
                    )?),
                    Some(super::glass::build_glass_mesh_pipeline_rt_textured(
                        &device, hot_reload,
                    )?),
                )
            } else {
                (None, None)
            };
        // Layer 2 see-through glass is opt-in per `Material` (the `see_through`
        // arg, which implies `transparent`): see-through only looks right when the
        // space behind the glass is modelled. A material that is `transparent` but
        // NOT `see_through` renders as Layer 1 (opaque, low roughness, scene
        // reflections) = tinted reflective glass that hides the interior. This list
        // drives the producer + the opaque-pass skip (`mesh_glass_active`) + the
        // RT-BLAS exclude together.
        let seethrough_mesh_indices: Vec<usize> = draw_objects
            .iter()
            .enumerate()
            .filter(|(_, o)| o.material.transparent != 0 && o.material.see_through != 0)
            .map(|(i, _)| i)
            .collect();
        // The Layer 2 path is enabled when at least one material opts into
        // see-through AND the mesh pipeline built (RT-capable device). Mirrors
        // `MtlContext::seethrough_meshes_enabled`; used here for the init-time BVH
        // build, which must exclude the see-through meshes it will reroute.
        let seethrough_enabled =
            !seethrough_mesh_indices.is_empty() && glass_mesh_pipeline_rt.is_some();

        // Planar reflection set: group every flat reflector (water surfaces +
        // glass panes) into a bounded number of distinct planes, one mirror render
        // each. Water planes are listed first so they take slots before glass when
        // the budget is tight. Each reflector records the slot it samples; planes
        // past the budget get no slot and keep the box-projected probe cube
        // (warned here, not silently dropped). The set is built only when the
        // world has >=1 reflector; the per-frame pass is additionally gated on RT
        // being off.
        let planar_reflection = {
            let mut planes: Vec<[f32; 4]> = Vec::new();
            for s in &water_surfaces {
                // Horizontal plane at the surface base height, normal +y.
                planes.push([0.0, 1.0, 0.0, -s.centre[1]]);
            }
            for g in &glass_panels {
                // The pane plane: normal (unit from `from_args`) through centre,
                // so `n . p + d = 0` on the pane.
                let n = g.normal;
                let d = -(n[0] * g.centre[0] + n[1] * g.centre[1] + n[2] * g.centre[2]);
                planes.push([n[0], n[1], n[2], d]);
            }
            // The budget is capped at the capacity ceiling the mirror targets + ICB
            // slots are sized to, so a stale/over-large preset value can never
            // over-allocate.
            let planar_budget = planar_planes.min(super::planar::MAX_PLANAR_PLANES);
            let assignment =
                crate::gfx::planar_reflection::assign_planar_slots(&planes, planar_budget);
            // Record each reflector's slot (water first, then glass, matching the
            // push order above).
            for (rec, slot) in water_records.iter_mut().zip(assignment.slots.iter()) {
                rec.planar_slot = *slot;
            }
            let glass_offset = water_records.len();
            for (rec, slot) in glass_records
                .iter_mut()
                .zip(assignment.slots[glass_offset..].iter())
            {
                rec.planar_slot = *slot;
            }
            let overflow = assignment.slots.iter().filter(|s| s.is_none()).count();
            if overflow > 0 {
                tracing::warn!(
                    "planar reflection: {} reflector plane(s) exceed the budget of {} \
                     and fall back to the box-projected probe cube",
                    overflow,
                    planar_budget
                );
            }
            if assignment.representatives.is_empty() {
                None
            } else {
                Some(super::planar::create_planar_set(
                    &device,
                    render_w,
                    render_h,
                    HDR_SAMPLE_COUNT,
                    &assignment.representatives,
                )?)
            }
        };

        // Raymarched SDF volumes. Each volume builds its own pipeline
        // from the wrapped user source (helpers + user + template) at
        // init time; the proxy-cube buffers are allocated once and
        // shared across all volumes. Empty input list → both stay
        // None / empty and the raymarch executor short-circuits.
        let (raymarch_records, raymarch_cube_vertex_buffer, raymarch_cube_index_buffer) =
            if sdf_volumes.is_empty() {
                (Vec::new(), None, None)
            } else {
                let mut records = Vec::with_capacity(sdf_volumes.len());
                for (volume, source_bytes, label) in &sdf_volumes {
                    records.push(super::raymarch::build_raymarch_volume_record(
                        &device,
                        volume,
                        source_bytes,
                        label,
                    )?);
                }
                let (vb, ib) = super::raymarch::build_raymarch_cube_buffers(&device)?;
                (records, Some(vb), Some(ib))
            };

        let (cull_bvh, always_draw) = crate::gfx::bvh::partition_draw_objects(&draw_objects);

        // Seed the draw-slot allocator and the always_draw membership map from
        // the initial draw set so runtime spawn/despawn can recycle vacated
        // slots and add a recycled slot to always_draw exactly once.
        let draw_slots = crate::gfx::draw_slot::DrawSlotAllocator::with_len(draw_objects.len());
        let mut always_draw_member = vec![false; draw_objects.len()];
        for &idx in &always_draw {
            always_draw_member[idx as usize] = true;
        }

        // Snapshot each object's initial model matrix as its "previous" frame
        // so the velocity pre-pass sees zero motion until the first update.
        let prev_draw_models: Vec<[[f32; 4]; 4]> = draw_objects.iter().map(|o| o.model).collect();

        // Shader hot-reload wiring. The atomic flag is shared between the
        // notify watcher thread and `draw_frame`, plus (eventually) the
        // debug WS `reload-shaders` command path via `GraphicsSystem`.
        // Watcher creation is best-effort: a missing source dir or a notify
        // error logs a warning and disables only the watcher half -- the
        // debug command still works on the same flag.
        let (shader_reload_pending, shader_watcher) = if hot_reload {
            let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let watcher = super::hot_reload::spawn(std::sync::Arc::clone(&flag));
            (Some(flag), watcher)
        } else {
            (None, None)
        };

        // Built before `device` moves into Self. Returns `None` when the
        // device does not expose the timestamp counter set; the per-pass
        // GPU timer then stays at zero for every pass.
        let pass_timing = super::pass_timing::PassTimingResources::new(&device);
        tracing::info!(
            "pass-timing: per-pass GPU sample buffers {}",
            if pass_timing.is_some() {
                "ready"
            } else {
                "unavailable (no MTLCommonCounterSetTimestamp)"
            }
        );

        // Capture the resolved EDR multiplier (or None on SDR) so the HUD +
        // debug WS can report it. The shader flag in `post_process.hdr_output`
        // tracks "is HDR on" as a bool; this captures the multiplier itself.
        let max_edr = match hdr_mode {
            crate::gfx::hdr_output::HdrOutputMode::Hdr { max_edr, .. } => Some(max_edr),
            crate::gfx::hdr_output::HdrOutputMode::Sdr => None,
        };

        // Build the scene acceleration structure for hardware ray-traced
        // reflections. Only when the world enabled RT (the RT pipeline is built
        // above) and the GPU supports ray tracing; `build_rt_accel` returns None
        // when the scene has no resident geometry, in which case the RT pass
        // stays a no-op (draw/mod gates `rt_reflections_enabled` on this being
        // Some). Built here because it needs the shared geometry buffers + draw
        // list, which exist by now; resolution-independent, so untouched on
        // resize. A wholesale rebuild on geometry change is the current update path.
        let rt_accel = if rt_reflection_settings.is_some()
            && super::raytrace::raytracing_supported(&device)
        {
            match super::raytrace::build_rt_accel(
                &device,
                &command_queue,
                &vertex_buffer,
                &index_buffer,
                &draw_objects,
                &instanced_clusters,
                gpu_textures.len(),
                gpu_normal_maps.len(),
                // Skinned meshes upload after `new`, so the initial BVH is
                // static + instanced; the first frame's update seeds the
                // skinned geometry once `upload_skinned` has run.
                None,
                seethrough_enabled,
            )? {
                Some(a) => {
                    tracing::info!(
                        "ray-traced reflections: built BVH over {} static objects",
                        a.blas.len()
                    );
                    Some(a)
                }
                None => {
                    tracing::warn!(
                        "ray-traced reflections requested but the scene has no static geometry to build a BVH from; reflections disabled"
                    );
                    None
                }
            }
        } else {
            if rt_reflection_settings.is_some() {
                tracing::warn!(
                    "ray-traced reflections requested but this GPU does not support hardware ray tracing; falling back (no RT reflections)"
                );
            }
            None
        };

        // How the BVH tracks moving props (CN_RT_DYNAMIC; Auto default:
        // dirty-gated TLAS rebuild, so a static scene never rebuilds).
        let rt_dynamic_mode = super::raytrace::RtDynamicMode::from_env();
        if rt_accel.is_some() {
            tracing::info!("ray-traced reflections: dynamic transform mode = {rt_dynamic_mode:?}");
        }

        // Fold every instanced-cluster instance into the GPU-driven bindless
        // main pass: each becomes a `GpuObjectData` record appended after the
        // static objects, drawn through the shared cull + indirect path
        // (`build_object_buffer` / `build_draw_args_buffer` re-append these every
        // frame; see `cull_count`). Built once here against the final bindless
        // pool counts (`gpu_textures` / `gpu_normal_maps`, the same counts the
        // static fill uses) via the Metal-local `metal_instance_records`, which
        // addresses the flat pool with Metal's CPU-bias convention (NOT the
        // shared core `instance_object_records`, which is the DX/VK raw-index
        // convention): instances are placed at world load and never move, so the
        // records are static. Per-instance LOD is deferred -- every instance uses
        // the cluster base index range, so its draw args never change either.
        let n_instances: usize = instanced_clusters.iter().map(|c| c.instances.len()).sum();
        let (instance_records, instance_draw_args) = {
            use crate::gfx::render_types::{GpuDrawArgs, draw_args_flags};
            let records = super::cull::metal_instance_records(
                &instanced_clusters,
                gpu_textures.len(),
                gpu_normal_maps.len(),
            );
            let mut args: Vec<GpuDrawArgs> = Vec::with_capacity(records.len());
            for cluster in &instanced_clusters {
                for _ in &cluster.instances {
                    args.push(GpuDrawArgs {
                        index_count: cluster.index_count as u32,
                        index_offset: cluster.index_offset as u32,
                        base_vertex: 0,
                        flags: draw_args_flags(true, true, true),
                    });
                }
            }
            (records, args)
        };

        Ok(Self {
            device,
            command_queue,
            swap_pixel_format,
            max_edr,
            hdr_encoding,
            last_present_texture: None,
            pipeline_state,
            bindless,
            cull: super::cull::CullState {
                pipeline: cull_pipeline,
                icb: None,
                icb_arg_encoder: cull_icb_arg_encoder,
                icb_arg_buffer: None,
                icb_capacity: 0,
                pipeline_phase2: cull_pipeline_phase2,
                icb_2: None,
                icb_2_arg_encoder: cull_icb2_arg_encoder,
                icb_2_arg_buffer: None,
                status_buffer: None,
                two_pass_occlusion,
                hiz,
                prev_view_proj: IDENTITY4,
                cur_view_proj: IDENTITY4,
                hiz_valid: false,
                shadow_pipeline: shadow_cull_pipeline,
                shadow_bindless_pipeline,
                shadow_icb: None,
                shadow_icb_arg_encoder,
                shadow_icb_arg_buffer: None,
                shadow_icb_capacity: 0,
                mirror_slots: Vec::new(),
                mirror_status: None,
                mirror_icb_capacity: 0,
            },
            bindless_tex_arg_encoder,
            depth_state,
            depth_state_read_only,
            vertex_buffer,
            index_buffer,
            draw_objects,
            cull_bvh,
            always_draw,
            always_draw_member,
            visible_scratch: Vec::new(),
            instanced_clusters,
            n_instances,
            instance_records,
            instance_draw_args,
            // Set by `upload_skinned` (when bindless + static geometry present);
            // 0 keeps the skinned fold inactive until a SkinnedMesh uploads.
            n_skinned: 0,
            instanced_pipeline_state,
            clear_color,
            geometry_less,
            view_matrix: IDENTITY4,
            textures: gpu_textures,
            normal_map_textures: gpu_normal_maps,
            light_uniforms,
            sampler,
            shadow_pipeline_state,
            shadow_map,
            shadow_map_size: effective_shadow_size,
            shadow_update,
            shadow_distance,
            shadow_cascades,
            shadow_scheduler: Default::default(),
            shadow_render_mask: 0,
            shadow_sampler,
            shadow_uniforms: shadow_uniforms_init,
            shadow_light_dir,
            env_map,
            probe_placements: Vec::new(),
            probe_maps: Vec::new(),
            // Empty until `set_reflection_probes` supplies placements.
            probe_bake_queue: crate::gfx::reflection_probe::ProbeBakeQueue::new(0),
            probe_set: crate::metal::uniforms::ProbeSet::EMPTY,
            probe_rendering: None,
            probe_converting: None,
            probe_retire_pool: super::transient::RetirePool::new(),
            cube_sampler,
            text_pipeline_state,
            text_atlas_textures: gpu_text_atlases,
            text_sampler,
            hdr_targets,
            post_pipeline_state,
            post_sampler,
            bloom_targets,
            bloom_pipelines,
            transient_pool,
            post_process,
            color_lut,
            taa: super::post::TaaState {
                enabled: effective_taa_enabled,
                pipeline_state: taa_pipeline_state,
                targets: taa_targets,
                dst: 0,
                history_valid: false,
                frame: 0,
            },
            prev_view_proj: IDENTITY4,
            upscale: super::post::UpscaleState {
                scaler: upscaler,
                scale: upscale_scale,
                jitter: Default::default(),
                reset_pending: std::sync::atomic::AtomicBool::new(true),
            },
            ssao,
            ssr,
            gbuffer,
            ssgi,
            rt: super::raytrace::RtState {
                settings: rt_reflection_settings,
                accel: rt_accel,
                dynamic_mode: rt_dynamic_mode,
                update_failed: false,
                topology_dirty: false,
                pipeline: rt_pipeline,
                pipeline_textured: rt_pipeline_textured,
                skin_pipeline: rt_skin_pipeline,
            },
            decal: super::decal::DecalState {
                records: decals.into_iter().map(Some).collect(),
                free_slots: Vec::new(),
                pipeline: decal_pipeline,
                cube_vertex_buffer: decal_cube_vertex_buffer,
                cube_index_buffer: decal_cube_index_buffer,
                sampler: decal_sampler,
            },
            fog: super::fog::FogState {
                settings: fog_settings,
                pipeline: fog_pipeline,
                froxel_pipeline: fog_froxel_pipeline,
                froxel_volume: fog_froxel_volume,
            },
            particle: super::particle::ParticleState {
                records: particles.into_iter().map(Some).collect(),
                emitter_state: particle_emitter_state.into_iter().map(Some).collect(),
                free_slots: Vec::new(),
                pipelines: particle_pipelines,
                last_elapsed: 0.0,
                frame_index: 0,
            },
            auto_exposure: super::auto_exposure::AutoExposureGpu {
                settings: auto_exposure_settings,
                state: auto_exposure_state,
                bias_ev: auto_exposure_bias,
                pipelines: auto_exposure_pipelines,
                histogram: auto_exposure_histogram,
                output: auto_exposure_output,
                last_elapsed: 0.0,
            },
            hot_reload,
            shader_reload_pending,
            shader_watcher,
            prev_draw_models,
            skinned: super::resources::skinning::SkinnedState {
                pipeline_state: None,
                shadow_pipeline_state: None,
                vertex_buffer: None,
                index_buffer: None,
                draw_objects: Vec::new(),
                joint_matrices: Vec::new(),
                prev_joint_matrices: Vec::new(),
                skin_pipeline: None,
                deformed: Vec::new(),
                deformed_primed: std::sync::atomic::AtomicBool::new(false),
            },
            mesh_vtx_alloc: crate::gfx::range_alloc::RangeAllocator::new(),
            mesh_idx_alloc: crate::gfx::range_alloc::RangeAllocator::new(),
            chunk_vtx_alloc: crate::gfx::range_alloc::RangeAllocator::new(),
            chunk_idx_alloc: crate::gfx::range_alloc::RangeAllocator::new(),
            draw_slots,
            // Seeded later by `seed_skinned_instance_pool` once skinned geometry
            // (with its pre-reserved copies) has been uploaded.
            skinned_pool: crate::gfx::skinned_pool::SkinnedInstancePool::new(),
            window,
            mtk_view,
            window_closed: false,
            pump_events,
            was_visible: false,
            cursor_captured: false,
            recapture_on_click: false,
            ui_cursor_hidden: false,
            menu_mode: false,
            fullscreen,
            window_delegate,
            keys: KeyState::default(),
            keymap: crate::gfx::keymap::KeyMap::default(),
            frame_stats: crate::gfx::profile::RenderStats::default(),
            gpu_time_us: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            render_fault_logged: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            pass_fault_count: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            pass_timing,
            pass_times_us: std::sync::Arc::new(std::array::from_fn(|_| {
                std::sync::atomic::AtomicU32::new(0)
            })),
            draw_calls_accum: std::sync::atomic::AtomicU32::new(0),
            frame_pacing: super::frame_pacing::FrameInFlight::new(frames_in_flight),
            frames_in_flight: frames_in_flight.max(1),
            frame_ring_index: 0,
            // The bindless buffers an async reflection-probe bake reads (object,
            // draw-args, bindless-texture-args, and the skinned joint palettes) get
            // one EXTRA ring slot. The frame only ever uses slots
            // `frame_ring_index % frames_in_flight` -- i.e. `[0, frames_in_flight)`
            // -- so slot `frames_in_flight` is reserved for the bake: a slot the
            // frame never overwrites, keeping the bake's CPU-written buffers valid
            // across its asynchronous (no `waitUntilCompleted`) GPU capture. See
            // metal/probe.rs `bake_ring_slot`.
            object_ring: super::transient::TransientRing::new(frames_in_flight.max(1) + 1),
            draw_args_ring: super::transient::TransientRing::new(frames_in_flight.max(1) + 1),
            prev_model_ring: super::transient::TransientRing::new(frames_in_flight),
            bindless_tex_ring: super::transient::TransientRing::new(frames_in_flight.max(1) + 1),
            joint_ring: super::transient::JointRing::new(frames_in_flight.max(1) + 1),
            prev_joint_ring: super::transient::JointRing::new(frames_in_flight),
            instance_ring: super::transient::InstanceRing::new(frames_in_flight),
            object_scratch: Vec::new(),
            draw_args_scratch: Vec::new(),
            prev_model_scratch: Vec::new(),
            water_pipeline,
            water_pipeline_rt,
            water_pipeline_rt_textured,
            water_surfaces: water_records,
            planar_reflection,
            glass_pipeline,
            glass_pipeline_rt,
            glass_pipeline_rt_textured,
            glass_mesh_pipeline_rt,
            glass_mesh_pipeline_rt_textured,
            seethrough_mesh_indices,
            glass_panels: glass_records,
            raymarch_volumes: raymarch_records,
            raymarch_cube_vertex_buffer,
            raymarch_cube_index_buffer,
        })
    }
}
