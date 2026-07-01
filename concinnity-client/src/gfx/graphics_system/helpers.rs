// Free-function helpers shared by GraphicsSystem's init / frame / streaming
// code: chunk conversion, camera-relative chunk placement, draw-object
// position extraction, streaming payload sources, and backend construction.

use crate::assets::{BlockType, WindowArgs};
use crate::gfx::mesh_payload::Vertex;

// Resolve a `BlockType` asset into the chunk mesher's palette entry. Per-face
// UV overrides fall back to the uv_min/uv_max rectangle, mirroring the
// build-time `geometry::resolve_block_type`. Used by every backend's
// chunk-streaming setup.
pub(super) fn block_type_to_chunk(bt: &BlockType) -> crate::geometry::ChunkBlockType {
    let default_rect = [bt.uv_min[0], bt.uv_min[1], bt.uv_max[0], bt.uv_max[1]];
    crate::geometry::ChunkBlockType {
        solid: bt.solid,
        uv_top: bt.uv_top.unwrap_or(default_rect),
        uv_bottom: bt.uv_bottom.unwrap_or(default_rect),
        uv_side: bt.uv_side.unwrap_or(default_rect),
    }
}

// Column-major model-to-world transform placing a chunk relative to the
// camera-relative render `origin` chunk: the translation is the chunk's offset
// from the origin in world units, computed from the integer chunk delta so it
// is exact and small regardless of how far the world origin is. The matching
// view matrix is rebased onto the same origin by `camera_relative_view`, which
// keeps an unbounded world's precision intact. Used by every
// backend's per-frame chunk-streaming drive.
pub(super) fn chunk_model_matrix(
    coord: crate::gfx::chunk_coord::ChunkCoord,
    origin: crate::gfx::chunk_coord::ChunkCoord,
    chunk_w: f32,
    chunk_d: f32,
) -> [[f32; 4]; 4] {
    let dx = (coord.x - origin.x) as f32 * chunk_w;
    let dz = (coord.z - origin.z) as f32 * chunk_d;
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [dx, 0.0, dz, 1.0],
    ]
}

// World-space position used to score a draw object for texture streaming:
// the AABB centre when bounds are finite, otherwise the model-matrix
// translation (dynamic props carry a non-finite sentinel AABB).
pub(super) fn draw_object_position(obj: &crate::gfx::render_types::DrawObject) -> [f32; 3] {
    let finite = obj
        .bb_min
        .iter()
        .chain(obj.bb_max.iter())
        .all(|v| v.is_finite());
    if finite {
        [
            0.5 * (obj.bb_min[0] + obj.bb_max[0]),
            0.5 * (obj.bb_min[1] + obj.bb_max[1]),
            0.5 * (obj.bb_min[2] + obj.bb_max[2]),
        ]
    } else {
        [obj.model[3][0], obj.model[3][1], obj.model[3][2]]
    }
}

// Above this triangle count the reflection-probe auto-seed skips the world-triangle
// gather and keeps coarse object-AABB occupancy, so a heavy import (Bistro is ~2.8M
// triangles) pays nothing extra at load. Small authored scenes stay well under it and
// get the finer surface-voxel interior detection (a watertight single-mesh room is then
// seen as hollow).
pub(super) const AUTO_SEED_MAX_TRIANGLES: usize = 200_000;

// Gather world-space triangles from the static draw list for reflection-probe auto-seed
// interior detection (surface voxelisation needs real geometry, not AABBs). Returns
// `None` when there is no cullable static geometry or the scene exceeds
// `AUTO_SEED_MAX_TRIANGLES` -- the caller then falls back to coarse AABB occupancy. Each
// cullable draw's indexed triangles are transformed to world space by its model matrix;
// `base_vertex` is honoured so streamed (mesh-relative) chunks resolve too, and every
// fetch is bounds-checked against the shared vertex buffer (build-time offsets should be
// in range, but a bad offset is skipped rather than risking an out-of-bounds index).
pub(super) fn gather_auto_seed_triangles(
    draw_objects: &[crate::gfx::render_types::DrawObject],
    all_vertices: &[Vertex],
    all_indices: &[u32],
) -> Option<Vec<[[f32; 3]; 3]>> {
    let eligible = |o: &crate::gfx::render_types::DrawObject| o.cullable() && o.index_count >= 3;
    let total_tris: usize = draw_objects
        .iter()
        .filter(|o| eligible(o))
        .map(|o| o.index_count / 3)
        .sum();
    if total_tris == 0 || total_tris > AUTO_SEED_MAX_TRIANGLES {
        return None;
    }

    // Column-major model-to-world transform of a model-space point.
    let xf = |m: &[[f32; 4]; 4], p: [f32; 3]| {
        [
            m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
            m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
            m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
        ]
    };

    let mut tris = Vec::with_capacity(total_tris);
    for o in draw_objects.iter().filter(|o| eligible(o)) {
        let iend = o.index_offset + o.index_count;
        if iend > all_indices.len() {
            continue;
        }
        for t in all_indices[o.index_offset..iend].chunks_exact(3) {
            let vi = |k: usize| (t[k] as i64 + o.base_vertex as i64) as usize;
            let (a, b, c) = (vi(0), vi(1), vi(2));
            if a >= all_vertices.len() || b >= all_vertices.len() || c >= all_vertices.len() {
                continue;
            }
            tris.push([
                xf(&o.model, all_vertices[a].pos),
                xf(&o.model, all_vertices[b].pos),
                xf(&o.model, all_vertices[c].pos),
            ]);
        }
    }
    (!tris.is_empty()).then_some(tris)
}

// Build the payload source for a streamed texture pool (albedo or normal-map).
//
// When `disk_backed`, each locator's payload-section offset is turned into an
// absolute file offset (the blob file's payload section starts past its header
// and defs) so the streamer can re-read payloads from disk without a RAM copy.
// Otherwise the retained `payloads` are wrapped RAM-resident. Used by the
// Metal, Vulkan, and DirectX texture-streaming paths.
pub(super) fn build_texture_payload_source(
    payloads: Vec<Vec<u8>>,
    locators: &[crate::ecs::PayloadLocator],
    disk_backed: bool,
) -> Result<std::sync::Arc<dyn crate::app::texture_stream::PayloadSource>, String> {
    if !disk_backed {
        return Ok(std::sync::Arc::new(
            crate::app::texture_stream::MemPayloadSource::new(payloads),
        ));
    }
    let mut section_starts: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    let mut disk_locators = Vec::with_capacity(locators.len());
    for loc in locators {
        let path = crate::blob::blob_path(loc.blob_index);
        let start = match section_starts.get(&loc.blob_index) {
            Some(&s) => s,
            None => {
                let s = crate::blob::payload_section_start(&path)
                    .map_err(|e| format!("blob {}: {:?}", loc.blob_index, e))?;
                section_starts.insert(loc.blob_index, s);
                s
            }
        };
        disk_locators.push(crate::app::texture_stream::DiskTextureLocator {
            path,
            file_offset: start + loc.offset,
            len: loc.len,
        });
    }
    Ok(std::sync::Arc::new(
        crate::app::texture_stream::DiskPayloadSource::new(disk_locators),
    ))
}

// Probe the active GPU's coarse performance profile before the backend is built,
// so the auto-config quality ceiling can influence the render targets / effect
// pipelines the backend sizes at init. Each backend creates only the cheap
// throwaway handle it needs and classifies it: Metal the default-device handle,
// DirectX the DXGI adapter (no device / swapchain), Vulkan a surface-free
// instance (destroyed immediately). The three `backend_*` cfgs are mutually
// exclusive, so exactly one arm compiles; the fallback returns `UNKNOWN` (which
// the resolver treats as "no ceiling") only when no backend is configured.
pub(super) fn probe_gpu_profile() -> crate::gfx::backend::GpuProfile {
    #[cfg(backend_dx)]
    {
        crate::directx::probe_gpu_profile()
    }
    #[cfg(backend_vk)]
    {
        crate::vulkan::probe_gpu_profile()
    }
    #[cfg(backend_metal)]
    {
        crate::metal::probe_gpu_profile()
    }
    #[cfg(not(any(backend_dx, backend_vk, backend_metal)))]
    {
        crate::gfx::backend::GpuProfile::UNKNOWN
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn init_backend(
    window_args: &WindowArgs,
    validation: bool,
    frames_in_flight: usize,
    // When false (the default) the renderer presents uncapped (DX: tearing
    // allowed; VK: mailbox). When true, presentation locks to the display
    // refresh. See `GraphicsConfig.vsync`.
    vsync: bool,
    clear_color: [f32; 4],
    vertices: &[Vertex],
    indices: &[u32],
    draw_objects: Vec<crate::gfx::render_types::DrawObject>,
    instanced_clusters: Vec<crate::gfx::render_types::InstancedCluster>,
    // Number of skinned draw objects (the world's `SkinnedMesh` count). Threaded
    // purely to size each backend's shared GPU-cull buffers for the merged total
    // (static + instances + skinned) at init; the skinned geometry itself is
    // uploaded later via `upload_skinned`.
    n_skinned: usize,
    // Worst-case resident chunk count for a streaming VoxelWorld (0 otherwise).
    // Threaded purely to reserve a chunk record region in each backend's shared
    // GPU-cull buffers at init; resident chunks fold into the indirect path each
    // frame. Honoured by DirectX + Vulkan; Metal accepts it for
    // signature parity (its per-frame full draw_objects rebuild already covers
    // chunks, so it needs no reserve).
    n_chunk_max: usize,
    vert_bytes: &[u8],
    frag_bytes: &[u8],
    // compiled shadow-pass vertex shader bytes; empty slice = shadows disabled
    shadow_bytes: &[u8],
    // compiled GPU-instanced vertex shader bytes; empty slice = no instanced pipeline
    // (any InstancedProp in the world will fail to render in v2).
    vert_instanced_bytes: &[u8],
    // decoded textures: (width, height, RGBA pixels) per texture slot
    textures: &[(u32, u32, Vec<u8>)],
    // decoded normal-map textures: slot 0 is the flat-normal fallback (added by MtlContext)
    normal_maps: &[(u32, u32, Vec<u8>)],
    light_uniforms: crate::gfx::render_types::LightUniforms,
    shadow_map_size: u32,
    // shadow-cascade update policy from GraphicsConfig.shadow_update. The shared
    // `ShadowCascadeScheduler` staggers which cascades re-render each frame.
    shadow_update: crate::assets::ShadowUpdate,
    // Shadow distance (world units) from GraphicsConfig.shadow_distance. Each
    // backend's per-frame cascade-split computation reads it (capped at the camera
    // far plane); live on Metal via set_shadow_distance.
    shadow_distance: u32,
    // Shadow cascade count (1..=4) from GraphicsConfig.shadow_cascades. Each
    // backend's per-frame split + schedule read it; live on Metal via
    // set_shadow_cascades.
    shadow_cascades: u32,
    // Scene-sampler max anisotropy from GraphicsConfig.anisotropy. Each backend
    // builds its albedo / normal-map sampler with this (clamped to the GPU's
    // 1..16 range) at init; restart-required.
    anisotropy: u32,
    // Distinct planar-reflection plane budget, resolved from the GPU tier / quality
    // preset ceiling. Each backend passes it to `assign_planar_slots` at init
    // (clamped to its capacity ceiling); reflectors past it fall back to the
    // box-projected probe cube. Restart-required -- the mirror targets are allocated
    // once at init.
    planar_planes: usize,
    // glyph atlas textures for text rendering; empty = no text support
    text_atlases: Vec<(u32, u32, Vec<u8>)>,
    // serialised EnvironmentMap payload (irradiance + prefilter cubemaps).
    // None disables IBL; the runtime then binds 1x1 grey fallback cubes.
    env_map_bytes: Option<&[u8]>,
    // post-process tunables (bloom intensity/threshold/knee).
    post_process: crate::gfx::render_types::PostProcessParams,
    // serialised ColorLut payload (3D grading LUT). None disables grading;
    // the runtime then binds an identity LUT.
    color_lut_bytes: Option<&[u8]>,
    // temporal anti-aliasing toggle from PostProcessConfig.aa_mode.
    taa_enabled: bool,
    // SSAO (GTAO) settings from PostProcessConfig; None disables SSAO.
    ssao_settings: Option<crate::gfx::ssao::SsaoSettings>,
    // SSR settings from PostProcessConfig; None disables SSR.
    ssr_settings: Option<crate::gfx::ssr::SsrSettings>,
    // SSGI settings from PostProcessConfig; None disables SSGI.
    ssgi_settings: Option<crate::gfx::ssgi::SsgiSettings>,
    // RT-reflection settings from PostProcessConfig; None disables RT
    // reflections. Requires a GPU with ray-tracing support, else the graph falls
    // back to SSR. Takes precedence over `ssr_settings` where RT is live (the
    // graph builder picks RtReflections over SsrResolve); SSR stays the fallback.
    rt_reflection_settings: Option<crate::gfx::rt_reflections::RtReflectionSettings>,
    // Per-axis divisor for the roughness-aware reflection blur target, resolved
    // from `PostProcessConfig.reflection_blur_resolution`. The reflection-composite
    // blur target is sized at render / this.
    reflection_blur_scale: u32,
    // Projected decals resolved from the world's `Decal` components.
    decals: Vec<crate::gfx::decal::DecalRecord>,
    // Particle-emitter records resolved from the world's `ParticleEmitter`
    // components.
    particles: Vec<crate::gfx::particles::ParticleEmitterRecord>,
    // Volumetric-fog settings resolved from `VolumetricFog`.
    fog_settings: Option<crate::gfx::volumetric_fog::FogSettings>,
    // Auto-exposure settings resolved from PostProcessConfig; None disables
    // auto-exposure.
    auto_exposure_settings: Option<crate::gfx::auto_exposure::AutoExposureSettings>,
    // Authored `exposure_ev` carried through as a bias on the adapted EV when
    // auto-exposure is on; ignored when it is off.
    auto_exposure_bias_ev: f32,
    // World-side HDR display request from `PostProcessConfig.hdr_display`. Each
    // backend gates this on its own HDR-capability check: Metal reconfigures the
    // CAMetalLayer + composite shader once the active panel's EDR capability
    // clears the threshold; DirectX picks the scRGB-linear DXGI colour space when
    // `CheckColorSpaceSupport` accepts it; Vulkan opts into the
    // `VK_EXT_swapchain_colorspace` instance extension and picks the
    // scRGB-linear surface format when both the loader exposes the
    // extension and the surface advertises the pair. Any of those gates
    // failing logs a warning and falls back to SDR.
    hdr_display: bool,
    // PQ-encoded HDR output request from `PostProcessConfig.hdr_pq`. Honoured
    // only by the Metal backend today (picks `kCGColorSpaceDisplayP3_PQ` for
    // the swapchain and flips the composite shader's PQ-encode branch);
    // DirectX / Vulkan accept and ignore until they wire their own PQ paths.
    // No effect when `hdr_display` is false or the active panel reports no
    // EDR headroom.
    hdr_pq: bool,
    // Temporal upscaling toggle from `PostProcessConfig.temporal_upscaling`. With
    // this on, the renderer draws the 3D scene at `(drawable * upscale_scale)` and
    // a per-backend upscaler reconstructs a drawable-resolution image.
    temporal_upscaling: bool,
    // Per-axis input-to-output ratio from `PostProcessConfig.upscale_quality`
    // (e.g. 2/3 for Quality, 0.5 for Performance). Ignored when
    // `temporal_upscaling` is false.
    upscale_scale: f32,
    // Upscaler backend selector from `PostProcessConfig.upscale_backend`.
    // Honoured by the DirectX and Vulkan backends (FSR3 / DLSS / XeSS); Metal
    // always uses MetalFX, so it ignores the selector and drops it below.
    upscale_backend: crate::assets::UpscalerBackend,
    // Two-pass Hi-Z occlusion toggle from
    // `PostProcessConfig.occlusion_two_pass`. Each backend gates it on the
    // bindless GPU-cull path being active (the phase-2 cull pipeline must exist).
    occlusion_two_pass: bool,
    // Transparent water surfaces drained from the world's `WaterSurface`
    // components. Honoured by the Metal backend; DirectX / Vulkan accept the
    // slice for parity but render no water yet.
    water_surfaces: Vec<crate::assets::WaterSurface>,
    // Translucent glass panels drained from the world's `GlassPanel`
    // components.
    glass_panels: Vec<crate::assets::GlassPanel>,
    // Raymarched SDF volumes drained from the world's `SdfVolume`
    // components, paired with their compiled-payload fragment shader
    // source bytes + asset label.
    sdf_volumes: Vec<(crate::assets::SdfVolume, Vec<u8>, String)>,
    // Shader hot-reload toggle, true only under `cn debug`.
    hot_reload: bool,
) -> Option<Box<dyn crate::gfx::backend::RenderBackend>> {
    // `water_surfaces` is Metal-only today (the Vulkan + DirectX GLSL /
    // HLSL water ports are still open); silence the unused-binding lint
    // on the Windows backends. `sdf_volumes` is consumed on DirectX
    // (`directx/raymarch.rs`) and Vulkan (`vulkan/raymarch.rs`).
    #[cfg(not(backend_metal))]
    let _ = &water_surfaces;
    // `glass_panels` is consumed on every backend now (the transparent-pass
    // glass producer: `directx/glass.rs`, `metal/glass.rs`, `vulkan/glass.rs`).
    // The upscaler backend selector drives DirectX (FSR3 / DLSS / XeSS) and
    // Vulkan (FSR-or-native); only Metal (MetalFX, single backend) ignores it.
    #[cfg(backend_metal)]
    let _ = upscale_backend;
    // Hardware ray-traced reflections are wired on every backend now (Metal +
    // DirectX inline DXR, Vulkan `VK_KHR_ray_query`); each constructor takes the
    // settings and falls back to SSR when the GPU lacks RT support.

    // The shadow shader is engine-internal on Metal (compiled from
    // `shadow_map.metal`); only the DX / Vulkan constructors still consume the
    // shadow payload bytes.
    #[cfg(backend_metal)]
    let _ = shadow_bytes;

    // The reflection-blur divisor feeds every backend's reflection composite
    // (each sizes its blur target at render / this).

    #[cfg(backend_dx)]
    {
        use crate::directx::DxContext;

        match DxContext::new(
            &window_args.title,
            window_args.width,
            window_args.height,
            validation,
            frames_in_flight,
            vsync,
            clear_color,
            vertices,
            indices,
            draw_objects,
            instanced_clusters,
            n_skinned,
            n_chunk_max,
            vert_bytes,
            frag_bytes,
            shadow_bytes,
            vert_instanced_bytes,
            textures,
            normal_maps,
            light_uniforms,
            shadow_map_size,
            shadow_update,
            shadow_distance,
            shadow_cascades,
            anisotropy,
            planar_planes,
            text_atlases,
            env_map_bytes,
            post_process,
            color_lut_bytes,
            taa_enabled,
            ssao_settings,
            ssr_settings,
            ssgi_settings,
            rt_reflection_settings,
            reflection_blur_scale,
            decals,
            particles,
            fog_settings,
            auto_exposure_settings,
            auto_exposure_bias_ev,
            hdr_display,
            hdr_pq,
            temporal_upscaling,
            upscale_scale,
            upscale_backend,
            occlusion_two_pass,
            sdf_volumes,
            glass_panels,
            hot_reload,
        ) {
            Ok(dx) => Some(Box::new(dx)),
            Err(e) => {
                tracing::error!("GraphicsSystem: D3D12 init failed: {}", e);
                None
            }
        }
    }

    #[cfg(backend_vk)]
    {
        use crate::vulkan::VkContext;

        match VkContext::new(
            &window_args.title,
            window_args.width,
            window_args.height,
            validation,
            frames_in_flight,
            vsync,
            clear_color,
            vertices,
            indices,
            draw_objects,
            instanced_clusters,
            n_skinned,
            n_chunk_max,
            vert_bytes,
            frag_bytes,
            shadow_bytes,
            vert_instanced_bytes,
            textures,
            normal_maps,
            light_uniforms,
            shadow_map_size,
            shadow_update,
            shadow_distance,
            shadow_cascades,
            anisotropy,
            planar_planes,
            text_atlases,
            env_map_bytes,
            post_process,
            color_lut_bytes,
            taa_enabled,
            ssao_settings,
            ssr_settings,
            ssgi_settings,
            rt_reflection_settings,
            reflection_blur_scale,
            decals,
            particles,
            fog_settings,
            auto_exposure_settings,
            auto_exposure_bias_ev,
            hdr_display,
            hdr_pq,
            temporal_upscaling,
            upscale_scale,
            upscale_backend,
            occlusion_two_pass,
            sdf_volumes,
            glass_panels,
            hot_reload,
        ) {
            Ok(vk) => Some(Box::new(vk)),
            Err(e) => {
                tracing::error!("GraphicsSystem: Vulkan init failed: {}", e);
                None
            }
        }
    }

    #[cfg(backend_metal)]
    {
        use crate::metal::MtlContext;

        match MtlContext::new(
            &window_args.title,
            window_args.width,
            window_args.height,
            validation,
            frames_in_flight,
            vsync,
            clear_color,
            vertices,
            indices,
            draw_objects,
            instanced_clusters,
            n_skinned,
            n_chunk_max,
            vert_bytes,
            frag_bytes,
            vert_instanced_bytes,
            textures,
            normal_maps,
            light_uniforms,
            shadow_map_size,
            shadow_update,
            shadow_distance,
            shadow_cascades,
            anisotropy,
            planar_planes,
            text_atlases,
            env_map_bytes,
            post_process,
            color_lut_bytes,
            taa_enabled,
            ssao_settings,
            ssr_settings,
            ssgi_settings,
            rt_reflection_settings,
            reflection_blur_scale,
            decals,
            particles,
            fog_settings,
            auto_exposure_settings,
            auto_exposure_bias_ev,
            hdr_display,
            hdr_pq,
            temporal_upscaling,
            upscale_scale,
            occlusion_two_pass,
            water_surfaces,
            glass_panels,
            sdf_volumes,
            hot_reload,
        ) {
            Ok(mtl) => Some(Box::new(mtl)),
            Err(e) => {
                tracing::error!("GraphicsSystem: Metal init failed: {}", e);
                None
            }
        }
    }
}
