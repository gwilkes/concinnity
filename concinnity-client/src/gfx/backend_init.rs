// src/gfx/backend_init.rs
//
// Grouped construction inputs for the render backends, plus the requirements
// derivation that trims scene-scoped features when a world has no 3D content.
// GraphicsSystem init assembles a `BackendInit` from the drained world assets,
// calls `resolve_requirements()`, and hands it to the backend constructor
// selected at compile time (Metal / DirectX / Vulkan). Every backend receives
// the same struct; each reads the fields its feature set consumes.

use crate::assets::{
    GlassPanel, SdfVolume, ShadowUpdate, UpscalerBackend, WaterSurface, WindowArgs,
};
use crate::gfx::auto_exposure::AutoExposureSettings;
use crate::gfx::decal::DecalRecord;
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::particles::ParticleEmitterRecord;
use crate::gfx::render_types::{DrawObject, InstancedCluster, LightUniforms, PostProcessParams};
use crate::gfx::rt_reflections::RtReflectionSettings;
use crate::gfx::ssao::SsaoSettings;
use crate::gfx::ssgi::SsgiSettings;
use crate::gfx::ssr::SsrSettings;
use crate::gfx::volumetric_fog::FogSettings;

// Static scene geometry and the draw lists built over it.
pub struct SceneData<'a> {
    pub vertices: &'a [Vertex],
    pub indices: &'a [u32],
    pub draw_objects: Vec<DrawObject>,
    pub instanced_clusters: Vec<InstancedCluster>,
    // Skinned draw-object count (the world's `SkinnedMesh` count). Sizes each
    // backend's shared GPU-cull buffers for the merged total (static +
    // instances + skinned) at init; the skinned geometry itself is uploaded
    // later via `upload_skinned`.
    pub n_skinned: usize,
    // Worst-case resident chunk count for a streaming VoxelWorld (0
    // otherwise). Reserves a chunk record region in the shared GPU-cull
    // buffers at init; resident chunks fold into the indirect path each
    // frame. Honoured by DirectX + Vulkan; Metal's per-frame rebuild already
    // covers chunks, so it needs no reserve.
    pub n_chunk_max: usize,
}

// Compiled shader payloads. Each backend loads the format its toolchain
// produced (metallib / DXBC / SPIR-V).
pub struct ShaderBytes<'a> {
    pub vert: &'a [u8],
    pub frag: &'a [u8],
    // Compiled shadow-pass vertex shader; consumed by DirectX / Vulkan. Metal
    // compiles its shadow shader internally (shadow_map.metal) and ignores it.
    pub shadow: &'a [u8],
    // Compiled GPU-instanced vertex shader; empty slice = no instanced
    // pipeline (any InstancedProp in the world will fail to render).
    pub vert_instanced: &'a [u8],
}

// Decoded image payloads: texture pools, glyph atlases, and the serialised
// IBL / grading payloads (None = the backend binds identity fallbacks).
pub struct MediaPayloads<'a> {
    // Decoded albedo textures: (width, height, RGBA pixels) per slot.
    pub textures: &'a [(u32, u32, Vec<u8>)],
    // Decoded normal maps; slot 0 is the backend-added flat-normal fallback.
    pub normal_maps: &'a [(u32, u32, Vec<u8>)],
    // Glyph atlas textures for text rendering; empty = no text support.
    pub text_atlases: Vec<(u32, u32, Vec<u8>)>,
    // Serialised EnvironmentMap payload (irradiance + prefilter cubemaps).
    // None disables IBL; the runtime binds 1x1 grey fallback cubes.
    pub env_map_bytes: Option<&'a [u8]>,
    // Serialised ColorLut payload (3D grading LUT). None = identity LUT.
    pub color_lut_bytes: Option<&'a [u8]>,
}

// Shadow-mapping knobs from GraphicsConfig. `map_size == 0` disables the
// shadow pipeline and cascade array entirely.
#[derive(Copy, Clone, Debug)]
pub struct ShadowParams {
    pub map_size: u32,
    // Cascade re-render policy: hybrid amortizes far cascades across frames.
    pub update: ShadowUpdate,
    // Shadow distance in world units, capped at the camera far plane by the
    // per-frame cascade split.
    pub distance: u32,
    // Cascade count (1..=4) the per-frame split + schedule render.
    pub cascades: u32,
}

// Post-process and display settings resolved from PostProcessConfig (plus
// the user's persisted overrides and the quality-preset ceiling). Every
// Option here is an init-time gate: None allocates nothing.
pub struct PostSettings {
    pub post_process: PostProcessParams,
    pub taa_enabled: bool,
    pub ssao: Option<SsaoSettings>,
    pub ssr: Option<SsrSettings>,
    pub ssgi: Option<SsgiSettings>,
    // Requires an RT-capable GPU; backends fall back to SSR without one.
    pub rt_reflections: Option<RtReflectionSettings>,
    // Per-axis divisor for the roughness-aware reflection blur target.
    pub reflection_blur_scale: u32,
    pub auto_exposure: Option<AutoExposureSettings>,
    // Authored exposure_ev carried as a bias on the adapted EV when
    // auto-exposure is on; otherwise baked into post_process.exposure.
    pub auto_exposure_bias_ev: f32,
    // HDR display request; each backend gates it on its own EDR / colour-
    // space capability probe and falls back to SDR with a warning.
    pub hdr_display: bool,
    // PQ-encoded HDR output; honoured by Metal today, accepted elsewhere.
    pub hdr_pq: bool,
    pub temporal_upscaling: bool,
    // Per-axis input-to-output ratio; ignored when upscaling is off.
    pub upscale_scale: f32,
    // Upscaler selector for DirectX / Vulkan (FSR3 / DLSS / XeSS); Metal
    // always uses MetalFX and ignores it.
    pub upscale_backend: UpscalerBackend,
    // Two-pass Hi-Z occlusion request; gated on the bindless cull path.
    pub occlusion_two_pass: bool,
}

// World-authored effect content drained from components. Empty / None means
// the backend builds no pipelines or pools for that feature.
pub struct WorldFx {
    pub decals: Vec<DecalRecord>,
    pub particles: Vec<ParticleEmitterRecord>,
    pub fog: Option<FogSettings>,
    // Transparent water surfaces; rendered by Metal today, accepted by the
    // other backends for parity until their water ports land.
    pub water_surfaces: Vec<WaterSurface>,
    pub glass_panels: Vec<GlassPanel>,
    // Raymarched SDF volumes as (volume, compiled fragment source bytes,
    // asset label for error messages).
    pub sdf_volumes: Vec<(SdfVolume, Vec<u8>, String)>,
}

// Everything a backend constructor needs, assembled once by GraphicsSystem
// init after the world's assets have been drained and settings resolved.
pub struct BackendInit<'a> {
    pub window: &'a WindowArgs,
    // Debug-layer toggle for the DirectX / Vulkan validation layers.
    pub validation: bool,
    pub frames_in_flight: usize,
    pub vsync: bool,
    pub clear_color: [f32; 4],
    // True only under `cn debug`: disk-first shader resolution + watcher.
    pub hot_reload: bool,
    pub scene: SceneData<'a>,
    pub shaders: ShaderBytes<'a>,
    pub media: MediaPayloads<'a>,
    pub light_uniforms: LightUniforms,
    pub shadows: ShadowParams,
    // Scene-sampler max anisotropy, clamped to the GPU's range at init.
    pub anisotropy: u32,
    // Distinct planar-reflection plane budget from the quality preset / GPU
    // tier ceiling; reflectors past it fall back to the probe cube.
    pub planar_planes: usize,
    pub post: PostSettings,
    pub fx: WorldFx,
    // Derived by `resolve_requirements()`; the conservative default assumes a
    // full scene so a caller that skips resolution never under-allocates.
    pub requirements: RenderRequirements,
}

// What the world's content requires of the renderer. Derived from the
// assembled scene + fx data, backend-agnostic, so all three backends make
// identical trimming decisions.
#[derive(Copy, Clone, Debug)]
pub struct RenderRequirements {
    // True when any 3D scene content exists (meshes, instances, skinned
    // meshes, streamed chunks, water, glass, SDF volumes, particles, or
    // decals). False = the world renders UI / text only: the backend skips
    // the scene pipelines and the frame collapses to a clear + composite.
    pub scene: bool,
}

impl Default for RenderRequirements {
    fn default() -> Self {
        RenderRequirements { scene: true }
    }
}

impl RenderRequirements {
    pub fn derive(scene: &SceneData, fx: &WorldFx) -> Self {
        let scene_present = !scene.vertices.is_empty()
            || !scene.draw_objects.is_empty()
            || !scene.instanced_clusters.is_empty()
            || scene.n_skinned > 0
            || scene.n_chunk_max > 0
            || !fx.water_surfaces.is_empty()
            || !fx.glass_panels.is_empty()
            || !fx.sdf_volumes.is_empty()
            || !fx.particles.is_empty()
            || !fx.decals.is_empty();
        RenderRequirements {
            scene: scene_present,
        }
    }
}

impl BackendInit<'_> {
    // Derive the requirements from the assembled content and trim
    // scene-scoped features accordingly. Runtime spawning can only clone
    // assets already declared in the world, so the derivation here is
    // complete: a world with no scene content at init can never grow one.
    pub fn resolve_requirements(&mut self) {
        let req = RenderRequirements::derive(&self.scene, &self.fx);
        if !req.scene {
            trim_scene_features(
                &mut self.shadows,
                &mut self.post,
                &mut self.fx,
                &mut self.planar_planes,
            );
            tracing::info!(
                "render requirements: no 3D scene content; scene-scoped features disabled"
            );
        }
        self.requirements = req;
    }
}

// Force off every feature that only decorates a 3D scene. All of these are
// existing init-time gates in the backends, so zeroing them here means every
// backend skips the matching resources with no backend-side changes.
fn trim_scene_features(
    shadows: &mut ShadowParams,
    post: &mut PostSettings,
    fx: &mut WorldFx,
    planar_planes: &mut usize,
) {
    shadows.map_size = 0;
    post.taa_enabled = false;
    post.ssao = None;
    post.ssr = None;
    post.ssgi = None;
    post.rt_reflections = None;
    post.auto_exposure = None;
    post.temporal_upscaling = false;
    post.occlusion_two_pass = false;
    fx.fog = None;
    *planar_planes = 0;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_scene() -> SceneData<'static> {
        SceneData {
            vertices: &[],
            indices: &[],
            draw_objects: Vec::new(),
            instanced_clusters: Vec::new(),
            n_skinned: 0,
            n_chunk_max: 0,
        }
    }

    fn empty_fx() -> WorldFx {
        WorldFx {
            decals: Vec::new(),
            particles: Vec::new(),
            fog: None,
            water_surfaces: Vec::new(),
            glass_panels: Vec::new(),
            sdf_volumes: Vec::new(),
        }
    }

    fn full_post() -> PostSettings {
        PostSettings {
            post_process: PostProcessParams::DEFAULT,
            taa_enabled: true,
            ssao: Some(SsaoSettings::resolve(0.5, 1.0)),
            ssr: None,
            ssgi: None,
            rt_reflections: None,
            reflection_blur_scale: 2,
            auto_exposure: None,
            auto_exposure_bias_ev: 0.0,
            hdr_display: false,
            hdr_pq: false,
            temporal_upscaling: true,
            upscale_scale: 0.5,
            upscale_backend: UpscalerBackend::Auto,
            occlusion_two_pass: true,
        }
    }

    #[test]
    fn text_only_world_derives_no_scene() {
        let req = RenderRequirements::derive(&empty_scene(), &empty_fx());
        assert!(!req.scene);
    }

    #[test]
    fn any_scene_content_derives_scene() {
        let mut scene = empty_scene();
        scene.n_skinned = 1;
        assert!(RenderRequirements::derive(&scene, &empty_fx()).scene);

        let mut scene = empty_scene();
        scene.n_chunk_max = 8;
        assert!(RenderRequirements::derive(&scene, &empty_fx()).scene);

        // FX content alone is scene content too (a water-only world still
        // renders into the HDR scene chain).
        let scene = empty_scene();
        let mut fx = empty_fx();
        fx.water_surfaces.push(WaterSurface::default());
        assert!(RenderRequirements::derive(&scene, &fx).scene);
    }

    #[test]
    fn sceneless_world_trims_scene_features() {
        let mut shadows = ShadowParams {
            map_size: 2048,
            update: ShadowUpdate::default(),
            distance: 120,
            cascades: 4,
        };
        let mut post = full_post();
        let mut fx = empty_fx();
        let mut planar = 3usize;
        trim_scene_features(&mut shadows, &mut post, &mut fx, &mut planar);
        assert_eq!(shadows.map_size, 0);
        assert!(!post.taa_enabled);
        assert!(post.ssao.is_none());
        assert!(!post.temporal_upscaling);
        assert!(!post.occlusion_two_pass);
        assert!(fx.fog.is_none());
        assert_eq!(planar, 0);
    }

    #[test]
    fn scene_world_keeps_settings() {
        // A world with content must pass its resolved settings through
        // untouched: derivation flags the scene, and nothing is trimmed.
        let mut scene = empty_scene();
        scene.n_skinned = 2;
        let fx = empty_fx();
        let req = RenderRequirements::derive(&scene, &fx);
        assert!(req.scene);
    }
}
