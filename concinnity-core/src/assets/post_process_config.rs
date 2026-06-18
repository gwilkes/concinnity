// src/assets/post_process_config.rs

use crate::ecs::{AssetOrigin, Component};
use crate::gfx::render_types::PostProcessParams;

/// Tunables for the post-process stack. One per world; the first declared
/// instance wins. With no `PostProcessConfig` present, the defaults below are
/// used (bloom on at a moderate intensity).
///
/// Colour-LUT grading is a separate [ColorLut](#colorlut) asset; `lut_strength`
/// here is the blend amount applied to whichever [ColorLut](#colorlut) the world
/// declares.
///
/// When `auto_exposure` is on, the scene's average brightness is measured each
/// frame and exposure adapts toward a balanced mid-tone. The authored
/// `exposure_ev` then acts as an additive bias (in stops) on top of the adapted
/// value.
///
/// ```jsonl
/// {"name":"post","type":"PostProcessConfig","args":{"bloom_intensity":0.8}}
/// {"name":"post_dim","type":"PostProcessConfig","args":{"exposure_ev":-1.0,"vignette_strength":0.4}}
/// {"name":"post_taa","type":"PostProcessConfig","args":{"taa":true}}
/// {"name":"post_ssao","type":"PostProcessConfig","args":{"ssao":true,"ssao_radius":0.6}}
/// {"name":"post_ssr","type":"PostProcessConfig","args":{"ssr":true,"ssr_intensity":0.8}}
/// {"name":"post_rt","type":"PostProcessConfig","args":{"ray_traced_reflections":true,"ssr_intensity":0.8}}
/// {"name":"post_ssgi","type":"PostProcessConfig","args":{"indirect_lighting":"ssgi","ssgi_intensity":0.6}}
/// {"name":"post_auto_ev","type":"PostProcessConfig","args":{"auto_exposure":true}}
/// {"name":"post_hdr","type":"PostProcessConfig","args":{"hdr_display":true}}
/// {"name":"post_upscale","type":"PostProcessConfig","args":{"temporal_upscaling":true,"upscale_quality":"balanced"}}
/// {"name":"post_dlss","type":"PostProcessConfig","args":{"temporal_upscaling":true,"upscale_backend":"dlss"}}
/// {"name":"post_occ2","type":"PostProcessConfig","args":{"occlusion_two_pass":true}}
/// {"name":"post_off","type":"PostProcessConfig","args":{"bloom_intensity":0.0}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PostProcessConfig {
    /// Additive bloom contribution. 0 skips bloom entirely.
    pub bloom_intensity: f32,
    /// Brightness threshold for bloom. Pixels brighter than this contribute
    /// fully; pixels within `bloom_knee` below it ramp in softly.
    pub bloom_threshold: f32,
    /// Width of the soft knee just below `bloom_threshold`.
    pub bloom_knee: f32,
    /// Exposure offset in photographic stops. Each +1 doubles scene
    /// brightness before bloom and tonemapping; 0 is neutral.
    pub exposure_ev: f32,
    /// Vignette strength in `[0, 1]`. 0 disables the corner darkening.
    pub vignette_strength: f32,
    /// Colour-LUT blend in `[0, 1]`. Mixes the graded colour over the ungraded
    /// one by this amount. Only matters when the world declares a
    /// [ColorLut](#colorlut); with none, grading is a no-op at any strength.
    pub lut_strength: f32,
    /// Temporal anti-aliasing toggle. Smooths edges by jittering and
    /// accumulating detail across frames.
    pub taa: bool,
    /// Screen-space ambient occlusion toggle. Darkens creases and contact areas
    /// where ambient light is occluded.
    pub ssao: bool,
    /// How far the ambient-occlusion search reaches for occluders, in world
    /// units. Larger values pick up broader, softer occlusion.
    pub ssao_radius: f32,
    /// Ambient-occlusion strength, clamped to `[0, 4]`. 1.0 is the natural
    /// amount; higher values exaggerate the contact darkening.
    pub ssao_intensity: f32,
    /// Screen-space reflection toggle. Mixes reflected scene colour over glossy
    /// surfaces (water, polished floors).
    pub ssr: bool,
    /// Reflection blend strength, clamped to `[0, 1]`. Scales the
    /// Fresnel-weighted reflection mixed over the base shading.
    pub ssr_intensity: f32,
    /// How far a reflection reaches, in world units. Longer reaches catch more
    /// distant reflections, more coarsely.
    pub ssr_max_distance: f32,
    /// Hardware ray-traced reflection toggle. When the GPU supports ray tracing,
    /// traces real reflection rays so off-screen geometry still appears, instead
    /// of the screen-space method. Reuses the `ssr_intensity` /
    /// `ssr_max_distance` tunables and takes precedence over `ssr`, falling back
    /// to it where ray tracing isn't available.
    pub ray_traced_reflections: bool,
    /// Indirect-diffuse lighting source. `ibl` (default) uses the environment
    /// map's ambient alone. `ssgi` adds a screen-space global-illumination pass
    /// on top, so nearby lit surfaces bleed colour onto one another; the
    /// environment ambient still covers the off-screen / sky fallback.
    pub indirect_lighting: IndirectLighting,
    /// Multiplier on the indirect (ambient / IBL) lighting term, clamped to
    /// `[0, 16]`. 1.0 (default) leaves the environment-derived ambient at its
    /// physical level. Raising it lifts fill light in areas the directional
    /// light cannot reach (shadowed facades, alleys) without brightening
    /// directly lit surfaces, which the sun already dominates. Scales the
    /// diffuse and specular IBL together, so reflections stay consistent with
    /// the brighter ambient. Useful for high-contrast exterior scenes where a
    /// strong sun would otherwise crush shadows to black.
    pub ambient_intensity: f32,
    /// Indirect-bounce strength, clamped to `[0, 4]`. Scales the gathered
    /// indirect light added on top of the existing shading; 0 makes it a no-op.
    /// Only matters when `indirect_lighting` is `ssgi`.
    pub ssgi_intensity: f32,
    /// How far the indirect-light gather reaches, in world units. A near-field
    /// effect, so it defaults well below `ssr_max_distance`. Only matters when
    /// `indirect_lighting` is `ssgi`.
    pub ssgi_max_distance: f32,
    /// Internal resolution of the SSGI gather. `half` (default) trades a little
    /// sharpness for a large performance saving; `full` is native; `quarter` is
    /// the cheapest. Only matters when `indirect_lighting` is `ssgi`.
    pub ssgi_resolution: SsgiResolution,
    /// Hemisphere rays cast per pixel by the SSGI gather, clamped to `[1, 32]`.
    /// More rays reduce noise at a higher cost. Only matters when
    /// `indirect_lighting` is `ssgi`.
    pub ssgi_rays: u32,
    /// Ray-march samples per SSGI ray, clamped to `[1, 64]`. More samples catch
    /// finer occlusion at a higher cost. Only matters when `indirect_lighting`
    /// is `ssgi`.
    pub ssgi_steps: u32,
    /// Auto-exposure toggle. Adapts exposure each frame toward a balanced
    /// mid-tone. The authored `exposure_ev` then acts as an additive bias in
    /// stops on top of the adapted value.
    pub auto_exposure: bool,
    /// Lower bound on the adapted exposure (EV). The `exposure_ev` bias is
    /// applied before this clamp.
    pub auto_exposure_min_ev: f32,
    /// Upper bound on the adapted exposure (EV).
    pub auto_exposure_max_ev: f32,
    /// How quickly exposure chases a new target (per second). Higher converges
    /// faster but can pump under flickering content; 1-3 is comfortable.
    pub auto_exposure_speed: f32,
    /// HDR display output toggle. On a capable display, emits extended-range
    /// HDR instead of the standard tonemapped output. Falls back to standard
    /// output when the display or platform doesn't support HDR.
    pub hdr_display: bool,
    /// PQ (HDR10) output mode. When true, and `hdr_display` is on, and the
    /// display has HDR headroom, output is PQ-encoded for HDR10 panels. No
    /// effect when `hdr_display` is off.
    pub hdr_pq: bool,
    /// Temporal upscaling toggle. Renders the 3D scene at a lower resolution
    /// (set by `upscale_quality`) and reconstructs a full-resolution image,
    /// trading some sharpness for performance. Replaces TAA while on (the `taa`
    /// flag is ignored).
    pub temporal_upscaling: bool,
    /// Render-scale preset for `temporal_upscaling`; each step progressively
    /// lowers the internal resolution. No effect when `temporal_upscaling` is
    /// off.
    pub upscale_quality: UpscaleQuality,
    /// Which upscaler backend `temporal_upscaling` uses. `auto` (default) picks
    /// the best available at runtime (DLSS on NVIDIA RTX, else XeSS, else FSR3);
    /// `fsr3` / `dlss` / `xess` request a specific one and fall back when it is
    /// unavailable on the current GPU or build. No effect when
    /// `temporal_upscaling` is off. DLSS and XeSS are DirectX-only.
    pub upscale_backend: UpscalerBackend,
    /// Two-pass occlusion culling toggle. Reduces objects popping in a frame
    /// late when they're revealed by camera or occluder motion, at the cost of
    /// extra culling work each frame.
    pub occlusion_two_pass: bool,
}

/// Render-scale preset for `PostProcessConfig.temporal_upscaling`. The ratio
/// applies to both axes (input pixel count = output * ratio per axis), so
/// `Quality` renders at 4/9 of the output pixel count, `Performance` at 1/4,
/// and `UltraPerformance` at 1/9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum UpscaleQuality {
    #[default]
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl UpscaleQuality {
    /// Per-axis input-to-output ratio. The render target's width/height are
    /// `(output_w * scale(), output_h * scale())`.
    pub fn scale(self) -> f32 {
        match self {
            UpscaleQuality::Quality => 2.0 / 3.0,
            UpscaleQuality::Balanced => 0.587,
            UpscaleQuality::Performance => 0.5,
            UpscaleQuality::UltraPerformance => 1.0 / 3.0,
        }
    }
}

/// Upscaler backend selector for `PostProcessConfig.temporal_upscaling`.
/// `Auto` resolves at runtime to the best available (DLSS, then XeSS, then
/// FSR3); the explicit variants request a specific backend and fall back when
/// it is unavailable. DLSS (NVIDIA NGX) and XeSS (Intel) are DirectX-only;
/// Metal uses MetalFX and Vulkan has no upscaler yet, so both treat any value
/// as their native path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum UpscalerBackend {
    #[default]
    Auto,
    Fsr3,
    Dlss,
    Xess,
}

/// Indirect-diffuse lighting source for `PostProcessConfig.indirect_lighting`.
/// `Ibl` is the image-based-lighting-only ambient term the renderer has always
/// used; `Ssgi` layers a screen-space global-illumination bounce on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum IndirectLighting {
    #[default]
    Ibl,
    Ssgi,
}

/// Internal render resolution of the SSGI gather pass (only meaningful when
/// `indirect_lighting` is `ssgi`). The gather is the expensive part (a
/// hemisphere ray-march per pixel), and its composite is a depth-aware
/// bilateral filter that upsamples a lower-resolution gather back to full
/// resolution at little visible cost. `half` (the default) gathers at a quarter
/// of the pixels for a large saving; `full` keeps the gather at native
/// resolution; `quarter` is the cheapest, for low-end GPUs or debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum SsgiResolution {
    Full,
    #[default]
    Half,
    Quarter,
}

impl SsgiResolution {
    /// Per-axis render-resolution divisor the gather target is scaled by.
    pub fn scale_divisor(self) -> u32 {
        match self {
            SsgiResolution::Full => 1,
            SsgiResolution::Half => 2,
            SsgiResolution::Quarter => 4,
        }
    }
}

/// `exposure_ev` is clamped to this range before resolving to a multiplier so
/// a stray value cannot push the scene to `inf` / `0`.
const EXPOSURE_EV_LIMIT: f32 = 16.0;

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            bloom_intensity: 0.6,
            bloom_threshold: 1.0,
            bloom_knee: 0.5,
            exposure_ev: 0.0,
            vignette_strength: 0.0,
            lut_strength: 1.0,
            taa: false,
            ssao: false,
            ssao_radius: 0.5,
            ssao_intensity: 1.0,
            ssr: false,
            ssr_intensity: 0.7,
            ssr_max_distance: 40.0,
            ray_traced_reflections: false,
            indirect_lighting: IndirectLighting::Ibl,
            ambient_intensity: 1.0,
            ssgi_intensity: 0.5,
            ssgi_max_distance: 8.0,
            ssgi_resolution: SsgiResolution::default(),
            ssgi_rays: crate::gfx::ssgi::DEFAULT_RAYS,
            ssgi_steps: crate::gfx::ssgi::DEFAULT_STEPS,
            auto_exposure: false,
            auto_exposure_min_ev: -8.0,
            auto_exposure_max_ev: 8.0,
            auto_exposure_speed: 1.5,
            hdr_display: false,
            hdr_pq: false,
            temporal_upscaling: false,
            upscale_quality: UpscaleQuality::default(),
            upscale_backend: UpscalerBackend::default(),
            occlusion_two_pass: false,
        }
    }
}

impl PostProcessConfig {
    /// Resolve the asset's authored fields into the GPU-facing
    /// `PostProcessParams`: clamps each tunable to a safe range and converts
    /// `exposure_ev` (stops) into the linear multiplier the shaders expect.
    ///
    /// `hdr_output` is left at 0.0 (SDR path) here; the backend overwrites it
    /// to 1.0 only after it has confirmed the active display reports EDR
    /// support. That keeps the asset-side resolve pure: a world authored for
    /// HDR still renders correctly on an SDR display.
    pub fn resolve(&self) -> PostProcessParams {
        let ev = self
            .exposure_ev
            .clamp(-EXPOSURE_EV_LIMIT, EXPOSURE_EV_LIMIT);
        PostProcessParams {
            bloom_intensity: self.bloom_intensity.max(0.0),
            bloom_threshold: self.bloom_threshold.max(0.0),
            bloom_knee: self.bloom_knee.max(0.0),
            exposure: ev.exp2(),
            vignette: self.vignette_strength.clamp(0.0, 1.0),
            lut_strength: self.lut_strength.clamp(0.0, 1.0),
            hdr_output: 0.0,
            pq_output: 0.0,
        }
    }

    /// Clamp the authored `ambient_intensity` to a safe `[0, 16]` multiplier
    /// the backend folds into `LightUniforms` so the main pass can scale its
    /// indirect (ambient / IBL) term.
    pub fn ambient_intensity(&self) -> f32 {
        self.ambient_intensity.clamp(0.0, 16.0)
    }

    /// Resolve the SSAO tunables into clamped `SsaoSettings`, or `None` when
    /// the `ssao` toggle is off so the backend can skip the SSAO passes
    /// entirely.
    pub fn ssao_settings(&self) -> Option<crate::gfx::ssao::SsaoSettings> {
        self.ssao
            .then(|| crate::gfx::ssao::SsaoSettings::resolve(self.ssao_radius, self.ssao_intensity))
    }

    /// Resolve the SSR tunables into clamped `SsrSettings`, or `None` when
    /// the `ssr` toggle is off so the backend can skip the SSR passes
    /// entirely.
    pub fn ssr_settings(&self) -> Option<crate::gfx::ssr::SsrSettings> {
        self.ssr.then(|| {
            crate::gfx::ssr::SsrSettings::resolve(self.ssr_intensity, self.ssr_max_distance)
        })
    }

    /// Resolve the ray-traced-reflection tunables into clamped
    /// `RtReflectionSettings`, or `None` when the `ray_traced_reflections`
    /// toggle is off. Reuses the SSR intensity / distance fields: RT
    /// reflections are the same effect with a hardware trace instead of a
    /// screen-space march. The backend additionally gates on GPU ray-tracing
    /// support; this asset-side resolve only reflects the authored intent.
    pub fn rt_reflection_settings(
        &self,
    ) -> Option<crate::gfx::rt_reflections::RtReflectionSettings> {
        self.ray_traced_reflections.then(|| {
            crate::gfx::rt_reflections::RtReflectionSettings::resolve(
                self.ssr_intensity,
                self.ssr_max_distance,
            )
        })
    }

    /// Resolve the SSGI tunables into clamped `SsgiSettings`, or `None` when
    /// `indirect_lighting` is not `Ssgi` so the backend can skip the SSGI
    /// gather + composite passes entirely.
    pub fn ssgi_settings(&self) -> Option<crate::gfx::ssgi::SsgiSettings> {
        (self.indirect_lighting == IndirectLighting::Ssgi).then(|| {
            crate::gfx::ssgi::SsgiSettings::resolve(
                self.ssgi_intensity,
                self.ssgi_max_distance,
                self.ssgi_rays,
                self.ssgi_steps,
                self.ssgi_resolution.scale_divisor(),
            )
        })
    }

    /// Resolve the auto-exposure tunables into clamped
    /// `AutoExposureSettings`, or `None` when the toggle is off so the
    /// backend can skip the histogram passes entirely.
    pub fn auto_exposure_settings(
        &self,
    ) -> Option<crate::gfx::auto_exposure::AutoExposureSettings> {
        self.auto_exposure.then(|| {
            // `hdr_display = true` shifts AE's pivot from scene-white
            // (legacy SDR + ACES) to perceptual middle-grey, so the
            // average pixel reads as a comfortable mid-tone on a panel
            // that does no implicit tonemap. Falls back gracefully: even
            // if the platform rejects the HDR request at swapchain time,
            // SDR + ACES still produces a sensible result (just slightly
            // darker overall: middle-grey lands at middle-grey instead
            // of bright mid-tone, which is the photographically correct
            // exposure anyway).
            crate::gfx::auto_exposure::AutoExposureSettings::resolve(
                self.auto_exposure_min_ev,
                self.auto_exposure_max_ev,
                self.auto_exposure_speed,
                self.hdr_display,
            )
        })
    }
}

impl Component for PostProcessConfig {
    const NAME: &'static str = "PostProcessConfig";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }

    fn from_args(args: Self) -> Self {
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_resolves_to_neutral_params() {
        let p = PostProcessConfig::default().resolve();
        assert_eq!(p.bloom_intensity, 0.6);
        assert_eq!(p.bloom_threshold, 1.0);
        assert_eq!(p.bloom_knee, 0.5);
        // No exposure offset and no vignette out of the box.
        assert_eq!(p.exposure, 1.0);
        assert_eq!(p.vignette, 0.0);
        // Full LUT blend by default: a no-op until a ColorLut is declared.
        assert_eq!(p.lut_strength, 1.0);
    }

    #[test]
    fn exposure_ev_resolves_to_power_of_two_multiplier() {
        let cfg = PostProcessConfig {
            exposure_ev: 2.0,
            ..Default::default()
        };
        assert_eq!(cfg.resolve().exposure, 4.0);

        let cfg = PostProcessConfig {
            exposure_ev: -1.0,
            ..Default::default()
        };
        assert_eq!(cfg.resolve().exposure, 0.5);
    }

    #[test]
    fn exposure_ev_is_clamped_to_a_finite_multiplier() {
        let cfg = PostProcessConfig {
            exposure_ev: 1.0e9,
            ..Default::default()
        };
        let exposure = cfg.resolve().exposure;
        assert!(exposure.is_finite());
        assert_eq!(exposure, EXPOSURE_EV_LIMIT.exp2());
    }

    #[test]
    fn negative_and_overrange_inputs_are_clamped() {
        let cfg = PostProcessConfig {
            bloom_intensity: -3.0,
            bloom_threshold: -1.0,
            bloom_knee: -0.2,
            vignette_strength: 5.0,
            lut_strength: -2.0,
            ..Default::default()
        };
        let p = cfg.resolve();
        assert_eq!(p.bloom_intensity, 0.0);
        assert_eq!(p.bloom_threshold, 0.0);
        assert_eq!(p.bloom_knee, 0.0);
        assert_eq!(p.vignette, 1.0);
        assert_eq!(p.lut_strength, 0.0);
    }

    #[test]
    fn lut_strength_is_clamped_to_unit_range() {
        let cfg = PostProcessConfig {
            lut_strength: 3.0,
            ..Default::default()
        };
        assert_eq!(cfg.resolve().lut_strength, 1.0);
    }

    #[test]
    fn taa_defaults_off_and_round_trips_through_args() {
        assert!(!PostProcessConfig::default().taa);
        let cfg = PostProcessConfig {
            taa: true,
            ..Default::default()
        };
        assert!(PostProcessConfig::from_args(cfg.to_args()).taa);
    }

    #[test]
    fn ssao_defaults_off_with_neutral_tunables() {
        let cfg = PostProcessConfig::default();
        assert!(!cfg.ssao);
        assert_eq!(cfg.ssao_radius, 0.5);
        assert_eq!(cfg.ssao_intensity, 1.0);
        // No SsaoSettings produced while the toggle is off.
        assert!(cfg.ssao_settings().is_none());
    }

    #[test]
    fn ssao_settings_resolve_and_clamp_when_enabled() {
        let cfg = PostProcessConfig {
            ssao: true,
            ssao_radius: -1.0,
            ssao_intensity: 99.0,
            ..Default::default()
        };
        let s = cfg.ssao_settings().expect("ssao on");
        assert!(s.radius > 0.0);
        assert_eq!(s.intensity, 4.0);
    }

    #[test]
    fn ssao_deserialises_from_jsonl_args() {
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"ssao":true,"ssao_radius":0.6}"#).expect("parse");
        assert!(cfg.ssao);
        assert_eq!(cfg.ssao_radius, 0.6);
        // Omitted intensity falls back to the default.
        assert_eq!(cfg.ssao_intensity, 1.0);
    }

    #[test]
    fn ssr_defaults_off_with_neutral_tunables() {
        let cfg = PostProcessConfig::default();
        assert!(!cfg.ssr);
        assert_eq!(cfg.ssr_intensity, 0.7);
        assert_eq!(cfg.ssr_max_distance, 40.0);
        // No SsrSettings produced while the toggle is off.
        assert!(cfg.ssr_settings().is_none());
    }

    #[test]
    fn ssr_settings_resolve_and_clamp_when_enabled() {
        let cfg = PostProcessConfig {
            ssr: true,
            ssr_intensity: 9.0,
            ssr_max_distance: 1.0e6,
            ..Default::default()
        };
        let s = cfg.ssr_settings().expect("ssr on");
        assert_eq!(s.intensity, 1.0);
        assert!(s.max_distance > 0.0 && s.max_distance.is_finite());
    }

    #[test]
    fn ssr_deserialises_from_jsonl_args() {
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"ssr":true,"ssr_intensity":0.5}"#).expect("parse");
        assert!(cfg.ssr);
        assert_eq!(cfg.ssr_intensity, 0.5);
        // Omitted distance falls back to the default.
        assert_eq!(cfg.ssr_max_distance, 40.0);
    }

    #[test]
    fn rt_reflections_default_off_and_resolve_to_none() {
        let cfg = PostProcessConfig::default();
        assert!(!cfg.ray_traced_reflections);
        // No RtReflectionSettings produced while the toggle is off.
        assert!(cfg.rt_reflection_settings().is_none());
    }

    #[test]
    fn rt_reflection_settings_reuse_ssr_tunables_when_enabled() {
        let cfg = PostProcessConfig {
            ray_traced_reflections: true,
            ssr_intensity: 9.0,
            ssr_max_distance: 1.0e6,
            ..Default::default()
        };
        let s = cfg.rt_reflection_settings().expect("rt on");
        // Reuses the SSR intensity / distance fields, clamped by the RT resolve.
        assert_eq!(s.intensity, 1.0);
        assert!(s.max_distance > 0.0 && s.max_distance.is_finite());
    }

    #[test]
    fn rt_reflections_deserialise_from_jsonl_args() {
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"ray_traced_reflections":true,"ssr_intensity":0.5}"#)
                .expect("parse");
        assert!(cfg.ray_traced_reflections);
        assert!(cfg.rt_reflection_settings().is_some());
        // Omitting the field leaves ray tracing off.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"bloom_intensity":0.5}"#).expect("parse");
        assert!(!cfg.ray_traced_reflections);
        assert!(cfg.rt_reflection_settings().is_none());
    }

    #[test]
    fn ambient_intensity_defaults_neutral_and_clamps() {
        // Default is a no-op multiplier.
        assert_eq!(PostProcessConfig::default().ambient_intensity(), 1.0);
        // Authored values clamp into [0, 16].
        let hot = PostProcessConfig {
            ambient_intensity: 100.0,
            ..Default::default()
        };
        assert_eq!(hot.ambient_intensity(), 16.0);
        let neg = PostProcessConfig {
            ambient_intensity: -2.0,
            ..Default::default()
        };
        assert_eq!(neg.ambient_intensity(), 0.0);
        // Round-trips through JSONL like any other tunable.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"ambient_intensity":3.5}"#).expect("parse");
        assert_eq!(cfg.ambient_intensity(), 3.5);
    }

    #[test]
    fn ssgi_defaults_to_ibl_with_neutral_tunables() {
        let cfg = PostProcessConfig::default();
        assert_eq!(cfg.indirect_lighting, IndirectLighting::Ibl);
        assert_eq!(cfg.ssgi_intensity, 0.5);
        assert_eq!(cfg.ssgi_max_distance, 8.0);
        // The gather defaults to half resolution with the historical 8x12
        // ray/step counts.
        assert_eq!(cfg.ssgi_resolution, SsgiResolution::Half);
        assert_eq!(cfg.ssgi_rays, 8);
        assert_eq!(cfg.ssgi_steps, 12);
        // No SsgiSettings produced while indirect lighting is IBL-only.
        assert!(cfg.ssgi_settings().is_none());
    }

    #[test]
    fn ssgi_resolution_maps_to_a_per_axis_divisor() {
        assert_eq!(SsgiResolution::Full.scale_divisor(), 1);
        assert_eq!(SsgiResolution::Half.scale_divisor(), 2);
        assert_eq!(SsgiResolution::Quarter.scale_divisor(), 4);
        assert_eq!(SsgiResolution::default(), SsgiResolution::Half);
    }

    #[test]
    fn ssgi_resolution_and_counts_flow_into_settings() {
        let cfg = PostProcessConfig {
            indirect_lighting: IndirectLighting::Ssgi,
            ssgi_resolution: SsgiResolution::Quarter,
            ssgi_rays: 4,
            ssgi_steps: 20,
            ..Default::default()
        };
        let s = cfg.ssgi_settings().expect("ssgi on");
        assert_eq!(s.rays, 4);
        assert_eq!(s.steps, 20);
        assert_eq!(s.gi_scale, 4);
    }

    #[test]
    fn ssgi_resolution_and_counts_deserialise_from_jsonl_args() {
        let cfg: PostProcessConfig = serde_json::from_str(
            r#"{"indirect_lighting":"ssgi","ssgi_resolution":"full","ssgi_rays":16,"ssgi_steps":8}"#,
        )
        .expect("parse");
        assert_eq!(cfg.ssgi_resolution, SsgiResolution::Full);
        assert_eq!(cfg.ssgi_rays, 16);
        assert_eq!(cfg.ssgi_steps, 8);
        // Omitting them falls back to the half-resolution 8x12 defaults.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"indirect_lighting":"ssgi"}"#).expect("parse");
        assert_eq!(cfg.ssgi_resolution, SsgiResolution::Half);
        assert_eq!(cfg.ssgi_rays, 8);
        assert_eq!(cfg.ssgi_steps, 12);
    }

    #[test]
    fn ssgi_settings_resolve_and_clamp_when_enabled() {
        let cfg = PostProcessConfig {
            indirect_lighting: IndirectLighting::Ssgi,
            ssgi_intensity: 99.0,
            ssgi_max_distance: 1.0e6,
            ..Default::default()
        };
        let s = cfg.ssgi_settings().expect("ssgi on");
        assert_eq!(s.intensity, 4.0);
        assert!(s.max_distance > 0.0 && s.max_distance.is_finite());
    }

    #[test]
    fn ssgi_deserialises_from_jsonl_args() {
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"indirect_lighting":"ssgi","ssgi_intensity":0.8}"#)
                .expect("parse");
        assert_eq!(cfg.indirect_lighting, IndirectLighting::Ssgi);
        assert_eq!(cfg.ssgi_intensity, 0.8);
        // Omitted distance falls back to the default.
        assert_eq!(cfg.ssgi_max_distance, 8.0);
        // Omitting the field leaves indirect lighting on IBL.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"bloom_intensity":0.5}"#).expect("parse");
        assert_eq!(cfg.indirect_lighting, IndirectLighting::Ibl);
        assert!(cfg.ssgi_settings().is_none());
    }

    #[test]
    fn auto_exposure_defaults_off_with_neutral_tunables() {
        let cfg = PostProcessConfig::default();
        assert!(!cfg.auto_exposure);
        assert_eq!(cfg.auto_exposure_min_ev, -8.0);
        assert_eq!(cfg.auto_exposure_max_ev, 8.0);
        assert_eq!(cfg.auto_exposure_speed, 1.5);
        assert!(cfg.auto_exposure_settings().is_none());
    }

    #[test]
    fn auto_exposure_settings_resolve_when_enabled() {
        let cfg = PostProcessConfig {
            auto_exposure: true,
            auto_exposure_min_ev: -4.0,
            auto_exposure_max_ev: 6.0,
            auto_exposure_speed: 2.0,
            ..Default::default()
        };
        let s = cfg.auto_exposure_settings().expect("auto-exposure on");
        assert_eq!(s.min_ev, -4.0);
        assert_eq!(s.max_ev, 6.0);
        assert_eq!(s.speed, 2.0);
    }

    #[test]
    fn auto_exposure_deserialises_from_jsonl_args() {
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"auto_exposure":true,"auto_exposure_speed":3.0}"#)
                .expect("parse");
        assert!(cfg.auto_exposure);
        assert_eq!(cfg.auto_exposure_speed, 3.0);
        // Omitted bounds fall back to the defaults.
        assert_eq!(cfg.auto_exposure_min_ev, -8.0);
        assert_eq!(cfg.auto_exposure_max_ev, 8.0);
    }

    #[test]
    fn taa_deserialises_from_jsonl_args() {
        let cfg: PostProcessConfig = serde_json::from_str(r#"{"taa":true}"#).expect("parse");
        assert!(cfg.taa);
        // Omitting the field leaves TAA off.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"bloom_intensity":0.5}"#).expect("parse");
        assert!(!cfg.taa);
    }

    #[test]
    fn hdr_display_defaults_off_and_resolve_leaves_flag_zero() {
        let cfg = PostProcessConfig::default();
        assert!(!cfg.hdr_display);
        // The asset-side resolve always emits the SDR path; the backend is
        // the one that promotes hdr_output to 1.0 after confirming EDR.
        assert_eq!(cfg.resolve().hdr_output, 0.0);
    }

    #[test]
    fn hdr_display_round_trips_through_args_and_jsonl() {
        let cfg = PostProcessConfig {
            hdr_display: true,
            ..Default::default()
        };
        assert!(PostProcessConfig::from_args(cfg.to_args()).hdr_display);

        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"hdr_display":true}"#).expect("parse");
        assert!(cfg.hdr_display);
    }

    #[test]
    fn temporal_upscaling_defaults_off_with_quality_preset() {
        let cfg = PostProcessConfig::default();
        assert!(!cfg.temporal_upscaling);
        assert_eq!(cfg.upscale_quality, UpscaleQuality::Quality);
    }

    #[test]
    fn upscale_quality_scales_are_monotonic() {
        // Each step down in quality must reduce the per-axis ratio so render
        // cost drops monotonically as users dial quality lower.
        let q = UpscaleQuality::Quality.scale();
        let b = UpscaleQuality::Balanced.scale();
        let p = UpscaleQuality::Performance.scale();
        let u = UpscaleQuality::UltraPerformance.scale();
        assert!(q > b && b > p && p > u);
        assert!(u > 0.0);
    }

    #[test]
    fn occlusion_two_pass_defaults_off_and_round_trips() {
        assert!(!PostProcessConfig::default().occlusion_two_pass);
        let cfg = PostProcessConfig {
            occlusion_two_pass: true,
            ..Default::default()
        };
        assert!(PostProcessConfig::from_args(cfg.to_args()).occlusion_two_pass);
        // Deserialises from jsonl args; omitting it leaves the feature off.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"occlusion_two_pass":true}"#).expect("parse");
        assert!(cfg.occlusion_two_pass);
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"bloom_intensity":0.5}"#).expect("parse");
        assert!(!cfg.occlusion_two_pass);
    }

    #[test]
    fn upscale_backend_defaults_to_auto() {
        assert_eq!(
            PostProcessConfig::default().upscale_backend,
            UpscalerBackend::Auto
        );
        assert_eq!(UpscalerBackend::default(), UpscalerBackend::Auto);
    }

    #[test]
    fn upscale_backend_round_trips_via_snake_case_json() {
        for (s, want) in [
            ("auto", UpscalerBackend::Auto),
            ("fsr3", UpscalerBackend::Fsr3),
            ("dlss", UpscalerBackend::Dlss),
            ("xess", UpscalerBackend::Xess),
        ] {
            let json = format!(r#"{{"temporal_upscaling":true,"upscale_backend":"{s}"}}"#);
            let cfg: PostProcessConfig = serde_json::from_str(&json).expect("parse");
            assert_eq!(cfg.upscale_backend, want, "for {s}");
        }
        // Omitting the field falls back to Auto.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"temporal_upscaling":true}"#).expect("parse");
        assert_eq!(cfg.upscale_backend, UpscalerBackend::Auto);
    }

    #[test]
    fn upscale_backend_round_trips_through_args() {
        let cfg = PostProcessConfig {
            upscale_backend: UpscalerBackend::Xess,
            ..Default::default()
        };
        assert_eq!(
            PostProcessConfig::from_args(cfg.to_args()).upscale_backend,
            UpscalerBackend::Xess
        );
    }

    #[test]
    fn upscale_quality_round_trips_via_snake_case_json() {
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"temporal_upscaling":true,"upscale_quality":"performance"}"#)
                .expect("parse");
        assert!(cfg.temporal_upscaling);
        assert_eq!(cfg.upscale_quality, UpscaleQuality::Performance);
        // Omitting the preset falls back to the default.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"temporal_upscaling":true}"#).expect("parse");
        assert_eq!(cfg.upscale_quality, UpscaleQuality::Quality);
    }
}
