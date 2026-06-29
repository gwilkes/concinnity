// src/gfx/quality_preset.rs
//
// The master "Graphics Quality" preset and how it resolves to a performance
// ceiling over a world's authored look. The preset is persisted in
// GraphicsSettings; at init it produces a QualityCeiling that clamps the
// perf-relevant Tier-A settings DOWN where the chosen tier (or detected GPU,
// under Auto) cannot honor them. A ceiling never turns a feature on -- it only
// reduces -- so a world authored conservatively is never "upgraded", and an
// explicit per-row user override always wins over the ceiling (applied by the
// caller). This keeps the per-field `None = use the world's value` contract: the
// only thing persisted is the one preset marker, not a bake of every field.

use serde::{Deserialize, Serialize};

use crate::assets::{ReflectionBlurResolution, ShadowUpdate, SsgiResolution, UpscaleQuality};
use crate::gfx::backend::{GpuProfile, GpuTier};

// Persisted master graphics-quality choice. `Auto` resolves from the detected
// GPU tier each launch; a named tier (Low..Ultra) is a fixed ceiling; `Custom`
// imposes no ceiling (the user's per-row overrides drive). In GraphicsSettings
// a `None` (never persisted) means "never configured": the first launch seeds
// `Auto` and saves once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityPreset {
    Auto,
    Low,
    Medium,
    High,
    Ultra,
    Custom,
}

// A ceiling on the perf-relevant Tier-A settings: which feature toggles are
// permitted (`false` forces the feature off), and the least aggressive render
// scale allowed. A ceiling only reduces quality; it cannot enable a feature the
// world did not author.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QualityCeiling {
    pub taa: bool,
    pub ssao: bool,
    pub ssr: bool,
    pub ray_traced_reflections: bool,
    pub ssgi: bool,
    pub auto_exposure: bool,
    // The minimum upscaling the ceiling forces: the effective render scale is the
    // more aggressive (lower internal resolution) of the world's choice and this.
    // `Quality` (the least aggressive) means "no forced upscaling" -- the world's
    // choice stands.
    pub min_upscale: UpscaleQuality,
    // Caps on the SSGI gather sub-quality (they only bite where `ssgi` is
    // permitted): the finest gather resolution, and the most rays / ray-march
    // steps per pixel. Each clamps DOWN: the effective value is the coarser
    // resolution / smaller count of the world's choice and the cap. The
    // no-ceiling values are the engine maxima (`Full`, 32, 64), so a world's
    // authored value always stands under them.
    pub ssgi_resolution: SsgiResolution,
    pub ssgi_rays: u32,
    pub ssgi_steps: u32,
    // Cap on the roughness-aware reflection blur resolution (only bites where
    // `ssr` or `ray_traced_reflections` is permitted): the finest blur the tier
    // allows, clamping the world's choice coarser. The no-ceiling value is `Full`
    // (finest), so a world's authored value always stands under it.
    pub reflection_blur_resolution: ReflectionBlurResolution,
    // Cap on the shadow-map cascade resolution in texels (restart-required): the
    // effective size is the smaller of the world's choice and this cap. The
    // no-ceiling value is `u32::MAX`, so a world's authored size always stands.
    pub shadow_map_size: u32,
    // Whether the tier permits the `EveryFrame` shadow re-render cadence (live).
    // When false the cadence is clamped to the cheaper `Hybrid`; the no-ceiling
    // value is `true`, so a world's authored cadence always stands.
    pub allow_every_frame_shadows: bool,
}

// The coarser (higher render-resolution divisor) of two SSGI resolutions, the
// resolution analogue of `more_aggressive_upscale`. Used to clamp a world's
// gather resolution under a ceiling without ever making it finer.
pub(crate) fn coarser_ssgi_resolution(a: SsgiResolution, b: SsgiResolution) -> SsgiResolution {
    if a.scale_divisor() >= b.scale_divisor() {
        a
    } else {
        b
    }
}

// The coarser of two reflection-blur resolutions (the SSGI helper's sibling for
// the reflection blur enum).
pub(crate) fn coarser_reflection_blur(
    a: ReflectionBlurResolution,
    b: ReflectionBlurResolution,
) -> ReflectionBlurResolution {
    if a.scale_divisor() >= b.scale_divisor() {
        a
    } else {
        b
    }
}

// No ceiling: everything permitted, no forced upscaling. The resolved ceiling
// for `Custom`, and for `Auto` on hardware we could not classify (clamping an
// unknown GPU on a guess would risk needlessly degrading a capable one; the
// world's authored look is the best signal we have there).
// The engine maxima for the SSGI sub-quality caps, used wherever a tier imposes
// no SSGI ceiling: `Full` gather resolution, and the upper clamp bounds the
// gather honours (rays <= 32, steps <= 64). A world's authored value always
// stands under these.
const SSGI_RES_MAX: SsgiResolution = SsgiResolution::Full;
const SSGI_RAYS_MAX: u32 = 32;
const SSGI_STEPS_MAX: u32 = 64;
// `Full` (finest) is the no-cap reflection-blur resolution: a world's choice
// always stands coarser-or-equal under it.
const REFLECTION_BLUR_MAX: ReflectionBlurResolution = ReflectionBlurResolution::Full;
// `u32::MAX` is the no-cap shadow-map resolution: a world's authored size always
// stands smaller-or-equal under it.
const SHADOW_SIZE_MAX: u32 = u32::MAX;

// The coarser (smaller) of two shadow-map resolutions, the shadow analogue of
// the SSGI / reflection-blur clamp helpers. Used to clamp a world's authored
// size DOWN under a ceiling without ever raising it. `0` (shadows disabled) is
// the smallest, so a ceiling never re-enables a world that authored it off.
pub(crate) fn clamp_shadow_map_size(authored: u32, ceiling: &QualityCeiling) -> u32 {
    authored.min(ceiling.shadow_map_size)
}

// The world's shadow re-render cadence clamped under the ceiling: a tier that
// disallows `EveryFrame` forces the cheaper `Hybrid`; otherwise the authored
// cadence stands. Never raises (`Hybrid` -> `EveryFrame`).
pub(crate) fn clamp_shadow_update(
    authored: ShadowUpdate,
    ceiling: &QualityCeiling,
) -> ShadowUpdate {
    if ceiling.allow_every_frame_shadows {
        authored
    } else {
        ShadowUpdate::Hybrid
    }
}

const NONE: QualityCeiling = QualityCeiling {
    taa: true,
    ssao: true,
    ssr: true,
    ray_traced_reflections: true,
    ssgi: true,
    auto_exposure: true,
    min_upscale: UpscaleQuality::Quality,
    ssgi_resolution: SSGI_RES_MAX,
    ssgi_rays: SSGI_RAYS_MAX,
    ssgi_steps: SSGI_STEPS_MAX,
    reflection_blur_resolution: REFLECTION_BLUR_MAX,
    shadow_map_size: SHADOW_SIZE_MAX,
    allow_every_frame_shadows: true,
};
const LOW: QualityCeiling = QualityCeiling {
    taa: true,
    ssao: false,
    ssr: false,
    ray_traced_reflections: false,
    ssgi: false,
    auto_exposure: true,
    min_upscale: UpscaleQuality::Performance,
    ssgi_resolution: SsgiResolution::Quarter,
    ssgi_rays: 4,
    ssgi_steps: 8,
    reflection_blur_resolution: ReflectionBlurResolution::Quarter,
    shadow_map_size: 1024,
    allow_every_frame_shadows: false,
};
const MEDIUM: QualityCeiling = QualityCeiling {
    taa: true,
    ssao: true,
    ssr: false,
    ray_traced_reflections: false,
    ssgi: false,
    auto_exposure: true,
    min_upscale: UpscaleQuality::Balanced,
    ssgi_resolution: SsgiResolution::Half,
    ssgi_rays: 8,
    ssgi_steps: 12,
    reflection_blur_resolution: ReflectionBlurResolution::Half,
    shadow_map_size: 2048,
    allow_every_frame_shadows: false,
};
const HIGH: QualityCeiling = QualityCeiling {
    taa: true,
    ssao: true,
    ssr: true,
    ray_traced_reflections: false,
    ssgi: true,
    auto_exposure: true,
    min_upscale: UpscaleQuality::Quality,
    ssgi_resolution: SsgiResolution::Half,
    ssgi_rays: 8,
    ssgi_steps: 12,
    reflection_blur_resolution: ReflectionBlurResolution::Half,
    shadow_map_size: 4096,
    allow_every_frame_shadows: false,
};
const ULTRA: QualityCeiling = QualityCeiling {
    taa: true,
    ssao: true,
    ssr: true,
    ray_traced_reflections: true,
    ssgi: true,
    auto_exposure: true,
    min_upscale: UpscaleQuality::Quality,
    ssgi_resolution: SSGI_RES_MAX,
    ssgi_rays: SSGI_RAYS_MAX,
    ssgi_steps: SSGI_STEPS_MAX,
    reflection_blur_resolution: REFLECTION_BLUR_MAX,
    shadow_map_size: SHADOW_SIZE_MAX,
    allow_every_frame_shadows: true,
};

// The active ceiling for the persisted preset and detected GPU. `Auto` maps the
// GPU tier to a named tier (Unknown -> no ceiling); `Custom` imposes no ceiling;
// a named tier is fixed.
pub(crate) fn resolve_ceiling(preset: QualityPreset, profile: &GpuProfile) -> QualityCeiling {
    match preset {
        QualityPreset::Custom => NONE,
        QualityPreset::Low => LOW,
        QualityPreset::Medium => MEDIUM,
        QualityPreset::High => HIGH,
        QualityPreset::Ultra => ULTRA,
        QualityPreset::Auto => match profile.tier {
            GpuTier::Unknown => NONE,
            GpuTier::Integrated => LOW,
            GpuTier::EntryDiscrete => MEDIUM,
            GpuTier::MidDiscrete => HIGH,
            GpuTier::HighDiscrete => ULTRA,
        },
    }
}

// The more aggressive (lower internal resolution) of two upscale qualities,
// ordered by `settings::render_scale_index` (Quality < Balanced < Performance <
// UltraPerformance). Used to clamp a world's render scale under a ceiling's
// `min_upscale` without ever raising it.
pub(crate) fn more_aggressive_upscale(a: UpscaleQuality, b: UpscaleQuality) -> UpscaleQuality {
    use crate::gfx::settings::render_scale_index;
    if render_scale_index(a) >= render_scale_index(b) {
        a
    } else {
        b
    }
}

impl QualityPreset {
    // The presets in menu-cycle order. The settings-menu master row cycles
    // through these; `GRAPHICS_QUALITY_OPTIONS` in `gfx::settings` holds the
    // matching display labels in the same order (locked by a test there).
    pub(crate) const ALL: [QualityPreset; 6] = [
        Self::Auto,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::Ultra,
        Self::Custom,
    ];

    // The display name for this preset (the bare label, without an `Auto`
    // tier suffix; see `preset_label`).
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Ultra => "Ultra",
            Self::Custom => "Custom",
        }
    }

    // Parse a preset from a string (case-insensitive), for the `CN_QUALITY_PRESET`
    // env override that lets a test / CI run force a preset (e.g. `custom` for no
    // clamp) without writing to settings.bin. `None` for an unrecognized value.
    pub(crate) fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "ultra" => Some(Self::Ultra),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }
}

// The cycle index of a preset, and the preset at an index, over `ALL`. The
// master settings row cycles indices; these convert to and from the live
// `QualityPreset`. An out-of-range index falls back to `Auto`.
pub(crate) fn preset_index(preset: QualityPreset) -> usize {
    QualityPreset::ALL
        .iter()
        .position(|&p| p == preset)
        .unwrap_or(0)
}
pub(crate) fn preset_at(index: usize) -> QualityPreset {
    QualityPreset::ALL
        .get(index)
        .copied()
        .unwrap_or(QualityPreset::Auto)
}

// The named tier `Auto` resolves to on this GPU, for the menu label (e.g.
// "Auto (High)"). `None` when the GPU is unclassified, where `Auto` imposes no
// ceiling and the bare "Auto" reads correctly.
pub(crate) fn auto_resolved_name(profile: &GpuProfile) -> Option<&'static str> {
    match profile.tier {
        GpuTier::Unknown => None,
        GpuTier::Integrated => Some("Low"),
        GpuTier::EntryDiscrete => Some("Medium"),
        GpuTier::MidDiscrete => Some("High"),
        GpuTier::HighDiscrete => Some("Ultra"),
    }
}

// The master row's display text for a preset: a named tier shows its own name,
// while `Auto` annotates the tier it resolved to on the detected GPU (e.g.
// "Auto (High)") so the user can see what the auto-config chose.
pub(crate) fn preset_label(preset: QualityPreset, profile: &GpuProfile) -> String {
    match preset {
        QualityPreset::Auto => match auto_resolved_name(profile) {
            Some(tier) => format!("Auto ({tier})"),
            None => "Auto".to_string(),
        },
        other => other.name().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile_with_tier(tier: GpuTier) -> GpuProfile {
        GpuProfile {
            tier,
            ..GpuProfile::UNKNOWN
        }
    }

    #[test]
    fn custom_and_unknown_impose_no_ceiling() {
        // Custom never clamps, regardless of hardware.
        assert_eq!(
            resolve_ceiling(
                QualityPreset::Custom,
                &profile_with_tier(GpuTier::Integrated)
            ),
            NONE
        );
        // Auto on an unclassified GPU does not clamp on a guess.
        assert_eq!(
            resolve_ceiling(QualityPreset::Auto, &profile_with_tier(GpuTier::Unknown)),
            NONE
        );
    }

    #[test]
    fn auto_maps_tier_to_named_ceiling() {
        assert_eq!(
            resolve_ceiling(QualityPreset::Auto, &profile_with_tier(GpuTier::Integrated)),
            LOW
        );
        assert_eq!(
            resolve_ceiling(
                QualityPreset::Auto,
                &profile_with_tier(GpuTier::EntryDiscrete)
            ),
            MEDIUM
        );
        assert_eq!(
            resolve_ceiling(
                QualityPreset::Auto,
                &profile_with_tier(GpuTier::MidDiscrete)
            ),
            HIGH
        );
        assert_eq!(
            resolve_ceiling(
                QualityPreset::Auto,
                &profile_with_tier(GpuTier::HighDiscrete)
            ),
            ULTRA
        );
    }

    #[test]
    fn named_presets_resolve_independent_of_hardware() {
        // A named preset ignores the GPU tier.
        let weak = profile_with_tier(GpuTier::Integrated);
        assert_eq!(resolve_ceiling(QualityPreset::Low, &weak), LOW);
        assert_eq!(resolve_ceiling(QualityPreset::Ultra, &weak), ULTRA);
    }

    #[test]
    fn ceilings_are_monotonic_in_tier() {
        // Each tier permits a superset of the next-lower tier's features, so a
        // higher tier never disables something a lower tier allows.
        let order = [LOW, MEDIUM, HIGH, ULTRA];
        for pair in order.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            for (lo_on, hi_on) in [
                (lo.taa, hi.taa),
                (lo.ssao, hi.ssao),
                (lo.ssr, hi.ssr),
                (lo.ray_traced_reflections, hi.ray_traced_reflections),
                (lo.ssgi, hi.ssgi),
                (lo.auto_exposure, hi.auto_exposure),
            ] {
                assert!(!lo_on || hi_on, "a higher tier dropped a feature");
            }
            // And never forces more aggressive upscaling than a lower tier.
            assert_eq!(
                more_aggressive_upscale(lo.min_upscale, hi.min_upscale),
                lo.min_upscale
            );
            // The SSGI sub-quality caps rise (or hold) with the tier too: a
            // higher tier never permits fewer rays / steps or a coarser gather.
            assert!(lo.ssgi_rays <= hi.ssgi_rays, "ssgi_rays cap dropped");
            assert!(lo.ssgi_steps <= hi.ssgi_steps, "ssgi_steps cap dropped");
            assert_eq!(
                coarser_ssgi_resolution(lo.ssgi_resolution, hi.ssgi_resolution),
                lo.ssgi_resolution,
                "a higher tier permitted a coarser SSGI gather"
            );
            assert_eq!(
                coarser_reflection_blur(
                    lo.reflection_blur_resolution,
                    hi.reflection_blur_resolution
                ),
                lo.reflection_blur_resolution,
                "a higher tier permitted a coarser reflection blur"
            );
            // The shadow caps rise (or hold) with the tier: a higher tier never
            // permits a smaller shadow map or forbids a cadence a lower tier
            // allowed.
            assert!(
                lo.shadow_map_size <= hi.shadow_map_size,
                "shadow_map_size cap dropped"
            );
            assert!(
                !lo.allow_every_frame_shadows || hi.allow_every_frame_shadows,
                "a higher tier forbade the EveryFrame shadow cadence"
            );
        }
    }

    #[test]
    fn shadow_caps_clamp_down_only() {
        use crate::assets::ShadowUpdate;
        // No ceiling (Custom / Ultra) leaves a world's authored shadows alone.
        let none = resolve_ceiling(QualityPreset::Custom, &GpuProfile::UNKNOWN);
        assert_eq!(clamp_shadow_map_size(8192, &none), 8192);
        assert_eq!(
            clamp_shadow_update(ShadowUpdate::EveryFrame, &none),
            ShadowUpdate::EveryFrame
        );
        // Low caps the map size hard and forces the cheaper Hybrid cadence.
        let low = resolve_ceiling(QualityPreset::Low, &GpuProfile::UNKNOWN);
        assert_eq!(clamp_shadow_map_size(4096, &low), 1024);
        assert_eq!(
            clamp_shadow_update(ShadowUpdate::EveryFrame, &low),
            ShadowUpdate::Hybrid
        );
        // The clamp never raises: a world authoring a smaller map keeps it, and a
        // tier permitting EveryFrame leaves Hybrid as Hybrid.
        assert_eq!(clamp_shadow_map_size(512, &low), 512);
        assert_eq!(
            clamp_shadow_update(ShadowUpdate::Hybrid, &none),
            ShadowUpdate::Hybrid
        );
        // Shadows authored off (size 0) stay off under any ceiling.
        assert_eq!(clamp_shadow_map_size(0, &none), 0);
        assert_eq!(clamp_shadow_map_size(0, &low), 0);
    }

    #[test]
    fn ssgi_caps_clamp_down_only() {
        // The no-ceiling values are the engine maxima, so any authored value
        // stands under them.
        assert_eq!(NONE.ssgi_rays, 32);
        assert_eq!(NONE.ssgi_steps, 64);
        assert_eq!(NONE.ssgi_resolution, SsgiResolution::Full);
        // The coarser-resolution helper picks the higher divisor (lower quality),
        // and an equal input is returned as-is.
        assert_eq!(
            coarser_ssgi_resolution(SsgiResolution::Full, SsgiResolution::Quarter),
            SsgiResolution::Quarter
        );
        assert_eq!(
            coarser_ssgi_resolution(SsgiResolution::Half, SsgiResolution::Half),
            SsgiResolution::Half
        );
        // Ultra imposes the maxima (no clamp); Low caps hard.
        let ultra = resolve_ceiling(QualityPreset::Ultra, &GpuProfile::UNKNOWN);
        assert_eq!(ultra.ssgi_rays, 32);
        let low = resolve_ceiling(QualityPreset::Low, &GpuProfile::UNKNOWN);
        assert_eq!(low.ssgi_rays, 4);
        assert_eq!(low.ssgi_resolution, SsgiResolution::Quarter);
    }

    #[test]
    fn low_disables_the_expensive_effects() {
        // Resolve through the public path so the assertions are on a runtime
        // value, not the `const LOW` directly.
        let low = resolve_ceiling(QualityPreset::Low, &GpuProfile::UNKNOWN);
        assert!(!low.ssr);
        assert!(!low.ssgi);
        assert!(!low.ray_traced_reflections);
        assert!(!low.ssao);
        // AA and auto-exposure are cheap enough to keep on even at Low.
        assert!(low.taa);
        assert!(low.auto_exposure);
    }

    #[test]
    fn parse_is_case_insensitive_and_rejects_garbage() {
        assert_eq!(QualityPreset::parse("custom"), Some(QualityPreset::Custom));
        assert_eq!(QualityPreset::parse("  Ultra "), Some(QualityPreset::Ultra));
        assert_eq!(QualityPreset::parse("AUTO"), Some(QualityPreset::Auto));
        assert_eq!(QualityPreset::parse("nonsense"), None);
        assert_eq!(QualityPreset::parse(""), None);
    }

    #[test]
    fn preset_index_and_at_round_trip() {
        for p in QualityPreset::ALL {
            assert_eq!(preset_at(preset_index(p)), p);
        }
        // Auto leads the cycle order; an out-of-range index falls back to Auto.
        assert_eq!(preset_at(0), QualityPreset::Auto);
        assert_eq!(preset_at(99), QualityPreset::Auto);
    }

    #[test]
    fn auto_label_annotates_the_resolved_tier() {
        // Auto shows the tier it resolved to, so the user sees the auto choice.
        assert_eq!(
            preset_label(
                QualityPreset::Auto,
                &profile_with_tier(GpuTier::MidDiscrete)
            ),
            "Auto (High)"
        );
        assert_eq!(
            preset_label(QualityPreset::Auto, &profile_with_tier(GpuTier::Integrated)),
            "Auto (Low)"
        );
        // An unclassified GPU imposes no ceiling, so bare "Auto" is honest.
        assert_eq!(
            preset_label(QualityPreset::Auto, &profile_with_tier(GpuTier::Unknown)),
            "Auto"
        );
        // A named preset is just its own name, hardware-independent.
        assert_eq!(
            preset_label(
                QualityPreset::Ultra,
                &profile_with_tier(GpuTier::Integrated)
            ),
            "Ultra"
        );
        // The Auto suffix tracks the resolved ceiling.
        for tier in [
            GpuTier::Integrated,
            GpuTier::EntryDiscrete,
            GpuTier::MidDiscrete,
            GpuTier::HighDiscrete,
        ] {
            let profile = profile_with_tier(tier);
            let suffix = auto_resolved_name(&profile).unwrap();
            assert_eq!(
                preset_label(QualityPreset::Auto, &profile),
                format!("Auto ({suffix})")
            );
        }
    }

    #[test]
    fn more_aggressive_picks_the_lower_resolution() {
        // Higher index = more aggressive (lower internal resolution).
        assert_eq!(
            more_aggressive_upscale(UpscaleQuality::Quality, UpscaleQuality::Performance),
            UpscaleQuality::Performance
        );
        assert_eq!(
            more_aggressive_upscale(UpscaleQuality::UltraPerformance, UpscaleQuality::Balanced),
            UpscaleQuality::UltraPerformance
        );
        // Equal inputs return that quality; a ceiling of Quality never raises.
        assert_eq!(
            more_aggressive_upscale(UpscaleQuality::Balanced, UpscaleQuality::Quality),
            UpscaleQuality::Balanced
        );
    }
}
