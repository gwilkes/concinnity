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

use crate::assets::UpscaleQuality;
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
}

// No ceiling: everything permitted, no forced upscaling. The resolved ceiling
// for `Custom`, and for `Auto` on hardware we could not classify (clamping an
// unknown GPU on a guess would risk needlessly degrading a capable one; the
// world's authored look is the best signal we have there).
const NONE: QualityCeiling = QualityCeiling {
    taa: true,
    ssao: true,
    ssr: true,
    ray_traced_reflections: true,
    ssgi: true,
    auto_exposure: true,
    min_upscale: UpscaleQuality::Quality,
};
const LOW: QualityCeiling = QualityCeiling {
    taa: true,
    ssao: false,
    ssr: false,
    ray_traced_reflections: false,
    ssgi: false,
    auto_exposure: true,
    min_upscale: UpscaleQuality::Performance,
};
const MEDIUM: QualityCeiling = QualityCeiling {
    taa: true,
    ssao: true,
    ssr: false,
    ray_traced_reflections: false,
    ssgi: false,
    auto_exposure: true,
    min_upscale: UpscaleQuality::Balanced,
};
const HIGH: QualityCeiling = QualityCeiling {
    taa: true,
    ssao: true,
    ssr: true,
    ray_traced_reflections: false,
    ssgi: true,
    auto_exposure: true,
    min_upscale: UpscaleQuality::Quality,
};
const ULTRA: QualityCeiling = QualityCeiling {
    taa: true,
    ssao: true,
    ssr: true,
    ray_traced_reflections: true,
    ssgi: true,
    auto_exposure: true,
    min_upscale: UpscaleQuality::Quality,
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
        }
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
