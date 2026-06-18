// src/gfx/settings.rs
//
// The engine-side registry of user-facing settings an OptionSelect row can
// cycle. Each known setting key maps to its ordered option labels; the labels
// are what the value TextLabel shows. How a chosen option is applied (which
// backend call, which persisted field) lives in GraphicsSystem's drain, keyed
// by the same string. Keeping the option list here (not in world data) means a
// row can only target a setting the engine actually knows how to apply.

use crate::assets::{SettingOp, UpscaleQuality, WindowMode};

// Ordered option labels for `vsync`: index 0 is off, index 1 is on.
const VSYNC_OPTIONS: [&str; 2] = ["Off", "On"];
// Shared Off/On labels for the boolean quality toggles. Index 0 is off, 1 on,
// so `bool as usize` indexes directly.
const OFF_ON_OPTIONS: [&str; 2] = ["Off", "On"];
// The quality-feature toggle keys (Video "Quality" group). Each gates a render
// pass whose GPU resources are built at init, so a change rebuilds those
// resources (live on Metal; persisted + applied at the next launch elsewhere).
// `GraphicsSystem` maps each key to the `PostProcessConfig` field it flips.
pub(crate) const QUALITY_TOGGLE_KEYS: [&str; 6] = [
    "taa",
    "ssao",
    "ssr",
    "ray_traced_reflections",
    "ssgi",
    "auto_exposure",
];

// Whether `key` is one of the boolean quality toggles.
pub(crate) fn is_quality_toggle(key: &str) -> bool {
    QUALITY_TOGGLE_KEYS.contains(&key)
}

// Key-rebind settings (Controls tab) are a third setting category alongside
// cycle rows (`options`) and sliders (`slider_range`): a rebind key's value is a
// physical `Key`, not an option index or a fraction. Their classification +
// per-action data live in `gfx/keymap.rs` (the `Bindable` / `KeyMap` types) and
// the live map is owned by `GraphicsSystem`, so there is nothing to register
// here; this comment just records the third category for the reader.

// Window mode options, in cycle order. Indices map via `window_mode_at` /
// `window_mode_index`.
const WINDOW_MODE_OPTIONS: [&str; 3] = ["Windowed", "Borderless", "Fullscreen"];
// Render-scale (upscaling quality) options, in cycle order. Indices map via
// `render_scale_at` / `render_scale_index`.
const RENDER_SCALE_OPTIONS: [&str; 4] = ["Quality", "Balanced", "Performance", "Ultra"];

// Master-volume options, in cycle order. Indices map to a linear gain via
// `master_volume_at` / `master_volume_index`.
const MASTER_VOLUME_OPTIONS: [&str; 5] = ["Off", "25%", "50%", "75%", "100%"];
const MASTER_VOLUME_GAINS: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
// Effective master volume when the user has never chosen one (full gain).
pub(crate) const DEFAULT_MASTER_VOLUME: f32 = 1.0;

// Effective mouse sensitivity (radians per pixel) when the user has never
// chosen one. Matches `CameraController`'s authored default. Mouse sensitivity
// is a slider (1..100 -> radians/pixel), not a cycle row; see
// `MOUSE_SENSITIVITY_RANGE` and `slider_apply_value`.
pub(crate) const DEFAULT_MOUSE_SENSITIVITY: f32 = 0.0015;

// Window-size presets [label, width, height], in cycle order.
const WINDOW_SIZE_PRESETS: [(&str, u32, u32); 4] = [
    ("1280x720", 1280, 720),
    ("1600x900", 1600, 900),
    ("1920x1080", 1920, 1080),
    ("2560x1440", 2560, 1440),
];

// The option labels for a known setting key, or `None` if the key is unknown.
pub(crate) fn options(key: &str) -> Option<&'static [&'static str]> {
    match key {
        "vsync" => Some(&VSYNC_OPTIONS),
        "window_mode" => Some(&WINDOW_MODE_OPTIONS),
        "render_scale" => Some(&RENDER_SCALE_OPTIONS),
        "window_size" => Some(&WINDOW_SIZE_LABELS),
        "master_volume" => Some(&MASTER_VOLUME_OPTIONS),
        key if is_quality_toggle(key) => Some(&OFF_ON_OPTIONS),
        _ => None,
    }
}

// The window-size preset labels, derived once from the preset table so the two
// never drift.
const WINDOW_SIZE_LABELS: [&str; 4] = [
    WINDOW_SIZE_PRESETS[0].0,
    WINDOW_SIZE_PRESETS[1].0,
    WINDOW_SIZE_PRESETS[2].0,
    WINDOW_SIZE_PRESETS[3].0,
];

// WindowMode for an option index, and the index for a WindowMode. Order matches
// WINDOW_MODE_OPTIONS, not the enum's declaration order.
pub(crate) fn window_mode_at(index: usize) -> WindowMode {
    match index {
        1 => WindowMode::Borderless,
        2 => WindowMode::Fullscreen,
        _ => WindowMode::Windowed,
    }
}
pub(crate) fn window_mode_index(mode: WindowMode) -> usize {
    match mode {
        WindowMode::Windowed => 0,
        WindowMode::Borderless => 1,
        WindowMode::Fullscreen => 2,
    }
}

// UpscaleQuality for an option index, and the index for a quality.
pub(crate) fn render_scale_at(index: usize) -> UpscaleQuality {
    match index {
        1 => UpscaleQuality::Balanced,
        2 => UpscaleQuality::Performance,
        3 => UpscaleQuality::UltraPerformance,
        _ => UpscaleQuality::Quality,
    }
}
pub(crate) fn render_scale_index(quality: UpscaleQuality) -> usize {
    match quality {
        UpscaleQuality::Quality => 0,
        UpscaleQuality::Balanced => 1,
        UpscaleQuality::Performance => 2,
        UpscaleQuality::UltraPerformance => 3,
    }
}

// Window (width, height) for a preset index.
pub(crate) fn window_size_at(index: usize) -> (u32, u32) {
    let p = WINDOW_SIZE_PRESETS
        .get(index)
        .unwrap_or(&WINDOW_SIZE_PRESETS[0]);
    (p.1, p.2)
}
// The preset index whose dimensions match (w, h), or 0 when none match (a
// custom / asset-authored size that isn't a preset).
pub(crate) fn window_size_index(w: u32, h: u32) -> usize {
    WINDOW_SIZE_PRESETS
        .iter()
        .position(|p| p.1 == w && p.2 == h)
        .unwrap_or(0)
}

// Linear gain for a master-volume option index, and the index for a gain. A
// gain that is not a preset (an authored value) falls back to the last index
// (full).
pub(crate) fn master_volume_at(index: usize) -> f32 {
    *MASTER_VOLUME_GAINS
        .get(index)
        .unwrap_or(&DEFAULT_MASTER_VOLUME)
}
pub(crate) fn master_volume_index(gain: f32) -> usize {
    MASTER_VOLUME_GAINS
        .iter()
        .position(|g| (g - gain).abs() < 1.0e-4)
        .unwrap_or(MASTER_VOLUME_GAINS.len() - 1)
}

// Advance an option index one step in the given direction, wrapping at the
// ends. `len` must be non-zero (a known setting always has options). A
// `SetFraction` op only applies to slider settings and never reaches here.
pub(crate) fn cycle(index: usize, len: usize, op: SettingOp) -> usize {
    debug_assert!(len > 0);
    match op {
        SettingOp::Prev => (index + len - 1) % len,
        // Next steps forward; the slider and rebind ops never reach a cycle
        // setting, so treating them as Next is harmless.
        SettingOp::Next | SettingOp::SetFraction(_) | SettingOp::Rebind(_) => (index + 1) % len,
    }
}

// Slider (continuous) settings. Unlike the cycle settings above, these map a
// fraction in `[0, 1]` to a value in a fixed range; the range and the display
// format live here so a Slider row can only target a setting the engine knows
// how to apply. `slider_range` returning `Some` is what marks a key as a
// slider (vs `options` for a cycle row).

// Exposure slider range, in photographic stops (EV). Centered on 0 (neutral),
// so a fresh world reads as the midpoint.
const EXPOSURE_EV_RANGE: (f32, f32) = (-3.0, 3.0);
// Post-process slider ranges. The upper bounds are practical UI ceilings; the
// engine clamps applied values in `PostProcessConfig::resolve` (bloom is
// lower-bounded only, vignette / LUT are [0,1], ambient is [0,16]).
const BLOOM_INTENSITY_RANGE: (f32, f32) = (0.0, 2.0);
const BLOOM_THRESHOLD_RANGE: (f32, f32) = (0.0, 4.0);
const VIGNETTE_RANGE: (f32, f32) = (0.0, 1.0);
const LUT_STRENGTH_RANGE: (f32, f32) = (0.0, 1.0);
const AMBIENT_RANGE: (f32, f32) = (0.0, 4.0);
// Mouse-sensitivity slider: a 1..100 UI scale (what the row shows) mapped
// linearly to a radians-per-pixel value in [MOUSE_SENS_MIN, MOUSE_SENS_MAX] by
// `slider_apply_value`. The endpoints span slow..fast; the camera's authored
// default (`DEFAULT_MOUSE_SENSITIVITY`) sits low on the track.
const MOUSE_SENSITIVITY_RANGE: (f32, f32) = (1.0, 100.0);
const MOUSE_SENS_MIN: f32 = 0.0003;
const MOUSE_SENS_MAX: f32 = 0.005;

// The (min, max) value range for a slider key, or `None` if the key is not a
// slider setting.
pub(crate) fn slider_range(key: &str) -> Option<(f32, f32)> {
    match key {
        "exposure" => Some(EXPOSURE_EV_RANGE),
        "bloom_intensity" => Some(BLOOM_INTENSITY_RANGE),
        "bloom_threshold" => Some(BLOOM_THRESHOLD_RANGE),
        "vignette" => Some(VIGNETTE_RANGE),
        "lut_strength" => Some(LUT_STRENGTH_RANGE),
        "ambient_intensity" => Some(AMBIENT_RANGE),
        "mouse_sensitivity" => Some(MOUSE_SENSITIVITY_RANGE),
        _ => None,
    }
}

// The setting value at a `0.0..=1.0` fraction of its range, or `None` for a
// non-slider key. The fraction is clamped.
pub(crate) fn slider_value_at(key: &str, fraction: f32) -> Option<f32> {
    let (lo, hi) = slider_range(key)?;
    Some(lo + (hi - lo) * fraction.clamp(0.0, 1.0))
}

// The `0.0..=1.0` fraction a value sits at within its range, or `None` for a
// non-slider key. The result is clamped, so an out-of-range authored value
// pins the handle to an end.
pub(crate) fn slider_fraction(key: &str, value: f32) -> Option<f32> {
    let (lo, hi) = slider_range(key)?;
    let span = hi - lo;
    if span.abs() < f32::EPSILON {
        return Some(0.0);
    }
    Some(((value - lo) / span).clamp(0.0, 1.0))
}

// Human-readable value text for a slider, shown in the row's value label.
pub(crate) fn format_slider_value(key: &str, value: f32) -> String {
    match key {
        "exposure" => format!("{value:+.1} EV"),
        // [0, 1] strengths read more naturally as a percentage.
        "vignette" | "lut_strength" => format!("{}%", (value * 100.0).round() as i32),
        // Mouse sensitivity is a whole-number 1..100 scale.
        "mouse_sensitivity" => format!("{}", value.round() as i32),
        _ => format!("{value:.2}"),
    }
}

// The value to store in the live render param for slider `key` at the given
// user-facing `value`, clamped to match `PostProcessConfig::resolve`. Exposure
// is authored in EV but stored as the linear multiplier 2^ev; the rest are
// stored as-is (only clamped). The single source of truth shared by the live
// drag-apply and the persisted re-apply at init, so those two cannot diverge.
// The 16.0 EV bound mirrors core's `EXPOSURE_EV_LIMIT`.
pub(crate) fn slider_apply_value(key: &str, value: f32) -> f32 {
    match key {
        "exposure" => value.clamp(-16.0, 16.0).exp2(),
        "bloom_intensity" | "bloom_threshold" => value.max(0.0),
        "vignette" | "lut_strength" => value.clamp(0.0, 1.0),
        "ambient_intensity" => value.clamp(0.0, 16.0),
        // 1..100 UI value -> radians/pixel, linearly across the sensitivity span.
        "mouse_sensitivity" => {
            let v = value.clamp(MOUSE_SENSITIVITY_RANGE.0, MOUSE_SENSITIVITY_RANGE.1);
            MOUSE_SENS_MIN + (MOUSE_SENS_MAX - MOUSE_SENS_MIN) * (v - 1.0) / 99.0
        }
        _ => value,
    }
}

// The user-facing value recovered from a stored render param, the inverse of
// `slider_apply_value`, so a slider's handle + label re-sync to the live value
// at init. Only exposure is non-identity (2^ev stored -> EV shown).
pub(crate) fn slider_recover_value(key: &str, stored: f32) -> f32 {
    match key {
        // Guard log2(0); the slider range keeps the multiplier well above this.
        "exposure" => stored.max(1.0e-6).log2(),
        // radians/pixel -> 1..100 UI value (inverse of the apply mapping).
        "mouse_sensitivity" => {
            1.0 + (stored - MOUSE_SENS_MIN) / (MOUSE_SENS_MAX - MOUSE_SENS_MIN) * 99.0
        }
        _ => stored,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vsync_options_are_off_then_on() {
        assert_eq!(options("vsync"), Some(&["Off", "On"][..]));
    }

    #[test]
    fn unknown_key_has_no_options() {
        assert!(options("does_not_exist").is_none());
    }

    #[test]
    fn quality_toggles_are_off_then_on_and_classified() {
        for key in QUALITY_TOGGLE_KEYS {
            assert!(is_quality_toggle(key), "{key} should classify as a toggle");
            assert_eq!(options(key), Some(&["Off", "On"][..]), "{key} options");
            // A quality toggle is a cycle row, never a slider.
            assert!(slider_range(key).is_none(), "{key} should not be a slider");
        }
        // Non-toggle keys are not misclassified.
        assert!(!is_quality_toggle("vsync"));
        assert!(!is_quality_toggle("exposure"));
        assert!(!is_quality_toggle("nope"));
    }

    #[test]
    fn rebind_keys_are_a_distinct_category() {
        use crate::gfx::keymap::Bindable;
        // A rebind key is neither a cycle row nor a slider, so the three setting
        // categories never collide on one key.
        for b in Bindable::ALL {
            let key = b.setting_key();
            assert!(options(key).is_none(), "{key} should not be a cycle row");
            assert!(slider_range(key).is_none(), "{key} should not be a slider");
        }
    }

    #[test]
    fn cycle_next_wraps() {
        assert_eq!(cycle(0, 2, SettingOp::Next), 1);
        assert_eq!(cycle(1, 2, SettingOp::Next), 0);
    }

    #[test]
    fn cycle_prev_wraps() {
        assert_eq!(cycle(0, 2, SettingOp::Prev), 1);
        assert_eq!(cycle(1, 2, SettingOp::Prev), 0);
    }

    #[test]
    fn cycle_three_options() {
        assert_eq!(cycle(2, 3, SettingOp::Next), 0);
        assert_eq!(cycle(0, 3, SettingOp::Prev), 2);
    }

    #[test]
    fn known_settings_have_options() {
        assert_eq!(options("window_mode").unwrap().len(), 3);
        assert_eq!(options("render_scale").unwrap().len(), 4);
        assert_eq!(options("window_size").unwrap().len(), 4);
        assert_eq!(options("master_volume").unwrap().len(), 5);
        // mouse_sensitivity is a slider, not a cycle row.
        assert!(options("mouse_sensitivity").is_none());
        assert!(slider_range("mouse_sensitivity").is_some());
    }

    #[test]
    fn master_volume_index_and_at_round_trip() {
        for i in 0..MASTER_VOLUME_GAINS.len() {
            assert_eq!(master_volume_index(master_volume_at(i)), i);
        }
        // A non-preset gain falls back to the full (last) index.
        assert_eq!(master_volume_index(0.33), MASTER_VOLUME_GAINS.len() - 1);
        // The default reads as the full preset.
        assert_eq!(
            master_volume_at(master_volume_index(DEFAULT_MASTER_VOLUME)),
            1.0
        );
    }

    #[test]
    fn mouse_sensitivity_is_a_slider_1_to_100() {
        // It is a slider (range present), not a cycle row.
        assert_eq!(slider_range("mouse_sensitivity"), Some((1.0, 100.0)));
        assert!(options("mouse_sensitivity").is_none());
        // The 1..100 UI value maps linearly to radians/pixel and back.
        for &ui in &[1.0_f32, 25.0, 50.0, 100.0] {
            let stored = slider_apply_value("mouse_sensitivity", ui);
            let back = slider_recover_value("mouse_sensitivity", stored);
            assert!((back - ui).abs() < 1.0e-2, "ui={ui} -> {stored} -> {back}");
        }
        // Endpoints land on the radians/pixel span; values rise with the UI value.
        assert!((slider_apply_value("mouse_sensitivity", 1.0) - MOUSE_SENS_MIN).abs() < 1.0e-9);
        assert!((slider_apply_value("mouse_sensitivity", 100.0) - MOUSE_SENS_MAX).abs() < 1.0e-9);
        assert!(
            slider_apply_value("mouse_sensitivity", 10.0)
                < slider_apply_value("mouse_sensitivity", 90.0)
        );
        // The label is a whole number.
        assert_eq!(format_slider_value("mouse_sensitivity", 26.3), "26");
        // The authored default recovers to a position inside the track.
        let def = slider_recover_value("mouse_sensitivity", DEFAULT_MOUSE_SENSITIVITY);
        assert!(
            (1.0..=100.0).contains(&def),
            "default UI value {def} in range"
        );
    }

    #[test]
    fn window_mode_index_and_at_round_trip() {
        for m in [
            WindowMode::Windowed,
            WindowMode::Borderless,
            WindowMode::Fullscreen,
        ] {
            assert_eq!(window_mode_at(window_mode_index(m)), m);
        }
    }

    #[test]
    fn render_scale_index_and_at_round_trip() {
        for q in [
            UpscaleQuality::Quality,
            UpscaleQuality::Balanced,
            UpscaleQuality::Performance,
            UpscaleQuality::UltraPerformance,
        ] {
            assert_eq!(render_scale_at(render_scale_index(q)), q);
        }
    }

    #[test]
    fn window_size_matches_preset_or_defaults_to_first() {
        assert_eq!(window_size_index(1920, 1080), 2);
        assert_eq!(window_size_at(2), (1920, 1080));
        // A non-preset (asset-authored) size falls back to index 0.
        assert_eq!(window_size_index(1024, 768), 0);
    }

    #[test]
    fn window_size_labels_track_the_preset_table() {
        for (i, (label, _, _)) in WINDOW_SIZE_PRESETS.iter().enumerate() {
            assert_eq!(WINDOW_SIZE_LABELS[i], *label);
        }
    }

    #[test]
    fn exposure_is_a_slider_not_a_cycle() {
        // A slider key has a range and no cycle option list, and vice versa.
        assert!(slider_range("exposure").is_some());
        assert!(options("exposure").is_none());
        assert!(slider_range("vsync").is_none());
    }

    #[test]
    fn slider_value_and_fraction_round_trip() {
        // Endpoints and the midpoint map exactly.
        assert_eq!(slider_value_at("exposure", 0.0), Some(-3.0));
        assert_eq!(slider_value_at("exposure", 1.0), Some(3.0));
        assert_eq!(slider_value_at("exposure", 0.5), Some(0.0));
        for &f in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let v = slider_value_at("exposure", f).unwrap();
            let back = slider_fraction("exposure", v).unwrap();
            assert!((back - f).abs() < 1.0e-5, "f={f} -> v={v} -> {back}");
        }
    }

    #[test]
    fn slider_fraction_clamps_out_of_range() {
        // A value past either end pins the handle to that end.
        assert_eq!(slider_fraction("exposure", -100.0), Some(0.0));
        assert_eq!(slider_fraction("exposure", 100.0), Some(1.0));
        // The neutral default sits at the midpoint.
        assert_eq!(slider_fraction("exposure", 0.0), Some(0.5));
    }

    #[test]
    fn unknown_slider_key_has_no_range() {
        assert!(slider_range("nope").is_none());
        assert!(slider_value_at("nope", 0.5).is_none());
        assert!(slider_fraction("nope", 0.0).is_none());
    }

    #[test]
    fn exposure_value_is_formatted_in_stops() {
        assert_eq!(format_slider_value("exposure", 0.0), "+0.0 EV");
        assert_eq!(format_slider_value("exposure", 1.5), "+1.5 EV");
        assert_eq!(format_slider_value("exposure", -2.0), "-2.0 EV");
    }

    #[test]
    fn post_process_sliders_have_ranges_and_round_trip() {
        // Every live post-process slider key is a slider (not a cycle row) and
        // round-trips value<->fraction across its range.
        for key in [
            "bloom_intensity",
            "bloom_threshold",
            "vignette",
            "lut_strength",
            "ambient_intensity",
        ] {
            assert!(slider_range(key).is_some(), "{key} should be a slider");
            assert!(options(key).is_none(), "{key} should not be a cycle row");
            let (lo, hi) = slider_range(key).unwrap();
            assert!(lo < hi, "{key} range must be non-empty");
            assert_eq!(slider_value_at(key, 0.0), Some(lo));
            assert_eq!(slider_value_at(key, 1.0), Some(hi));
            for &f in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
                let v = slider_value_at(key, f).unwrap();
                let back = slider_fraction(key, v).unwrap();
                assert!((back - f).abs() < 1.0e-5, "{key}: f={f} -> {v} -> {back}");
            }
        }
    }

    #[test]
    fn slider_apply_and_recover_round_trip() {
        // Applying a slider value to the live param then recovering it must
        // return the same value, so the handle never jumps when a persisted
        // choice is re-applied at the next launch. Locks the shared transform
        // used by both the live drag-apply and the init re-apply.
        for key in [
            "exposure",
            "bloom_intensity",
            "bloom_threshold",
            "vignette",
            "lut_strength",
            "ambient_intensity",
        ] {
            for &f in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
                let v = slider_value_at(key, f).unwrap();
                let stored = slider_apply_value(key, v);
                let recovered = slider_recover_value(key, stored);
                assert!(
                    (recovered - v).abs() < 1.0e-4,
                    "{key}: v={v} stored={stored} recovered={recovered}"
                );
            }
        }
    }

    #[test]
    fn slider_apply_value_clamps_match_resolve() {
        // Out-of-range inputs (e.g. a hand-edited settings.bin) clamp to the
        // engine's domain, matching PostProcessConfig::resolve.
        assert_eq!(slider_apply_value("bloom_intensity", -5.0), 0.0);
        assert_eq!(slider_apply_value("vignette", 2.0), 1.0);
        assert_eq!(slider_apply_value("lut_strength", -1.0), 0.0);
        assert_eq!(slider_apply_value("ambient_intensity", 100.0), 16.0);
        // Exposure stores the linear multiplier 2^ev (clamped EV).
        assert_eq!(slider_apply_value("exposure", 2.0), 4.0);
        assert!((slider_recover_value("exposure", 4.0) - 2.0).abs() < 1.0e-5);
    }

    #[test]
    fn strength_sliders_format_as_percent() {
        assert_eq!(format_slider_value("vignette", 0.0), "0%");
        assert_eq!(format_slider_value("vignette", 0.5), "50%");
        assert_eq!(format_slider_value("lut_strength", 1.0), "100%");
        // Bloom / ambient use the plain two-decimal fallback.
        assert_eq!(format_slider_value("bloom_intensity", 0.6), "0.60");
        assert_eq!(format_slider_value("ambient_intensity", 1.25), "1.25");
    }
}
