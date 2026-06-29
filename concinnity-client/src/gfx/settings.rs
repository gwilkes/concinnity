// src/gfx/settings.rs
//
// The engine-side registry of user-facing settings an OptionSelect row can
// cycle. Each known setting key maps to its ordered option labels; the labels
// are what the value TextLabel shows. How a chosen option is applied (which
// backend call, which persisted field) lives in GraphicsSystem's drain, keyed
// by the same string. Keeping the option list here (not in world data) means a
// row can only target a setting the engine actually knows how to apply.

use crate::assets::{
    AaMode, ReflectionBlurResolution, SettingOp, ShadowUpdate, SsgiResolution, UpscaleQuality,
    WindowMode,
};

// Ordered option labels for `vsync`: index 0 is off, index 1 is on.
const VSYNC_OPTIONS: [&str; 2] = ["Off", "On"];
// Shared Off/On labels for the boolean quality toggles. Index 0 is off, 1 on,
// so `bool as usize` indexes directly.
const OFF_ON_OPTIONS: [&str; 2] = ["Off", "On"];
// The quality-feature toggle keys (Video "Quality" group). Each gates a render
// pass whose GPU resources are built at init, so a change rebuilds those
// resources (live on Metal; persisted + applied at the next launch elsewhere).
// `GraphicsSystem` maps each key to the `PostProcessConfig` field it flips.
pub(crate) const QUALITY_TOGGLE_KEYS: [&str; 5] = [
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

// Whether setting `key` can be changed on a device with the given capabilities.
// A capability-gated setting (e.g. `ray_traced_reflections`, which needs
// hardware ray tracing) is unavailable when the device lacks that capability;
// every other setting is always available. The settings menu grays out and
// disables an unavailable row. This is the one place to gate a future
// capability-dependent toggle.
pub(crate) fn setting_available(key: &str, caps: &crate::gfx::backend::DeviceCapabilities) -> bool {
    match key {
        "ray_traced_reflections" => caps.ray_tracing,
        _ => true,
    }
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

// Frame-rate cap options (Video Display group), in cycle order. "Unlimited" is 0
// (no cap); the rest are target FPS. A CPU-side frame pacer in the render loop
// enforces them, so they compose with vsync (the more restrictive wins). Indices
// map via the `fps_cap_*` helpers below (an authored cap off the discrete levels
// snaps to the nearest). Applied live (the pacer reads the value each frame).
const FPS_CAP_OPTIONS: [&str; 6] = ["Unlimited", "30", "60", "120", "144", "240"];
const FPS_CAP_VALUES: [u32; 6] = [0, 30, 60, 120, 144, 240];

// SSGI gather sub-quality dropdowns. The gather clamps rays to [1,32] and steps
// to [1,64]; these are the menu's discrete levels, in cycle order. Resolution is
// finest-first (Full/Half/Quarter), matching the enum. Indices map via the
// `ssgi_*_at` / `ssgi_*_index` helpers below.
const SSGI_RESOLUTION_OPTIONS: [&str; 3] = ["Full", "Half", "Quarter"];
const SSGI_RAYS_OPTIONS: [&str; 4] = ["4", "8", "16", "32"];
const SSGI_RAYS_COUNTS: [u32; 4] = [4, 8, 16, 32];
const SSGI_STEPS_OPTIONS: [&str; 4] = ["8", "12", "24", "48"];
const SSGI_STEPS_COUNTS: [u32; 4] = [8, 12, 24, 48];
// Reflection blur resolution options, finest-first (matches the enum).
const REFLECTION_BLUR_OPTIONS: [&str; 3] = ["Full", "Half", "Quarter"];

// Anti-aliasing mode options (Video "Quality" group), in cycle order matching
// the AaMode enum: Off (no edge smoothing), FXAA (cheap composite edge filter),
// TAA (temporal accumulation, the cleanest and most expensive). Rides the
// feature's live-reinit like the other quality cycles. Indices map via the
// `aa_mode_*` helpers below.
const AA_MODE_OPTIONS: [&str; 3] = ["Off", "FXAA", "TAA"];

// Shadow-map cascade resolution options (texels), in cycle order. "Off" disables
// shadows (size 0); the rest are the per-cascade texel dimensions. Indices map
// via the `shadow_resolution_*` helpers below (an authored size off the discrete
// levels snaps to the nearest). Restart-required (the shadow map array is sized
// once at backend init).
const SHADOW_RESOLUTION_OPTIONS: [&str; 4] = ["Off", "1024", "2048", "4096"];
const SHADOW_RESOLUTION_SIZES: [u32; 4] = [0, 1024, 2048, 4096];
// Shadow re-render cadence options, best (most expensive) first: "Every Frame"
// re-renders every cascade each frame, "Hybrid" amortizes the far cascades.
// Applied live (the cascade scheduler reads the policy each frame).
const SHADOW_UPDATE_OPTIONS: [&str; 2] = ["Every Frame", "Hybrid"];

// Shadow-distance options (world units the cascades cover), in cycle order. A
// larger distance shadows more of the scene but spreads the same resolution over
// more area. Indices map via the `shadow_distance_*` helpers below (an authored
// distance off the discrete levels snaps to the nearest). Applied live (the
// per-frame cascade-split computation reads the distance each draw).
const SHADOW_DISTANCE_OPTIONS: [&str; 4] = ["40 m", "80 m", "160 m", "320 m"];
const SHADOW_DISTANCE_VALUES: [u32; 4] = [40, 80, 160, 320];

// Shadow cascade-count options, in cycle order. More cascades keep distant
// shadows sharper (finer view-range slices) at the cost of an extra shadow-map
// render each; fewer is cheaper but blockier far away. Indices map via the
// `shadow_cascades_*` helpers below. Applied live (the per-frame split + schedule
// read the count each draw); preset-governed.
const SHADOW_CASCADES_OPTIONS: [&str; 3] = ["2", "3", "4"];
const SHADOW_CASCADES_VALUES: [u32; 3] = [2, 3, 4];

// Anisotropic-filtering degree options for the scene sampler, in cycle order.
// "Off" is 1x (plain trilinear); the rest are the max anisotropy degree. Indices
// map via the `anisotropy_*` helpers below (an authored degree off the discrete
// levels snaps to the nearest). Restart-required (the sampler is built once at
// backend init).
const ANISOTROPY_OPTIONS: [&str; 5] = ["Off", "2x", "4x", "8x", "16x"];
const ANISOTROPY_LEVELS: [u32; 5] = [1, 2, 4, 8, 16];

// Frame-buffering (ring-buffer depth / frames-in-flight) options, in cycle order.
// Lower is less latency, higher is smoother pacing. Restart-required (the ring
// buffers are sized once at init). Indices map via `frames_in_flight_at/_index`.
const FRAME_BUFFERING_OPTIONS: [&str; 3] = ["1", "2", "3"];
const FRAME_BUFFERING_COUNTS: [u32; 3] = [1, 2, 3];
// Texture-quality options: one row drives both the streaming pool size
// (`texture_cap`, how many high-resolution textures stay resident) and the
// per-frame upload budget (`texture_budget`, how fast they stream in). Restart-
// required (the pool is sized once at init). Indices map via
// `texture_quality_at/_index`; the index is recovered from the pool size.
const TEXTURE_QUALITY_OPTIONS: [&str; 4] = ["Low", "Medium", "High", "Ultra"];
const TEXTURE_QUALITY_CAPS: [u32; 4] = [48, 96, 192, 384];
const TEXTURE_QUALITY_BUDGETS: [u32; 4] = [2, 4, 8, 12];

// The cycle (dropdown) quality knobs governed by the preset ceiling like the
// boolean QUALITY_TOGGLE_KEYS. Each rides the feature's live-reinit rebuild
// (`apply_quality_settings`) -- the sub-tunable travels in its settings payload,
// so no new backend method is needed. `GraphicsSystem` maps each key to the
// `PostProcessConfig` field it cycles.
pub(crate) const QUALITY_CYCLE_KEYS: [&str; 5] = [
    "aa_mode",
    "ssgi_resolution",
    "ssgi_rays",
    "ssgi_steps",
    "reflection_blur_resolution",
];

// Master "Graphics Quality" preset options, in cycle order. The labels mirror
// `QualityPreset::ALL`'s order (locked by `graphics_quality_options_match_preset_order`);
// `quality_preset::preset_index` / `preset_at` map an index to the live preset.
// The `Auto` row is relabeled with its resolved tier (e.g. "Auto (High)") by the
// graphics system, which the static table cannot express.
const GRAPHICS_QUALITY_OPTIONS: [&str; 6] = ["Auto", "Low", "Medium", "High", "Ultra", "Custom"];

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
        "graphics_quality" => Some(&GRAPHICS_QUALITY_OPTIONS),
        "vsync" => Some(&VSYNC_OPTIONS),
        "window_mode" => Some(&WINDOW_MODE_OPTIONS),
        "render_scale" => Some(&RENDER_SCALE_OPTIONS),
        "fps_cap" => Some(&FPS_CAP_OPTIONS),
        "window_size" => Some(&WINDOW_SIZE_LABELS),
        "master_volume" => Some(&MASTER_VOLUME_OPTIONS),
        "aa_mode" => Some(&AA_MODE_OPTIONS),
        "ssgi_resolution" => Some(&SSGI_RESOLUTION_OPTIONS),
        "ssgi_rays" => Some(&SSGI_RAYS_OPTIONS),
        "ssgi_steps" => Some(&SSGI_STEPS_OPTIONS),
        "reflection_blur_resolution" => Some(&REFLECTION_BLUR_OPTIONS),
        "shadow_map_size" => Some(&SHADOW_RESOLUTION_OPTIONS),
        "shadow_update" => Some(&SHADOW_UPDATE_OPTIONS),
        "shadow_distance" => Some(&SHADOW_DISTANCE_OPTIONS),
        "shadow_cascades" => Some(&SHADOW_CASCADES_OPTIONS),
        "anisotropy" => Some(&ANISOTROPY_OPTIONS),
        "frames_in_flight" => Some(&FRAME_BUFFERING_OPTIONS),
        "texture_quality" => Some(&TEXTURE_QUALITY_OPTIONS),
        // Display-output / upscaling preference + occlusion toggles (Off/On).
        "temporal_upscaling" | "hdr_display" | "hdr_pq" | "occlusion_two_pass" => {
            Some(&OFF_ON_OPTIONS)
        }
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

// Anti-aliasing mode for an option index, and the index for a mode. Order
// matches AA_MODE_OPTIONS (Off, FXAA, TAA), which is also ascending cost so the
// index doubles as the aggressiveness rank the preset ceiling clamps against.
pub(crate) fn aa_mode_at(index: usize) -> AaMode {
    match index {
        0 => AaMode::Off,
        2 => AaMode::Taa,
        _ => AaMode::Fxaa,
    }
}
pub(crate) fn aa_mode_index(mode: AaMode) -> usize {
    match mode {
        AaMode::Off => 0,
        AaMode::Fxaa => 1,
        AaMode::Taa => 2,
    }
}

// SSGI gather resolution for an option index, and the index for a resolution.
// Order matches SSGI_RESOLUTION_OPTIONS (finest first).
pub(crate) fn ssgi_resolution_at(index: usize) -> SsgiResolution {
    match index {
        0 => SsgiResolution::Full,
        2 => SsgiResolution::Quarter,
        _ => SsgiResolution::Half,
    }
}
pub(crate) fn ssgi_resolution_index(res: SsgiResolution) -> usize {
    match res {
        SsgiResolution::Full => 0,
        SsgiResolution::Half => 1,
        SsgiResolution::Quarter => 2,
    }
}

// SSGI ray / step counts for an option index, and the menu index nearest an
// authored count (the world may author a value off the discrete levels; the row
// then shows the closest one).
pub(crate) fn ssgi_rays_at(index: usize) -> u32 {
    *SSGI_RAYS_COUNTS.get(index).unwrap_or(&SSGI_RAYS_COUNTS[1])
}
pub(crate) fn ssgi_rays_index(count: u32) -> usize {
    nearest_count_index(&SSGI_RAYS_COUNTS, count)
}
pub(crate) fn ssgi_steps_at(index: usize) -> u32 {
    *SSGI_STEPS_COUNTS
        .get(index)
        .unwrap_or(&SSGI_STEPS_COUNTS[1])
}
pub(crate) fn ssgi_steps_index(count: u32) -> usize {
    nearest_count_index(&SSGI_STEPS_COUNTS, count)
}
// The index of the level closest to `count` (ties pick the lower level).
fn nearest_count_index(levels: &[u32], count: u32) -> usize {
    levels
        .iter()
        .enumerate()
        .min_by_key(|&(_, &v)| v.abs_diff(count))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// Reflection blur resolution for an option index, and the index for a
// resolution. Order matches REFLECTION_BLUR_OPTIONS (finest first).
pub(crate) fn reflection_blur_at(index: usize) -> ReflectionBlurResolution {
    match index {
        0 => ReflectionBlurResolution::Full,
        2 => ReflectionBlurResolution::Quarter,
        _ => ReflectionBlurResolution::Half,
    }
}
pub(crate) fn reflection_blur_index(res: ReflectionBlurResolution) -> usize {
    match res {
        ReflectionBlurResolution::Full => 0,
        ReflectionBlurResolution::Half => 1,
        ReflectionBlurResolution::Quarter => 2,
    }
}

// Shadow-map resolution (texels) for an option index, and the menu index nearest
// an authored size (the world may author a size off the discrete levels; the row
// then shows the closest one). The default fallback is the world default (2048).
pub(crate) fn shadow_resolution_at(index: usize) -> u32 {
    *SHADOW_RESOLUTION_SIZES
        .get(index)
        .unwrap_or(&SHADOW_RESOLUTION_SIZES[2])
}
pub(crate) fn shadow_resolution_index(size: u32) -> usize {
    nearest_count_index(&SHADOW_RESOLUTION_SIZES, size)
}

// Shadow re-render cadence for an option index, and the index for a cadence.
// Order matches SHADOW_UPDATE_OPTIONS (EveryFrame first).
pub(crate) fn shadow_update_at(index: usize) -> ShadowUpdate {
    match index {
        0 => ShadowUpdate::EveryFrame,
        _ => ShadowUpdate::Hybrid,
    }
}
pub(crate) fn shadow_update_index(update: ShadowUpdate) -> usize {
    match update {
        ShadowUpdate::EveryFrame => 0,
        ShadowUpdate::Hybrid => 1,
    }
}

// Shadow distance (world units) for an option index, and the menu index nearest
// an authored distance (the world may author a distance off the discrete levels;
// the row then shows the closest one). The default fallback is the world default
// (80).
pub(crate) fn shadow_distance_at(index: usize) -> u32 {
    *SHADOW_DISTANCE_VALUES
        .get(index)
        .unwrap_or(&SHADOW_DISTANCE_VALUES[1])
}
pub(crate) fn shadow_distance_index(distance: u32) -> usize {
    nearest_count_index(&SHADOW_DISTANCE_VALUES, distance)
}

// Shadow cascade count for an option index, and the menu index nearest an
// authored count. The default fallback is the world default (4, the last index).
pub(crate) fn shadow_cascades_at(index: usize) -> u32 {
    *SHADOW_CASCADES_VALUES
        .get(index)
        .unwrap_or(&SHADOW_CASCADES_VALUES[2])
}
pub(crate) fn shadow_cascades_index(count: u32) -> usize {
    nearest_count_index(&SHADOW_CASCADES_VALUES, count)
}

// Anisotropic-filtering degree for an option index, and the menu index nearest an
// authored degree (the world may author a degree off the discrete levels; the row
// then shows the closest one). The default fallback is the world default (8x).
pub(crate) fn anisotropy_at(index: usize) -> u32 {
    *ANISOTROPY_LEVELS
        .get(index)
        .unwrap_or(&ANISOTROPY_LEVELS[3])
}
pub(crate) fn anisotropy_index(level: u32) -> usize {
    nearest_count_index(&ANISOTROPY_LEVELS, level)
}

// Frame-rate cap (FPS) for an option index, and the menu index nearest an
// authored cap (the world may author a cap off the discrete levels; the row then
// shows the closest one). The default fallback is "Unlimited" (index 0).
pub(crate) fn fps_cap_at(index: usize) -> u32 {
    *FPS_CAP_VALUES.get(index).unwrap_or(&FPS_CAP_VALUES[0])
}
pub(crate) fn fps_cap_index(cap: u32) -> usize {
    nearest_count_index(&FPS_CAP_VALUES, cap)
}

// Frames-in-flight (ring-buffer depth) for an option index, and the index for a
// count. Order matches FRAME_BUFFERING_OPTIONS (1, 2, 3); an out-of-range count
// snaps to the nearest level.
pub(crate) fn frames_in_flight_at(index: usize) -> u32 {
    *FRAME_BUFFERING_COUNTS
        .get(index)
        .unwrap_or(&FRAME_BUFFERING_COUNTS[1])
}
pub(crate) fn frames_in_flight_index(count: u32) -> usize {
    nearest_count_index(&FRAME_BUFFERING_COUNTS, count)
}

// Texture-quality level for an option index -> the (pool cap, per-frame budget)
// pair it sets, and the index recovered from a pool cap (the quality axis). An
// authored cap off the discrete levels snaps to the nearest level.
pub(crate) fn texture_quality_at(index: usize) -> (u32, u32) {
    let i = index.min(TEXTURE_QUALITY_CAPS.len() - 1);
    (TEXTURE_QUALITY_CAPS[i], TEXTURE_QUALITY_BUDGETS[i])
}
pub(crate) fn texture_quality_index(cap: u32) -> usize {
    nearest_count_index(&TEXTURE_QUALITY_CAPS, cap)
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
// Soft-knee width below the bloom threshold. Rides the live `update_post_process`
// path alongside the other bloom sliders (a `PostProcessParams` field).
const BLOOM_KNEE_RANGE: (f32, f32) = (0.0, 1.0);
// Per-feature sub-quality slider ranges. The UI ceilings are practical; the
// engine clamps the applied value in each feature's `*Settings::resolve`, mirrored
// by `slider_apply_value`. These ride the live `update_quality_params` path (the
// backend re-reads them into a per-frame uniform, no pass rebuild).
const SSAO_RADIUS_RANGE: (f32, f32) = (0.05, 2.0);
const SSAO_INTENSITY_RANGE: (f32, f32) = (0.0, 4.0);
const SSR_INTENSITY_RANGE: (f32, f32) = (0.0, 1.0);
const SSR_MAX_DISTANCE_RANGE: (f32, f32) = (1.0, 200.0);
const SSGI_INTENSITY_RANGE: (f32, f32) = (0.0, 4.0);
const SSGI_MAX_DISTANCE_RANGE: (f32, f32) = (0.5, 40.0);
const AE_MIN_EV_RANGE: (f32, f32) = (-16.0, 16.0);
const AE_MAX_EV_RANGE: (f32, f32) = (-16.0, 16.0);
const AE_SPEED_RANGE: (f32, f32) = (0.1, 6.0);

// The per-feature sub-quality slider keys, applied live by mutating the backend's
// stored `*Settings` (via `update_quality_params`) rather than rebuilding the pass.
// `bloom_knee` is deliberately NOT here: it is a `PostProcessParams` field and
// rides `update_post_process` like the other bloom sliders. These are look-tuning
// knobs, independent of the master quality preset (no ceiling, no Custom-flip),
// like the exposure / bloom / ambient sliders.
pub(crate) const QUALITY_PARAM_SLIDER_KEYS: [&str; 9] = [
    "ssao_radius",
    "ssao_intensity",
    "ssr_intensity",
    "ssr_max_distance",
    "ssgi_intensity",
    "ssgi_max_distance",
    "auto_exposure_min_ev",
    "auto_exposure_max_ev",
    "auto_exposure_speed",
];

// Whether `key` is one of the per-feature sub-quality sliders (applied live via
// `update_quality_params`, the stored-settings mutation path).
pub(crate) fn is_quality_param_slider(key: &str) -> bool {
    QUALITY_PARAM_SLIDER_KEYS.contains(&key)
}
// Mouse-sensitivity slider: a 1..100 UI scale (what the row shows) mapped
// linearly to a radians-per-pixel value in [MOUSE_SENS_MIN, MOUSE_SENS_MAX] by
// `slider_apply_value`. The endpoints span slow..fast; the camera's authored
// default (`DEFAULT_MOUSE_SENSITIVITY`) sits low on the track.
const MOUSE_SENSITIVITY_RANGE: (f32, f32) = (1.0, 100.0);
const MOUSE_SENS_MIN: f32 = 0.0003;
const MOUSE_SENS_MAX: f32 = 0.005;

// Field-of-view slider: a vertical FOV in degrees applied directly (the slider
// value IS the degrees, so `slider_apply_value` only clamps and the recover is
// the identity) to every Camera3D's `fov_y_degrees`. The range spans a narrow to
// a wide view; the engine's authored default (`DEFAULT_FOV`) sits mid-track.
const FOV_RANGE: (f32, f32) = (50.0, 100.0);
// Effective vertical FOV in degrees when the user has never chosen one. Matches
// Camera3D's authored default.
pub(crate) const DEFAULT_FOV: f32 = 75.0;

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
        "bloom_knee" => Some(BLOOM_KNEE_RANGE),
        "ssao_radius" => Some(SSAO_RADIUS_RANGE),
        "ssao_intensity" => Some(SSAO_INTENSITY_RANGE),
        "ssr_intensity" => Some(SSR_INTENSITY_RANGE),
        "ssr_max_distance" => Some(SSR_MAX_DISTANCE_RANGE),
        "ssgi_intensity" => Some(SSGI_INTENSITY_RANGE),
        "ssgi_max_distance" => Some(SSGI_MAX_DISTANCE_RANGE),
        "auto_exposure_min_ev" => Some(AE_MIN_EV_RANGE),
        "auto_exposure_max_ev" => Some(AE_MAX_EV_RANGE),
        "auto_exposure_speed" => Some(AE_SPEED_RANGE),
        "mouse_sensitivity" => Some(MOUSE_SENSITIVITY_RANGE),
        "fov" => Some(FOV_RANGE),
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
        // Exposure and the auto-exposure EV bounds read in photographic stops.
        "exposure" | "auto_exposure_min_ev" | "auto_exposure_max_ev" => {
            format!("{value:+.1} EV")
        }
        // World-space distances / radii read in metres.
        "ssr_max_distance" | "ssgi_max_distance" | "ssao_radius" => format!("{value:.1} m"),
        // [0, 1] strengths read more naturally as a percentage.
        "vignette" | "lut_strength" => format!("{}%", (value * 100.0).round() as i32),
        // Mouse sensitivity is a whole-number 1..100 scale.
        "mouse_sensitivity" => format!("{}", value.round() as i32),
        // Field of view reads in whole degrees.
        "fov" => format!("{}\u{00b0}", value.round() as i32),
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
        // Bloom soft-knee: lower-bounded only, like the other bloom params
        // (`PostProcessConfig::resolve` floors it at 0).
        "bloom_knee" => value.max(0.0),
        "vignette" | "lut_strength" => value.clamp(0.0, 1.0),
        "ambient_intensity" => value.clamp(0.0, 16.0),
        // Per-feature sub-quality clamps, mirroring each `*Settings::resolve`.
        "ssao_radius" => value.max(1.0e-3),
        "ssao_intensity" => value.clamp(0.0, 4.0),
        "ssr_intensity" => value.clamp(0.0, 1.0),
        "ssr_max_distance" => value.clamp(1.0, 200.0),
        "ssgi_intensity" => value.clamp(0.0, 4.0),
        "ssgi_max_distance" => value.clamp(0.5, 100.0),
        // The min/max EV bounds clamp to the engine EV limit; the resolve also
        // orders them (min <= max), which happens when the config is resolved.
        "auto_exposure_min_ev" | "auto_exposure_max_ev" => value.clamp(-16.0, 16.0),
        "auto_exposure_speed" => value.clamp(1.0e-3, 20.0),
        // 1..100 UI value -> radians/pixel, linearly across the sensitivity span.
        "mouse_sensitivity" => {
            let v = value.clamp(MOUSE_SENSITIVITY_RANGE.0, MOUSE_SENSITIVITY_RANGE.1);
            MOUSE_SENS_MIN + (MOUSE_SENS_MAX - MOUSE_SENS_MIN) * (v - 1.0) / 99.0
        }
        // FOV is stored as degrees, only clamped to the slider range.
        "fov" => value.clamp(FOV_RANGE.0, FOV_RANGE.1),
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
    fn graphics_quality_options_match_preset_order() {
        use crate::gfx::quality_preset::QualityPreset;
        // The master row's labels must line up 1:1 with the preset cycle order,
        // so an index from `preset_index` selects the right label and vice versa.
        assert_eq!(GRAPHICS_QUALITY_OPTIONS.len(), QualityPreset::ALL.len());
        for (i, p) in QualityPreset::ALL.iter().enumerate() {
            assert_eq!(GRAPHICS_QUALITY_OPTIONS[i], p.name(), "label {i}");
        }
        assert_eq!(
            options("graphics_quality"),
            Some(&GRAPHICS_QUALITY_OPTIONS[..])
        );
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
    fn rt_toggle_gated_on_ray_tracing_capability() {
        use crate::gfx::backend::DeviceCapabilities;
        let capable = DeviceCapabilities { ray_tracing: true };
        let incapable = DeviceCapabilities { ray_tracing: false };
        // RT reflections follow the device's ray-tracing capability.
        assert!(setting_available("ray_traced_reflections", &capable));
        assert!(!setting_available("ray_traced_reflections", &incapable));
        // Every other setting is always available, regardless of capability.
        for key in ["vsync", "aa_mode", "ssao", "ssr", "ssgi", "auto_exposure"] {
            assert!(
                setting_available(key, &incapable),
                "{key} should be available"
            );
        }
        // The default reports all capabilities present (an unwired backend keeps
        // every toggle live).
        assert!(setting_available(
            "ray_traced_reflections",
            &DeviceCapabilities::default()
        ));
    }

    #[test]
    fn aa_mode_round_trips_and_orders_by_cost() {
        // Index order is ascending cost (Off < FXAA < TAA), so it doubles as the
        // aggressiveness rank the preset ceiling clamps against.
        for (i, mode) in [AaMode::Off, AaMode::Fxaa, AaMode::Taa]
            .into_iter()
            .enumerate()
        {
            assert_eq!(aa_mode_index(mode), i);
            assert_eq!(aa_mode_at(i), mode);
        }
        assert_eq!(AA_MODE_OPTIONS.len(), 3);
        // An out-of-range index falls back to the FXAA default.
        assert_eq!(aa_mode_at(9), AaMode::Fxaa);
    }

    #[test]
    fn fps_cap_round_trips_and_snaps() {
        assert_eq!(FPS_CAP_OPTIONS.len(), FPS_CAP_VALUES.len());
        for (i, &cap) in FPS_CAP_VALUES.iter().enumerate() {
            assert_eq!(fps_cap_index(cap), i);
            assert_eq!(fps_cap_at(i), cap);
        }
        // "Unlimited" is 0 and the index-0 fallback.
        assert_eq!(fps_cap_at(0), 0);
        assert_eq!(fps_cap_at(99), 0);
        // An authored cap off the discrete levels snaps to the nearest.
        assert_eq!(fps_cap_index(58), fps_cap_index(60));
        assert_eq!(fps_cap_index(1000), FPS_CAP_VALUES.len() - 1);
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
    fn fov_is_a_degrees_slider() {
        // It is a slider (range present), not a cycle row.
        assert_eq!(slider_range("fov"), Some((50.0, 100.0)));
        assert!(options("fov").is_none());
        // The slider value IS the degrees: apply only clamps, recover is identity.
        for &deg in &[50.0_f32, 75.0, 100.0] {
            let stored = slider_apply_value("fov", deg);
            assert!((stored - deg).abs() < 1.0e-6);
            assert!((slider_recover_value("fov", stored) - deg).abs() < 1.0e-6);
        }
        // Out-of-range values clamp to the span.
        assert_eq!(slider_apply_value("fov", 10.0), 50.0);
        assert_eq!(slider_apply_value("fov", 200.0), 100.0);
        // The label reads in whole degrees, and the default sits inside the track.
        assert_eq!(format_slider_value("fov", 74.6), "75\u{00b0}");
        assert!((50.0..=100.0).contains(&DEFAULT_FOV));
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
    fn ssgi_sub_quality_round_trips_and_snaps() {
        // Resolution round-trips across every option.
        for r in [
            SsgiResolution::Full,
            SsgiResolution::Half,
            SsgiResolution::Quarter,
        ] {
            assert_eq!(ssgi_resolution_at(ssgi_resolution_index(r)), r);
        }
        // Ray / step levels round-trip on their preset values.
        for i in 0..SSGI_RAYS_COUNTS.len() {
            assert_eq!(ssgi_rays_index(ssgi_rays_at(i)), i);
        }
        for i in 0..SSGI_STEPS_COUNTS.len() {
            assert_eq!(ssgi_steps_index(ssgi_steps_at(i)), i);
        }
        // An authored value off the discrete levels snaps to the nearest.
        assert_eq!(ssgi_rays_index(7), 1); // 7 -> 8
        assert_eq!(ssgi_rays_index(20), 2); // 20 -> 16
        assert_eq!(ssgi_steps_index(40), 3); // 40 -> 48
        // The three SSGI sub-quality keys are cycle rows, not sliders.
        for key in ["ssgi_resolution", "ssgi_rays", "ssgi_steps"] {
            assert!(options(key).is_some(), "{key} should be a cycle row");
            assert!(slider_range(key).is_none(), "{key} should not be a slider");
        }
    }

    #[test]
    fn reflection_blur_round_trips() {
        for r in [
            ReflectionBlurResolution::Full,
            ReflectionBlurResolution::Half,
            ReflectionBlurResolution::Quarter,
        ] {
            assert_eq!(reflection_blur_at(reflection_blur_index(r)), r);
        }
        // It is registered as a cycle row + a governed cycle quality knob.
        assert_eq!(
            options("reflection_blur_resolution").map(|o| o.len()),
            Some(3)
        );
        assert!(QUALITY_CYCLE_KEYS.contains(&"reflection_blur_resolution"));
    }

    #[test]
    fn display_toggles_are_off_on_cycle_rows() {
        // The display-output / upscaling preferences are Off/On cycle rows, and
        // are NOT quality knobs (independent of the preset ceiling).
        for key in ["temporal_upscaling", "hdr_display", "hdr_pq"] {
            assert_eq!(options(key), Some(&["Off", "On"][..]), "{key} options");
            assert!(slider_range(key).is_none(), "{key} should not be a slider");
            assert!(!is_quality_toggle(key), "{key} is not a quality toggle");
            assert!(
                !QUALITY_CYCLE_KEYS.contains(&key),
                "{key} is not a quality cycle knob"
            );
        }
    }

    #[test]
    fn shadow_resolution_round_trips_and_snaps() {
        // Each discrete level round-trips through its index.
        for i in 0..SHADOW_RESOLUTION_SIZES.len() {
            assert_eq!(shadow_resolution_index(shadow_resolution_at(i)), i);
        }
        // "Off" is size 0 at index 0; the world default 2048 is index 2.
        assert_eq!(shadow_resolution_at(0), 0);
        assert_eq!(shadow_resolution_index(2048), 2);
        // An authored size off the discrete levels snaps to the nearest, and a
        // size above the top level snaps down to it.
        assert_eq!(shadow_resolution_index(1500), 1); // 1500 -> 1024
        assert_eq!(shadow_resolution_index(8192), 3); // 8192 -> 4096
        // It is a cycle row, not a slider.
        assert!(options("shadow_map_size").is_some());
        assert!(slider_range("shadow_map_size").is_none());
    }

    #[test]
    fn anisotropy_round_trips_and_snaps() {
        // Each discrete level round-trips through its index.
        for i in 0..ANISOTROPY_LEVELS.len() {
            assert_eq!(anisotropy_index(anisotropy_at(i)), i);
        }
        // "Off" is 1x at index 0; the world default 8x is index 3.
        assert_eq!(anisotropy_at(0), 1);
        assert_eq!(anisotropy_index(8), 3);
        // An authored degree off the discrete levels snaps to the nearest, and a
        // degree above the top level snaps down to it.
        assert_eq!(anisotropy_index(3), 1); // 3 -> 2x
        assert_eq!(anisotropy_index(32), 4); // 32 -> 16x
        // It is a cycle row, not a slider.
        assert!(options("anisotropy").is_some());
        assert!(slider_range("anisotropy").is_none());
    }

    #[test]
    fn shadow_distance_round_trips_and_snaps() {
        // Each discrete level round-trips through its index.
        for i in 0..SHADOW_DISTANCE_VALUES.len() {
            assert_eq!(shadow_distance_index(shadow_distance_at(i)), i);
        }
        // The world default 80 is index 1.
        assert_eq!(shadow_distance_at(1), 80);
        assert_eq!(shadow_distance_index(80), 1);
        // An authored distance off the discrete levels snaps to the nearest, and
        // one above the top level snaps down to it.
        assert_eq!(shadow_distance_index(50), 0); // 50 -> 40
        assert_eq!(shadow_distance_index(1000), 3); // 1000 -> 320
        // It is a cycle row, not a slider.
        assert!(options("shadow_distance").is_some());
        assert!(slider_range("shadow_distance").is_none());
    }

    #[test]
    fn shadow_cascades_round_trips_and_snaps() {
        for i in 0..SHADOW_CASCADES_VALUES.len() {
            assert_eq!(shadow_cascades_index(shadow_cascades_at(i)), i);
        }
        // The world default 4 is the last index; out-of-range falls back to it.
        assert_eq!(shadow_cascades_at(2), 4);
        assert_eq!(shadow_cascades_index(4), 2);
        assert_eq!(shadow_cascades_at(9), 4);
        // An authored count off the levels snaps to the nearest.
        assert_eq!(shadow_cascades_index(1), 0); // 1 -> 2
        assert!(options("shadow_cascades").is_some());
        assert!(slider_range("shadow_cascades").is_none());
    }

    #[test]
    fn shadow_update_round_trips() {
        for u in [ShadowUpdate::EveryFrame, ShadowUpdate::Hybrid] {
            assert_eq!(shadow_update_at(shadow_update_index(u)), u);
        }
        // EveryFrame leads the cycle (best / most expensive first).
        assert_eq!(shadow_update_at(0), ShadowUpdate::EveryFrame);
        assert_eq!(options("shadow_update").map(|o| o.len()), Some(2));
    }

    #[test]
    fn frame_buffering_round_trips_and_snaps() {
        for i in 0..FRAME_BUFFERING_COUNTS.len() {
            assert_eq!(frames_in_flight_index(frames_in_flight_at(i)), i);
        }
        assert_eq!(frames_in_flight_at(0), 1);
        // An out-of-range depth snaps to the nearest level.
        assert_eq!(frames_in_flight_index(4), 2); // 4 -> 3
        assert!(options("frames_in_flight").is_some());
    }

    #[test]
    fn texture_quality_pairs_cap_and_budget() {
        // Each level round-trips through its index (recovered from the pool cap),
        // and sets both the pool cap and the per-frame upload budget.
        for i in 0..TEXTURE_QUALITY_CAPS.len() {
            let (cap, budget) = texture_quality_at(i);
            assert_eq!(texture_quality_index(cap), i);
            assert_eq!(cap, TEXTURE_QUALITY_CAPS[i]);
            assert_eq!(budget, TEXTURE_QUALITY_BUDGETS[i]);
        }
        // The default world cap (96) reads as "Medium"; an authored cap off the
        // levels snaps to the nearest.
        assert_eq!(texture_quality_index(96), 1);
        assert_eq!(texture_quality_index(300), 3); // 300 -> 384 (Ultra)
        // occlusion_two_pass is an Off/On row, not a slider or preset knob.
        assert_eq!(options("occlusion_two_pass"), Some(&["Off", "On"][..]));
        assert!(slider_range("occlusion_two_pass").is_none());
        assert!(!is_quality_toggle("occlusion_two_pass"));
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
        // Per-feature sub-quality sliders clamp to their `*Settings::resolve`
        // domains; bloom_knee is lower-bounded like the other bloom params.
        assert_eq!(slider_apply_value("bloom_knee", -1.0), 0.0);
        assert_eq!(slider_apply_value("ssao_intensity", 100.0), 4.0);
        assert_eq!(slider_apply_value("ssr_intensity", 9.0), 1.0);
        assert_eq!(slider_apply_value("ssr_max_distance", 1.0e6), 200.0);
        assert_eq!(slider_apply_value("ssgi_intensity", 99.0), 4.0);
        assert_eq!(slider_apply_value("ssgi_max_distance", 1.0e6), 100.0);
        assert_eq!(slider_apply_value("auto_exposure_min_ev", -100.0), -16.0);
        assert_eq!(slider_apply_value("auto_exposure_max_ev", 100.0), 16.0);
        assert_eq!(slider_apply_value("auto_exposure_speed", 100.0), 20.0);
    }

    #[test]
    fn quality_param_sliders_are_independent_sliders() {
        // Each sub-quality slider is registered as a slider (has a range) and is
        // NOT a cycle row or a preset-governed quality knob (look-tuning, like the
        // exposure / bloom sliders).
        for key in QUALITY_PARAM_SLIDER_KEYS {
            assert!(
                is_quality_param_slider(key),
                "{key} should be a qparam slider"
            );
            assert!(
                slider_range(key).is_some(),
                "{key} should have a slider range"
            );
            assert!(options(key).is_none(), "{key} should not be a cycle row");
            assert!(
                !QUALITY_CYCLE_KEYS.contains(&key),
                "{key} should not be preset-governed"
            );
        }
        // bloom_knee is a slider but rides update_post_process, not the qparam path.
        assert!(slider_range("bloom_knee").is_some());
        assert!(!is_quality_param_slider("bloom_knee"));
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
