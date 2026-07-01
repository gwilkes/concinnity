// src/gfx/settings.rs
//
// The ordered option labels for every user-facing cycle setting, shared by the
// client renderer and the build pipeline. The client's settings drain reads a
// key's labels to display + persist the chosen value; the cook reads the label
// count to decide whether a settings row expands into a `<`/`>` stepper (two
// options) or a click-to-open dropdown (more than two). Keeping the labels here
// (not duplicated per crate) means the two never drift on a setting's option
// count. How a chosen option is applied stays in the client, keyed by the same
// string.

// Ordered option labels for `vsync`: index 0 is off, index 1 is on.
pub const VSYNC_OPTIONS: [&str; 2] = ["Off", "On"];
// Shared Off/On labels for the boolean quality toggles. Index 0 is off, 1 on,
// so `bool as usize` indexes directly.
pub const OFF_ON_OPTIONS: [&str; 2] = ["Off", "On"];

// The quality-feature toggle keys (Video "Quality" group). Each gates a render
// pass whose GPU resources are built at init, so a change rebuilds those
// resources (live on Metal; persisted + applied at the next launch elsewhere).
// The client maps each key to the `PostProcessConfig` field it flips.
pub const QUALITY_TOGGLE_KEYS: [&str; 5] = [
    "ssao",
    "ssr",
    "ray_traced_reflections",
    "ssgi",
    "auto_exposure",
];

// Whether `key` is one of the boolean quality toggles.
pub fn is_quality_toggle(key: &str) -> bool {
    QUALITY_TOGGLE_KEYS.contains(&key)
}

// Window mode options, in cycle order.
pub const WINDOW_MODE_OPTIONS: [&str; 3] = ["Windowed", "Borderless", "Fullscreen"];
// Render-scale (upscaling quality) options, in cycle order.
pub const RENDER_SCALE_OPTIONS: [&str; 4] = ["Quality", "Balanced", "Performance", "Ultra"];
// Upscaler-backend options, in cycle order matching the UpscalerBackend enum
// (Auto / FSR3 / DLSS / XeSS). DirectX / Vulkan only (Metal uses MetalFX).
pub const UPSCALE_BACKEND_OPTIONS: [&str; 4] = ["Auto", "FSR 3", "DLSS", "XeSS"];

// Frame-rate cap options (Video Display group), in cycle order. "Unlimited" is
// no cap; the rest are target FPS. The client pairs these with the numeric caps.
pub const FPS_CAP_OPTIONS: [&str; 6] = ["Unlimited", "30", "60", "120", "144", "240"];

// SSGI gather sub-quality dropdowns, in cycle order. Resolution is finest-first
// (Full/Half/Quarter), matching the enum.
pub const SSGI_RESOLUTION_OPTIONS: [&str; 3] = ["Full", "Half", "Quarter"];
pub const SSGI_RAYS_OPTIONS: [&str; 4] = ["4", "8", "16", "32"];
pub const SSGI_STEPS_OPTIONS: [&str; 4] = ["8", "12", "24", "48"];
// Reflection blur resolution options, finest-first (matches the enum).
pub const REFLECTION_BLUR_OPTIONS: [&str; 3] = ["Full", "Half", "Quarter"];

// Anti-aliasing mode options, in cycle order matching the AaMode enum: Off,
// FXAA (cheap composite edge filter), TAA (temporal accumulation). Ascending
// cost, so the index doubles as the aggressiveness rank the preset clamps.
pub const AA_MODE_OPTIONS: [&str; 3] = ["Off", "FXAA", "TAA"];

// Shadow-map cascade resolution options (texels), in cycle order. "Off"
// disables shadows; the rest are the per-cascade texel dimensions.
pub const SHADOW_RESOLUTION_OPTIONS: [&str; 4] = ["Off", "1024", "2048", "4096"];
// Shadow re-render cadence options, best (most expensive) first.
pub const SHADOW_UPDATE_OPTIONS: [&str; 2] = ["Every Frame", "Hybrid"];
// Shadow-distance options (world units the cascades cover), in cycle order.
pub const SHADOW_DISTANCE_OPTIONS: [&str; 4] = ["40 m", "80 m", "160 m", "320 m"];
// Shadow cascade-count options, in cycle order.
pub const SHADOW_CASCADES_OPTIONS: [&str; 3] = ["2", "3", "4"];
// Anisotropic-filtering degree options for the scene sampler, in cycle order.
// "Off" is 1x (plain trilinear); the rest are the max anisotropy degree.
pub const ANISOTROPY_OPTIONS: [&str; 5] = ["Off", "2x", "4x", "8x", "16x"];
// Frame-buffering (ring-buffer depth / frames-in-flight) options, in cycle
// order. Lower is less latency, higher is smoother pacing.
pub const FRAME_BUFFERING_OPTIONS: [&str; 3] = ["1", "2", "3"];
// Texture-quality options, in cycle order (drive the streaming pool cap + the
// per-frame upload budget on the client).
pub const TEXTURE_QUALITY_OPTIONS: [&str; 4] = ["Low", "Medium", "High", "Ultra"];

// Master "Graphics Quality" preset options, in cycle order. The labels mirror
// `QualityPreset::ALL`'s order (locked by a client test); the `Auto` row is
// relabeled with its resolved tier by the client, which this static table
// cannot express.
pub const GRAPHICS_QUALITY_OPTIONS: [&str; 6] =
    ["Auto", "Low", "Medium", "High", "Ultra", "Custom"];

// Master-volume options, in cycle order. The client maps each to a linear gain.
pub const MASTER_VOLUME_OPTIONS: [&str; 5] = ["Off", "25%", "50%", "75%", "100%"];

// Window-size presets [label, width, height], in cycle order. The label is the
// option text; the dimensions are read by the client's window_size mapping.
pub const WINDOW_SIZE_PRESETS: [(&str, u32, u32); 4] = [
    ("1280x720", 1280, 720),
    ("1600x900", 1600, 900),
    ("1920x1080", 1920, 1080),
    ("2560x1440", 2560, 1440),
];

// The window-size preset labels, derived once from the preset table so the two
// never drift.
pub const WINDOW_SIZE_LABELS: [&str; 4] = [
    WINDOW_SIZE_PRESETS[0].0,
    WINDOW_SIZE_PRESETS[1].0,
    WINDOW_SIZE_PRESETS[2].0,
    WINDOW_SIZE_PRESETS[3].0,
];

// The option labels for a known setting key, or `None` if the key is unknown
// (a slider or rebind key, or a typo). A key with more than two labels renders
// as a dropdown; two labels render as a `<`/`>` stepper.
pub fn options(key: &str) -> Option<&'static [&'static str]> {
    match key {
        "graphics_quality" => Some(&GRAPHICS_QUALITY_OPTIONS),
        "vsync" => Some(&VSYNC_OPTIONS),
        "window_mode" => Some(&WINDOW_MODE_OPTIONS),
        "render_scale" => Some(&RENDER_SCALE_OPTIONS),
        "upscale_backend" => Some(&UPSCALE_BACKEND_OPTIONS),
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
        // Stats-HUD display toggles: a master and one per readout (Off/On).
        "perf_stats" | "show_fps" | "show_vram" => Some(&OFF_ON_OPTIONS),
        key if is_quality_toggle(key) => Some(&OFF_ON_OPTIONS),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_size_labels_track_the_preset_table() {
        for (i, (label, _, _)) in WINDOW_SIZE_PRESETS.iter().enumerate() {
            assert_eq!(WINDOW_SIZE_LABELS[i], *label);
        }
    }

    #[test]
    fn quality_toggles_are_off_then_on() {
        for key in QUALITY_TOGGLE_KEYS {
            assert!(is_quality_toggle(key), "{key} should classify as a toggle");
            assert_eq!(options(key), Some(&["Off", "On"][..]), "{key} options");
        }
        assert!(!is_quality_toggle("vsync"));
        assert!(!is_quality_toggle("nope"));
    }

    #[test]
    fn unknown_key_has_no_options() {
        assert!(options("does_not_exist").is_none());
    }

    #[test]
    fn multi_option_settings_are_dropdowns_and_toggles_are_steppers() {
        // A setting with more than two options is a dropdown; exactly two is a
        // stepper. The cook keys its row expansion off this length.
        for key in [
            "graphics_quality",
            "window_mode",
            "render_scale",
            "fps_cap",
            "window_size",
            "master_volume",
            "aa_mode",
            "shadow_map_size",
            "anisotropy",
        ] {
            assert!(
                options(key).unwrap().len() > 2,
                "{key} should be a dropdown"
            );
        }
        for key in ["vsync", "shadow_update", "temporal_upscaling", "ssao"] {
            assert_eq!(options(key).unwrap().len(), 2, "{key} should be a stepper");
        }
    }
}
