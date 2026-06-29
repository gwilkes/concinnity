// src/config.rs: persistent client state, split by lifetime and ownership.
//
// - `Config` (login / connection: server URL + account) lives in the OS
//   user-config dir as JSON (e.g.
//   ~/Library/Application Support/com.Concinnity.Concinnity/config.json on
//   macOS, ~/.config/concinnity/config.json on Linux). It is per-developer-
//   machine and stays out of the project tree.
//
// - `Settings` (runtime choices made in the in-engine settings menu: graphics,
//   audio, controls) lives in the project at `.concinnity/config/settings.bin`,
//   the mutable sibling of the build-regenerated `.concinnity/data`. It is
//   stored as CBOR: binary like the data blobs, but self-describing, so adding
//   or removing a setting never invalidates an existing file (a missing field
//   falls back to its default, an unknown field is ignored). bincode, which the
//   data blobs use, would be wrong here: it is positional, so it is safe only
//   because the data blobs are regenerated each build, whereas settings persist.
//
// Unknown fields are ignored on load so future additions are forwards-compatible.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const DEFAULT_SERVER: &str = "http://127.0.0.1:8080";
const SETTINGS_FILE: &str = "settings.bin";
// Environment override for the directory that holds `settings.bin`. When set,
// `Settings::load`/`save` read and write there instead of the project-relative
// `.concinnity/config`, so a sandbox (CI, a second instance, or a test that
// drives the binary) keeps its settings off the developer's real file. Unset in
// normal use.
const CONFIG_DIR_ENV: &str = "CN_CONFIG_DIR";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // Base HTTP URL of the concinnity-infra server.
    #[serde(default = "default_server")]
    pub server: String,

    // Account ID used for ?account_id= authenticated endpoints.
    pub user: Option<String>,
}

// The runtime settings store: choices made in the in-engine settings menu.
// Persisted as CBOR at `.concinnity/config/settings.bin`. Each field is
// `Option` (via the sub-structs): `None` means "use the world's default" so an
// unchanged setting never overrides the authored value.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default)]
    pub graphics: GraphicsSettings,
    #[serde(default)]
    pub audio: AudioSettings,
    #[serde(default)]
    pub controls: ControlsSettings,
}

// Persisted overrides for graphics settings. Missing fields stay `None` and
// fall back to the world's GraphicsConfig / Window / PostProcessConfig defaults.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphicsSettings {
    // Master graphics-quality preset. `None` means never configured: the first
    // launch seeds `Auto` (detect the GPU tier and clamp quality under the
    // world's authored look) and saves once. `Auto` re-resolves from the
    // detected tier each launch; a named tier is a fixed ceiling; `Custom`
    // imposes no ceiling (only the per-field overrides below apply).
    #[serde(default)]
    pub quality_preset: Option<crate::gfx::quality_preset::QualityPreset>,
    // Display sync (vsync). `None` uses the world's `GraphicsConfig.vsync`.
    #[serde(default)]
    pub vsync: Option<bool>,
    // Window mode (windowed / borderless / fullscreen). `None` uses the world's
    // `Window.mode`. Applied live.
    #[serde(default)]
    pub window_mode: Option<crate::assets::WindowMode>,
    // Window size [width, height] in pixels (windowed mode only). `None` uses
    // the world's `Window.width`/`Window.height`. Applied live.
    #[serde(default)]
    pub window_size: Option<[u32; 2]>,
    // Render-scale preset (upscaling quality). `None` uses the world's
    // `PostProcessConfig.upscale_quality`. Applied at next launch (the upscaler
    // and render targets are sized once at init).
    #[serde(default)]
    pub render_scale: Option<crate::assets::UpscaleQuality>,
    // Exposure offset in photographic stops. `None` uses the world's
    // `PostProcessConfig.exposure_ev`. Applied live (a pure post-process
    // uniform), and re-applied at init for a persisted choice.
    #[serde(default)]
    pub exposure_ev: Option<f32>,
    // Bloom additive strength. `None` uses the world's
    // `PostProcessConfig.bloom_intensity`. Applied live.
    #[serde(default)]
    pub bloom_intensity: Option<f32>,
    // Bloom luminance threshold. `None` uses the world's
    // `PostProcessConfig.bloom_threshold`. Applied live.
    #[serde(default)]
    pub bloom_threshold: Option<f32>,
    // Bloom soft-knee width. `None` uses the world's `PostProcessConfig.bloom_knee`.
    // Applied live (a `PostProcessParams` field, like the other bloom sliders).
    #[serde(default)]
    pub bloom_knee: Option<f32>,
    // Vignette strength in [0, 1]. `None` uses the world's
    // `PostProcessConfig.vignette_strength`. Applied live.
    #[serde(default)]
    pub vignette: Option<f32>,
    // Colour-LUT blend in [0, 1]. `None` uses the world's
    // `PostProcessConfig.lut_strength`. Applied live.
    #[serde(default)]
    pub lut_strength: Option<f32>,
    // Ambient (IBL) light scale. `None` uses the world's
    // `PostProcessConfig.ambient_intensity`. Applied live on Metal (it rides
    // `LightUniforms`, not `PostProcessParams`); re-applied at init.
    #[serde(default)]
    pub ambient_intensity: Option<f32>,
    // Quality-feature toggles. Each `None` uses the world's
    // `PostProcessConfig` value. They gate render passes whose GPU resources
    // (pipelines, targets, acceleration structures) are built at init, so a
    // change rebuilds those resources: applied live on Metal (the backend
    // rebuilds the affected effects in place); on backends without a live path
    // the choice persists and applies at the next launch.
    #[serde(default)]
    pub taa: Option<bool>,
    #[serde(default)]
    pub ssao: Option<bool>,
    #[serde(default)]
    pub ssr: Option<bool>,
    // Hardware ray-traced reflections (`PostProcessConfig.ray_traced_reflections`).
    #[serde(default)]
    pub ray_traced_reflections: Option<bool>,
    // Screen-space global illumination (`PostProcessConfig.indirect_lighting ==
    // ssgi`).
    #[serde(default)]
    pub ssgi: Option<bool>,
    #[serde(default)]
    pub auto_exposure: Option<bool>,
    // SSGI gather sub-quality: internal resolution, hemisphere rays per pixel,
    // and ray-march steps per ray (`PostProcessConfig.ssgi_resolution`/`_rays`/
    // `_steps`). Each `None` uses the world's value. Applied live on Metal (the
    // backend rebuilds the SSGI pass in place); persisted + applied at the next
    // launch on backends without a live path. Governed by the quality preset
    // ceiling like the toggles above.
    #[serde(default)]
    pub ssgi_resolution: Option<crate::assets::SsgiResolution>,
    #[serde(default)]
    pub ssgi_rays: Option<u32>,
    #[serde(default)]
    pub ssgi_steps: Option<u32>,
    // Roughness-aware reflection blur resolution
    // (`PostProcessConfig.reflection_blur_resolution`). `None` uses the world's
    // value. Applied live on Metal; governed by the quality preset ceiling like
    // the SSGI sub-quality above (only bites when a reflection feature is on).
    #[serde(default)]
    pub reflection_blur_resolution: Option<crate::assets::ReflectionBlurResolution>,
    // Per-feature sub-quality tunables (SSAO radius / intensity, SSR intensity /
    // distance, SSGI intensity / distance, auto-exposure EV bounds + speed). Each
    // `None` uses the world's `PostProcessConfig` value. Applied live on Metal via
    // `update_quality_params` (the backend re-reads them into a per-frame uniform,
    // no pass rebuild); look-tuning knobs, independent of the quality preset.
    #[serde(default)]
    pub ssao_radius: Option<f32>,
    #[serde(default)]
    pub ssao_intensity: Option<f32>,
    #[serde(default)]
    pub ssr_intensity: Option<f32>,
    #[serde(default)]
    pub ssr_max_distance: Option<f32>,
    #[serde(default)]
    pub ssgi_intensity: Option<f32>,
    #[serde(default)]
    pub ssgi_max_distance: Option<f32>,
    #[serde(default)]
    pub auto_exposure_min_ev: Option<f32>,
    #[serde(default)]
    pub auto_exposure_max_ev: Option<f32>,
    #[serde(default)]
    pub auto_exposure_speed: Option<f32>,
    // Shadow quality: cascade map resolution in texels (0 disables shadows) and
    // re-render cadence (`GraphicsConfig.shadow_map_size` / `shadow_update`).
    // `None` uses the world's value. Resolution is restart-required (the shadow
    // map array is sized once at backend init); cadence is applied live on Metal.
    // Both are governed by the quality preset ceiling like the toggles above.
    #[serde(default)]
    pub shadow_map_size: Option<u32>,
    #[serde(default)]
    pub shadow_update: Option<crate::assets::ShadowUpdate>,
    // Display-output / upscaling preferences. Unlike the quality knobs above,
    // these are independent of the master preset (a user choice, not a tier), and
    // each is restart-required: the swapchain format / render targets are sized
    // once at backend init, so a change persists and applies at the next launch.
    // `None` uses the world's `PostProcessConfig` value.
    #[serde(default)]
    pub temporal_upscaling: Option<bool>,
    #[serde(default)]
    pub hdr_display: Option<bool>,
    #[serde(default)]
    pub hdr_pq: Option<bool>,
    // System / streaming restart preferences, independent of the master preset
    // (like the display rows above) and each restart-required: ring-buffer depth
    // (`GraphicsConfig.frames_in_flight`), two-pass occlusion culling
    // (`PostProcessConfig.occlusion_two_pass`), and the texture-streaming pool /
    // per-frame upload budget (`StreamingConfig.texture_cap` / `texture_budget`,
    // driven together by one "Texture Quality" row). `None` uses the world's value.
    #[serde(default)]
    pub frames_in_flight: Option<u32>,
    #[serde(default)]
    pub occlusion_two_pass: Option<bool>,
    #[serde(default)]
    pub texture_cap: Option<u32>,
    #[serde(default)]
    pub texture_budget: Option<u32>,
}

// Persisted overrides for audio settings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioSettings {
    // Master output volume as a linear gain (0.0 = silent, 1.0 = full). `None`
    // leaves each emitter at its authored `AudioEmitter.volume`. Applied when a
    // world's audio initializes (the main menu itself has no audio).
    #[serde(default)]
    pub master_volume: Option<f32>,
}

// Persisted overrides for control settings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ControlsSettings {
    // Mouse-look sensitivity in radians per pixel. `None` uses the controlling
    // camera's authored `CameraController.mouse_sensitivity`. Applied when the
    // camera controller initializes.
    #[serde(default)]
    pub mouse_sensitivity: Option<f32>,
    // Gameplay movement key bindings (forward/back/strafe/sprint/jump/interact).
    // `None` uses the engine defaults (W/S/A/D/Shift/Space/E). Applied live: the
    // active backend decodes physical keys through this map.
    #[serde(default)]
    pub keymap: Option<crate::gfx::keymap::KeyMap>,
}

fn default_server() -> String {
    DEFAULT_SERVER.to_string()
}

impl Default for Config {
    fn default() -> Self {
        Config {
            server: default_server(),
            user: None,
        }
    }
}

impl Config {
    // Used by the CLI binary (main.rs, cli/login.rs), not lib code.
    #[allow(dead_code)]
    pub fn load() -> Self {
        let path = config_path();
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    // Persist the current config to disk. Creates parent directories as needed.
    #[allow(dead_code)]
    pub fn save(&self) -> std::io::Result<()> {
        let path = config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(&path, json)
    }
}

impl Settings {
    // Load from `<config dir>/settings.bin` (CBOR). When the file is absent,
    // fall back to migrating any graphics/audio/controls choices from the legacy
    // `config.json` (where they used to live) so an existing user's choices are
    // not silently reset. The migrated values are persisted on the next `save()`
    // (a settings change). Returns defaults when nothing is stored or the file
    // is unreadable.
    pub fn load() -> Self {
        Self::load_from(&config_dir())
    }

    // Persist to `<config dir>/settings.bin` as CBOR. Creates the config
    // directory as needed.
    pub fn save(&self) -> std::io::Result<()> {
        self.save_to(&config_dir())
    }

    // Read settings from `dir`. Split from `load` so the serialize-read path can
    // be tested against a sandbox dir, never the developer's real file.
    fn load_from(dir: &Path) -> Self {
        match std::fs::read(dir.join(SETTINGS_FILE)) {
            Ok(bytes) => ciborium::from_reader(&bytes[..]).unwrap_or_else(|e| {
                // A truncated / incompatible file: use defaults rather than
                // wiping silently mid-run (CBOR + serde defaults make this rare).
                tracing::warn!("settings store unreadable, using defaults: {e}");
                Settings::default()
            }),
            // No settings file yet: try a one-time read-fallback migration.
            Err(_) => migrate_from_legacy().unwrap_or_default(),
        }
    }

    // Write settings to `dir`. Split from `save` so the serialize-write path can
    // be tested against a sandbox dir, never the developer's real file.
    fn save_to(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes).map_err(std::io::Error::other)?;
        std::fs::write(dir.join(SETTINGS_FILE), bytes)
    }
}

// The directory holding `settings.bin`: the `CN_CONFIG_DIR` override when set,
// otherwise the project-relative `.concinnity/config`.
fn config_dir() -> PathBuf {
    std::env::var_os(CONFIG_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concinnity_core::world::CONCINNITY_CONFIG_DIR))
}

// Read the legacy `config.json` and lift any graphics/audio/controls sections
// it still carries into a `Settings`, or `None` when it has none of them. Used
// once, when no `settings.bin` exists yet.
fn migrate_from_legacy() -> Option<Settings> {
    let text = std::fs::read_to_string(config_path()).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    settings_from_legacy_value(&value)
}

// Pure mapping from a legacy config JSON value to a `Settings`. Returns `None`
// when none of the three sections are present (nothing to migrate).
fn settings_from_legacy_value(value: &serde_json::Value) -> Option<Settings> {
    let present = ["graphics", "audio", "controls"]
        .iter()
        .any(|k| value.get(k).is_some());
    if !present {
        return None;
    }
    Some(Settings {
        graphics: legacy_section(value, "graphics"),
        audio: legacy_section(value, "audio"),
        controls: legacy_section(value, "controls"),
    })
}

// Deserialize one section of a legacy config JSON value, falling back to the
// type's default when the section is absent or malformed.
fn legacy_section<T: serde::de::DeserializeOwned + Default>(
    value: &serde_json::Value,
    key: &str,
) -> T {
    value
        .get(key)
        .cloned()
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default()
}

fn config_path() -> PathBuf {
    directories::ProjectDirs::from("com", "Concinnity", "Concinnity")
        .map(|dirs| dirs.config_dir().join("config.json"))
        .unwrap_or_else(|| {
            directories::BaseDirs::new()
                .map(|b| b.home_dir().join(".concinnity").join("config.json"))
                .unwrap_or_else(|| PathBuf::from(".concinnity-config.json"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_correct_server() {
        let cfg = Config::default();
        assert_eq!(cfg.server, DEFAULT_SERVER);
        assert!(cfg.user.is_none());
    }

    #[test]
    fn config_login_roundtrip_through_json() {
        let cfg = Config {
            server: "http://10.0.0.1:9090".to_string(),
            user: Some("alice".to_string()),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let loaded: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.server, cfg.server);
        assert_eq!(loaded.user, cfg.user);
    }

    #[test]
    fn missing_user_field_uses_none() {
        let json = r#"{"server":"http://localhost:8080"}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.user.is_none());
    }

    #[test]
    fn missing_server_field_uses_default() {
        let json = r#"{"user":"bob"}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.server, DEFAULT_SERVER);
    }

    #[test]
    fn settings_cbor_roundtrip() {
        let s = Settings {
            graphics: GraphicsSettings {
                quality_preset: Some(crate::gfx::quality_preset::QualityPreset::High),
                vsync: Some(true),
                window_size: Some([1920, 1080]),
                exposure_ev: Some(-1.5),
                bloom_intensity: Some(0.8),
                bloom_threshold: Some(1.2),
                vignette: Some(0.3),
                lut_strength: Some(0.75),
                ambient_intensity: Some(1.5),
                taa: Some(true),
                ssao: Some(false),
                ssr: Some(true),
                ray_traced_reflections: Some(false),
                ssgi: Some(true),
                auto_exposure: Some(false),
                ssgi_resolution: Some(crate::assets::SsgiResolution::Quarter),
                ssgi_rays: Some(16),
                ssgi_steps: Some(24),
                reflection_blur_resolution: Some(crate::assets::ReflectionBlurResolution::Full),
                bloom_knee: Some(0.4),
                ssao_radius: Some(0.6),
                ssao_intensity: Some(1.2),
                ssr_intensity: Some(0.8),
                ssr_max_distance: Some(50.0),
                ssgi_intensity: Some(0.7),
                ssgi_max_distance: Some(10.0),
                auto_exposure_min_ev: Some(-6.0),
                auto_exposure_max_ev: Some(6.0),
                auto_exposure_speed: Some(2.0),
                shadow_map_size: Some(4096),
                shadow_update: Some(crate::assets::ShadowUpdate::EveryFrame),
                temporal_upscaling: Some(true),
                hdr_display: Some(true),
                hdr_pq: Some(false),
                frames_in_flight: Some(3),
                occlusion_two_pass: Some(true),
                texture_cap: Some(192),
                texture_budget: Some(8),
                ..Default::default()
            },
            audio: AudioSettings {
                master_volume: Some(0.5),
            },
            controls: ControlsSettings {
                mouse_sensitivity: Some(0.0025),
                keymap: Some(crate::gfx::keymap::KeyMap {
                    forward: crate::assets::Key::Up,
                    ..crate::gfx::keymap::KeyMap::default()
                }),
            },
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&s, &mut bytes).unwrap();
        let loaded: Settings = ciborium::from_reader(&bytes[..]).unwrap();
        assert_eq!(loaded, s);
    }

    #[test]
    fn settings_empty_cbor_map_is_all_defaults() {
        // An empty CBOR map deserializes to all-default (every section's fields
        // are `#[serde(default)]`), i.e. "use the world's values".
        let mut bytes = Vec::new();
        ciborium::into_writer(&std::collections::BTreeMap::<String, u8>::new(), &mut bytes)
            .unwrap();
        let loaded: Settings = ciborium::from_reader(&bytes[..]).unwrap();
        assert_eq!(loaded, Settings::default());
    }

    // Schema evolution: a file written by an older build (fewer fields) and one
    // written by a newer build (an extra field) both still load. This is the
    // whole reason for choosing self-describing CBOR over positional bincode.
    #[test]
    fn settings_tolerate_missing_and_unknown_fields() {
        #[derive(Serialize)]
        struct OtherShape {
            // Only one known section present...
            graphics: GraphicsSettings,
            // ...plus a field this build has never heard of.
            some_future_setting: u32,
        }
        let other = OtherShape {
            graphics: GraphicsSettings {
                vsync: Some(false),
                ..Default::default()
            },
            some_future_setting: 7,
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&other, &mut bytes).unwrap();
        let loaded: Settings = ciborium::from_reader(&bytes[..]).unwrap();
        // Known field carried; missing sections defaulted; unknown field ignored.
        assert_eq!(loaded.graphics.vsync, Some(false));
        assert_eq!(loaded.audio, AudioSettings::default());
        assert_eq!(loaded.controls, ControlsSettings::default());
    }

    #[test]
    fn migrates_legacy_settings_sections() {
        let value = serde_json::json!({
            "server": "http://x",
            "user": null,
            "graphics": { "vsync": true },
            "controls": { "mouse_sensitivity": 0.0025 },
        });
        let s = settings_from_legacy_value(&value).unwrap();
        assert_eq!(s.graphics.vsync, Some(true));
        assert_eq!(s.controls.mouse_sensitivity, Some(0.0025));
        assert_eq!(s.audio.master_volume, None);
    }

    #[test]
    fn no_legacy_sections_means_no_migration() {
        let value = serde_json::json!({ "server": "http://x", "user": "bob" });
        assert!(settings_from_legacy_value(&value).is_none());
    }

    // Regression guard: the on-disk `save`/`load` path must stay sandboxable so a
    // test can never clobber the developer's real `.concinnity/config/settings.bin`.
    // Drives the real serialize-write-read cycle, but against a temp dir. A
    // non-default `render_scale` mirrors a real persisted choice and proves a
    // populated field survives the round trip (it is the field whose loss the
    // original 271 -> 260 byte clobber would have shown).
    #[test]
    fn settings_save_load_roundtrip_is_sandboxed() {
        let s = Settings {
            graphics: GraphicsSettings {
                render_scale: Some(crate::assets::UpscaleQuality::Performance),
                vsync: Some(true),
                exposure_ev: Some(-1.5),
                ..Default::default()
            },
            ..Default::default()
        };

        let dir = tempfile::tempdir().unwrap();
        // The sandbox is somewhere else entirely, never the real settings dir.
        assert_ne!(dir.path(), config_dir());

        s.save_to(dir.path()).unwrap();
        // The write landed in the sandbox under the expected file name.
        assert!(dir.path().join(SETTINGS_FILE).exists());

        let loaded = Settings::load_from(dir.path());
        assert_eq!(loaded, s);
    }

    // With no override set, settings resolve to the project-relative config dir.
    // The env is read-only here (mutating it is process-global and would race
    // other tests), so the override branch is left to integration use.
    #[test]
    fn config_dir_defaults_to_project_dir() {
        if std::env::var_os(CONFIG_DIR_ENV).is_none() {
            assert_eq!(
                config_dir(),
                PathBuf::from(concinnity_core::world::CONCINNITY_CONFIG_DIR)
            );
        }
    }
}
