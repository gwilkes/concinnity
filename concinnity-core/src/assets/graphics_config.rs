// src/assets/graphics_config.rs

use crate::ecs::{AssetOrigin, CompanionSpec, Component};

/// How often each cascaded-shadow-map slice is re-rendered. The shadow pass
/// re-rasterizes all scene geometry into every cascade, so it is one of the
/// heavier passes; updating distant cascades less often cuts that cost.
///
/// `hybrid` (the default) re-renders the nearest cascade every frame (so close
/// shadows stay crisp) and rotates through the farther cascades one per frame.
/// Distant shadows then lag a few frames while the camera moves, which is
/// imperceptible at that range. `every_frame` re-renders all cascades every
/// frame: pick it for scenes with fast-moving shadow casters where even distant
/// shadow lag is unacceptable. Each cascade is always primed (rendered once)
/// before it is sampled, so there is never missing shadow data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ShadowUpdate {
    EveryFrame,
    #[default]
    Hybrid,
}

/// Rendering settings for the world: frame pacing, shadows, and clear colour.
/// One per world. The GPU backend is chosen by the engine for the platform and
/// is not user-configurable.
///
/// ```json
/// {
///   "name": "gfx",
///   "type": "GraphicsConfig",
///   "args": { "clear_color": [0.1, 0.1, 0.15, 1.0], "frames_in_flight": 2 }
/// }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GraphicsConfig {
    /// Cap the render loop at this many frames, then exit. Unset runs until the
    /// window is closed.
    pub max_frames: Option<u64>,
    /// Preferred number of frames in flight (1-3). Higher can smooth pacing at
    /// the cost of input latency.
    pub frames_in_flight: u32,
    /// Cap the frame rate to the display refresh (vsync). Defaults to `false`:
    /// the render loop runs uncapped (DirectX presents with tearing allowed,
    /// Vulkan uses a mailbox present mode), which is what a benchmark wants. Set
    /// to `true` to lock presentation to the monitor refresh, eliminating tearing
    /// and the wasted frames that never reach the screen.
    pub vsync: bool,
    /// Cap the frame rate to this many frames per second. `0` (default) leaves
    /// the loop uncapped. The cap is a CPU-side frame pacer, so it composes with
    /// `vsync`: the more restrictive of the two wins. Useful for limiting heat,
    /// fan noise, and power draw, or matching a fixed refresh.
    pub fps_cap: u32,
    /// Background clear colour [r, g, b, a] in linear 0..1 space.
    pub clear_color: [f32; 4],
    /// Rotation speed of the demo object in radians per second. Only used when
    /// no camera is present.
    pub rotation_speed: f32,
    /// Shadow map resolution in texels (e.g. 2048). Set to 0 to disable shadows.
    pub shadow_map_size: u32,
    /// How often shadow cascades are re-rendered. `hybrid` (default) amortizes
    /// the far cascades across frames; `every_frame` refreshes them all every
    /// frame. See [ShadowUpdate].
    pub shadow_update: ShadowUpdate,
    /// How far from the camera shadows are cast, in world units (e.g. 80). The
    /// cascades cover from the near plane out to this distance; a larger value
    /// shadows more of the scene but spreads the same shadow-map resolution over
    /// more area (softer, blockier shadows). Capped at the camera far plane.
    pub shadow_distance: u32,
    /// Number of shadow cascades, 1 to 4 (`4` is the default and the maximum).
    /// More cascades keep distant shadows sharper by splitting the view range
    /// into finer slices, at the cost of an extra shadow-map render per cascade;
    /// fewer is cheaper but blockier far from the camera. The slice count covers
    /// the same `shadow_distance` regardless.
    pub shadow_cascades: u32,
    /// Maximum anisotropic-filtering degree for the scene texture sampler
    /// (albedo + normal maps), e.g. 8. Higher keeps textures viewed at a grazing
    /// angle (floors, walls receding into the distance) sharp instead of blurring
    /// along the minor axis, at a small sampling cost. `1` disables anisotropy
    /// (plain trilinear). Clamped to the GPU's supported range (1..16) at init.
    pub anisotropy: u32,
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            max_frames: None,
            frames_in_flight: 2,
            vsync: false,
            fps_cap: 0,
            clear_color: [0.01, 0.01, 0.02, 1.0],
            rotation_speed: 1.0,
            shadow_map_size: 2048,
            shadow_update: ShadowUpdate::default(),
            shadow_distance: 80,
            shadow_cascades: 4,
            anisotropy: 8,
        }
    }
}

impl Component for GraphicsConfig {
    const NAME: &'static str = "GraphicsConfig";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    // GraphicsConfig is the marker that a world renders: its presence gates the
    // internal GraphicsSystem at runtime and pulls in the assets that system
    // needs: a Window and, when the world declares no ShaderStage of its own,
    // the bundled default shader set.
    fn companions(_args: &serde_json::Value, world: &[serde_json::Value]) -> Vec<CompanionSpec> {
        let mut specs = vec![CompanionSpec {
            name: "Window",
            asset_type: "Window",
            args: serde_json::json!({}),
        }];

        // Inject the bundled default vertex + fragment ShaderStages as a set
        // only when the world declares no ShaderStage at all. A world with even
        // one custom ShaderStage owns its pipeline and is left alone.
        let has_shader = world.iter().any(|v| {
            v.get("type")
                .and_then(|t| t.as_str())
                .map(|s| s.to_lowercase().replace('_', "") == "shaderstage")
                .unwrap_or(false)
        });
        if !has_shader {
            specs.push(CompanionSpec {
                name: "default_vertex_shader",
                asset_type: "ShaderStage",
                args: serde_json::json!({
                    "kind": "vertex",
                    "sources": {"metal": "default.metal", "hlsl": "default_vert.hlsl"}
                }),
            });
            specs.push(CompanionSpec {
                name: "default_fragment_shader",
                asset_type: "ShaderStage",
                args: serde_json::json!({
                    "kind": "fragment",
                    "sources": {"metal": "default.metal", "hlsl": "default_frag.hlsl"}
                }),
            });
            // The GPU-instanced vertex stage is only needed when the world has
            // an InstancedProp; without it the backend cannot build the
            // instanced main/SSR/SSAO/velocity pipelines and instanced clusters
            // silently never rasterize. Inject it on demand so worlds without
            // instancing don't pay for an extra shader compile.
            let has_instanced = world.iter().any(|v| {
                v.get("type")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_lowercase().replace('_', "") == "instancedprop")
                    .unwrap_or(false)
            });
            if has_instanced {
                specs.push(CompanionSpec {
                    name: "default_instanced_vertex_shader",
                    asset_type: "ShaderStage",
                    args: serde_json::json!({
                        "kind": "vertex_instanced",
                        "sources": {"metal": "default.metal", "hlsl": "default_vert_instanced.hlsl"}
                    }),
                });
            }
        }
        specs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_update_defaults_to_hybrid() {
        assert_eq!(
            GraphicsConfig::default().shadow_update,
            ShadowUpdate::Hybrid
        );
        assert_eq!(ShadowUpdate::default(), ShadowUpdate::Hybrid);
    }

    #[test]
    fn shadow_update_round_trips_via_snake_case_json() {
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_update":"every_frame"}"#).expect("parse");
        assert_eq!(cfg.shadow_update, ShadowUpdate::EveryFrame);
        // Omitting the field falls back to the hybrid default.
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.shadow_update, ShadowUpdate::Hybrid);
    }

    #[test]
    fn vsync_defaults_off_and_round_trips() {
        // Omitted -> uncapped (false).
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert!(!cfg.vsync);
        // Explicit true is honoured.
        let cfg: GraphicsConfig = serde_json::from_str(r#"{"vsync":true}"#).expect("parse");
        assert!(cfg.vsync);
    }

    #[test]
    fn fps_cap_defaults_to_unlimited_and_round_trips() {
        // Omitted -> 0 (uncapped).
        assert_eq!(GraphicsConfig::default().fps_cap, 0);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.fps_cap, 0);
        // Explicit cap is honoured.
        let cfg: GraphicsConfig = serde_json::from_str(r#"{"fps_cap":60}"#).expect("parse");
        assert_eq!(cfg.fps_cap, 60);
    }

    #[test]
    fn shadow_distance_defaults_to_80_and_round_trips() {
        assert_eq!(GraphicsConfig::default().shadow_distance, 80);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_distance":160}"#).expect("parse");
        assert_eq!(cfg.shadow_distance, 160);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.shadow_distance, 80);
    }

    #[test]
    fn shadow_cascades_defaults_to_4_and_round_trips() {
        assert_eq!(GraphicsConfig::default().shadow_cascades, 4);
        let cfg: GraphicsConfig = serde_json::from_str(r#"{"shadow_cascades":2}"#).expect("parse");
        assert_eq!(cfg.shadow_cascades, 2);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.shadow_cascades, 4);
    }

    #[test]
    fn anisotropy_defaults_to_8_and_round_trips() {
        // The default matches the value the backends historically hardcoded.
        assert_eq!(GraphicsConfig::default().anisotropy, 8);
        // An authored value is honoured; omitting the field falls back to 8.
        let cfg: GraphicsConfig = serde_json::from_str(r#"{"anisotropy":16}"#).expect("parse");
        assert_eq!(cfg.anisotropy, 16);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.anisotropy, 8);
    }

    #[test]
    fn shadow_update_round_trips_through_args() {
        let cfg = GraphicsConfig {
            shadow_update: ShadowUpdate::EveryFrame,
            ..Default::default()
        };
        assert_eq!(
            GraphicsConfig::from_args(cfg.to_args()).shadow_update,
            ShadowUpdate::EveryFrame
        );
    }
}
