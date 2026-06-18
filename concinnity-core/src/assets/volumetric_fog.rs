// src/assets/volumetric_fog.rs

use crate::ecs::{AssetOrigin, Component};

/// Environmental volumetric fog: a single lit medium that wraps the scene,
/// thicker near the ground and thinning with height, with extra glow around the
/// sun.
///
/// Only one `VolumetricFog` is honoured: the first declared instance wins;
/// later instances are silently dropped. With none declared, there is no fog.
///
/// ```jsonl
/// {"name":"fog","type":"VolumetricFog","args":{"density":0.08,"color":[0.75,0.82,0.95],"height_falloff":0.18,"max_distance":160.0,"phase_g":0.5}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct VolumetricFog {
    /// Master toggle. `false` disables the fog even when this asset is present.
    pub enabled: bool,
    /// Linear-space RGB tint of the fog: the colour the camera sees in the far
    /// distance.
    pub color: [f32; 3],
    /// Base thickness of the fog at `height_reference` (per world unit). Higher
    /// is thicker. Floored at 0.
    pub density: f32,
    /// How quickly the fog thins with height above `height_reference`. 0 keeps
    /// it uniform; larger values pin it to the ground.
    pub height_falloff: f32,
    /// World-space Y at which the fog reaches full `density`. It thickens below
    /// this height and thins above it.
    pub height_reference: f32,
    /// Maximum distance the fog covers from the camera, in world units. Past
    /// this, distant geometry stays clear.
    pub max_distance: f32,
    /// Sun-glow anisotropy in `(-1, 1)`. Positive values concentrate brightness
    /// around the sun (haloes), negative values scatter away from it, 0 is
    /// uniform.
    pub phase_g: f32,
    /// Constant ambient brightness so the fog keeps some colour in shaded areas.
    pub ambient: f32,
}

impl Default for VolumetricFog {
    fn default() -> Self {
        Self {
            enabled: true,
            color: [0.7, 0.78, 0.85],
            density: 0.05,
            height_falloff: 0.2,
            height_reference: 0.0,
            max_distance: 200.0,
            phase_g: 0.4,
            ambient: 0.15,
        }
    }
}

impl Component for VolumetricFog {
    const NAME: &'static str = "VolumetricFog";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        // Density / falloff / ambient floor at 0; max_distance must stay
        // positive so the gfx-side resolver does not divide by zero when
        // computing the per-step length.
        args.density = args.density.max(0.0);
        args.height_falloff = args.height_falloff.max(0.0);
        args.ambient = args.ambient.max(0.0);
        if args.max_distance <= 0.0 || !args.max_distance.is_finite() {
            args.max_distance = 1.0;
        }
        // Henyey-Greenstein blows up at |g| = 1; clamp inside the open
        // interval so the closed-form `(1 - g²)` factor stays positive.
        args.phase_g = args.phase_g.clamp(-0.95, 0.95);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_with_defaults() {
        let f: VolumetricFog = serde_json::from_str("{}").unwrap();
        assert!(f.enabled);
        assert_eq!(f.color, [0.7, 0.78, 0.85]);
        assert!((f.density - 0.05).abs() < 1e-6);
        assert!((f.max_distance - 200.0).abs() < 1e-6);
    }

    #[test]
    fn deserialises_with_explicit_fields() {
        let json = r#"{
            "enabled":false,"density":0.12,"color":[0.5,0.6,0.7],
            "height_falloff":0.3,"height_reference":1.5,
            "max_distance":80.0,"phase_g":0.7,"ambient":0.25
        }"#;
        let f: VolumetricFog = serde_json::from_str(json).unwrap();
        assert!(!f.enabled);
        assert_eq!(f.color, [0.5, 0.6, 0.7]);
        assert!((f.phase_g - 0.7).abs() < 1e-6);
    }

    #[test]
    fn from_args_clamps_invalid_inputs() {
        let a = VolumetricFog {
            density: -1.0,
            height_falloff: -0.4,
            ambient: -2.0,
            max_distance: -1.0,
            phase_g: 1.4,
            ..Default::default()
        };
        let n = VolumetricFog::from_args(a);
        assert_eq!(n.density, 0.0);
        assert_eq!(n.height_falloff, 0.0);
        assert_eq!(n.ambient, 0.0);
        assert!(n.max_distance > 0.0);
        assert!(n.phase_g <= 0.95 && n.phase_g > 0.0);
    }

    #[test]
    fn from_args_passes_through_valid_inputs() {
        let a = VolumetricFog {
            density: 0.08,
            phase_g: -0.3,
            ..Default::default()
        };
        let n = VolumetricFog::from_args(a);
        assert!((n.density - 0.08).abs() < 1e-6);
        assert!((n.phase_g - (-0.3)).abs() < 1e-6);
    }
}
