// src/assets/directional_light.rs

use crate::ecs::{AssetOrigin, Component};

/// An infinitely distant directional light (sun, moon, or sky fill).
///
/// Up to 4 directional lights may be declared; extras beyond 4 are silently ignored.
/// When no directional light is present, a built-in warm sun is used as a fallback.
///
/// ```jsonl
/// {"name":"sun","type":"DirectionalLight","args":{"direction":[-0.3,0.85,0.4],"color":[1.0,0.95,0.8],"intensity":1.0}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DirectionalLight {
    /// Direction pointing toward the light source. Does not need to be
    /// normalised.
    pub direction: [f32; 3],
    /// Linear-space RGB colour of the light.
    pub color: [f32; 3],
    /// Intensity multiplier applied to the colour.
    pub intensity: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: [-0.3, 0.85, 0.4],
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
        }
    }
}

impl Component for DirectionalLight {
    const NAME: &'static str = "DirectionalLight";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        args.intensity = args.intensity.max(0.0);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}
