// src/assets/point_light.rs

use crate::ecs::{AssetOrigin, Component};

/// A spherical point light with quadratic distance attenuation.
///
/// Up to 8 point lights may be declared; extras beyond 8 are silently ignored.
///
/// ```jsonl
/// {"name":"lamp","type":"PointLight","args":{"position":[2.0,2.5,-3.0],"color":[1.0,0.8,0.5],"intensity":8.0,"range":6.0}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PointLight {
    /// World-space position of the light source.
    pub position: [f32; 3],
    /// Linear-space RGB colour of the light.
    pub color: [f32; 3],
    /// Intensity multiplier applied to the colour.
    pub intensity: f32,
    /// Maximum reach in world units; attenuation is zero at this distance.
    pub range: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: [0.0, 2.5, 0.0],
            color: [1.0, 1.0, 1.0],
            intensity: 8.0,
            range: 6.0,
        }
    }
}

impl Component for PointLight {
    const NAME: &'static str = "PointLight";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        args.intensity = args.intensity.max(0.0);
        args.range = args.range.max(0.0);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}
