// src/assets/glass_panel.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

/// A flat translucent panel of coloured glass. A fixed-orientation rectangular
/// quad that refracts and tints the scene behind it and brightens the
/// grazing-angle rim with a Fresnel highlight.
///
/// Unlike [WaterSurface](#watersurface) it has no animation, no surface
/// displacement, and no depth-based colour. It's a simple building block for
/// translucent surfaces such as windows, ice, holograms, or force fields.
///
/// The panel is positioned by `centre`, oriented by `normal` (the facing
/// direction), and sized by `half_size` (half-width along the panel's tangent,
/// half-height along its bitangent).
///
/// ```jsonl
/// {"name":"window","type":"GlassPanel","args":{
///   "centre":[0.0,2.0,-3.0],
///   "normal":[0.0,0.0,1.0],
///   "half_size":[2.0,1.5],
///   "tint":[0.6,0.85,0.9],
///   "opacity":0.45,
///   "refraction_strength":0.04,
///   "fresnel_power":4.0
/// }}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GlassPanel {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// World-space position of the panel's centre.
    pub centre: [f32; 3],
    /// Facing direction of the panel. Normalised on load; defaults to +Z when
    /// degenerate.
    pub normal: [f32; 3],
    /// Half-width and half-height of the panel, in world units.
    pub half_size: [f32; 2],
    /// Linear-space RGB colour the glass tints the scene behind it.
    pub tint: [f32; 3],
    /// How opaque the glass is, in [0, 1]. 0 = clear, 1 = fully opaque tint.
    pub opacity: f32,
    /// How strongly the glass bends the view of what's behind it. 0 = no
    /// refraction.
    pub refraction_strength: f32,
    /// Sharpness of the grazing-angle rim highlight. Higher values confine the
    /// brightening to steeper viewing angles.
    pub fresnel_power: f32,
    /// When false the panel is skipped each frame.
    pub visible: bool,
}

impl Default for GlassPanel {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            centre: [0.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            half_size: [1.0, 1.0],
            tint: [0.7, 0.85, 0.95],
            opacity: 0.5,
            refraction_strength: 0.04,
            fresnel_power: 4.0,
            visible: true,
        }
    }
}

impl GlassPanel {
    /// Unit-length facing direction, falling back to `+Z` when the authored
    /// `normal` is degenerate. The build-time quad generator and the runtime
    /// shader both rely on a usable normal.
    pub fn unit_normal(&self) -> [f32; 3] {
        let n = self.normal;
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len < 1e-6 {
            [0.0, 0.0, 1.0]
        } else {
            [n[0] / len, n[1] / len, n[2] / len]
        }
    }
}

impl Component for GlassPanel {
    const NAME: &'static str = "GlassPanel";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        args.normal = args.unit_normal();
        args.half_size[0] = args.half_size[0].max(1e-3);
        args.half_size[1] = args.half_size[1].max(1e-3);
        args.opacity = args.opacity.clamp(0.0, 1.0);
        args.refraction_strength = args.refraction_strength.max(0.0);
        args.fresnel_power = args.fresnel_power.max(0.0);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::Component;

    #[test]
    fn from_args_normalizes_normal() {
        let g = GlassPanel::from_args(GlassPanel {
            normal: [0.0, 0.0, 4.0],
            ..Default::default()
        });
        let len = (g.normal[0].powi(2) + g.normal[1].powi(2) + g.normal[2].powi(2)).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
        assert!((g.normal[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn from_args_falls_back_on_degenerate_normal() {
        let g = GlassPanel::from_args(GlassPanel {
            normal: [0.0, 0.0, 0.0],
            ..Default::default()
        });
        assert_eq!(g.normal, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn from_args_clamps_ranges() {
        let g = GlassPanel::from_args(GlassPanel {
            half_size: [-2.0, 0.0],
            opacity: 1.5,
            refraction_strength: -0.1,
            fresnel_power: -3.0,
            ..Default::default()
        });
        assert!(g.half_size[0] > 0.0 && g.half_size[1] > 0.0);
        assert_eq!(g.opacity, 1.0);
        assert_eq!(g.refraction_strength, 0.0);
        assert_eq!(g.fresnel_power, 0.0);
    }
}
