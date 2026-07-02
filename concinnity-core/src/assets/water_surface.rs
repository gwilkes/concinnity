// src/assets/water_surface.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

/// Maximum number of waves per water surface.
pub const MAX_WATER_WAVES: usize = 4;

/// One wave in a water surface's motion. A surface sums up to
/// [`MAX_WATER_WAVES`] of these to displace its flat grid. Each wave travels
/// horizontally along `direction`, rising and falling with `amplitude` peak
/// height, `wavelength` distance between crests, and `speed` metres per second.
/// `steepness` in [0, 1] pinches the crests and broadens the troughs (choppier
/// water).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct WaterWave {
    /// Peak height of the wave, in world units.
    pub amplitude: f32,
    /// Distance between successive crests, in world units.
    pub wavelength: f32,
    /// Horizontal travel speed, in metres per second.
    pub speed: f32,
    /// Horizontal travel direction `[x, z]`.
    pub direction: [f32; 2],
    /// Crest sharpness in [0, 1]. 0 is a smooth sine; higher pinches crests and
    /// broadens troughs.
    pub steepness: f32,
}

impl Default for WaterWave {
    fn default() -> Self {
        Self {
            amplitude: 0.15,
            wavelength: 4.0,
            speed: 1.0,
            direction: [1.0, 0.0],
            steepness: 0.4,
        }
    }
}

/// A translucent animated water surface.
///
/// A flat, subdivided horizontal surface whose vertices ripple with summed
/// waves. It refracts and reflects the scene, blends from a shallow to a deep
/// colour with depth, and adds shoreline foam.
///
/// The surface is positioned by `centre` and sized by `extent` (XZ
/// half-widths). The mesh itself is flat; all height variation comes from the
/// animated waves.
///
/// ```jsonl
/// {"name":"pond","type":"WaterSurface","args":{
///   "centre":[0.0,0.4,0.0],
///   "extent":[12.0,8.0],
///   "subdivisions":96,
///   "waves":[
///     {"amplitude":0.10,"wavelength":3.0,"speed":0.7,"direction":[1.0,0.0],"steepness":0.4},
///     {"amplitude":0.05,"wavelength":1.5,"speed":1.1,"direction":[-0.4,0.8],"steepness":0.3}
///   ]
/// }}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct WaterSurface {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// World-space position of the surface's centre.
    pub centre: [f32; 3],
    /// Half-width and half-depth of the surface `[x, z]`, in world units.
    pub extent: [f32; 2],
    /// Grid subdivisions across the surface. Higher gives smoother waves.
    /// Clamped to [8, 255].
    pub subdivisions: u32,
    /// The waves summed to animate the surface (up to 4). Defaults to a single
    /// gentle wave.
    pub waves: Vec<WaterWave>,
    /// Linear-space RGB colour of deep water.
    pub deep_colour: [f32; 3],
    /// Linear-space RGB colour of shallow water near the shore.
    pub shallow_colour: [f32; 3],
    /// Depth over which the colour blends from shallow to deep, in metres.
    pub depth_falloff_metres: f32,
    /// Width of the shoreline foam band, in metres.
    pub foam_width_metres: f32,
    /// Strength of the shoreline foam, in [0, 1].
    pub foam_intensity: f32,
    /// Sharpness of the grazing-angle reflection. Higher confines reflections to
    /// steeper viewing angles.
    pub fresnel_power: f32,
    /// Surface roughness in [0, 1]. Higher gives blurrier reflections.
    pub roughness: f32,
    /// How strongly the surface bends the view of what's beneath it.
    pub refraction_strength: f32,
    /// When false the surface is skipped each frame.
    pub visible: bool,
}

impl Default for WaterSurface {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            centre: [0.0, 0.0, 0.0],
            extent: [10.0, 10.0],
            subdivisions: 64,
            waves: vec![WaterWave::default()],
            deep_colour: [0.02, 0.05, 0.15],
            shallow_colour: [0.20, 0.50, 0.55],
            depth_falloff_metres: 4.0,
            foam_width_metres: 0.30,
            foam_intensity: 0.8,
            fresnel_power: 5.0,
            roughness: 0.05,
            refraction_strength: 0.15,
            visible: true,
        }
    }
}

impl Component for WaterSurface {
    const NAME: &'static str = "WaterSurface";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        args.subdivisions = args.subdivisions.clamp(8, 255);
        if args.waves.len() > MAX_WATER_WAVES {
            args.waves.truncate(MAX_WATER_WAVES);
        }
        if args.waves.is_empty() {
            args.waves.push(WaterWave::default());
        }
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }

    fn companions(_args: &serde_json::Value, _world: &[serde_json::Value]) -> Vec<CompanionSpec> {
        vec![CompanionSpec {
            name: "GraphicsConfig",
            asset_type: "GraphicsConfig",
            args: serde_json::json!({}),
        }]
    }
}
