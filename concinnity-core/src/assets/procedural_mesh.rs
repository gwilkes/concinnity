// src/assets/procedural_mesh.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

/// Geometry built by a named generator at compile time. Use for standard shapes.
///
/// For custom / hand-authored geometry use [Mesh](#mesh) instead.
///
/// **Built-in generators:**
///
/// ```jsonl
/// {"name":"room_mesh","type":"ProceduralMesh","args":{"generator":"room","half_width":16.0,"half_depth":20.0,"ceiling_height":3.5}}
/// {"name":"box_mesh","type":"ProceduralMesh","args":{"generator":"box","half_extents":[0.4,0.4,0.4]}}
/// {"name":"column_mesh","type":"ProceduralMesh","args":{"generator":"cylinder","radius":0.18,"height":3.4,"segments":14}}
/// {"name":"sphere_mesh","type":"ProceduralMesh","args":{"generator":"sphere","radius":0.5,"rings":16,"segments":16}}
/// {"name":"terrain_mesh","type":"ProceduralMesh","args":{"generator":"terrain","half_width":64.0,"half_depth":64.0,"subdivisions":64,"amplitude":4.0}}
/// {"name":"alpine_mesh","type":"ProceduralMesh","args":{"generator":"heightfield","half_width":64.0,"half_depth":64.0,"subdivisions":128,"source":"../concinnity-infra/assets/heightmaps/alpine_512.png","elevation_max":20.0}}
/// {"name":"sky_mesh","type":"ProceduralMesh","args":{"generator":"skybox","size":490.0}}
/// {"name":"plus_mesh","type":"ProceduralMesh","args":{"generator":"extrude","profile":[[-1,-3],[1,-3],[1,-1],[3,-1],[3,1],[1,1],[1,3],[-1,3],[-1,1],[-3,1],[-3,-1],[-1,-1]],"height":0.5,"corner_radius":0.2}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProceduralMesh {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Built-in generator name (required), e.g. `room`, `box`, `cylinder`,
    /// `sphere`, `terrain`, `heightfield`, `skybox`, or `extrude`.
    pub generator: String,

    // Room / box / plane dimensions
    /// Half-width along X (room / box / plane / terrain), in world units.
    pub half_width: f32,
    /// Half-depth along Z (room / box / plane / terrain), in world units.
    pub half_depth: f32,
    /// Ceiling height for the `room` generator, in world units.
    pub ceiling_height: f32,

    // Box
    /// Half-extents `[x, y, z]` for the `box` generator.
    pub half_extents: Option<[f32; 3]>,

    // Cylinder / sphere
    /// Radius for the `cylinder` and `sphere` generators.
    pub radius: Option<f32>,
    /// Height for the `cylinder` and `extrude` generators.
    pub height: Option<f32>,
    /// Number of radial segments around the `cylinder` and `sphere` generators.
    pub segments: Option<u32>,

    // Sphere
    /// Number of horizontal rings on the `sphere` generator.
    pub rings: Option<u32>,

    // Terrain
    /// Grid subdivisions for the `terrain` and `heightfield` generators. Higher
    /// is more detailed.
    pub subdivisions: Option<u32>,
    /// Maximum height variation for the `terrain` generator, in world units.
    pub amplitude: Option<f32>,

    // Heightfield (grayscale image → height grid)
    /// Path to a grayscale heightmap image for the `heightfield` generator.
    pub source: Option<String>,
    /// Height mapped to black pixels in the `heightfield` source, in world units.
    pub elevation_min: Option<f32>,
    /// Height mapped to white pixels in the `heightfield` source, in world units.
    pub elevation_max: Option<f32>,

    // Skybox
    /// Half-extent on all axes for the `skybox` generator, in world units.
    /// Keep it below the camera's `far` plane so the sky is not clipped.
    pub size: Option<f32>,

    // Extrude
    /// 2D outline `[[x, z], ...]` extruded by the `extrude` generator.
    pub profile: Option<Vec<[f32; 2]>>,
    /// Corner-rounding radius for the `extrude` generator. 0 keeps sharp corners.
    pub corner_radius: Option<f32>,
    /// Number of segments used to round each corner in the `extrude` generator.
    pub corner_segments: Option<u32>,

    /// Number of level-of-detail versions to generate, including the original.
    /// `1` (the default) generates none; values are clamped to `[1, 8]`.
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version; length
    /// should be `lod_levels - 1`. Empty lets the build choose defaults.
    pub lod_distances: Vec<f32>,

    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for ProceduralMesh {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            generator: String::new(),
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            half_extents: None,
            radius: None,
            height: None,
            segments: None,
            rings: None,
            subdivisions: None,
            amplitude: None,
            source: None,
            elevation_min: None,
            elevation_max: None,
            size: None,
            profile: None,
            corner_radius: None,
            corner_segments: None,
            lod_levels: 1,
            lod_distances: Vec::new(),
            locator: None,
        }
    }
}

impl Component for ProceduralMesh {
    const NAME: &'static str = "ProceduralMesh";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }
    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

// Blob indices of heightfield-generator ProceduralMeshes. GraphicsSystem's
// init release sweep must spare these blobs: PhysicsSystem inits afterwards and
// reads the baked heightfield collider grid from the payload, mirroring the
// AudioClip / SdfVolume precedent of holding a blob resident for a later system.
pub fn heightfield_blob_indices(
    ctx: &crate::ecs::PipelineContext,
) -> std::collections::HashSet<u32> {
    ctx.query::<ProceduralMesh>()
        .filter(|m| m.generator == "heightfield")
        .filter_map(|m| m.locator.as_ref().map(|l| l.blob_index))
        .collect()
}
