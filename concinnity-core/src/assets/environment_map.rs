// src/assets/environment_map.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, CompanionSpec, Component, PayloadLocator};

/// A baked lighting environment built from a Radiance HDR equirectangular
/// source (or a built-in generator). It provides the scene's ambient
/// image-based lighting (soft diffuse fill plus glossy reflections that follow
/// surface roughness) and the on-screen sky.
///
/// **`prefilter_face_size` note:** this controls both the reflection detail and
/// the on-screen sky sharpness. 512 is the default balance: 256 visibly
/// pixelates a 4K-source HDR sky, 1024 sharpens it further at 4× the size.
///
/// **Built-in generators:** `sky` produces a procedural blue sky with a soft
/// sun, useful when no HDR file is available.
///
/// The sky mesh that displays the map (a skybox
/// [ProceduralMesh](#proceduralmesh) plus its [Material](#material) and
/// [Prop](#prop)) is injected at build time when the world declares no skybox
/// mesh of its own. Declare an [EngineDefaults](#enginedefaults) with
/// `"sky": false` to use the map for image-based lighting only, with the
/// background left to `clear_color` or your own geometry.
///
/// ```jsonl
/// {"name":"env_studio","type":"EnvironmentMap","args":{"source":"assets/hdri/studio.hdr"}}
/// {"name":"env_outdoor","type":"EnvironmentMap","args":{"source":"assets/hdri/sky.hdr","prefilter_face_size":512}}
/// {"name":"env_proc","type":"EnvironmentMap","args":{"generator":"sky"}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EnvironmentMap {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Path to the source equirectangular HDR (`.hdr`) file, relative to the
    /// project root. Mutually exclusive with `generator`.
    pub source: String,
    /// Built-in source name (e.g. "sky"). Mutually exclusive with `source`.
    pub generator: String,
    /// Face size of the reflection/sky cubemap, in pixels. Higher is sharper
    /// but larger.
    pub prefilter_face_size: u32,
    /// Face size of the diffuse ambient cubemap, in pixels.
    pub irradiance_face_size: u32,
    /// Number of samples used to filter each reflection texel. Higher reduces
    /// noise at the cost of build time.
    pub prefilter_samples: u32,
    /// Upper bound on how bright a single source texel may count while building
    /// the glossy reflection mips. A clear-sky HDR holds a few sun or sky
    /// texels thousands of times brighter than their surroundings; left
    /// unbounded they survive into the small (coarse) reflection mips as lone
    /// hot texels and smear across glossy floors as hard bright squares. This
    /// caps each sampled texel so that energy spreads smoothly across the
    /// reflection instead. It affects reflections only, never the on-screen
    /// sky. Set to `0` to disable (no cap); lower values clamp harder.
    pub prefilter_clamp: f32,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

// The face-size / sample-count defaults below are the single source of truth:
// the build pipeline deserialises args through this struct, so a field absent
// from a JSONL entry inherits these values rather than a constant duplicated in
// the build crate. They are chosen for ~32 MB payloads and a few seconds of
// build cost on the dev box. `prefilter_face_size` does double duty: mips 1..N
// feed the GGX specular IBL lookup (fine at low resolution) while mip 0 is
// sampled directly by the skybox sentinel branch in the fragment shaders, so it
// has to be large enough that the displayed sky doesn't look blocky. 512 is the
// balance point; 256 visibly pixelates a 4K HDR sky, 1024 quadruples the payload
// for sharpness only the skybox (not the IBL math) actually uses.
//
// `prefilter_clamp` defaults to a moderate cap rather than off: an unbounded
// clear-sky HDR aliases its sun and bright sky into hard squares on glossy
// floors (the coarse reflection mips hold only a handful of texels, so one hot
// texel paints a whole region). The cap spreads that energy without touching
// the on-screen sky, and a uniform sky below the cap is unchanged.
impl Default for EnvironmentMap {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            source: String::new(),
            generator: String::new(),
            prefilter_face_size: 512,
            irradiance_face_size: 8,
            prefilter_samples: 1024,
            prefilter_clamp: 12.0,
            locator: None,
        }
    }
}

impl Component for EnvironmentMap {
    const NAME: &'static str = "EnvironmentMap";

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

    fn companions(_args: &serde_json::Value, _world: &[serde_json::Value]) -> Vec<CompanionSpec> {
        vec![CompanionSpec {
            name: "GraphicsConfig",
            asset_type: "GraphicsConfig",
            args: serde_json::json!({}),
        }]
    }
}

impl crate::build::SourceBacked for EnvironmentMap {
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        // Procedural generators have no source file.
        if args
            .get("generator")
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false)
        {
            return None;
        }
        args.get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}
