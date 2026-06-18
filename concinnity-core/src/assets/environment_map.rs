// src/assets/environment_map.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

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
impl Default for EnvironmentMap {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            source: String::new(),
            generator: String::new(),
            prefilter_face_size: 512,
            irradiance_face_size: 8,
            prefilter_samples: 1024,
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
