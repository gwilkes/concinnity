// src/assets/cubemap_texture.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

/// A six-face HDR cubemap baked from an equirectangular Radiance HDR source.
///
/// The build resamples the source into six square HDR faces of `face_size`
/// pixels each, used as an environment / image-based-lighting source.
///
/// ```jsonl
/// {"name":"env_studio","type":"CubemapTexture","args":{"source":"assets/hdri/studio.hdr","face_size":512}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CubemapTexture {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Path to the source equirectangular HDR (`.hdr`) file, relative to the
    /// project root.
    pub source: String,
    /// Edge length of each cube face in pixels. Must be a power of two.
    pub face_size: u32,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for CubemapTexture {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            source: String::new(),
            face_size: 256,
            locator: None,
        }
    }
}

impl Component for CubemapTexture {
    const NAME: &'static str = "CubemapTexture";

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

impl crate::build::SourceBacked for CubemapTexture {
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}
