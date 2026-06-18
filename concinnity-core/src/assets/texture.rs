// src/assets/texture.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

/// A 2D texture image.
///
/// Use the `generator` field for built-in patterns or supply a `source` file path.
///
/// **Built-in generators:**
///
/// **Choosing a room texture**: for neutral indoor spaces prefer `plaster` (cream-white) or `concrete` (grey). `brick` is reddish-orange, only use it when you explicitly want that look. `stone` (dark grey-blue) suits dungeons or medieval rooms.
///
/// ```jsonl
/// {"name":"tex_brick","type":"Texture","args":{"generator":"brick","resolution":512}}
/// {"name":"tex_grass","type":"Texture","args":{"generator":"grass","resolution":256}}
/// {"name":"tex_checker","type":"Texture","args":{"generator":"checker","resolution":128}}
/// {"name":"tex_stone","type":"Texture","args":{"generator":"stone","resolution":512}}
/// {"name":"tex_plaster","type":"Texture","args":{"generator":"plaster","resolution":512}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Texture {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Procedural generator name. Empty or omitted means use `source` instead.
    pub generator: String,
    /// Path to the source image, relative to the project root.
    /// Used only when `generator` is empty. A `.glb` path is allowed, use
    /// `image_index` to pick which embedded image to use.
    pub source: String,
    /// When `source` points to a `.glb` file, which embedded image to import.
    /// Ignored for regular image files.
    pub image_index: u32,
    /// Resolution hint for procedural generators (width = height). Defaults to
    /// 512. Ignored for file-backed textures.
    pub resolution: u32,
    /// Optional ceiling on the longest edge of a file-backed image, in pixels.
    /// `0` (the default) keeps the source resolution. When set and the source is
    /// larger, the image is box-filtered down so its longest edge is at most this
    /// value. Useful to keep very large source maps (4K+) from bloating the
    /// compiled scene, which stores uncompressed pixels.
    pub max_size: u32,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for Texture {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            generator: String::new(),
            source: String::new(),
            image_index: 0,
            resolution: 512,
            max_size: 0,
            locator: None,
        }
    }
}

impl Component for Texture {
    const NAME: &'static str = "Texture";

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

impl crate::build::SourceBacked for Texture {
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        // Procedural textures (with a non-empty `generator`) have no source file.
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
