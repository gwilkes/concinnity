// src/assets/mesh.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

/// A single vertex as supplied in raw Mesh args.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VertexData {
    /// Vertex position `[x, y, z]` in model space.
    pub pos: [f32; 3],
    /// Vertex colour `[r, g, b]` in [0, 1]. Use `[0.75, 0.74, 0.72]` for a
    /// neutral surface that takes the material albedo.
    pub color: [f32; 3],
    /// Texture coordinates in [0, 1] space.  Defaults to [0, 0] when omitted.
    #[serde(default)]
    pub uv: [f32; 2],
}

/// Raw geometry. Supply `vertices` and `indices` directly, or import them from
/// a binary glTF file with `source` + `primitive_index`.
///
/// Use when you want full control over shape: custom furniture,
/// architectural details, signage, or any form a generator cannot
/// produce. For standard shapes use [ProceduralMesh](#proceduralmesh).
///
/// Normals and tangents are computed automatically at build time.
/// **Do not supply normals or tangents.**
///
/// **Vertex color:** use `[0.75, 0.74, 0.72]` for a neutral surface that takes
/// the material albedo, or `[1, 1, 1]` to pass through unmodified.
///
/// **Winding:** triangles must be counter-clockwise when viewed from the front.
/// Reversed winding = invisible face.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Mesh {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Optional path to a `.glb` file. When set, the build imports
    /// `vertices` / `indices` from it; inline geometry leaves this empty.
    pub source: String,
    /// Which primitive (counted across all meshes in the file) to import from
    /// `source`. Ignored when `source` is empty.
    pub primitive_index: u32,
    /// Pick a single chunk of an oversized imported primitive. `None` (the
    /// default) imports the whole primitive, which is fine whenever its vertex
    /// count fits in 16-bit indices; larger primitives are split into chunks on
    /// import, one Mesh per chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_index: Option<u32>,
    /// Vertex list.  Each vertex: `{"pos":[x,y,z], "color":[r,g,b], "uv":[u,v]}`.
    pub vertices: Vec<VertexData>,
    /// Triangle index list (16-bit values).
    pub indices: Vec<u16>,
    /// Number of level-of-detail versions to generate, including the original.
    /// `1` (the default) generates none; values are clamped to `[1, 8]`.
    #[serde(default = "default_lod_levels")]
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version. Length
    /// should be `lod_levels - 1`; empty lets the build derive a default
    /// sequence. The version for index `i` is used at camera distance ≥
    /// `lod_distances[i]`.
    pub lod_distances: Vec<f32>,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

fn default_lod_levels() -> u32 {
    1
}

impl Default for Mesh {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            source: String::new(),
            primitive_index: 0,
            chunk_index: None,
            vertices: Vec::new(),
            indices: Vec::new(),
            lod_levels: 1,
            lod_distances: Vec::new(),
            locator: None,
        }
    }
}

impl Component for Mesh {
    const NAME: &'static str = "Mesh";
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

impl crate::build::SourceBacked for Mesh {
    // A glTF-sourced Mesh needs its `.glb` fetched before the build's desugar
    // pass can expand it; an inline-authored mesh has no source.
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}
