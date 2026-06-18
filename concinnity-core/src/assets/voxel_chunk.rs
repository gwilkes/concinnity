// src/assets/voxel_chunk.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

/// A voxel grid that compiles into a single mesh.
///
/// A dense grid of blocks compiled into a single mesh at build time. Use one
/// chunk per region of a voxel/Minecraft-style world; reference it from a
/// [Prop](#prop)'s `mesh` field. Hidden faces between two solid blocks are
/// dropped, so a fully filled chunk contributes zero triangles to its interior.
///
/// The palette must contain at least one entry whose [BlockType](#blocktype) has
/// `solid: false` (typically named `air`); cells whose palette entry is
/// non-solid emit no faces. Faces are only emitted between a solid block and
/// either an empty neighbour or the outside of the chunk.
///
/// ```jsonl
/// {"name":"air","type":"BlockType","args":{"solid":false}}
/// {"name":"stone","type":"BlockType","args":{"uv_min":[0,0],"uv_max":[1,1]}}
/// {"name":"my_chunk","type":"VoxelChunk","args":{
///   "palette":["air","stone"],
///   "dim":[2,1,1],
///   "blocks":[1,1]
/// }}
/// {"name":"chunk_prop","type":"Prop","args":{"mesh":"my_chunk","material":"mat_stone"}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct VoxelChunk {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// [BlockType](#blocktype) asset names. `blocks[i]` is an index into this list.
    pub palette: Vec<AssetId>,
    /// Chunk dimensions `[dx, dy, dz]` in blocks.
    pub dim: [u32; 3],
    /// World units per block edge.
    pub block_size: f32,
    /// Flat block array, length `dx*dy*dz`. Index = `x + y*dx + z*dx*dy`.
    pub blocks: Vec<u32>,
    /// Number of level-of-detail versions to generate, including the original.
    /// `1` (the default) generates none.
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version; empty
    /// lets the build choose defaults.
    #[serde(default)]
    pub lod_distances: Vec<f32>,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for VoxelChunk {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            palette: Vec::new(),
            dim: [0, 0, 0],
            block_size: 1.0,
            blocks: Vec::new(),
            lod_levels: 1,
            lod_distances: Vec::new(),
            locator: None,
        }
    }
}

impl Component for VoxelChunk {
    const NAME: &'static str = "VoxelChunk";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        args.block_size = args.block_size.max(0.0);
        if args.lod_levels == 0 {
            args.lod_levels = 1;
        }
        args.lod_levels = args.lod_levels.min(8);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }
    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::check::cross_reference::CrossReferenced for VoxelChunk {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let mut refs = Vec::new();

        let palette = args
            .get("palette")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]);
        for (i, entry) in palette.iter().enumerate() {
            let bt_name = entry.as_str().unwrap_or("");
            if bt_name.is_empty() {
                refs.push(CrossRef::Issue(format!(
                    "VoxelChunk '{}': palette[{}] is not a valid BlockType name",
                    name, i
                )));
            } else {
                refs.push(CrossRef::Resolve {
                    kind: RefKind::BlockType,
                    target: bt_name.to_string(),
                    error: format!(
                        "VoxelChunk '{}': palette[{}] BlockType '{}' not found, add a BlockType asset with that name",
                        name, i, bt_name
                    ),
                });
            }
        }

        refs
    }
}
