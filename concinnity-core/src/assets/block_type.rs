// src/assets/block_type.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

/// Describes one entry in a [VoxelChunk](#voxelchunk) palette.
///
/// Each BlockType represents either a solid block (with UVs into the chunk's atlas texture)
/// or an empty/air marker.
///
/// Per-face fields fall back to `uv_min`/`uv_max` when omitted. Set `solid=false`
/// on the air/empty palette entry; faces between solid blocks and air blocks are
/// the only faces the chunk emits.
///
/// ```jsonl
/// {"name":"air","type":"BlockType","args":{"solid":false}}
/// {"name":"stone","type":"BlockType","args":{"uv_min":[0,0],"uv_max":[0.25,0.25]}}
/// {"name":"grass","type":"BlockType","args":{"uv_side":[0.25,0,0.5,0.25],"uv_top":[0.5,0,0.75,0.25],"uv_bottom":[0,0.25,0.25,0.5]}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct BlockType {
    /// Asset identity; injected via `inject_name`. Not part of `args`. Lets the
    /// runtime resolve a `VoxelWorld` palette (a list of `BlockType` ids) back
    /// to the block data the chunk generator needs.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// When false the block is treated as air -- no faces are emitted for it
    /// and it does not occlude neighboring faces.
    pub solid: bool,
    /// Default atlas UV at the (0,0) corner of each face.
    pub uv_min: [f32; 2],
    /// Default atlas UV at the (1,1) corner of each face.
    pub uv_max: [f32; 2],
    /// Optional per-face override for the +Y face: `[u_min, v_min, u_max, v_max]`.
    pub uv_top: Option<[f32; 4]>,
    /// Optional per-face override for the -Y face.
    pub uv_bottom: Option<[f32; 4]>,
    /// Optional per-face override applied to all four side faces (±X, ±Z).
    pub uv_side: Option<[f32; 4]>,
}

impl Default for BlockType {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            solid: true,
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
            uv_top: None,
            uv_bottom: None,
            uv_side: None,
        }
    }
}

impl Component for BlockType {
    const NAME: &'static str = "BlockType";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}
