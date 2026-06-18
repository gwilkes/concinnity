// src/assets/voxel_world.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};

/// An infinite, procedurally generated voxel world.
///
/// Where a [VoxelChunk](#voxelchunk) is one authored chunk compiled to a fixed
/// mesh at build time, a `VoxelWorld` describes an *unbounded* world: chunks are
/// generated on demand from `seed` as the camera moves and streamed in and out
/// around it. The grid is infinite on X/Z and a single chunk tall on Y.
/// Declaring one opts the world into chunk streaming; with no `VoxelWorld`
/// present nothing changes.
///
/// The `palette` lists [BlockType](#blocktype) assets; the generator uses index
/// 0 as air, index 1 as the surface block, and index 2 (when present) as the
/// subsurface block. `material` supplies the textures and lighting shared by
/// every chunk.
///
/// ```jsonl
/// {"name":"air","type":"BlockType","args":{"solid":false}}
/// {"name":"grass","type":"BlockType","args":{"uv_min":[0,0],"uv_max":[1,1]}}
/// {"name":"stone","type":"BlockType","args":{"uv_min":[0,0],"uv_max":[1,1]}}
/// {"name":"overworld","type":"VoxelWorld","args":{
///   "seed":42,"view_radius":6,"palette":["air","grass","stone"],"material":"mat_ground"
/// }}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct VoxelWorld {
    /// Deterministic terrain seed. The same seed always generates the same
    /// world, so a chunk regenerates identically each time it streams back in.
    pub seed: u64,
    /// Blocks per chunk `[dx, dy, dz]`. Y is the world's fixed vertical extent.
    pub chunk_blocks: [u32; 3],
    /// World units per block edge.
    pub block_size: f32,
    /// Chunk radius streamed around the camera at full voxel detail.
    pub view_radius: u32,
    /// Outer chunk radius streamed as cheap coarse impostors. Chunks farther
    /// than `view_radius` but within `impostor_radius` render as a low-detail
    /// surface mesh instead of full voxel geometry. `0` (the default) or any
    /// value `<= view_radius` disables impostors.
    pub impostor_radius: u32,
    /// Coarse-grid step (in blocks) for distant-chunk impostors: the surface is
    /// sampled every `impostor_step` blocks. Higher = cheaper and coarser.
    pub impostor_step: u32,
    /// Maximum number of chunks generated and loaded per frame.
    pub load_budget: u32,
    /// [BlockType](#blocktype) asset names. Index 0 is air; 1 is the surface
    /// block; 2, when present, is the subsurface block.
    pub palette: Vec<AssetId>,
    /// [Material](#material) shared by every chunk: textures and lighting.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub material: Option<AssetId>,
}

impl Default for VoxelWorld {
    fn default() -> Self {
        Self {
            seed: 0,
            chunk_blocks: [16, 24, 16],
            block_size: 1.0,
            view_radius: 5,
            impostor_radius: 0,
            impostor_step: 4,
            load_budget: 3,
            palette: Vec::new(),
            material: None,
        }
    }
}

// These accessors feed the Metal chunk-streaming path for now
// (Vulkan / DirectX catch-up is a follow-up).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
impl VoxelWorld {
    /// Blocks per chunk, each axis floored at 1 so a chunk is never degenerate.
    pub fn chunk_blocks(&self) -> [u32; 3] {
        [
            self.chunk_blocks[0].max(1),
            self.chunk_blocks[1].max(1),
            self.chunk_blocks[2].max(1),
        ]
    }

    /// World units per block edge, floored at a small positive value.
    pub fn block_size(&self) -> f32 {
        self.block_size.max(0.01)
    }

    /// World-space `(x, z)` size of one chunk.
    pub fn chunk_world_size(&self) -> (f32, f32) {
        let b = self.chunk_blocks();
        let s = self.block_size();
        (b[0] as f32 * s, b[2] as f32 * s)
    }

    /// View radius in chunks, floored at 0 and capped so a typo cannot ask for
    /// a multi-thousand-chunk window.
    pub fn view_radius(&self) -> i32 {
        (self.view_radius as i32).clamp(0, 32)
    }

    /// Effective impostor (far) radius in chunks. Capped well above the
    /// full-detail cap since impostors are cheap, and floored at `view_radius`
    /// (a smaller value disables impostors, there is no far band to fill).
    pub fn impostor_radius(&self) -> i32 {
        (self.impostor_radius as i32)
            .clamp(0, 96)
            .max(self.view_radius())
    }

    /// Coarse-grid step in blocks for distant impostors, floored at 1 and
    /// capped so a typo cannot collapse the whole surface to a single quad on a
    /// huge chunk (still valid, just degenerate).
    pub fn impostor_step(&self) -> u32 {
        self.impostor_step.clamp(1, 64)
    }

    /// Whether the distant-impostor far band is active: an impostor radius
    /// strictly beyond the full-detail radius.
    pub fn impostors_enabled(&self) -> bool {
        self.impostor_radius() > self.view_radius()
    }

    /// Per-frame chunk load budget as a `usize`, floored at 1 so a stray 0
    /// cannot wedge streaming permanently.
    pub fn load_budget(&self) -> usize {
        (self.load_budget as usize).max(1)
    }
}

impl Component for VoxelWorld {
    const NAME: &'static str = "VoxelWorld";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}

impl crate::check::cross_reference::CrossReferenced for VoxelWorld {
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
                    "VoxelWorld '{}': palette[{}] is not a valid BlockType name",
                    name, i
                )));
            } else {
                refs.push(CrossRef::Resolve {
                    kind: RefKind::BlockType,
                    target: bt_name.to_string(),
                    error: format!(
                        "VoxelWorld '{}': palette[{}] BlockType '{}' not found, add a BlockType asset with that name",
                        name, i, bt_name
                    ),
                });
            }
        }

        if let Some(mat) = args.get("material").and_then(|v| v.as_str())
            && !mat.is_empty()
        {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Material,
                target: mat.to_string(),
                error: format!(
                    "VoxelWorld '{}': material '{}' not found, add a Material asset with that name",
                    name, mat
                ),
            });
        }

        refs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_a_modest_window() {
        let w = VoxelWorld::default();
        assert_eq!(w.chunk_blocks(), [16, 24, 16]);
        assert_eq!(w.view_radius(), 5);
        assert_eq!(w.load_budget(), 3);
        assert_eq!(w.chunk_world_size(), (16.0, 16.0));
    }

    #[test]
    fn degenerate_args_are_floored_and_clamped() {
        let w = VoxelWorld {
            chunk_blocks: [0, 0, 0],
            block_size: -1.0,
            view_radius: 9999,
            load_budget: 0,
            ..VoxelWorld::default()
        };
        assert_eq!(w.chunk_blocks(), [1, 1, 1]);
        assert!(w.block_size() > 0.0);
        assert_eq!(w.view_radius(), 32);
        assert_eq!(w.load_budget(), 1);
    }

    #[test]
    fn deserialises_from_jsonl_args_with_defaults_for_omitted_fields() {
        let w: VoxelWorld = serde_json::from_str(r#"{"seed":7,"view_radius":8}"#).expect("parse");
        assert_eq!(w.seed, 7);
        assert_eq!(w.view_radius(), 8);
        // omitted fields fall back to the defaults
        assert_eq!(w.chunk_blocks(), [16, 24, 16]);
        assert_eq!(w.load_budget(), 3);
    }

    #[test]
    fn round_trips_through_args() {
        let w = VoxelWorld {
            seed: 99,
            chunk_blocks: [8, 32, 8],
            block_size: 2.0,
            view_radius: 4,
            impostor_radius: 12,
            impostor_step: 2,
            load_budget: 5,
            palette: Vec::new(),
            material: None,
        };
        let back = VoxelWorld::from_args(w.to_args());
        assert_eq!(back.seed, 99);
        assert_eq!(back.chunk_blocks, [8, 32, 8]);
        assert_eq!(back.block_size, 2.0);
        assert_eq!(back.impostor_radius, 12);
        assert_eq!(back.impostor_step, 2);
        assert_eq!(back.load_budget, 5);
    }

    #[test]
    fn impostors_disabled_by_default() {
        let w = VoxelWorld::default();
        // Default impostor_radius 0 -> clamped up to view_radius -> no far band.
        assert_eq!(w.impostor_radius(), w.view_radius());
        assert!(!w.impostors_enabled());
        assert_eq!(w.impostor_step(), 4);
    }

    #[test]
    fn impostor_radius_enables_the_far_band_and_clamps() {
        let w = VoxelWorld {
            view_radius: 5,
            impostor_radius: 16,
            impostor_step: 0,
            ..VoxelWorld::default()
        };
        assert_eq!(w.impostor_radius(), 16);
        assert!(w.impostors_enabled());
        // step floored at 1.
        assert_eq!(w.impostor_step(), 1);

        // An impostor radius below the view radius disables impostors.
        let w2 = VoxelWorld {
            view_radius: 8,
            impostor_radius: 4,
            ..VoxelWorld::default()
        };
        assert_eq!(w2.impostor_radius(), w2.view_radius());
        assert!(!w2.impostors_enabled());
    }
}
