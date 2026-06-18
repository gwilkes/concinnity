// src/assets/room.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RoomArgs {
    /// Half the room's width along X, in world units. Ignored when `size` is set.
    pub half_width: f32,
    /// Half the room's depth along Z, in world units. Ignored when `size` is set.
    pub half_depth: f32,
    /// Floor-to-ceiling height in world units. Ignored when `size` is set.
    pub ceiling_height: f32,
    /// Shorthand for the full dimensions `[width, depth, height]`. When set, it
    /// overrides `half_width`, `half_depth`, and `ceiling_height`.
    pub size: Option<[f32; 3]>,
    /// [Texture](#texture) applied to all surfaces. Falls back to `wall_texture`
    /// when unset. Generator names such as `"brick"` or `"concrete"` resolve to
    /// a matching texture at build time.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub texture: Option<AssetId>,
    /// [Texture](#texture) for the walls. Currently all surfaces share one
    /// texture; per-surface texturing is reserved for a future update.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub wall_texture: Option<AssetId>,
    /// [Texture](#texture) for the floor (see `wall_texture`).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub floor_texture: Option<AssetId>,
    /// [Texture](#texture) for the ceiling (see `wall_texture`).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub ceiling_texture: Option<AssetId>,
    /// Number of level-of-detail versions to generate, including the original.
    /// `1` (the default) generates no alternates.
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version. Empty
    /// lets the build choose defaults.
    #[serde(default)]
    pub lod_distances: Vec<f32>,
}

impl Default for RoomArgs {
    fn default() -> Self {
        Self {
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            size: None,
            texture: None,
            wall_texture: None,
            floor_texture: None,
            ceiling_texture: None,
            lod_levels: 1,
            lod_distances: Vec::new(),
        }
    }
}

/// A self-contained room (floor, ceiling, four walls), with optional texturing.
///
/// Prefer `Room` over a [ProceduralMesh](#proceduralmesh) (generator `"room"`) +
/// [Prop](#prop) pair for a shorter declaration. The room is placed at the world
/// origin.
///
/// Dimensions can be given as `size: [width, depth, height]` (full extents) or
/// as `half_width`, `half_depth`, and `ceiling_height` individually.
///
/// `texture`, `wall_texture`, `floor_texture`, and `ceiling_texture` are checked
/// in that order; the first set value wins. Generator names such as `"brick"` or
/// `"concrete"` resolve to a matching [Texture](#texture) at build time.
///
/// ```jsonl
/// {"name":"room","type":"Room","args":{"size":[16,20,3.5],"texture":"tex_plaster"}}
/// ```
#[derive(Debug)]
pub struct Room {
    pub asset_id: AssetId,
    pub half_width: f32,
    pub half_depth: f32,
    pub ceiling_height: f32,
    pub texture: Option<AssetId>,
    pub wall_texture: Option<AssetId>,
    pub floor_texture: Option<AssetId>,
    pub ceiling_texture: Option<AssetId>,
    pub locator: Option<PayloadLocator>,
}

impl Room {
    // Returns the first set texture reference across all texture fields.
    pub fn effective_texture(&self) -> Option<AssetId> {
        [
            self.texture,
            self.wall_texture,
            self.floor_texture,
            self.ceiling_texture,
        ]
        .into_iter()
        .flatten()
        .next()
    }
}

impl Component for Room {
    const NAME: &'static str = "Room";

    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;
    type Args = RoomArgs;

    fn to_args(&self) -> RoomArgs {
        RoomArgs {
            half_width: self.half_width,
            half_depth: self.half_depth,
            ceiling_height: self.ceiling_height,
            size: None,
            texture: self.texture,
            wall_texture: self.wall_texture,
            floor_texture: self.floor_texture,
            ceiling_texture: self.ceiling_texture,
            lod_levels: 1,
            lod_distances: Vec::new(),
        }
    }

    fn from_args(args: RoomArgs) -> Self {
        // Resolve size shorthand into half_width / half_depth / ceiling_height.
        let (half_width, half_depth, ceiling_height) = if let Some([w, d, h]) = args.size {
            (w / 2.0, d / 2.0, h)
        } else {
            (args.half_width, args.half_depth, args.ceiling_height)
        };
        Self {
            asset_id: AssetId::default(),
            half_width,
            half_depth,
            ceiling_height,
            texture: args.texture,
            wall_texture: args.wall_texture,
            floor_texture: args.floor_texture,
            ceiling_texture: args.ceiling_texture,
            locator: None,
        }
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_texture_returns_texture_field_first() {
        let room = Room {
            asset_id: AssetId::default(),
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            texture: Some(AssetId(1)),
            wall_texture: Some(AssetId(2)),
            floor_texture: None,
            ceiling_texture: None,
            locator: None,
        };
        assert_eq!(room.effective_texture(), Some(AssetId(1)));
    }

    #[test]
    fn effective_texture_falls_back_to_wall_texture() {
        let room = Room {
            asset_id: AssetId::default(),
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            texture: None,
            wall_texture: Some(AssetId(7)),
            floor_texture: None,
            ceiling_texture: None,
            locator: None,
        };
        assert_eq!(room.effective_texture(), Some(AssetId(7)));
    }

    #[test]
    fn effective_texture_returns_none_when_all_unset() {
        let room = Room::from_args(RoomArgs::default());
        assert_eq!(room.effective_texture(), None);
    }

    #[test]
    fn from_args_resolves_size_shorthand() {
        let args = RoomArgs {
            size: Some([16.0, 20.0, 3.5]),
            ..RoomArgs::default()
        };
        let room = Room::from_args(args);
        assert_eq!(room.half_width, 8.0);
        assert_eq!(room.half_depth, 10.0);
        assert_eq!(room.ceiling_height, 3.5);
    }

    #[test]
    fn from_args_uses_explicit_half_extents_when_no_size() {
        let args = RoomArgs {
            half_width: 5.0,
            half_depth: 7.0,
            ceiling_height: 4.0,
            size: None,
            ..RoomArgs::default()
        };
        let room = Room::from_args(args);
        assert_eq!(room.half_width, 5.0);
        assert_eq!(room.half_depth, 7.0);
        assert_eq!(room.ceiling_height, 4.0);
    }
}
