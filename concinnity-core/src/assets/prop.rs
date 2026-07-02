// src/assets/prop.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

/// Collision volume attached to a [Prop](#prop).
///
/// The shape dimensions are in the prop's local space and are scaled by the
/// prop's `scale`. `ball` and `capsule` use the X scale component (they assume
/// uniform scaling).
///
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PropCollider {
    /// Collision shape: "aabb" (alias "cuboid"), "ball", or "capsule".
    pub shape: String,
    /// Box half-extents in local space [x, y, z]. Used by cuboid shapes.
    pub half_extents: [f32; 3],
    /// Radius in local space. Used by ball and capsule shapes.
    pub radius: f32,
    /// Half the cylinder height in local space. Used by capsule shapes.
    pub half_height: f32,
}

impl Default for PropCollider {
    fn default() -> Self {
        Self {
            shape: "cuboid".to_string(),
            half_extents: [0.5, 0.5, 0.5],
            radius: 0.5,
            half_height: 0.5,
        }
    }
}

/// A scene object: places geometry at a world-space transform.
///
/// Reference either a [Model](#model) (multi-mesh) or a single
/// [Mesh](#mesh)/[ProceduralMesh](#proceduralmesh). `model` takes precedence
/// when both are set.
///
/// ```jsonl
/// // single mesh
/// {"name":"crate_a","type":"Prop","args":{"mesh":"box_mesh","material":"mat_brick","position":[4.0,0.4,-8.0],"collider":{"shape":"aabb","half_extents":[0.4,0.4,0.4]}}}
/// {"name":"column_ne","type":"Prop","args":{"mesh":"column_mesh","material":"mat_stone","position":[8.0,1.7,-10.0],"collider":{"shape":"aabb","half_extents":[0.18,1.7,0.18]}}}
/// {"name":"room_floor","type":"Prop","args":{"mesh":"room_mesh","material":"mat_plaster","position":[0.0,0.0,0.0]}}
///
/// // multi-mesh model
/// {"name":"crate_a","type":"Prop","args":{"model":"wooden_crate","position":[2.0,0.3,-4.0],"collider":{"shape":"aabb","half_extents":[0.3,0.3,0.3]}}}
///
/// // parent-child hierarchy: door panel inherits the frame's world transform
/// {"name":"door_frame","type":"Prop","args":{"model":"wooden_frame","position":[3,0,-2]}}
/// {"name":"door_panel","type":"Prop","args":{"model":"door","parent":"door_frame","position":[0,0,0.05]}}
/// ```
///
/// Rotation notes:
/// - `rotation_deg[0]` = pitch (tilt forward/back)
/// - `rotation_deg[1]` = yaw (spin on vertical axis), most common
/// - `rotation_deg[2]` = roll (tilt side-to-side)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Prop {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// A [Model](#model) asset. When set, the prop renders all sub-meshes of
    /// that model (each with its own material) sharing this prop's transform.
    /// Takes precedence over `mesh` and `material`.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub model: Option<AssetId>,
    /// A [Mesh](#mesh) or [ProceduralMesh](#proceduralmesh) asset this prop
    /// renders. Used when `model` is unset.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub mesh: Option<AssetId>,
    /// A [Material](#material) to use for this prop. When set it takes
    /// precedence over `texture` and provides the albedo texture plus the
    /// lighting parameters (roughness, metallic, tint, emissive). Used when
    /// `model` is unset.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub material: Option<AssetId>,
    /// A [Texture](#texture) to use for this prop. Older field: ignored when
    /// `material` is set. Unset uses the first declared texture (or a white
    /// fallback).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub texture: Option<AssetId>,
    /// World-space position [x, y, z].
    pub position: [f32; 3],
    /// Euler rotation in degrees [pitch, yaw, roll], applied in YXZ order
    /// (yaw first so that rotating around the vertical axis is intuitive).
    pub rotation_deg: [f32; 3],
    /// Non-uniform scale [x, y, z]. Defaults to [1, 1, 1].
    pub scale: [f32; 3],
    /// Optional collision volume. When present, the prop blocks the player; when
    /// absent the prop is non-solid.
    pub collider: Option<PropCollider>,
    /// When true, the player can interact with this prop: pressing the interact
    /// key (E) while close and facing it triggers its rotation behaviour.
    pub interactable: bool,
    /// When true, the player can pick up and carry this prop with the interact
    /// key (E). A companion [PropBody](#propbody) must also be declared so the
    /// prop falls correctly after being dropped.
    pub pickup: bool,
    /// Another [Prop](#prop) whose world transform this prop inherits. When set,
    /// `position`, `rotation_deg`, and `scale` are relative to the parent's
    /// world transform. The parent must be declared in the same world; circular
    /// chains are treated as an error.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub parent: Option<AssetId>,
    /// [Scene](#scene) this prop belongs to. Resolved automatically from the
    /// naming convention (a prop named `<scene>_*` belongs to scene `<scene>`);
    /// you don't set this directly. `None` means the prop is visible in every
    /// scene. Used by [SceneReel](#scenereel) for per-scene visibility.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub scene: Option<AssetId>,
    /// Name of a [Prefab](#prefab) to instantiate at this prop's transform. When
    /// set, it expands into concrete child props and lights, replacing this
    /// prop. Cannot be combined with `model` or `mesh`.
    pub prefab: String,
    /// Optional view-distance cutoff in world units. When > 0 the prop is hidden
    /// once the camera is further than this from it. 0 (default) keeps the prop
    /// visible at any distance.
    pub cull_distance: f32,
    /// Set at runtime while the prop is being carried. Not serialised.
    /// While true, PhysicsSystem drives the prop as a kinematic body that
    /// follows the camera instead of simulating it dynamically.
    #[serde(skip)]
    pub is_held: bool,
}

impl Default for Prop {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            model: None,
            mesh: None,
            material: None,
            texture: None,
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            collider: None,
            interactable: false,
            pickup: false,
            parent: None,
            scene: None,
            prefab: String::new(),
            cull_distance: 0.0,
            is_held: false,
        }
    }
}

impl Prop {
    /// Build a column-major model matrix from this prop's transform.
    /// Order: scale, then YXZ Euler rotation, then translation.
    pub fn model_matrix(&self) -> [[f32; 4]; 4] {
        let [px, py, pz] = self.position;
        let [pitch_deg, yaw_deg, roll_deg] = self.rotation_deg;
        let [sx, sy, sz] = self.scale;

        let (pr, yr, rr) = (
            pitch_deg.to_radians(),
            yaw_deg.to_radians(),
            roll_deg.to_radians(),
        );
        let (sp, cp) = (pr.sin(), pr.cos());
        let (sy_, cy) = (yr.sin(), yr.cos());
        let (sr, cr) = (rr.sin(), rr.cos());

        // YXZ rotation: R = Ry * Rx * Rz
        // Combined and scaled, column-major storage: out[col][row].
        [
            [
                sx * (cy * cr + sy_ * sp * sr),
                sx * (cp * sr),
                sx * (-sy_ * cr + cy * sp * sr),
                0.0,
            ],
            [
                sy * (-cy * sr + sy_ * sp * cr),
                sy * (cp * cr),
                sy * (sy_ * sr + cy * sp * cr),
                0.0,
            ],
            [sz * (sy_ * cp), sz * (-sp), sz * (cy * cp), 0.0],
            [px, py, pz, 1.0],
        ]
    }
}

impl Component for Prop {
    const NAME: &'static str = "Prop";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(mut args: Self) -> Self {
        args.cull_distance = args.cull_distance.max(0.0);
        args.is_held = false;
        args
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

impl crate::check::cross_reference::CrossReferenced for Prop {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();

        // A Model takes precedence over a Mesh; only the one in effect is checked.
        let model_ref = arg("model");
        let mesh_ref = arg("mesh");
        if !model_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Model,
                target: model_ref.to_string(),
                error: format!(
                    "Prop '{}': model '{}' not found, add a Model asset with that name",
                    name, model_ref
                ),
            });
        } else if !mesh_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::MeshSource,
                target: mesh_ref.to_string(),
                error: format!(
                    "Prop '{}': mesh '{}' not found, add a Mesh, ProceduralMesh, or File (obj) asset with that name",
                    name, mesh_ref
                ),
            });
        }

        let mat_ref = arg("material");
        if !mat_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Material,
                target: mat_ref.to_string(),
                error: format!(
                    "Prop '{}': material '{}' not found, add a Material asset with that name",
                    name, mat_ref
                ),
            });
        }

        let tex_ref = arg("texture");
        if !tex_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: tex_ref.to_string(),
                error: format!(
                    "Prop '{}': texture '{}' not found, add a Texture asset with that name",
                    name, tex_ref
                ),
            });
        }

        let parent_ref = arg("parent");
        if !parent_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Prop,
                target: parent_ref.to_string(),
                error: format!(
                    "Prop '{}': parent '{}' not found, add a Prop asset with that name",
                    name, parent_ref
                ),
            });
        }

        refs
    }
}
