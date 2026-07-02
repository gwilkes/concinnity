// src/assets/instanced_prop.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

/// Per-instance transform within an `InstancedProp`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct InstanceTransform {
    /// World-space position `[x, y, z]`.
    pub position: [f32; 3],
    /// Euler rotation in degrees `[pitch, yaw, roll]`, applied in YXZ order.
    pub rotation_deg: [f32; 3],
    /// Non-uniform scale `[x, y, z]`.
    pub scale: [f32; 3],
}

impl Default for InstanceTransform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

/// A single mesh + material drawn at many world-space transforms.
///
/// Use for foliage, debris, projectiles, or any content that repeats the same
/// shape with varied placement. Each instance gets its own world transform and
/// culling without the overhead of declaring many separate [Prop](#prop)s.
///
/// Each `instances` entry has the shape `{"position":[x,y,z], "rotation_deg":[p,y,r], "scale":[sx,sy,sz]}`.
/// `rotation_deg` and `scale` may be omitted (defaults `[0,0,0]` and `[1,1,1]`).
///
/// ```jsonl
/// {"name":"rock_mesh","type":"ProceduralMesh","args":{"generator":"sphere","radius":0.4,"rings":8,"segments":10}}
/// {"name":"mat_stone","type":"Material","args":{"albedo":"tex_stone","roughness":0.9}}
/// {"name":"rocks","type":"InstancedProp","args":{
///   "mesh":"rock_mesh",
///   "material":"mat_stone",
///   "cull_distance":80.0,
///   "instances":[
///     {"position":[ 2.0, 0.4, -3.0]},
///     {"position":[-5.0, 0.4,  1.0], "rotation_deg":[0, 45, 0]},
///     {"position":[ 4.0, 0.4,  7.0], "scale":[1.5, 1.5, 1.5]}
///   ]
/// }}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct InstancedProp {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// A [Mesh](#mesh), [ProceduralMesh](#proceduralmesh),
    /// [VoxelChunk](#voxelchunk), or mesh-kind [File](#file) asset.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub mesh: Option<AssetId>,
    /// A [Material](#material); takes precedence over `texture` when set.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub material: Option<AssetId>,
    /// Older texture-only reference; ignored when `material` is set.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub texture: Option<AssetId>,
    /// Per-instance transforms. Empty list renders nothing.
    pub instances: Vec<InstanceTransform>,
    /// View-distance cutoff in world units per instance. 0 = always draw.
    pub cull_distance: f32,
}

impl Default for InstancedProp {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            mesh: None,
            material: None,
            texture: None,
            instances: Vec::new(),
            cull_distance: 0.0,
        }
    }
}

impl InstancedProp {
    /// Build a column-major model matrix for the i-th instance.
    /// Order matches `Prop::model_matrix`: scale, then YXZ rotation, then translation.
    pub fn instance_model_matrix(&self, idx: usize) -> Option<[[f32; 4]; 4]> {
        let xform = self.instances.get(idx)?;
        let [px, py, pz] = xform.position;
        let [pitch_deg, yaw_deg, roll_deg] = xform.rotation_deg;
        let [sx, sy, sz] = xform.scale;

        let (pr, yr, rr) = (
            pitch_deg.to_radians(),
            yaw_deg.to_radians(),
            roll_deg.to_radians(),
        );
        let (sp, cp) = (pr.sin(), pr.cos());
        let (sy_, cy) = (yr.sin(), yr.cos());
        let (sr, cr) = (rr.sin(), rr.cos());

        Some([
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
        ])
    }
}

impl Component for InstancedProp {
    const NAME: &'static str = "InstancedProp";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        args.cull_distance = args.cull_distance.max(0.0);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
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

impl crate::check::cross_reference::CrossReferenced for InstancedProp {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();

        let mesh_ref = arg("mesh");
        if mesh_ref.is_empty() {
            refs.push(CrossRef::Issue(format!(
                "InstancedProp '{}': `mesh` field is required",
                name
            )));
        } else {
            refs.push(CrossRef::Resolve {
                kind: RefKind::MeshSource,
                target: mesh_ref.to_string(),
                error: format!(
                    "InstancedProp '{}': mesh '{}' not found, add a Mesh, ProceduralMesh, VoxelChunk, or File (obj) asset with that name",
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
                    "InstancedProp '{}': material '{}' not found, add a Material asset with that name",
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
                    "InstancedProp '{}': texture '{}' not found, add a Texture asset with that name",
                    name, tex_ref
                ),
            });
        }

        refs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty() -> InstancedProp {
        InstancedProp {
            asset_id: AssetId::default(),
            mesh: None,
            material: None,
            texture: None,
            instances: Vec::new(),
            cull_distance: 0.0,
        }
    }

    #[test]
    fn instance_model_matrix_default_is_identity() {
        let mut p = empty();
        p.instances.push(InstanceTransform::default());
        let m = p.instance_model_matrix(0).unwrap();
        assert_eq!(m[3], [0.0, 0.0, 0.0, 1.0]);
        assert!((m[0][0] - 1.0).abs() < 1e-5);
        assert!((m[1][1] - 1.0).abs() < 1e-5);
        assert!((m[2][2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn instance_model_matrix_translates() {
        let mut p = empty();
        p.instances.push(InstanceTransform {
            position: [5.0, -2.0, 3.0],
            ..InstanceTransform::default()
        });
        let m = p.instance_model_matrix(0).unwrap();
        assert_eq!(m[3], [5.0, -2.0, 3.0, 1.0]);
    }

    #[test]
    fn instance_model_matrix_scales() {
        let mut p = empty();
        p.instances.push(InstanceTransform {
            scale: [2.0, 3.0, 4.0],
            ..InstanceTransform::default()
        });
        let m = p.instance_model_matrix(0).unwrap();
        // diagonal entries should be the scale factors (no rotation)
        assert!((m[0][0] - 2.0).abs() < 1e-5);
        assert!((m[1][1] - 3.0).abs() < 1e-5);
        assert!((m[2][2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn instance_model_matrix_out_of_range_returns_none() {
        let p = empty();
        assert!(p.instance_model_matrix(0).is_none());
    }

    #[test]
    fn from_args_clamps_negative_cull_distance() {
        let args = InstancedProp {
            cull_distance: -5.0,
            ..InstancedProp::default()
        };
        let p = InstancedProp::from_args(args);
        assert_eq!(p.cull_distance, 0.0);
    }
}
