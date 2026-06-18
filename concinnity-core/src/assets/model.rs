// src/assets/model.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};

/// One geometric part of a Model, referencing a mesh and its surface material.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SubMeshRef {
    /// A [Mesh](#mesh) or [ProceduralMesh](#proceduralmesh) asset.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub mesh: Option<AssetId>,
    /// A [Material](#material) asset.  `None` uses the default material.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub material: Option<AssetId>,
}

/// An ordered list of sub-meshes, each with its own material.
///
/// Use via the `model` field on a [Prop](#prop) instead of `mesh`. Each
/// sub-mesh is drawn with its own material, all sharing the prop's transform.
///
/// Each `mesh` must name a [Mesh](#mesh) or [ProceduralMesh](#proceduralmesh)
/// asset present in the scene. `material` may be empty to use the default
/// material.
///
/// ```jsonl
/// {"name":"crate_body","type":"ProceduralMesh","args":{"generator":"box","half_extents":[0.3,0.3,0.3]}}
/// {"name":"crate_bands","type":"ProceduralMesh","args":{"generator":"box","half_extents":[0.31,0.04,0.31]}}
/// {"name":"mat_wood","type":"Material","args":{"albedo":"tex_wood","roughness":0.75,"metallic":0.0}}
/// {"name":"mat_metal","type":"Material","args":{"albedo":"tex_metal","roughness":0.4,"metallic":1.0}}
/// {"name":"wooden_crate","type":"Model","args":{"meshes":[
///   {"mesh":"crate_body", "material":"mat_wood"},
///   {"mesh":"crate_bands","material":"mat_metal"}
/// ]}}
/// {"name":"crate_a","type":"Prop","args":{"model":"wooden_crate","position":[2.0,0.3,-4.0]}}
/// {"name":"crate_b","type":"Prop","args":{"model":"wooden_crate","position":[-1.5,0.3,-6.0],"rotation_deg":[0,45,0]}}
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Model {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Ordered list of sub-meshes that make up this model.
    pub meshes: Vec<SubMeshRef>,
}

impl Component for Model {
    const NAME: &'static str = "Model";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::check::cross_reference::CrossReferenced for Model {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let mut refs = Vec::new();

        if let Some(meshes) = args.get("meshes").and_then(|v| v.as_array()) {
            for (i, sub) in meshes.iter().enumerate() {
                let sub_mesh = sub.get("mesh").and_then(|v| v.as_str()).unwrap_or("");
                if sub_mesh.is_empty() {
                    refs.push(CrossRef::Issue(format!(
                        "Model '{}': submesh[{}] is missing a 'mesh' field",
                        name, i
                    )));
                } else {
                    refs.push(CrossRef::Resolve {
                        kind: RefKind::MeshSource,
                        target: sub_mesh.to_string(),
                        error: format!(
                            "Model '{}': submesh[{}] mesh '{}' not found, add a Mesh, ProceduralMesh, or File (obj) asset with that name",
                            name, i, sub_mesh
                        ),
                    });
                }

                let sub_mat = sub.get("material").and_then(|v| v.as_str()).unwrap_or("");
                if !sub_mat.is_empty() {
                    refs.push(CrossRef::Resolve {
                        kind: RefKind::Material,
                        target: sub_mat.to_string(),
                        error: format!(
                            "Model '{}': submesh[{}] material '{}' not found, add a Material asset with that name",
                            name, i, sub_mat
                        ),
                    });
                }
            }
        }

        refs
    }
}
