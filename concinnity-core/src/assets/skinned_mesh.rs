// src/assets/skinned_mesh.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

fn white() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn unit_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn first_weight() -> [f32; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

/// One vertex of a skinned mesh. Beyond position / colour / uv it carries up
/// to four joint bindings: `joints[k]` indexes the skeleton, `weights[k]` is
/// its blend weight. Weights are normalised at build time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SkinnedVertexData {
    /// Vertex position `[x, y, z]` in model space.
    pub pos: [f32; 3],
    /// Vertex colour `[r, g, b]` in [0, 1]. Defaults to white.
    #[serde(default = "white")]
    pub color: [f32; 3],
    /// Texture coordinates in [0, 1] space. Defaults to [0, 0].
    #[serde(default)]
    pub uv: [f32; 2],
    /// Joint indices this vertex is bound to. Unused slots can be 0.
    #[serde(default)]
    pub joints: [u32; 4],
    /// Blend weights parallel to `joints`. Defaults to fully bound to joint 0.
    #[serde(default = "first_weight")]
    pub weights: [f32; 4],
}

/// One joint of a skeleton's bind pose.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct JointDef {
    /// Human-readable joint name (animation tracks may reference it later).
    pub name: String,
    /// Parent joint index, or -1 for a root. Parents must appear before their
    /// children in the `skeleton` list.
    pub parent: i32,
    /// Local bind translation relative to the parent.
    pub translation: [f32; 3],
    /// Local bind rotation, Euler degrees [pitch, yaw, roll], YXZ order.
    pub rotation_deg: [f32; 3],
    /// Local bind scale.
    pub scale: [f32; 3],
}

impl Default for JointDef {
    fn default() -> Self {
        Self {
            name: String::new(),
            parent: -1,
            translation: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

/// Build a runtime `Skeleton` from authored joint definitions. Mirrors the
/// conversion `GraphicsSystem::init` does at world
/// load time: each `JointDef.parent` becomes `Some(usize)` for valid indices
/// (negative values mark roots), and each `JointDef`'s translation /
/// rotation / scale becomes the joint's bind `JointPose`. Used at init and
/// by the asset hot-reload's skeleton-shape change path.
pub fn build_skeleton_from_joint_defs(defs: &[JointDef]) -> crate::gfx::skinning::Skeleton {
    use crate::gfx::skinning;
    let joints = defs
        .iter()
        .map(|jd| skinning::Joint {
            parent: (jd.parent >= 0).then_some(jd.parent as usize),
            bind: skinning::JointPose {
                translation: jd.translation,
                rotation_deg: jd.rotation_deg,
                scale: jd.scale,
            },
        })
        .collect();
    skinning::Skeleton::new(joints)
}

/// A skeletally animated mesh placed directly in the world.
///
/// Unlike a [Mesh](#mesh), a `SkinnedMesh` carries its own world transform and a
/// `skeleton` (a joint hierarchy with a bind pose). Each vertex is bound to up
/// to four joints; an [Animation](#animation) targeting this mesh deforms it at
/// runtime. With no animation the mesh renders in its bind pose.
///
/// The geometry + skeleton may be authored inline (`vertices` / `indices` /
/// `skeleton`) or imported from a binary glTF file with `source`. Only the
/// `.glb` container is supported, and only the mesh + skeleton bind pose are
/// imported (glTF animations are not yet brought in).
///
/// The `skeleton` (joint hierarchy and bind pose) is provided as an arg
/// (authored inline alongside `vertices`/`indices`, or filled in from the
/// imported `.glb`) and is baked into the mesh at build time.
///
/// Normals and tangents are computed automatically at build time. Do not
/// supply them.
///
/// ```jsonl
/// {"name":"flag","type":"SkinnedMesh","args":{"position":[0,1,0],"material":"mat_cloth","skeleton":[{"parent":-1},{"parent":0,"translation":[0,1,0]}],"vertices":[{"pos":[0,0,0],"joints":[0,0,0,0],"weights":[1,0,0,0]}],"indices":[0,0,0]}}
/// {"name":"hero","type":"SkinnedMesh","args":{"source":"models/hero.glb","position":[0,0,0],"material":"mat_skin"}}
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SkinnedMesh {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Optional path to a `.glb` file. When set, the build imports
    /// `vertices` / `indices` / `skeleton` from it; an inline-authored mesh
    /// leaves this empty.
    pub source: String,
    /// Skinned vertex list.
    pub vertices: Vec<SkinnedVertexData>,
    /// Triangle index list.
    pub indices: Vec<u16>,
    /// [Material](#material); provides the albedo texture plus lighting
    /// parameters.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub material: Option<AssetId>,
    /// [Texture](#texture) (older path); ignored when `material` is set.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub texture: Option<AssetId>,
    /// World-space position.
    pub position: [f32; 3],
    /// World rotation, Euler degrees [pitch, yaw, roll], YXZ order.
    pub rotation_deg: [f32; 3],
    /// World scale.
    pub scale: [f32; 3],
    /// Number of level-of-detail versions to generate, including the original.
    /// `1` (the default) generates none; values are clamped to `[1, 8]`.
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version. When
    /// non-empty, must have exactly `lod_levels - 1` entries; empty lets the
    /// build choose defaults.
    #[serde(default)]
    pub lod_distances: Vec<f32>,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl SkinnedMesh {
    /// Column-major world matrix from this mesh's transform. Same construction
    /// order (scale, YXZ rotation, translate) as `Prop::model_matrix`.
    pub fn model_matrix(&self) -> [[f32; 4]; 4] {
        crate::gfx::skinning::JointPose {
            translation: self.position,
            rotation_deg: self.rotation_deg,
            scale: self.scale,
        }
        .to_matrix()
    }
}

impl Component for SkinnedMesh {
    const NAME: &'static str = "SkinnedMesh";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(mut args: Self) -> Self {
        // A zero scale would collapse the world matrix; clamp to a sane unit.
        if args.scale == [0.0, 0.0, 0.0] {
            args.scale = unit_scale();
        }
        // `lod_levels` defaults to 0 via #[derive(Default)] on u32. 0 and 1
        // are equivalent (no alternates); cap at 8 to mirror the static
        // `Mesh` LOD ceiling.
        if args.lod_levels == 0 {
            args.lod_levels = 1;
        }
        args.lod_levels = args.lod_levels.min(8);
        args
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }
    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::build::SourceBacked for SkinnedMesh {
    // A glTF-sourced SkinnedMesh needs its `.glb` fetched before the build's
    // desugar pass can expand it; an inline-authored mesh has no source.
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_skeleton_from_joint_defs_preserves_count_and_parent_links() {
        let defs = vec![
            JointDef {
                name: "root".into(),
                parent: -1,
                translation: [0.0, 0.0, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
            JointDef {
                name: "tip".into(),
                parent: 0,
                translation: [0.0, 1.0, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
            JointDef {
                name: "tail".into(),
                parent: 1,
                translation: [0.0, 1.0, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
        ];
        let skel = build_skeleton_from_joint_defs(&defs);
        assert_eq!(skel.len(), 3);
        let joints = skel.joints();
        assert_eq!(joints[0].parent, None);
        assert_eq!(joints[1].parent, Some(0));
        assert_eq!(joints[2].parent, Some(1));
    }

    #[test]
    fn build_skeleton_from_joint_defs_treats_negative_parent_as_root() {
        // Any negative parent (not just -1) collapses to None; mirrors the
        // init-time semantics so a hot-reload from the same JointDef shape
        // produces the same Skeleton.
        let defs = vec![JointDef {
            name: "root".into(),
            parent: -42,
            translation: [1.0, 2.0, 3.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }];
        let skel = build_skeleton_from_joint_defs(&defs);
        assert_eq!(skel.joints()[0].parent, None);
    }
}
