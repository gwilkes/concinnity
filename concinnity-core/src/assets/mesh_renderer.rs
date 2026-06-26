// src/assets/mesh_renderer.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

/// Single-mesh render description for an entity: which mesh, material, and
/// optional legacy texture to draw, plus an optional view-distance cutoff.
///
/// Runtime-only. Mutually exclusive with `ModelRenderer` on an entity (an
/// entity has one or the other), which encodes the mesh-vs-model choice a
/// `Prop` expresses with its `model` field taking precedence.
#[derive(Debug, Clone, Default)]
pub struct MeshRenderer {
    /// A `Mesh` or `ProceduralMesh` to render.
    pub mesh: Option<AssetId>,
    /// A `Material` providing albedo plus lighting parameters.
    pub material: Option<AssetId>,
    /// Legacy texture, used only when `material` is unset.
    pub texture: Option<AssetId>,
    /// View-distance cutoff in world units; 0 keeps the draw visible at any
    /// distance.
    pub cull_distance: f32,
}

/// `MeshRenderer` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct MeshRendererArgs {}

impl Component for MeshRenderer {
    const NAME: &'static str = "MeshRenderer";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = MeshRendererArgs;

    fn to_args(&self) -> MeshRendererArgs {
        MeshRendererArgs {}
    }
    fn from_args(_: MeshRendererArgs) -> Self {
        Self::default()
    }
}
