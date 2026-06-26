// src/assets/model_renderer.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

/// Multi-mesh render description for an entity: a `Model` whose sub-meshes all
/// share this entity's transform, each with its own material.
///
/// Runtime-only. Mutually exclusive with `MeshRenderer` on an entity.
#[derive(Debug, Clone)]
pub struct ModelRenderer {
    /// The `Model` to render.
    pub model: AssetId,
    /// View-distance cutoff in world units; 0 keeps the draw visible at any
    /// distance.
    pub cull_distance: f32,
}

impl Default for ModelRenderer {
    fn default() -> Self {
        Self {
            model: AssetId::default(),
            cull_distance: 0.0,
        }
    }
}

/// `ModelRenderer` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelRendererArgs {}

impl Component for ModelRenderer {
    const NAME: &'static str = "ModelRenderer";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = ModelRendererArgs;

    fn to_args(&self) -> ModelRendererArgs {
        ModelRendererArgs {}
    }
    fn from_args(_: ModelRendererArgs) -> Self {
        Self::default()
    }
}
