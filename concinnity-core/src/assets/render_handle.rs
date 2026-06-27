// src/assets/render_handle.rs

use crate::ecs::{AssetOrigin, Component};

/// The backend draw-object slot(s) an entity occupies.
///
/// Runtime-only. The renderer writes one of these per renderable entity so
/// per-frame model-matrix and visibility updates address the GPU slots by
/// entity rather than by storage row. A mesh-backed entity has one slot; a
/// model-backed entity has one per sub-mesh.
#[derive(Debug, Clone, Default)]
pub struct RenderHandle {
    /// Backend draw-object indices owned by this entity.
    pub draws: Vec<u32>,
}

/// `RenderHandle` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct RenderHandleArgs {}

impl Component for RenderHandle {
    const NAME: &'static str = "RenderHandle";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = RenderHandleArgs;

    fn to_args(&self) -> RenderHandleArgs {
        RenderHandleArgs {}
    }
    fn from_args(_: RenderHandleArgs) -> Self {
        Self::default()
    }
}
