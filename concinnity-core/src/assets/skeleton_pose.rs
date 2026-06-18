// src/assets/skeleton_pose.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};
use crate::gfx::skinning::{self, Mat4, Skeleton};

/// Runtime-only link between a skinned mesh and its animation state.
///
/// `GraphicsSystem` publishes one `SkeletonPose` per `SkinnedMesh` during
/// init: it carries the resolved bind-pose `Skeleton` and the index of the
/// mesh's skinned draw object in the backend. `AnimationSystem` then ticks the
/// matching `Animation` clip each frame and writes the resulting skinning
/// matrices into `joint_matrices`; `GraphicsSystem` reads them back and
/// uploads them to the GPU. The one-frame producer/consumer hand-off is
/// invisible at animation rates.
///
/// Not authored in world files: it has no `args`.
#[derive(Debug)]
pub struct SkeletonPose {
    /// The `SkinnedMesh` asset this pose belongs to. Used by `AnimationSystem`
    /// to match an `Animation` clip to its target.
    pub mesh_id: AssetId,
    /// Index of this mesh's skinned draw object in the render backend.
    /// Read only by the Metal skinned pipeline for now.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub skinned_index: usize,
    /// Bind-pose joint hierarchy, used to compose skinning matrices.
    pub skeleton: Skeleton,
    /// Current skinning matrices, one per joint. Seeded to the bind pose
    /// (identity skinning) and overwritten by `AnimationSystem` each frame.
    pub joint_matrices: Vec<Mat4>,
}

impl SkeletonPose {
    /// Build a pose for `mesh_id`'s skinned draw object, seeded to the bind
    /// pose so the mesh renders undeformed until an animation drives it.
    pub fn new(mesh_id: AssetId, skinned_index: usize, skeleton: Skeleton) -> Self {
        let joint_matrices = skeleton.bind_skinning_matrices();
        Self {
            mesh_id,
            skinned_index,
            skeleton,
            joint_matrices,
        }
    }
}

/// `SkeletonPose` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct SkeletonPoseArgs {}

impl Component for SkeletonPose {
    const NAME: &'static str = "SkeletonPose";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = SkeletonPoseArgs;

    fn to_args(&self) -> SkeletonPoseArgs {
        SkeletonPoseArgs {}
    }
    fn from_args(_: SkeletonPoseArgs) -> Self {
        // A RuntimeOnly component never round-trips through args; real
        // instances are built by GraphicsSystem via `new`.
        Self::new(AssetId::default(), 0, skinning::Skeleton::new(Vec::new()))
    }
}
