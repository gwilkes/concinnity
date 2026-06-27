// src/assets/reparent_request.rs

use crate::ecs::asset_id::AssetId;

// Runtime-only event requesting that an authored placement be re-parented at
// runtime. Addressed by stable asset names: `child` is moved under `parent`, or
// detached to a root when `parent` is None. GraphicsSystem reads these from its
// Events<ReparentRequest> queue each step, resolves the names to their entities,
// re-points the child's Parent edge, and recomposes world matrices. World
// authors never declare this type directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReparentRequest {
    pub child: AssetId,
    pub parent: Option<AssetId>,
}
