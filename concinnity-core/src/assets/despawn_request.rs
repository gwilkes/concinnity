// src/assets/despawn_request.rs

use crate::ecs::asset_id::AssetId;

// Runtime-only event requesting that an authored placement be removed from the
// world at runtime. Addressed by the placement's stable asset name (entities
// are volatile), so a producer needs no live Entity handle. GraphicsSystem
// reads these from its Events<DespawnRequest> queue each step, resolves the
// name to its entity, hides that entity's GPU draw slots, and despawns it and
// its descendants. World authors never declare this type directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct DespawnRequest {
    pub name: AssetId,
}
