// src/assets/spawn_request.rs

use crate::assets::Transform;
use crate::ecs::asset_id::AssetId;

// Runtime-only event requesting that a copy of an existing placement be created
// in the world at runtime. The symmetric counterpart to DespawnRequest.
//
// `template` names a placement already present in the world; the new instance
// reuses that placement's geometry and material at a fresh `transform`, and is
// registered under `name` so it can later be addressed (despawned, reparented)
// like any authored placement. An optional `lifetime_secs` attaches a Lifetime
// so the instance auto-despawns after that many seconds, the churn that lets
// freed draw slots be recycled. GraphicsSystem reads these from its
// Events<SpawnRequest> queue each step. World authors never declare this type
// directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpawnRequest {
    pub template: AssetId,
    pub name: AssetId,
    pub transform: Transform,
    pub lifetime_secs: Option<f32>,
}
