// src/assets/spawner.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

/// Periodically instantiates copies of an existing placement at this entity's
/// position.
///
/// A spawner clones `template` (the name of another placement in the world)
/// every `interval` seconds, giving each copy a `lifetime` after which it is
/// automatically removed. Pairing a short lifetime with a short interval keeps a
/// bounded population churning (an enemy wave, a particle of debris, a fountain
/// of props) and is what exercises GPU draw-slot recycling: each expiry frees a
/// slot the next spawn reuses.
///
/// The spawner's own `Transform` (its position) is where copies appear, so place
/// the spawner where you want the stream to originate.
///
/// ```jsonl
/// {"name":"crate","type":"Prop","args":{"mesh":"box_mesh","material":"mat_brick","position":[0.0,1.0,-6.0]}}
/// {"name":"fountain","type":"Prop","args":{"mesh":"box_mesh","position":[0.0,1.0,-3.0]}}
/// {"name":"fountain_spawner","type":"Spawner","args":{"template":"crate","interval":0.5,"lifetime":2.0}}
/// ```
#[derive(Debug, Clone)]
pub struct Spawner {
    /// Name of the placement to copy on each spawn.
    pub template: AssetId,
    /// Seconds between spawns.
    pub interval: f32,
    /// Seconds each spawned copy lives before auto-removal; 0 keeps it forever.
    pub lifetime: f32,
    /// Runtime: seconds accumulated toward the next spawn.
    pub elapsed: f32,
    /// Runtime: number of copies spawned so far.
    pub count: u32,
}

/// Authored fields of a [`Spawner`]; the runtime accumulator is not declared.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SpawnerArgs {
    /// Name of the placement to copy on each spawn.
    pub template: AssetId,
    /// Seconds between spawns.
    pub interval: f32,
    /// Seconds each spawned copy lives before auto-removal; 0 keeps it forever.
    pub lifetime: f32,
}

impl Default for SpawnerArgs {
    fn default() -> Self {
        Self {
            template: AssetId::default(),
            interval: 1.0,
            lifetime: 0.0,
        }
    }
}

impl Component for Spawner {
    const NAME: &'static str = "Spawner";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = SpawnerArgs;

    fn from_args(args: SpawnerArgs) -> Self {
        Self {
            template: args.template,
            interval: args.interval.max(0.0),
            lifetime: args.lifetime.max(0.0),
            elapsed: 0.0,
            count: 0,
        }
    }
    fn to_args(&self) -> SpawnerArgs {
        SpawnerArgs {
            template: self.template,
            interval: self.interval,
            lifetime: self.lifetime,
        }
    }
}
