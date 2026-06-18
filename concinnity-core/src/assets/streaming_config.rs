// src/assets/streaming_config.rs

use crate::ecs::{AssetOrigin, Component};

/// Enables and tunes asset streaming.
///
/// When no `StreamingConfig` is declared, streaming is off and every texture and
/// mesh is loaded up front. When one is present, textures and static mesh
/// geometry load in gradually after startup: each frame the nearest not-yet-
/// loaded items are brought in, up to a per-frame budget, prioritised by camera
/// distance. Once more than the cap would be loaded at once, the farthest are
/// dropped to make room.
///
/// Texture streaming covers the colour and normal-map textures (each capped
/// independently via `texture_budget` / `texture_cap`). Mesh streaming covers
/// static geometry; the skybox, rooms, and moving props always stay loaded.
///
/// ```jsonl
/// {"name":"streaming","type":"StreamingConfig","args":{}}
/// {"name":"streaming_slow","type":"StreamingConfig","args":{"texture_budget":1}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StreamingConfig {
    /// Maximum number of textures whose load is started per frame, applied
    /// independently to the colour and normal-map pools. A low value spreads the
    /// cost over more frames.
    pub texture_budget: u32,
    /// Maximum number of textures kept loaded at once, applied independently to
    /// the colour and normal-map pools. When exceeded, the farthest-from-camera
    /// textures are dropped.
    pub texture_cap: u32,
    /// Maximum number of mesh regions whose load is started per frame. A low
    /// value spreads the cost over more frames.
    pub mesh_budget: u32,
    /// Maximum number of meshes kept loaded at once. When exceeded, the
    /// farthest-from-camera meshes are dropped.
    pub mesh_cap: u32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            texture_budget: 4,
            texture_cap: 96,
            mesh_budget: 4,
            mesh_cap: 4096,
        }
    }
}

// These accessors feed the Metal asset-streaming path for now
// (Vulkan / DirectX catch-up is a follow-up).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
impl StreamingConfig {
    /// Per-frame texture load budget as a `usize`, floored at 1 so a stray 0
    /// cannot wedge streaming permanently.
    pub fn budget(&self) -> usize {
        (self.texture_budget as usize).max(1)
    }

    /// Resident-texture cap as a `usize`, floored at 1.
    pub fn cap(&self) -> usize {
        (self.texture_cap as usize).max(1)
    }

    /// Per-frame mesh load budget as a `usize`, floored at 1.
    pub fn mesh_budget(&self) -> usize {
        (self.mesh_budget as usize).max(1)
    }

    /// Resident-mesh cap as a `usize`, floored at 1.
    pub fn mesh_cap(&self) -> usize {
        (self.mesh_cap as usize).max(1)
    }
}

impl Component for StreamingConfig {
    const NAME: &'static str = "StreamingConfig";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }

    fn from_args(args: Self) -> Self {
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_a_moderate_budget_and_a_full_cap() {
        let c = StreamingConfig::default();
        assert_eq!(c.texture_budget, 4);
        assert_eq!(c.texture_cap, 96);
        assert_eq!(c.budget(), 4);
        assert_eq!(c.cap(), 96);
        assert_eq!(c.mesh_budget(), 4);
        assert_eq!(c.mesh_cap(), 4096);
    }

    #[test]
    fn zero_budget_and_cap_are_floored_at_one() {
        let c = StreamingConfig {
            texture_budget: 0,
            texture_cap: 0,
            mesh_budget: 0,
            mesh_cap: 0,
        };
        // A 0 here would otherwise stall streaming forever.
        assert_eq!(c.budget(), 1);
        assert_eq!(c.cap(), 1);
        assert_eq!(c.mesh_budget(), 1);
        assert_eq!(c.mesh_cap(), 1);
    }

    #[test]
    fn deserialises_from_jsonl_args_with_defaults_for_omitted_fields() {
        let c: StreamingConfig =
            serde_json::from_str(r#"{"texture_budget":2,"mesh_budget":2}"#).expect("parse");
        assert_eq!(c.texture_budget, 2);
        assert_eq!(c.mesh_budget, 2);
        // Omitted fields fall back to the defaults.
        assert_eq!(c.texture_cap, 96);
        assert_eq!(c.mesh_cap, 4096);

        // An empty object is all defaults.
        let c: StreamingConfig = serde_json::from_str("{}").expect("parse");
        assert_eq!(c.texture_budget, 4);
        assert_eq!(c.texture_cap, 96);
        assert_eq!(c.mesh_budget, 4);
        assert_eq!(c.mesh_cap, 4096);
    }

    #[test]
    fn round_trips_through_args() {
        let c = StreamingConfig {
            texture_budget: 7,
            texture_cap: 32,
            mesh_budget: 3,
            mesh_cap: 64,
        };
        let back = StreamingConfig::from_args(c.to_args());
        assert_eq!(back.texture_budget, 7);
        assert_eq!(back.texture_cap, 32);
        assert_eq!(back.mesh_budget, 3);
        assert_eq!(back.mesh_cap, 64);
    }
}
