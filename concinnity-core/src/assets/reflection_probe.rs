// src/assets/reflection_probe.rs

use crate::ecs::{AssetOrigin, Component};

/// A localized reflection probe. The renderer captures the surrounding scene
/// into a cubemap from `position` and uses it for the specular reflection on
/// glossy surfaces within the influence box (`position` plus or minus
/// `half_extents`). The box is also the parallax-correction volume, so a
/// reflection stays anchored to the surrounding geometry as the camera moves.
///
/// Place several across a level so reflections stay accurate as a first-person
/// camera moves between areas (a room, a courtyard, a corridor): each surface
/// uses the probe whose box it sits deepest inside, and cross-fades into the
/// neighbouring box near a shared boundary so reflections don't pop as the camera
/// crosses between them. When a world declares no `ReflectionProbe`, the renderer
/// auto-seeds a small grid of probes from the scene bounds, so existing scenes
/// still get local reflections without authoring.
///
/// Reflections are most accurate near `position`; a tighter box around a
/// distinct space (a room) parallax-corrects better than one large box. Boxes may
/// overlap freely: a surface inside several boxes blends all of them, so reflections
/// cross-fade smoothly as the camera moves between probes.
///
/// ```jsonl
/// {"name":"lobby_probe","type":"ReflectionProbe","args":{"position":[0.0,1.7,0.0],"half_extents":[8.0,4.0,8.0]}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ReflectionProbe {
    /// World-space capture point the cubemap is rendered from. Put it at roughly
    /// eye height in open space (not inside geometry) for the area it serves.
    pub position: [f32; 3],
    /// Half-size of the influence box around `position`, per axis. A surface
    /// inside `position` plus or minus `half_extents` may select this probe, and
    /// the box is the parallax-correction volume. Make it span the local space
    /// the probe represents (e.g. a room's walls).
    pub half_extents: [f32; 3],
}

impl Default for ReflectionProbe {
    fn default() -> Self {
        Self {
            position: [0.0, 1.7, 0.0],
            half_extents: [10.0, 5.0, 10.0],
        }
    }
}

impl Component for ReflectionProbe {
    const NAME: &'static str = "ReflectionProbe";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        // Half-extents are sizes: keep them non-negative so the influence box is
        // never inverted.
        for e in &mut args.half_extents {
            *e = e.max(0.0);
        }
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}
