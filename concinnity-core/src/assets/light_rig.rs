// src/assets/light_rig.rs

use crate::ecs::{AssetOrigin, Component};

/// A named grouping of lights.
///
/// Use `preset` to expand a built-in setup into named
/// [DirectionalLight](#directionallight)/[PointLight](#pointlight) assets
/// (`<rig_name>_<light_name>`), or declare lights directly and list their names
/// in `lights`.
///
/// **Library presets:**
///
/// ```jsonl
/// // From preset (expands into rig_sun + rig_fill):
/// {"name":"rig","type":"LightRig","args":{"preset":"rig_outdoor_sun_fill"}}
///
/// // Referencing pre-declared lights:
/// {"name":"sun",  "type":"DirectionalLight","args":{"direction":[-0.4,0.7,0.3],"color":[1.0,0.95,0.8],"intensity":1.2}}
/// {"name":"torch","type":"PointLight",      "args":{"position":[3.0,2.0,-5.0],"color":[1.0,0.7,0.3],"intensity":10.0,"range":6.0}}
/// {"name":"rig",  "type":"LightRig","args":{"lights":["sun","torch"]}}
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LightRig {
    /// Name of a built-in or file-backed preset (e.g. "rig_outdoor_sun_fill").
    /// When set, `lights` is ignored.
    pub preset: String,
    /// Names of existing [DirectionalLight](#directionallight) or
    /// [PointLight](#pointlight) assets to include in this rig. Ignored when
    /// `preset` is set.
    pub lights: Vec<String>,
}

impl Component for LightRig {
    const NAME: &'static str = "LightRig";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}
