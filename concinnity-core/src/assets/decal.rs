// src/assets/decal.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};

/// A projected texture stamped onto whatever scene geometry sits inside the
/// decal's oriented box.
///
/// The decal is a box volume positioned by `position`/`rotation_deg`/`size` in
/// world space. The texture is projected down the box's local +Y axis onto the
/// local X-Z plane and stamped onto the surfaces inside the box; anything
/// outside the box is unaffected. Surfaces near the box's top and bottom faces
/// fade out so the stamp doesn't show a hard edge on a curved surface.
///
/// The defaults orient the decal as a ground stamp: a flat 1×1 m square laid on
/// the world X-Z plane, projecting down from +Y. To stamp a wall, rotate so
/// local +Y points into the surface (e.g. `rotation_deg:[0,0,90]` for a +X
/// wall).
///
/// Decals blend over the lit image without affecting depth, so they layer on
/// top of the surfaces they stamp.
///
/// ```jsonl
/// // ground stamp (1.5 m square, projects down)
/// {"name":"footprint_a","type":"Decal","args":{"texture":"tex_footprint","position":[2.0,0.01,-1.5],"size":[1.5,0.5,1.5]}}
///
/// // wall stamp (rotated so local +Y faces +X, into the wall)
/// {"name":"bullet_hole_a","type":"Decal","args":{"texture":"tex_bullet","position":[3.0,1.6,-2.0],"rotation_deg":[0,0,90],"size":[0.4,0.2,0.4]}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Decal {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The [Texture](#texture) asset projected onto the scene.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub texture: Option<AssetId>,
    /// World-space position of the decal box's centre.
    pub position: [f32; 3],
    /// Euler rotation in degrees [pitch, yaw, roll], YXZ order, same as
    /// [Prop](#prop).
    pub rotation_deg: [f32; 3],
    /// Local-space box extents. Local +Y is the projection axis; the texture
    /// is sampled on the local X-Z plane. A non-positive component disables
    /// the decal.
    pub size: [f32; 3],
    /// Linear-space RGBA tint multiplied with the sampled texture. The alpha
    /// channel scales the final blend, so `[1,1,1,0]` hides the decal.
    pub tint: [f32; 4],
    /// When false the decal is skipped each frame.
    pub visible: bool,
}

impl Default for Decal {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            texture: None,
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            size: [1.0, 1.0, 1.0],
            tint: [1.0, 1.0, 1.0, 1.0],
            visible: true,
        }
    }
}

impl Component for Decal {
    const NAME: &'static str = "Decal";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        // Clamp the alpha to [0, 1] so a stray > 1 doesn't blow out the
        // composite. The size components are left as-authored: a non-positive
        // value silently disables the decal in the gfx-side resolver below.
        args.tint[3] = args.tint[3].clamp(0.0, 1.0);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::check::cross_reference::CrossReferenced for Decal {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();
        let tex = arg("texture");
        if !tex.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: tex.to_string(),
                error: format!(
                    "Decal '{}': texture '{}' not found, add a Texture asset with that name",
                    name, tex
                ),
            });
        }
        refs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_with_defaults() {
        let d: Decal = serde_json::from_str("{}").unwrap();
        assert_eq!(d.position, [0.0, 0.0, 0.0]);
        assert_eq!(d.size, [1.0, 1.0, 1.0]);
        assert_eq!(d.tint, [1.0, 1.0, 1.0, 1.0]);
        assert!(d.visible);
        assert!(d.texture.is_none());
    }

    #[test]
    fn deserialises_with_all_fields() {
        let json = r#"{
            "texture":"tex_bullet",
            "position":[1.0,2.0,3.0],
            "rotation_deg":[0,90,0],
            "size":[0.4,0.2,0.4],
            "tint":[0.9,0.2,0.1,0.8],
            "visible":false
        }"#;
        let d: Decal = serde_json::from_str(json).unwrap();
        assert_eq!(d.position, [1.0, 2.0, 3.0]);
        assert_eq!(d.rotation_deg, [0.0, 90.0, 0.0]);
        assert_eq!(d.size, [0.4, 0.2, 0.4]);
        assert_eq!(d.tint, [0.9, 0.2, 0.1, 0.8]);
        assert!(!d.visible);
        assert!(d.texture.is_some());
    }

    #[test]
    fn clamps_alpha_through_from_args() {
        let json = r#"{"tint":[1,1,1,5.0]}"#;
        let parsed: Decal = serde_json::from_str(json).unwrap();
        let normalised = Decal::from_args(parsed);
        assert_eq!(normalised.tint[3], 1.0);

        let json = r#"{"tint":[1,1,1,-0.5]}"#;
        let parsed: Decal = serde_json::from_str(json).unwrap();
        let normalised = Decal::from_args(parsed);
        assert_eq!(normalised.tint[3], 0.0);
    }
}
