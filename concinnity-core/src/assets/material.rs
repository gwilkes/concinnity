// src/assets/material.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};

/// A Material bundles the surface parameters that control how a [Prop](#prop) is
/// lit and shaded.
///
/// Reference it from a [Prop](#prop)'s `material` field. The `material` field takes
/// precedence over the older `texture` field.
///
/// ```jsonl
/// {"name":"mat_brick","type":"Material","args":{"albedo":"tex_brick","roughness":0.85,"metallic":0.0}}
/// {"name":"mat_floor","type":"Material","args":{"albedo":"tex_wood","roughness":0.6,"metallic":0.0}}
/// {"name":"mat_metal","type":"Material","args":{"albedo":"tex_metal","roughness":0.3,"metallic":1.0}}
/// {"name":"mat_glow","type":"Material","args":{"albedo":"tex_plaster","roughness":0.9,"emissive_factor":[0.5,0.3,0.0]}}
///
/// // Prop referencing a material:
/// {"name":"crate","type":"Prop","args":{"mesh":"box_mesh","material":"mat_brick","position":[2.0,0.4,-3.0]}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Material {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The [Texture](#texture) asset used as the base colour (albedo) map.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub albedo: Option<AssetId>,
    /// The [Texture](#texture) asset used as a tangent-space normal map.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub normal_map: Option<AssetId>,
    /// The [Texture](#texture) asset used as an emissive map. Multiplied by
    /// `emissive_factor` to drive the glow; when omitted, only the scalar
    /// `emissive_factor` is used. Pair a textured emissive with an
    /// `emissive_factor` above 1 to make the bright parts bloom.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub emissive_map: Option<AssetId>,
    /// The [Texture](#texture) asset used as a packed surface map: green =
    /// roughness, blue = metalness. When present it overrides the scalar
    /// `roughness` and `metallic` per-texel; when omitted those scalars are
    /// used. The red channel is reserved and not read as ambient occlusion:
    /// packed maps in the wild (glTF metallic-roughness, FBX specular maps)
    /// leave red empty, so treating it as occlusion would darken indirect
    /// light to black. Ambient occlusion comes from the screen-space pass.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub orm_map: Option<AssetId>,
    /// Perceptual roughness in [0, 1]. 0 = mirror, 1 = fully diffuse.
    /// Controls the width of the specular highlight.
    pub roughness: f32,
    /// Metallic factor in [0, 1]. 0 = dielectric (plastic/stone), 1 = metal.
    /// Metallic surfaces tint their reflections with the albedo colour and show
    /// almost no diffuse; dielectrics keep a neutral, dim reflection.
    pub metallic: f32,
    /// Linear-space RGB multiplier applied to the albedo sample. Useful for
    /// tinting a shared texture without a separate asset (e.g. coloured brick).
    pub tint: [f32; 3],
    /// Additive emission colour in linear space. Non-zero values make the
    /// surface appear to glow independently of the scene lighting.
    pub emissive_factor: [f32; 3],
    /// Macro-variation strength in [0, 1]. When non-zero, a large-scale,
    /// world-space noise modulates the albedo so a tiled texture on a big
    /// surface (terrain, floors) stops reading as an obvious repeating grid.
    /// 0 disables it.
    pub macro_variation: f32,
    /// Terrain-shading blend in [0, 1]. When non-zero, the albedo and normal
    /// are sampled by a world-space projection blended from the three world
    /// axes (instead of a single UV lookup), and the surface shifts toward a
    /// darker rocky tint on steep slopes. This removes the obvious UV-stretch
    /// banding that heightfield ground shows when stretched across a big mesh,
    /// and gives "grass on top, rock on the cliffs" variation for free.
    /// 0 disables it.
    pub terrain_blend: f32,
    /// Optional second albedo [Texture](#texture) for the slope-based terrain
    /// blend. When present, the steep / cliff regions sample this texture and
    /// blend with the primary `albedo` over the flat regions, using the
    /// surface's up-facing component (softened by a per-pixel noise so the
    /// transition doesn't read as a clean line). Without it, a rocky-tint
    /// multiplier is applied to the primary texture instead. Only used when
    /// `terrain_blend > 0`.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub albedo_secondary: Option<AssetId>,
    /// Tangent-space normal map paired with `albedo_secondary`. Only used when
    /// both that field and `terrain_blend` are set.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub normal_secondary: Option<AssetId>,
    /// Sharpness of the slope-based blend in [0, 1]. 0 = wide soft
    /// gradient between the two layers; 1 = nearly hard cliff edge.
    /// Default `0.5` matches the "smooth but visible" transition AAA
    /// terrain materials typically tune to.
    pub secondary_blend_sharpness: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            albedo: None,
            normal_map: None,
            emissive_map: None,
            orm_map: None,
            roughness: 0.8,
            metallic: 0.0,
            tint: [1.0, 1.0, 1.0],
            emissive_factor: [0.0, 0.0, 0.0],
            macro_variation: 0.0,
            terrain_blend: 0.0,
            albedo_secondary: None,
            normal_secondary: None,
            secondary_blend_sharpness: 0.5,
        }
    }
}

impl Component for Material {
    const NAME: &'static str = "Material";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        args.roughness = args.roughness.clamp(0.0, 1.0);
        args.metallic = args.metallic.clamp(0.0, 1.0);
        args.macro_variation = args.macro_variation.clamp(0.0, 1.0);
        args.terrain_blend = args.terrain_blend.clamp(0.0, 1.0);
        args.secondary_blend_sharpness = args.secondary_blend_sharpness.clamp(0.0, 1.0);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::check::cross_reference::CrossReferenced for Material {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();

        let albedo = arg("albedo");
        if !albedo.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: albedo.to_string(),
                error: format!(
                    "Material '{}': albedo texture '{}' not found, add a Texture asset with that name",
                    name, albedo
                ),
            });
        }

        let normal_map = arg("normal_map");
        if !normal_map.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: normal_map.to_string(),
                error: format!(
                    "Material '{}': normal_map texture '{}' not found, add a Texture asset with that name",
                    name, normal_map
                ),
            });
        }

        let emissive_map = arg("emissive_map");
        if !emissive_map.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: emissive_map.to_string(),
                error: format!(
                    "Material '{}': emissive_map texture '{}' not found, add a Texture asset with that name",
                    name, emissive_map
                ),
            });
        }

        let orm_map = arg("orm_map");
        if !orm_map.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: orm_map.to_string(),
                error: format!(
                    "Material '{}': orm_map texture '{}' not found, add a Texture asset with that name",
                    name, orm_map
                ),
            });
        }

        let albedo_secondary = arg("albedo_secondary");
        if !albedo_secondary.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: albedo_secondary.to_string(),
                error: format!(
                    "Material '{}': albedo_secondary texture '{}' not found, add a Texture asset with that name",
                    name, albedo_secondary
                ),
            });
        }

        let normal_secondary = arg("normal_secondary");
        if !normal_secondary.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: normal_secondary.to_string(),
                error: format!(
                    "Material '{}': normal_secondary texture '{}' not found, add a Texture asset with that name",
                    name, normal_secondary
                ),
            });
        }

        refs
    }
}
