<!-- Auto-generated - do not edit. -->

# Material

A Material bundles the surface parameters that control how a [Prop](Prop.md) is
lit and shaded.

Reference it from a [Prop](Prop.md)'s `material` field. The `material` field takes
precedence over the older `texture` field.

```jsonl
{"name":"mat_brick","type":"Material","args":{"albedo":"tex_brick","roughness":0.85,"metallic":0.0}}
{"name":"mat_floor","type":"Material","args":{"albedo":"tex_wood","roughness":0.6,"metallic":0.0}}
{"name":"mat_metal","type":"Material","args":{"albedo":"tex_metal","roughness":0.3,"metallic":1.0}}
{"name":"mat_glow","type":"Material","args":{"albedo":"tex_plaster","roughness":0.9,"emissive_factor":[0.5,0.3,0.0]}}

// Prop referencing a material:
{"name":"crate","type":"Prop","args":{"mesh":"box_mesh","material":"mat_brick","position":[2.0,0.4,-3.0]}}
```

## Parameters

- `albedo`: A string. The [Texture](Texture.md) asset used as the base colour (albedo) map. Optional.
- `normal_map`: A string. The [Texture](Texture.md) asset used as a tangent-space normal map. Optional.
- `emissive_map`: A string. The [Texture](Texture.md) asset used as an emissive map. Multiplied by `emissive_factor` to drive the glow; when omitted, only the scalar `emissive_factor` is used. Pair a textured emissive with an `emissive_factor` above 1 to make the bright parts bloom.
- `orm_map`: A string. The [Texture](Texture.md) asset used as a packed surface map: green = roughness, blue = metalness. When present it overrides the scalar `roughness` and `metallic` per-texel; when omitted those scalars are used. The red channel is reserved and not read as ambient occlusion: packed maps in the wild (glTF metallic-roughness, FBX specular maps) leave red empty, so treating it as occlusion would darken indirect light to black. Ambient occlusion comes from the screen-space pass.
- `roughness`: A float. Perceptual roughness in [0, 1]. 0 = mirror, 1 = fully diffuse. Controls the width of the specular highlight. Defaults to `0.8`.
- `metallic`: A float. Metallic factor in [0, 1]. 0 = dielectric (plastic/stone), 1 = metal. Metallic surfaces tint their reflections with the albedo colour and show almost no diffuse; dielectrics keep a neutral, dim reflection. Defaults to `0.0`.
- `tint`: An array of 3 floats. Linear-space RGB multiplier applied to the albedo sample. Useful for tinting a shared texture without a separate asset (e.g. coloured brick). Defaults to `[1.0, 1.0, 1.0]`.
- `emissive_factor`: An array of 3 floats. Additive emission colour in linear space. Non-zero values make the surface appear to glow independently of the scene lighting. Defaults to `[0.0, 0.0, 0.0]`.
- `macro_variation`: A float. Macro-variation strength in [0, 1]. When non-zero, a large-scale, world-space noise modulates the albedo so a tiled texture on a big surface (terrain, floors) stops reading as an obvious repeating grid. 0 disables it. Defaults to `0.0`.
- `terrain_blend`: A float. Terrain-shading blend in [0, 1]. When non-zero, the albedo and normal are sampled by a world-space projection blended from the three world axes (instead of a single UV lookup), and the surface shifts toward a darker rocky tint on steep slopes. This removes the obvious UV-stretch banding that heightfield ground shows when stretched across a big mesh, and gives "grass on top, rock on the cliffs" variation for free. 0 disables it. Defaults to `0.0`.
- `albedo_secondary`: A string. Optional second albedo [Texture](Texture.md) for the slope-based terrain blend. When present, the steep / cliff regions sample this texture and blend with the primary `albedo` over the flat regions, using the surface's up-facing component (softened by a per-pixel noise so the transition doesn't read as a clean line). Without it, a rocky-tint multiplier is applied to the primary texture instead. Only used when `terrain_blend > 0`.
- `normal_secondary`: A string. Tangent-space normal map paired with `albedo_secondary`. Only used when both that field and `terrain_blend` are set. Optional.
- `secondary_blend_sharpness`: A float. Sharpness of the slope-based blend in [0, 1]. 0 = wide soft gradient between the two layers; 1 = nearly hard cliff edge. Default `0.5` matches the "smooth but visible" transition AAA terrain materials typically tune to.
- `opacity`: A float. Surface opacity in [0, 1]. 1 = fully opaque (the default). Only meaningful when `transparent` is set: it drives how much of the scene behind the surface shows through the glass.
- `transparent`: A boolean. When true, the surface is a translucent dielectric (glass): it renders in the engine's transparent pass instead of the opaque pass, refracting and reflecting the scene rather than writing solid colour + depth. The importer sets this for materials it detects as glass; authored materials can opt in directly. Defaults to false (opaque).
- `see_through`: A boolean. When true, the glass is rendered as genuinely see-through: the scene behind it shows through with a sharp per-pixel reflection (requires a ray-tracing-capable GPU). When false (the default), a `transparent` surface still renders as low-roughness reflective glass that hides whatever is behind it. See-through only looks right when the space behind the glass is actually modelled, so it is opt-in per material. Setting it implies `transparent`.
