<!-- Auto-generated - do not edit. -->

# GlassPanel

A flat translucent panel of coloured glass. A fixed-orientation rectangular
quad that refracts and tints the scene behind it and brightens the
grazing-angle rim with a Fresnel highlight.

Unlike [WaterSurface](WaterSurface.md) it has no animation, no surface
displacement, and no depth-based colour. It's a simple building block for
translucent surfaces such as windows, ice, holograms, or force fields.

The panel is positioned by `centre`, oriented by `normal` (the facing
direction), and sized by `half_size` (half-width along the panel's tangent,
half-height along its bitangent).

```jsonl
{"name":"window","type":"GlassPanel","args":{
  "centre":[0.0,2.0,-3.0],
  "normal":[0.0,0.0,1.0],
  "half_size":[2.0,1.5],
  "tint":[0.6,0.85,0.9],
  "opacity":0.45,
  "refraction_strength":0.04,
  "fresnel_power":4.0
}}
```

## Parameters

- `centre`: An array of 3 floats. World-space position of the panel's centre. Defaults to `[0.0, 1.0, 0.0]`.
- `normal`: An array of 3 floats. Facing direction of the panel. Normalised on load; defaults to +Z when degenerate.
- `half_size`: An array of 2 floats. Half-width and half-height of the panel, in world units. Defaults to `[1.0, 1.0]`.
- `tint`: An array of 3 floats. Linear-space RGB colour the glass tints the scene behind it. Defaults to `[0.7, 0.85, 0.95]`.
- `opacity`: A float. How opaque the glass is, in [0, 1]. 0 = clear, 1 = fully opaque tint. Defaults to `0.5`.
- `refraction_strength`: A float. How strongly the glass bends the view of what's behind it. 0 = no refraction. Defaults to `0.04`.
- `fresnel_power`: A float. Sharpness of the grazing-angle rim highlight. Higher values confine the brightening to steeper viewing angles. Defaults to `4.0`.
- `visible`: A boolean. When false the panel is skipped each frame. Defaults to `true`.
