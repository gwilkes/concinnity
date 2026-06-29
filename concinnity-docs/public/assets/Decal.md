<!-- Auto-generated - do not edit. -->

# Decal

A projected texture stamped onto whatever scene geometry sits inside the
decal's oriented box.

The decal is a box volume positioned by `position`/`rotation_deg`/`size` in
world space. The texture is projected down the box's local +Y axis onto the
local X-Z plane and stamped onto the surfaces inside the box; anything
outside the box is unaffected. Surfaces near the box's top and bottom faces
fade out so the stamp doesn't show a hard edge on a curved surface.

The defaults orient the decal as a ground stamp: a flat 1×1 m square laid on
the world X-Z plane, projecting down from +Y. To stamp a wall, rotate so
local +Y points into the surface (e.g. `rotation_deg:[0,0,90]` for a +X
wall).

Decals blend over the lit image without affecting depth, so they layer on
top of the surfaces they stamp.

```jsonl
// ground stamp (1.5 m square, projects down)
{"name":"footprint_a","type":"Decal","args":{"texture":"tex_footprint","position":[2.0,0.01,-1.5],"size":[1.5,0.5,1.5]}}

// wall stamp (rotated so local +Y faces +X, into the wall)
{"name":"bullet_hole_a","type":"Decal","args":{"texture":"tex_bullet","position":[3.0,1.6,-2.0],"rotation_deg":[0,0,90],"size":[0.4,0.2,0.4]}}
```

## Parameters

- `texture`: A string. The [Texture](Texture.md) asset projected onto the scene. Optional.
- `position`: An array of 3 floats. World-space position of the decal box's centre. Defaults to `[0.0, 0.0, 0.0]`.
- `rotation_deg`: An array of 3 floats. Euler rotation in degrees [pitch, yaw, roll], YXZ order, same as [Prop](Prop.md). Defaults to `[0.0, 0.0, 0.0]`.
- `size`: An array of 3 floats. Local-space box extents. Local +Y is the projection axis; the texture is sampled on the local X-Z plane. A non-positive component disables the decal. Defaults to `[1.0, 1.0, 1.0]`.
- `tint`: An array of 4 floats. Linear-space RGBA tint multiplied with the sampled texture. The alpha channel scales the final blend, so `[1,1,1,0]` hides the decal. Defaults to `[1.0, 1.0, 1.0, 1.0]`.
- `visible`: A boolean. When false the decal is skipped each frame. Defaults to `true`.
