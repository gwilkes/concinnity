<!-- Auto-generated - do not edit. -->

# Room

A self-contained room (floor, ceiling, four walls), with optional texturing.

Prefer `Room` over a [ProceduralMesh](ProceduralMesh.md) (generator `"room"`) +
[Prop](Prop.md) pair for a shorter declaration. The room is placed at the world
origin.

Dimensions can be given as `size: [width, depth, height]` (full extents) or
as `half_width`, `half_depth`, and `ceiling_height` individually.

`texture`, `wall_texture`, `floor_texture`, and `ceiling_texture` are checked
in that order; the first set value wins. Generator names such as `"brick"` or
`"concrete"` resolve to a matching [Texture](Texture.md) at build time.

```jsonl
{"name":"room","type":"Room","args":{"size":[16,20,3.5],"texture":"tex_plaster"}}
```

## Parameters

- `half_width`: A float. Half the room's width along X, in world units. Ignored when `size` is set. Defaults to `8.0`.
- `half_depth`: A float. Half the room's depth along Z, in world units. Ignored when `size` is set. Defaults to `10.0`.
- `ceiling_height`: A float. Floor-to-ceiling height in world units. Ignored when `size` is set. Defaults to `3.5`.
- `size`: An array of 3 floats. Shorthand for the full dimensions `[width, depth, height]`. When set, it overrides `half_width`, `half_depth`, and `ceiling_height`. Optional.
- `texture`: A string. [Texture](Texture.md) applied to all surfaces. Falls back to `wall_texture` when unset. Generator names such as `"brick"` or `"concrete"` resolve to a matching texture at build time.
- `wall_texture`: A string. [Texture](Texture.md) for the walls. Currently all surfaces share one texture; per-surface texturing is reserved for a future update. Optional.
- `floor_texture`: A string. [Texture](Texture.md) for the floor (see `wall_texture`). Optional.
- `ceiling_texture`: A string. [Texture](Texture.md) for the ceiling (see `wall_texture`). Optional.
- `lod_levels`: An integer. Number of level-of-detail versions to generate, including the original. `1` (the default) generates no alternates.
- `lod_distances`: An array of floats. Camera distances at which to switch to each lower-detail version. Empty lets the build choose defaults.
