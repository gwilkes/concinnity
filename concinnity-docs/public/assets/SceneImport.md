<!-- Auto-generated - do not edit. -->

# SceneImport

Imports a 3D scene file as a single declaration.

One `SceneImport` stands in for the whole asset graph a scene file
describes: its [Texture](Texture.md)s, [Material](Material.md)s,
[Mesh](Mesh.md)es, [Model](Model.md)s, and [Prop](Prop.md)s. The build expands the
import into those concrete assets, so `world.jsonl` stays small and
human-editable while the full graph lives in the lock file and compiled
blob. Geometry and texture pixels are never inlined into `world.jsonl`.

Supported `source` formats: `.fbx` and `.glb`.

**Generated names** are prefixed with the import's own asset `name`
(`<name>_mat_0`, `<name>_prim_0`, `<name>_model_0`, ...), so they never
clash with hand-authored assets. Because they only appear in the lock file
and blob, you never reference them by hand.

**Camera:** the import frames a [Camera3D](Camera3D.md) to the scene's bounds
so a freshly imported scene is immediately viewable. It is suppressed when
the world already declares a `Camera3D` (yours wins) or when `emit_camera`
is set to `false`.

```jsonl
{"name":"bistro","type":"SceneImport","args":{"source":"assets/Bistro/BistroExterior.fbx","texture_max_size":512}}
```

## Parameters

- `source`: A string. Path to the scene file, relative to the project root. `.fbx` or `.glb`.
- `texture_max_size`: An integer. Ceiling on the longest edge of each imported texture, in pixels. Large source maps (2K-4K) are box-filtered down so the compiled scene, which stores uncompressed pixels, stays within a sane memory budget. `0` keeps each texture at its source resolution. Defaults to `512`.
- `emissive_map_strength`: A float. Emissive factor applied to a material that carries an emissive map. Scene files often ship a zero emissive factor that would cancel the map, so a textured emissive gets this punchy factor instead. Defaults to `3.0`.
- `emit_camera`: A boolean. Whether to emit a [Camera3D](Camera3D.md) framed to the scene's bounds. Suppressed automatically when the world already declares a `Camera3D`. Defaults to `true`.
