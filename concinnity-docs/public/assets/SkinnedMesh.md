<!-- Auto-generated - do not edit. -->

# SkinnedMesh

A skeletally animated mesh placed directly in the world.

Unlike a [Mesh](Mesh.md), a `SkinnedMesh` carries its own world transform and a
`skeleton` (a joint hierarchy with a bind pose). Each vertex is bound to up
to four joints; an [Animation](Animation.md) targeting this mesh deforms it at
runtime. With no animation the mesh renders in its bind pose.

The geometry + skeleton may be authored inline (`vertices` / `indices` /
`skeleton`) or imported from a binary glTF file with `source`. Only the
`.glb` container is supported, and only the mesh + skeleton bind pose are
imported (glTF animations are not yet brought in).

The `skeleton` (joint hierarchy and bind pose) is provided as an arg
(authored inline alongside `vertices`/`indices`, or filled in from the
imported `.glb`) and is baked into the mesh at build time.

Normals and tangents are computed automatically at build time. Do not
supply them.

```jsonl
{"name":"flag","type":"SkinnedMesh","args":{"position":[0,1,0],"material":"mat_cloth","skeleton":[{"parent":-1},{"parent":0,"translation":[0,1,0]}],"vertices":[{"pos":[0,0,0],"joints":[0,0,0,0],"weights":[1,0,0,0]}],"indices":[0,0,0]}}
{"name":"hero","type":"SkinnedMesh","args":{"source":"models/hero.glb","position":[0,0,0],"material":"mat_skin"}}
```

## Parameters

- `source`: A string. Optional path to a `.glb` file. When set, the build imports `vertices` / `indices` / `skeleton` from it; an inline-authored mesh leaves this empty.
- `vertices`: An array of [SkinnedVertexData](SkinnedVertexData.md) objects. Skinned vertex list.
- `indices`: An array of integers. Triangle index list.
- `material`: A string. [Material](Material.md); provides the albedo texture plus lighting parameters. Optional.
- `texture`: A string. [Texture](Texture.md) (older path); ignored when `material` is set. Optional.
- `position`: An array of 3 floats. World-space position.
- `rotation_deg`: An array of 3 floats. World rotation, Euler degrees [pitch, yaw, roll], YXZ order.
- `scale`: An array of 3 floats. World scale.
- `lod_levels`: An integer. Number of level-of-detail versions to generate, including the original. `1` (the default) generates none; values are clamped to `[1, 8]`.
- `lod_distances`: An array of floats. Camera distances at which to switch to each lower-detail version. When non-empty, must have exactly `lod_levels - 1` entries; empty lets the build choose defaults.
- `max_instances`: An integer. How many runtime copies of this mesh may exist at once beyond the authored one. `0` (the default) means the mesh is not runtime-spawnable. A non-zero value pre-reserves that many extra instance slots at load: the engine appends that many hidden bind-pose copies to the skinned geometry so a runtime spawn can claim one without growing any GPU buffer, and a despawn returns it to the pool. Spawns past the reserve are dropped (a warning is logged). Capped at 4096.
