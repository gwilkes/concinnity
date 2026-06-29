<!-- Auto-generated - do not edit. -->

# StreamingConfig

Enables and tunes asset streaming.

When no `StreamingConfig` is declared, streaming is off and every texture and
mesh is loaded up front. When one is present, textures and static mesh
geometry load in gradually after startup: each frame the nearest not-yet-
loaded items are brought in, up to a per-frame budget, prioritised by camera
distance. Once more than the cap would be loaded at once, the farthest are
dropped to make room.

Texture streaming covers the colour and normal-map textures (each capped
independently via `texture_budget` / `texture_cap`). Mesh streaming covers
static geometry; the skybox, rooms, and moving props always stay loaded.

```jsonl
{"name":"streaming","type":"StreamingConfig","args":{}}
{"name":"streaming_slow","type":"StreamingConfig","args":{"texture_budget":1}}
```

## Parameters

- `texture_budget`: An integer. Maximum number of textures whose load is started per frame, applied independently to the colour and normal-map pools. A low value spreads the cost over more frames. Defaults to `4`.
- `texture_cap`: An integer. Maximum number of textures kept loaded at once, applied independently to the colour and normal-map pools. When exceeded, the farthest-from-camera textures are dropped. Defaults to `96`.
- `mesh_budget`: An integer. Maximum number of mesh regions whose load is started per frame. A low value spreads the cost over more frames. Defaults to `4`.
- `mesh_cap`: An integer. Maximum number of meshes kept loaded at once. When exceeded, the farthest-from-camera meshes are dropped. Defaults to `4096`.
