<!-- Auto-generated - do not edit. -->

# InstancedProp

A single mesh + material drawn at many world-space transforms.

Use for foliage, debris, projectiles, or any content that repeats the same
shape with varied placement. Each instance gets its own world transform and
culling without the overhead of declaring many separate [Prop](Prop.md)s.

Each `instances` entry has the shape `{"position":[x,y,z], "rotation_deg":[p,y,r], "scale":[sx,sy,sz]}`.
`rotation_deg` and `scale` may be omitted (defaults `[0,0,0]` and `[1,1,1]`).

```jsonl
{"name":"rock_mesh","type":"ProceduralMesh","args":{"generator":"sphere","radius":0.4,"rings":8,"segments":10}}
{"name":"mat_stone","type":"Material","args":{"albedo":"tex_stone","roughness":0.9}}
{"name":"rocks","type":"InstancedProp","args":{
  "mesh":"rock_mesh",
  "material":"mat_stone",
  "cull_distance":80.0,
  "instances":[
    {"position":[ 2.0, 0.4, -3.0]},
    {"position":[-5.0, 0.4,  1.0], "rotation_deg":[0, 45, 0]},
    {"position":[ 4.0, 0.4,  7.0], "scale":[1.5, 1.5, 1.5]}
  ]
}}
```

## Parameters

- `mesh`: A string. A [Mesh](Mesh.md), [ProceduralMesh](ProceduralMesh.md), [VoxelChunk](VoxelChunk.md), or mesh-kind [File](File.md) asset. Optional.
- `material`: A string. A [Material](Material.md); takes precedence over `texture` when set. Optional.
- `texture`: A string. Older texture-only reference; ignored when `material` is set. Optional.
- `instances`: An array of [InstanceTransform](InstanceTransform.md) objects. Per-instance transforms. Empty list renders nothing.
- `cull_distance`: A float. View-distance cutoff in world units per instance. 0 = always draw. Defaults to `0.0`.
