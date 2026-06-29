<!-- Auto-generated - do not edit. -->

# Model

An ordered list of sub-meshes, each with its own material.

Use via the `model` field on a [Prop](Prop.md) instead of `mesh`. Each
sub-mesh is drawn with its own material, all sharing the prop's transform.

Each `mesh` must name a [Mesh](Mesh.md) or [ProceduralMesh](ProceduralMesh.md)
asset present in the scene. `material` may be empty to use the default
material.

```jsonl
{"name":"crate_body","type":"ProceduralMesh","args":{"generator":"box","half_extents":[0.3,0.3,0.3]}}
{"name":"crate_bands","type":"ProceduralMesh","args":{"generator":"box","half_extents":[0.31,0.04,0.31]}}
{"name":"mat_wood","type":"Material","args":{"albedo":"tex_wood","roughness":0.75,"metallic":0.0}}
{"name":"mat_metal","type":"Material","args":{"albedo":"tex_metal","roughness":0.4,"metallic":1.0}}
{"name":"wooden_crate","type":"Model","args":{"meshes":[
  {"mesh":"crate_body", "material":"mat_wood"},
  {"mesh":"crate_bands","material":"mat_metal"}
]}}
{"name":"crate_a","type":"Prop","args":{"model":"wooden_crate","position":[2.0,0.3,-4.0]}}
{"name":"crate_b","type":"Prop","args":{"model":"wooden_crate","position":[-1.5,0.3,-6.0],"rotation_deg":[0,45,0]}}
```

## Parameters

- `meshes`: An array of [SubMeshRef](SubMeshRef.md) objects. Ordered list of sub-meshes that make up this model.
