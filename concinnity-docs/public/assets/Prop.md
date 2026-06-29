<!-- Auto-generated - do not edit. -->

# Prop

A scene object: places geometry at a world-space transform.

Reference either a [Model](Model.md) (multi-mesh) or a single
[Mesh](Mesh.md)/[ProceduralMesh](ProceduralMesh.md). `model` takes precedence
when both are set.

```jsonl
// single mesh
{"name":"crate_a","type":"Prop","args":{"mesh":"box_mesh","material":"mat_brick","position":[4.0,0.4,-8.0],"collider":{"shape":"aabb","half_extents":[0.4,0.4,0.4]}}}
{"name":"column_ne","type":"Prop","args":{"mesh":"column_mesh","material":"mat_stone","position":[8.0,1.7,-10.0],"collider":{"shape":"aabb","half_extents":[0.18,1.7,0.18]}}}
{"name":"room_floor","type":"Prop","args":{"mesh":"room_mesh","material":"mat_plaster","position":[0.0,0.0,0.0]}}

// multi-mesh model
{"name":"crate_a","type":"Prop","args":{"model":"wooden_crate","position":[2.0,0.3,-4.0],"collider":{"shape":"aabb","half_extents":[0.3,0.3,0.3]}}}

// parent-child hierarchy: door panel inherits the frame's world transform
{"name":"door_frame","type":"Prop","args":{"model":"wooden_frame","position":[3,0,-2]}}
{"name":"door_panel","type":"Prop","args":{"model":"door","parent":"door_frame","position":[0,0,0.05]}}
```

Rotation notes:
- `rotation_deg[0]` = pitch (tilt forward/back)
- `rotation_deg[1]` = yaw (spin on vertical axis), most common
- `rotation_deg[2]` = roll (tilt side-to-side)

## Parameters

- `model`: A string. A [Model](Model.md) asset. When set, the prop renders all sub-meshes of that model (each with its own material) sharing this prop's transform. Takes precedence over `mesh` and `material`. Optional.
- `mesh`: A string. A [Mesh](Mesh.md) or [ProceduralMesh](ProceduralMesh.md) asset this prop renders. Used when `model` is unset.
- `material`: A string. A [Material](Material.md) to use for this prop. When set it takes precedence over `texture` and provides the albedo texture plus the lighting parameters (roughness, metallic, tint, emissive). Used when `model` is unset.
- `texture`: A string. A [Texture](Texture.md) to use for this prop. Older field: ignored when `material` is set. Unset uses the first declared texture (or a white fallback).
- `position`: An array of 3 floats. World-space position [x, y, z]. Defaults to `[0.0, 0.0, 0.0]`.
- `rotation_deg`: An array of 3 floats. Euler rotation in degrees [pitch, yaw, roll], applied in YXZ order (yaw first so that rotating around the vertical axis is intuitive). Defaults to `[0.0, 0.0, 0.0]`.
- `scale`: An array of 3 floats. Non-uniform scale [x, y, z]. Defaults to [1, 1, 1].
- `collider`: A [PropCollider](PropCollider.md) object. Optional collision volume. When present, the prop blocks the player; when absent the prop is non-solid.
- `interactable`: A boolean. When true, the player can interact with this prop: pressing the interact key (E) while close and facing it triggers its rotation behaviour. Defaults to `false`.
- `pickup`: A boolean. When true, the player can pick up and carry this prop with the interact key (E). A companion [PropBody](PropBody.md) must also be declared so the prop falls correctly after being dropped. Defaults to `false`.
- `parent`: A string. Another [Prop](Prop.md) whose world transform this prop inherits. When set, `position`, `rotation_deg`, and `scale` are relative to the parent's world transform. The parent must be declared in the same world; circular chains are treated as an error. Optional.
- `scene`: A string. [Scene](Scene.md) this prop belongs to. Resolved automatically from the naming convention (a prop named `<scene>_*` belongs to scene `<scene>`); you don't set this directly. `None` means the prop is visible in every scene. Used by [SceneReel](SceneReel.md) for per-scene visibility. Optional.
- `prefab`: A string. Name of a [Prefab](Prefab.md) to instantiate at this prop's transform. When set, it expands into concrete child props and lights, replacing this prop. Cannot be combined with `model` or `mesh`.
- `cull_distance`: A float. Optional view-distance cutoff in world units. When > 0 the prop is hidden once the camera is further than this from it. 0 (default) keeps the prop visible at any distance.
