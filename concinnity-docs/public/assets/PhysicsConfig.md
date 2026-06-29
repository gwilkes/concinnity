<!-- Auto-generated - do not edit. -->

# PhysicsConfig

Configures the world's physics floor / terrain.

Optional: a world with physics bodies but no `PhysicsConfig` simulates over a
flat floor at Y = 0. Physics runs whenever the world declares a
`PhysicsConfig`, a [RigidBody](RigidBody.md), or a [PropBody](PropBody.md).
Declare a `PhysicsConfig` to put bodies on terrain or a non-zero floor.

For terrain-based outdoor scenes the terrain parameters must match the
terrain mesh exactly.

```jsonl
// Indoor (flat floor): no PhysicsConfig needed, just declare bodies.

// Outdoor (heightfield terrain):
{"name":"physics","type":"PhysicsConfig","args":{"terrain_mesh":"ground_heightfield_mesh","terrain_offset_y":-0.5}}
```

## Parameters

- `floor_y`: A float. Y coordinate of the floor. When left at 0.0 it is auto-detected from the camera; set it explicitly to override.
- `terrain_half_width`: A float. Half-width of the terrain mesh along X. Must match the terrain mesh. Leave at 0.0 (with `terrain_subdivisions` = 0) for flat-floor scenes.
- `terrain_half_depth`: A float. Half-depth of the terrain mesh along Z. Must match the terrain mesh.
- `terrain_subdivisions`: An integer. Subdivision count of the terrain mesh. When 0, a flat floor at Y = 0 is used instead of a heightfield.
- `terrain_amplitude`: A float. Height variation of the terrain mesh. Must match the terrain mesh.
- `terrain_offset_y`: A float. World-space Y offset of the terrain: the height of the prop that renders the terrain mesh. Leave at 0.0 when the terrain sits at the origin.
- `terrain_mesh`: A string. Name of a [ProceduralMesh](ProceduralMesh.md) with `generator: "heightfield"`. When set, the physics surface is built from that mesh's source image so props rest on the visible terrain. Takes precedence over the `terrain_*` values above. Optional.
