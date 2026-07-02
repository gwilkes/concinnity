<!-- Auto-generated - do not edit. -->

# ProceduralMesh

Geometry built by a named generator at compile time. Use for standard shapes.

For custom / hand-authored geometry use [Mesh](Mesh.md) instead.

**Built-in generators:**

```jsonl
{"name":"room_mesh","type":"ProceduralMesh","args":{"generator":"room","half_width":16.0,"half_depth":20.0,"ceiling_height":3.5}}
{"name":"box_mesh","type":"ProceduralMesh","args":{"generator":"box","half_extents":[0.4,0.4,0.4]}}
{"name":"column_mesh","type":"ProceduralMesh","args":{"generator":"cylinder","radius":0.18,"height":3.4,"segments":14}}
{"name":"sphere_mesh","type":"ProceduralMesh","args":{"generator":"sphere","radius":0.5,"rings":16,"segments":16}}
{"name":"terrain_mesh","type":"ProceduralMesh","args":{"generator":"terrain","half_width":64.0,"half_depth":64.0,"subdivisions":64,"amplitude":4.0}}
{"name":"alpine_mesh","type":"ProceduralMesh","args":{"generator":"heightfield","half_width":64.0,"half_depth":64.0,"subdivisions":128,"source":"../concinnity-infra/assets/heightmaps/alpine_512.png","elevation_max":20.0}}
{"name":"sky_mesh","type":"ProceduralMesh","args":{"generator":"skybox","size":490.0}}
{"name":"plus_mesh","type":"ProceduralMesh","args":{"generator":"extrude","profile":[[-1,-3],[1,-3],[1,-1],[3,-1],[3,1],[1,1],[1,3],[-1,3],[-1,1],[-3,1],[-3,-1],[-1,-1]],"height":0.5,"corner_radius":0.2}}
```

## Parameters

- `generator`: A string. Built-in generator name (required), e.g. `room`, `box`, `cylinder`, `sphere`, `terrain`, `heightfield`, `skybox`, or `extrude`.
- `half_width`: A float. Half-width along X (room / box / plane / terrain), in world units. Defaults to `8.0`.
- `half_depth`: A float. Half-depth along Z (room / box / plane / terrain), in world units. Defaults to `10.0`.
- `ceiling_height`: A float. Ceiling height for the `room` generator, in world units. Defaults to `3.5`.
- `half_extents`: An array of 3 floats. Half-extents `[x, y, z]` for the `box` generator. Optional.
- `radius`: A float. Radius for the `cylinder` and `sphere` generators. Optional.
- `height`: A float. Height for the `cylinder` and `extrude` generators. Optional.
- `segments`: An integer. Number of radial segments around the `cylinder` and `sphere` generators. Optional.
- `rings`: An integer. Number of horizontal rings on the `sphere` generator. Optional.
- `subdivisions`: An integer. Grid subdivisions for the `terrain` and `heightfield` generators. Higher is more detailed. Optional.
- `amplitude`: A float. Maximum height variation for the `terrain` generator, in world units. Optional.
- `source`: A string. Path to a grayscale heightmap image for the `heightfield` generator. Optional.
- `elevation_min`: A float. Height mapped to black pixels in the `heightfield` source, in world units. Optional.
- `elevation_max`: A float. Height mapped to white pixels in the `heightfield` source, in world units. Optional.
- `size`: A float. Half-extent on all axes for the `skybox` generator, in world units. Keep it below the camera's `far` plane so the sky is not clipped. Optional.
- `profile`: An array of arrays of 2 floats. 2D outline `[[x, z], ...]` extruded by the `extrude` generator. Optional.
- `corner_radius`: A float. Corner-rounding radius for the `extrude` generator. 0 keeps sharp corners. Optional.
- `corner_segments`: An integer. Number of segments used to round each corner in the `extrude` generator. Optional.
- `lod_levels`: An integer. Number of level-of-detail versions to generate, including the original. `1` (the default) generates none; values are clamped to `[1, 8]`.
- `lod_distances`: An array of floats. Camera distances at which to switch to each lower-detail version; length should be `lod_levels - 1`. Empty lets the build choose defaults.
