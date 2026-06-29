<!-- Auto-generated - do not edit. -->

# Mesh

Raw geometry. Supply `vertices` and `indices` directly, or import them from
a binary glTF file with `source` + `primitive_index`.

Use when you want full control over shape: custom furniture,
architectural details, signage, or any form a generator cannot
produce. For standard shapes use [ProceduralMesh](ProceduralMesh.md).

Normals and tangents are computed automatically at build time.
**Do not supply normals or tangents.**

**Vertex color:** use `[0.75, 0.74, 0.72]` for a neutral surface that takes
the material albedo, or `[1, 1, 1]` to pass through unmodified.

**Winding:** triangles must be counter-clockwise when viewed from the front.
Reversed winding = invisible face.

## Parameters

- `source`: A string. Optional path to a `.glb` file. When set, the build imports `vertices` / `indices` from it; inline geometry leaves this empty.
- `primitive_index`: An integer. Which primitive (counted across all meshes in the file) to import from `source`. Ignored when `source` is empty. Defaults to `0`.
- `chunk_index`: An integer. Pick a single chunk of an oversized imported primitive. `None` (the default) imports the whole primitive, which is fine whenever its vertex count fits in 16-bit indices; larger primitives are split into chunks on import, one Mesh per chunk.
- `vertices`: An array of [VertexData](VertexData.md) objects. Vertex list.  Each vertex: `{"pos":[x,y,z], "color":[r,g,b], "uv":[u,v]}`.
- `indices`: An array of integers. Triangle index list (16-bit values).
- `lod_levels`: An integer. Number of level-of-detail versions to generate, including the original. `1` (the default) generates none; values are clamped to `[1, 8]`.
- `lod_distances`: An array of floats. Camera distances at which to switch to each lower-detail version. Length should be `lod_levels - 1`; empty lets the build derive a default sequence. The version for index `i` is used at camera distance ≥ `lod_distances[i]`.
