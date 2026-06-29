<!-- Auto-generated - do not edit. -->

# VoxelChunk

A voxel grid that compiles into a single mesh.

A dense grid of blocks compiled into a single mesh at build time. Use one
chunk per region of a voxel/Minecraft-style world; reference it from a
[Prop](Prop.md)'s `mesh` field. Hidden faces between two solid blocks are
dropped, so a fully filled chunk contributes zero triangles to its interior.

The palette must contain at least one entry whose [BlockType](BlockType.md) has
`solid: false` (typically named `air`); cells whose palette entry is
non-solid emit no faces. Faces are only emitted between a solid block and
either an empty neighbour or the outside of the chunk.

```jsonl
{"name":"air","type":"BlockType","args":{"solid":false}}
{"name":"stone","type":"BlockType","args":{"uv_min":[0,0],"uv_max":[1,1]}}
{"name":"my_chunk","type":"VoxelChunk","args":{
  "palette":["air","stone"],
  "dim":[2,1,1],
  "blocks":[1,1]
}}
{"name":"chunk_prop","type":"Prop","args":{"mesh":"my_chunk","material":"mat_stone"}}
```

## Parameters

- `palette`: An array of strings. [BlockType](BlockType.md) asset names. `blocks[i]` is an index into this list.
- `dim`: An array of 3 integers. Chunk dimensions `[dx, dy, dz]` in blocks. Defaults to `[0, 0, 0]`.
- `block_size`: A float. World units per block edge. Defaults to `1.0`.
- `blocks`: An array of integers. Flat block array, length `dx*dy*dz`. Index = `x + y*dx + z*dx*dy`.
- `lod_levels`: An integer. Number of level-of-detail versions to generate, including the original. `1` (the default) generates none.
- `lod_distances`: An array of floats. Camera distances at which to switch to each lower-detail version; empty lets the build choose defaults.
