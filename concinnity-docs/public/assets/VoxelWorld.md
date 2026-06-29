<!-- Auto-generated - do not edit. -->

# VoxelWorld

An infinite, procedurally generated voxel world.

Where a [VoxelChunk](VoxelChunk.md) is one authored chunk compiled to a fixed
mesh at build time, a `VoxelWorld` describes an *unbounded* world: chunks are
generated on demand from `seed` as the camera moves and streamed in and out
around it. The grid is infinite on X/Z and a single chunk tall on Y.
Declaring one opts the world into chunk streaming; with no `VoxelWorld`
present nothing changes.

The `palette` lists [BlockType](BlockType.md) assets; the generator uses index
0 as air, index 1 as the surface block, and index 2 (when present) as the
subsurface block. `material` supplies the textures and lighting shared by
every chunk.

```jsonl
{"name":"air","type":"BlockType","args":{"solid":false}}
{"name":"grass","type":"BlockType","args":{"uv_min":[0,0],"uv_max":[1,1]}}
{"name":"stone","type":"BlockType","args":{"uv_min":[0,0],"uv_max":[1,1]}}
{"name":"overworld","type":"VoxelWorld","args":{
  "seed":42,"view_radius":6,"palette":["air","grass","stone"],"material":"mat_ground"
}}
```

## Parameters

- `seed`: An integer. Deterministic terrain seed. The same seed always generates the same world, so a chunk regenerates identically each time it streams back in. Defaults to `0`.
- `chunk_blocks`: An array of 3 integers. Blocks per chunk `[dx, dy, dz]`. Y is the world's fixed vertical extent. Defaults to `[16, 24, 16]`.
- `block_size`: A float. World units per block edge. Defaults to `1.0`.
- `view_radius`: An integer. Chunk radius streamed around the camera at full voxel detail. Defaults to `5`.
- `impostor_radius`: An integer. Outer chunk radius streamed as cheap coarse impostors. Chunks farther than `view_radius` but within `impostor_radius` render as a low-detail surface mesh instead of full voxel geometry. `0` (the default) or any value `<= view_radius` disables impostors.
- `impostor_step`: An integer. Coarse-grid step (in blocks) for distant-chunk impostors: the surface is sampled every `impostor_step` blocks. Higher = cheaper and coarser. Defaults to `4`.
- `load_budget`: An integer. Maximum number of chunks generated and loaded per frame. Defaults to `3`.
- `palette`: An array of strings. [BlockType](BlockType.md) asset names. Index 0 is air; 1 is the surface block; 2, when present, is the subsurface block.
- `material`: A string. [Material](Material.md) shared by every chunk: textures and lighting. Optional.
