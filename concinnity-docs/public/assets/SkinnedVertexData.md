<!-- Auto-generated - do not edit. -->

# SkinnedVertexData

One vertex of a skinned mesh. Beyond position / colour / uv it carries up
to four joint bindings: `joints[k]` indexes the skeleton, `weights[k]` is
its blend weight. Weights are normalised at build time.

## Parameters

- `pos`: An array of 3 floats. Vertex position `[x, y, z]` in model space.
- `color`: An array of 3 floats. Vertex colour `[r, g, b]` in [0, 1]. Defaults to white.
- `uv`: An array of 2 floats. Texture coordinates in [0, 1] space. Defaults to [0, 0].
- `joints`: An array of 4 integers. Joint indices this vertex is bound to. Unused slots can be 0.
- `weights`: An array of 4 floats. Blend weights parallel to `joints`. Defaults to fully bound to joint 0.
