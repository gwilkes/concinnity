<!-- Auto-generated - do not edit. -->

# ShaderStage

Declares a compiled shader stage.

**Vertex and fragment stages are required for anything to render.** The
shadow pass is engine-internal (no `ShaderStage` of its own); enable or
size it with `shadow_map_size` in [GraphicsConfig](GraphicsConfig.md).

Provide either `source` (single platform) or `sources` (multi-platform). When both are
present, `sources` takes priority for the current platform.

**Platform keys:** `"metal"` (macOS), `"hlsl"` (Windows), `"glsl"` (Linux/Vulkan).

**Bundled shaders:**

- `"default.metal"` / `"default_vert.hlsl"` / `"default_frag.hlsl"`: standard diffuse/specular lighting.

The engine-internal shadow map covers a ±20 m world-space region centred at
the origin with 80 m depth. For larger scenes, increase `shadow_map_size` in
[GraphicsConfig](GraphicsConfig.md) to maintain resolution.

```jsonl
// Multi-platform standard scene:
{"name":"vert","type":"ShaderStage","args":{"kind":"vertex","sources":{"metal":"default.metal","hlsl":"default_vert.hlsl"}}}
{"name":"frag","type":"ShaderStage","args":{"kind":"fragment","sources":{"metal":"default.metal","hlsl":"default_frag.hlsl"}}}

// Single-platform (macOS only):
{"name":"vert","type":"ShaderStage","args":{"kind":"vertex","source":"default.metal"}}
```

**Custom shader vertex layout**: the engine always supplies vertices with 5
attributes at a fixed 56-byte stride. Any custom `.metal` shader **must** declare
`struct Vertex` exactly as shown below: wrong attribute indices cause tangent
data to be read as vertex colour, producing red/green/blue geometry:

```metal
struct Vertex {
    float3 pos     [[attribute(0)]];  // offset  0
    float3 normal  [[attribute(1)]];  // offset 12
    float3 tangent [[attribute(2)]];  // offset 24
    float3 color   [[attribute(3)]];  // offset 36
    float2 uv      [[attribute(4)]];  // offset 48
};
```

Buffer and texture bindings that must match:

```metal
struct DirectionalLightData {
    packed_float3 direction;
    float         intensity;
    packed_float3 color;
    float         _pad;
};

struct PointLightData {
    packed_float3 position;
    float         range;
    packed_float3 color;
    float         intensity;
};

struct ShadowUniforms {
    float4x4 light_vp;
};
```

## Parameters

- `kind`: A string (see [ShaderKind](ShaderKind.md)). Which stage this shader drives.
- `source`: A string. Single-platform source path; used when `sources` is absent or lacks the current platform key.
- `sources`: An object. Per-platform source paths keyed by `"metal"`, `"hlsl"`, or `"glsl"`. Takes priority over `source`. Optional.
