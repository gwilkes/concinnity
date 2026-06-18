#pragma pack_matrix(column_major)

// Depth-only vertex shader for the GPU-driven shadow pass. Mirrors the bindless
// main VS: the per-command b0 object-id root constant indexes the per-frame
// GpuObjectData buffer for the model matrix, so the CPU never pushes a per-draw
// model. The cascade index is a per-ExecuteIndirect root constant at b2 (one
// indirect draw per cascade), selecting which `light_vps[i]` to project through.
//
// Layout (160 bytes) must match the Rust GpuObjectData in gfx::render_types and
// the bindless fragment shader's struct. The VS only reads `model`, but the full
// layout is required so `objects[object_id]` strides correctly.
struct GpuObjectData
{
    float4x4 model;
    float3 tint;
    float roughness;
    float3 emissive;
    float metallic;
    uint albedo_index;
    uint normal_index;
    float macro_variation;
    float terrain_blend;
    float3 bb_min;
    float cull_distance;
    float3 bb_max;
    float secondary_blend_sharpness;
    uint albedo_secondary_index;
    uint normal_secondary_index;
    uint emissive_map_index;
    uint orm_map_index;
};

cbuffer ObjId : register(b0)
{
    uint object_id;
}

cbuffer ShadowBlock : register(b1)
{
    float4x4 light_vps[4];
    float4   cascade_splits;
}

cbuffer Cascade : register(b2)
{
    uint cascade_idx;
}

StructuredBuffer<GpuObjectData> objects : register(t0);

struct VsIn
{
    float3 pos     : POSITION;
    float3 normal  : NORMAL;
    float3 tangent : TANGENT;
    float3 color   : COLOR;
    float2 uv      : TEXCOORD0;
};

float4 main(VsIn v) : SV_POSITION
{
    float4x4 model = objects[object_id].model;
    float4 world = mul(model, float4(v.pos, 1.0));
    return mul(light_vps[cascade_idx], world);
}
