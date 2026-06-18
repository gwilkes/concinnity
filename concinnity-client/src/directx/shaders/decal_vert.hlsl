#pragma pack_matrix(column_major)

// Projected (deferred) decal pass - vertex shader. Mirrors decal_vertex in
// src/metal/shaders/decal.metal. Rasterises a per-decal unit cube (positions
// in [-0.5, 0.5]^3) under the decal's local→world model matrix and the
// camera VP (jittered when TAA is on so the rasterised pixel grid matches
// the main pass exactly).

cbuffer DecalView : register(b0)
{
    float4x4 vp;
    float4x4 inv_vp;
    float2   viewport;
    float2   _pad;
}

cbuffer DecalParams : register(b1)
{
    float4x4 model;
    float4x4 inv_model;
    float4   tint;
    float    fade_pow;
    float    _p0;
    float    _p1;
    float    _p2;
}

struct VsIn
{
    float3 pos : POSITION;
};

struct VsOut
{
    float4 sv_pos : SV_POSITION;
};

VsOut main(VsIn v)
{
    VsOut o;
    float4 world = mul(model, float4(v.pos, 1.0));
    o.sv_pos     = mul(vp, world);
    return o;
}
