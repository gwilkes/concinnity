#pragma pack_matrix(column_major)

// GPU-instanced sibling of gbuffer_prepass_vert. Per-instance model matrices
// live in a root SRV at t0. Instance transforms are immutable, so cur == prev
// (the motion vector is camera-only). Fuses ssr_prepass_vert_instanced +
// velocity_vert_instanced.

cbuffer GbView : register(b0)
{
    float4x4 jittered_vp;
    float4x4 cur_vp;
    float4x4 prev_vp;
    float4x4 view_mat;
}

// Wrap in a column_major-qualified struct: see skinned_vert.hlsl for why.
struct ColMat4 { column_major float4x4 m; };
StructuredBuffer<ColMat4> instances : register(t0);

struct VsIn
{
    float3 pos     : POSITION;
    float3 normal  : NORMAL;
    float3 tangent : TANGENT;
    float3 color   : COLOR;
    float2 uv      : TEXCOORD0;
};

struct VsOut
{
    float4 sv_pos      : SV_POSITION;
    float3 view_normal : TEXCOORD0;
    float  view_depth  : TEXCOORD1;
    float4 cur_clip    : TEXCOORD2;
    float4 prev_clip   : TEXCOORD3;
};

VsOut main(VsIn v, uint iid : SV_InstanceID)
{
    VsOut o;
    float4x4 model  = instances[iid].m;
    float4 world    = mul(model, float4(v.pos, 1.0));
    float4 view_pos = mul(view_mat, world);
    o.sv_pos        = mul(jittered_vp, world);
    float3 world_n  = normalize(mul((float3x3)model, v.normal));
    o.view_normal   = mul((float3x3)view_mat, world_n);
    o.view_depth    = -view_pos.z;
    o.cur_clip      = mul(cur_vp,  world);
    o.prev_clip     = mul(prev_vp, world);
    return o;
}
