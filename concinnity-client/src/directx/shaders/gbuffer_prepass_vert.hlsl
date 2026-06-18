#pragma pack_matrix(column_major)

// Unified G-buffer pre-pass vertex shader (static geometry). One jittered
// traversal feeds the SSR / SSAO / SSGI / TAA readers: rasterise position via
// the jittered VP (matching the main pass coverage), pass through the
// view-space normal + linear view depth for the normal+depth target, and the
// un-jittered current / previous clip positions so the fragment shader can
// derive a jitter-free motion vector. Fuses ssr_prepass_vert + velocity_vert.

cbuffer GbView : register(b0)
{
    float4x4 jittered_vp;
    float4x4 cur_vp;
    float4x4 prev_vp;
    float4x4 view_mat;
}

cbuffer GbModel : register(b1)
{
    float4x4 cur_model;
    float4x4 prev_model;
}

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

VsOut main(VsIn v)
{
    VsOut o;
    float4 cur_world  = mul(cur_model,  float4(v.pos, 1.0));
    float4 prev_world = mul(prev_model, float4(v.pos, 1.0));
    float4 view_pos   = mul(view_mat, cur_world);
    o.sv_pos        = mul(jittered_vp, cur_world);
    float3 world_n  = normalize(mul((float3x3)cur_model, v.normal));
    o.view_normal   = mul((float3x3)view_mat, world_n);
    o.view_depth    = -view_pos.z;
    o.cur_clip      = mul(cur_vp,  cur_world);
    o.prev_clip     = mul(prev_vp, prev_world);
    // Skybox sentinel: pin to the far plane so sky never occludes scene.
    if (v.color.b > 1.5)
    {
        o.sv_pos.z = o.sv_pos.w * (1.0 - 1e-6);
    }
    return o;
}
