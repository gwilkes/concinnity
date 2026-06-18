// Unified G-buffer pre-pass fragment shader. One traversal writes the three
// targets every screen-space consumer reads:
//   SV_TARGET0 RGBA16F: rgb = unit view-space normal, a = positive linear view
//              depth (-view_z); alpha 0 marks "no geometry" (cleared background).
//   SV_TARGET1 R8:      perceptual roughness (1.0 = fully rough background).
//   SV_TARGET2 RG16F:   screen-space motion (prev_uv - cur_uv), derived from the
//              un-jittered clip positions so projection jitter never leaks into
//              the motion vector.
// Fuses ssr_prepass_frag + velocity_frag.

cbuffer GbMat : register(b0)
{
    float roughness;
    float _pad0;
    float _pad1;
    float _pad2;
}

struct PsIn
{
    float4 sv_pos      : SV_POSITION;
    float3 view_normal : TEXCOORD0;
    float  view_depth  : TEXCOORD1;
    float4 cur_clip    : TEXCOORD2;
    float4 prev_clip   : TEXCOORD3;
};

struct PsOut
{
    float4 nd    : SV_TARGET0;
    float  rough : SV_TARGET1;
    float2 vel   : SV_TARGET2;
};

PsOut main(PsIn p)
{
    PsOut o;
    o.nd    = float4(normalize(p.view_normal), p.view_depth);
    o.rough = roughness;
    float2 cur_ndc  = p.cur_clip.xy  / p.cur_clip.w;
    float2 prev_ndc = p.prev_clip.xy / p.prev_clip.w;
    float2 cur_uv  = float2(cur_ndc.x  * 0.5 + 0.5, 0.5 - cur_ndc.y  * 0.5);
    float2 prev_uv = float2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    o.vel = prev_uv - cur_uv;
    return o;
}
