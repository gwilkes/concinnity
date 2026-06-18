// Shadow pass D3D12 vertex shader for Concinnity scenes.
//
// Renders scene depth from the first directional light's perspective into one
// slice of a Depth32Float Texture2DArray. The host loops this pass once per
// cascade, pushing `cascade_idx` so the VS picks the right `light_vps` matrix.
// No pixel shader is needed - depth writes are automatic.
//
// Root signature layout (must match create_shadow_root_signature in directx/pipeline.rs):
//   b0 PushConstants : model mat4 + cascade_idx + 3 pad (20 DWORDs)
//   b1 ShadowBlock   : light_vps[4] + cascade_splits

#pragma pack_matrix(column_major)

cbuffer PushConstants : register(b0)
{
    float4x4 model;
    uint cascade_idx;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}

cbuffer ShadowBlock : register(b1)
{
    float4x4 light_vps[4];
    float4   cascade_splits;
}

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
    return mul(light_vps[cascade_idx], mul(model, float4(v.pos, 1.0)));
}
