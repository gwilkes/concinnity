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

// Wrap in a column_major-qualified struct: see skinned_vert.hlsl for why.
struct ColMat4 { column_major float4x4 m; };
StructuredBuffer<ColMat4> joint_mats : register(t0);

struct VsIn
{
    float3 pos     : POSITION;
    float3 normal  : NORMAL;
    float3 tangent : TANGENT;
    float3 color   : COLOR;
    float2 uv      : TEXCOORD0;
    uint4  joints  : BLENDINDICES;
    float4 weights : BLENDWEIGHT;
};

float4 main(VsIn v) : SV_POSITION
{
    float4x4 skin = v.weights.x * joint_mats[v.joints.x].m
                  + v.weights.y * joint_mats[v.joints.y].m
                  + v.weights.z * joint_mats[v.joints.z].m
                  + v.weights.w * joint_mats[v.joints.w].m;
    float4 skinned_pos = mul(skin, float4(v.pos, 1.0));
    return mul(light_vps[cascade_idx], mul(model, skinned_pos));
}
