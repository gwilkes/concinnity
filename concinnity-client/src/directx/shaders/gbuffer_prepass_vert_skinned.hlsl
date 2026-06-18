#pragma pack_matrix(column_major)

// Skinned sibling of gbuffer_prepass_vert. 4-influence linear blend skinning
// using the current and previous frame's joint matrices, so per-vertex skinned
// motion produces a correct screen-space motion vector. The model matrix is the
// same for cur and prev (skinned meshes are self-placing); the cbuffer shape
// matches the static path for layout parity. Fuses ssr_prepass_vert_skinned +
// velocity_vert_skinned.

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

// Wrap in a column_major-qualified struct: see skinned_vert.hlsl for why.
struct ColMat4 { column_major float4x4 m; };
StructuredBuffer<ColMat4> cur_joint_mats  : register(t0);
StructuredBuffer<ColMat4> prev_joint_mats : register(t1);

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
    float4x4 cur_skin  = v.weights.x * cur_joint_mats[v.joints.x].m
                       + v.weights.y * cur_joint_mats[v.joints.y].m
                       + v.weights.z * cur_joint_mats[v.joints.z].m
                       + v.weights.w * cur_joint_mats[v.joints.w].m;
    float4x4 prev_skin = v.weights.x * prev_joint_mats[v.joints.x].m
                       + v.weights.y * prev_joint_mats[v.joints.y].m
                       + v.weights.z * prev_joint_mats[v.joints.z].m
                       + v.weights.w * prev_joint_mats[v.joints.w].m;
    float4 cur_world      = mul(cur_model,  mul(cur_skin,  float4(v.pos, 1.0)));
    float4 prev_world     = mul(prev_model, mul(prev_skin, float4(v.pos, 1.0)));
    float3 skinned_normal = mul((float3x3)cur_skin, v.normal);
    float4 view_pos       = mul(view_mat, cur_world);
    o.sv_pos        = mul(jittered_vp, cur_world);
    float3 world_n  = normalize(mul((float3x3)cur_model, skinned_normal));
    o.view_normal   = mul((float3x3)view_mat, world_n);
    o.view_depth    = -view_pos.z;
    o.cur_clip      = mul(cur_vp,  cur_world);
    o.prev_clip     = mul(prev_vp, prev_world);
    return o;
}
