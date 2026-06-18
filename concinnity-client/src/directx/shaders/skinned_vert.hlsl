#pragma pack_matrix(column_major)

cbuffer PushConstants : register(b0)
{
    float4x4 model;
    float roughness;
    float metallic;
    float _mpad0;
    float _mpad1;
    float3 tint;
    float _mpad2;
    float3 emissive;
    float _mpad3;
}

cbuffer ViewBlock : register(b1)
{
    float4x4 vp;
    float4x4 view_mat;
    float elapsed;
    float _pad0;
    float cam_x;
    float cam_y;
    float cam_z;
    float prefilter_mip_count;
    float _ep0;
    float _ep1;
}

cbuffer ShadowBlock : register(b3)
{
    float4x4 light_vps[4];
    float4   cascade_splits;
}

// Wrapping the matrix in a struct with an explicit `column_major` qualifier
// pins the storage layout regardless of FXC's handling of `pack_matrix` for
// `StructuredBuffer<float4x4>` element-typed matrices (the pragma is honoured
// for struct members but its behaviour for raw element-type matrices in a
// StructuredBuffer is ambiguous in FXC). The Rust upload writes
// `[[f32;4];4]` matrices in column-major order, so this guarantees the
// shader reads them back unchanged.
struct ColMat4 { column_major float4x4 m; };
StructuredBuffer<ColMat4> joint_mats : register(t3);

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
    float4 sv_pos     : SV_POSITION;
    float3 world_pos  : TEXCOORD0;
    float3 normal     : TEXCOORD1;
    float3 tangent    : TEXCOORD2;
    float3 bitangent  : TEXCOORD3;
    float2 uv         : TEXCOORD4;
    float  view_depth : TEXCOORD5;
    float3 color      : TEXCOORD6;
};

VsOut main(VsIn v)
{
    VsOut o;
    // Linear blend skinning: weighted sum of the bound joints' matrices.
    float4x4 skin = v.weights.x * joint_mats[v.joints.x].m
                  + v.weights.y * joint_mats[v.joints.y].m
                  + v.weights.z * joint_mats[v.joints.z].m
                  + v.weights.w * joint_mats[v.joints.w].m;

    float4 skinned_pos     = mul(skin, float4(v.pos, 1.0));
    float3 skinned_normal  = mul((float3x3)skin, v.normal);
    float3 skinned_tangent = mul((float3x3)skin, v.tangent);

    float4 world = mul(model, skinned_pos);
    o.world_pos = world.xyz;

    float3x3 nm = (float3x3)model;
    o.normal    = normalize(mul(nm, skinned_normal));
    o.tangent   = normalize(mul(nm, skinned_tangent));
    o.bitangent = cross(o.normal, o.tangent);

    o.uv    = v.uv;
    o.color = v.color;

    o.view_depth = -mul(view_mat, world).z;
    o.sv_pos     = mul(vp, world);
    return o;
}
