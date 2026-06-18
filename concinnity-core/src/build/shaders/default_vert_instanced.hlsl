// GPU-instanced default vertex shader for Concinnity scenes.
//
// Sibling of default_vert.hlsl. Reads per-instance world matrices from a
// structured buffer at t3 instead of the PushConstants `model` field; that
// field is ignored here. Paired with the existing default_frag.hlsl.
//
// Root signature layout (must match the instanced PSO in directx/pipeline.rs):
//   b0 PushConstants : model mat4 + material (28 DWORDs)  -- model unused
//   b1 ViewBlock     : vp mat4, elapsed, cam xyz
//   b3 ShadowBlock   : light_vp mat4
//   t3 InstanceBlock : StructuredBuffer<float4x4> of per-instance world matrices
//
// Input layout matches the standard 56-byte Vertex.

#pragma pack_matrix(column_major)

cbuffer PushConstants : register(b0)
{
    float4x4 model_unused;
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
    float _pad1;
    float _ep0;
    float _ep1;
}

cbuffer ShadowBlock : register(b3)
{
    float4x4 light_vps[4];
    float4   cascade_splits;
}

// FXC quirk: `pack_matrix(column_major)` reliably applies to matrices that
// are STRUCT MEMBERS inside a StructuredBuffer, but its behaviour for raw
// element-type matrices (`StructuredBuffer<float4x4>`) is ambiguous. Wrapping
// in a struct with an explicit `column_major` qualifier pins the storage
// layout so the matrix reads back as Rust uploaded it (column-major).
struct ColMat4 { column_major float4x4 m; };
StructuredBuffer<ColMat4> instances : register(t3);

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
    float4 sv_pos     : SV_POSITION;
    float3 world_pos  : TEXCOORD0;
    float3 normal     : TEXCOORD1;
    float3 tangent    : TEXCOORD2;
    float3 bitangent  : TEXCOORD3;
    float2 uv         : TEXCOORD4;
    float  view_depth : TEXCOORD5;
    float3 color      : TEXCOORD6;
};

VsOut main(VsIn v, uint iid : SV_InstanceID)
{
    VsOut o;
    float4x4 model = instances[iid].m;

    float4 world = mul(model, float4(v.pos, 1.0));
    o.world_pos = world.xyz;

    float3x3 nm = (float3x3)model;
    o.normal    = normalize(mul(nm, v.normal));
    o.tangent   = normalize(mul(nm, v.tangent));
    o.bitangent = cross(o.normal, o.tangent);

    o.uv    = v.uv;
    o.color = v.color;

    o.view_depth = -mul(view_mat, world).z;
    o.sv_pos     = mul(vp, world);

    return o;
}
