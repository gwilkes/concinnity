// Default D3D12 vertex shader for Concinnity scenes.
//
// Root signature layout (must match directx/pipeline.rs):
//   b0 PushConstants : model mat4 + material (28 DWORDs)
//   b1 ViewBlock     : vp mat4, view mat4, elapsed, cam xyz
//   b3 ShadowBlock   : light_vps[4] mat4 + cascade_splits float4
//
// Input layout (56-byte Vertex, must match main_input_layout() in directx/pipeline.rs):
//   POSITION  float3  offset  0
//   NORMAL    float3  offset 12
//   TANGENT   float3  offset 24
//   COLOR     float3  offset 36
//   TEXCOORD0 float2  offset 48

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
    float _pad1;
    float _ep0;
    float _ep1;
}

cbuffer ShadowBlock : register(b3)
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

struct VsOut
{
    float4 sv_pos      : SV_POSITION;
    float3 world_pos   : TEXCOORD0;
    float3 normal      : TEXCOORD1;
    float3 tangent     : TEXCOORD2;
    float3 bitangent   : TEXCOORD3;
    float2 uv          : TEXCOORD4;
    float  view_depth  : TEXCOORD5;
    float3 color       : TEXCOORD6;
};

VsOut main(VsIn v)
{
    VsOut o;
    float4 world = mul(model, float4(v.pos, 1.0));
    o.world_pos = world.xyz;

    float3x3 nm = (float3x3)model;
    o.normal    = normalize(mul(nm, v.normal));
    o.tangent   = normalize(mul(nm, v.tangent));
    o.bitangent = cross(o.normal, o.tangent);

    o.uv    = v.uv;
    o.color = v.color;

    // View-space depth (positive in front of camera) for cascade selection.
    o.view_depth = -mul(view_mat, world).z;

    // Skybox sentinel: blue channel > 1.5 forces sky to far plane.
    o.sv_pos = mul(vp, world);
    if (v.color.b > 1.5)
        o.sv_pos.z = o.sv_pos.w * (1.0 - 1e-6);

    return o;
}
