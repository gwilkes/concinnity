#pragma pack_matrix(column_major)

// GPU-driven G-buffer pre-pass vertex shader. Bindless sibling of
// gbuffer_prepass_vert: the per-command b0 object-id root constant indexes the
// per-frame GpuObjectData buffer for the model matrix + roughness (so the CPU
// never pushes a per-draw model/material), and a parallel prev_models buffer
// supplies the previous-frame model for the motion vector. The previous-frame
// vertex position rides a second vertex stream (slot 1): the static prefix binds
// the static VB to both slots (prev_pos == cur_pos, so motion is the model delta
// plus camera), and the skinned tail binds the previous-frame deformed buffer to
// slot 1 (so per-vertex skin deformation produces a correct motion vector).
//
// Layout (160 bytes) must match the Rust GpuObjectData in gfx::render_types and
// the shadow bindless VS's struct. The VS reads `model` + `roughness`; the full
// layout is required so `objects[object_id]` strides correctly.
struct GpuObjectData
{
    float4x4 model;
    float3 tint;
    float roughness;
    float3 emissive;
    float metallic;
    uint albedo_index;
    uint normal_index;
    float macro_variation;
    float terrain_blend;
    float3 bb_min;
    float cull_distance;
    float3 bb_max;
    float secondary_blend_sharpness;
    uint albedo_secondary_index;
    uint normal_secondary_index;
    uint emissive_map_index;
    uint orm_map_index;
};

cbuffer ObjId : register(b0)
{
    uint object_id;
}

cbuffer GbView : register(b1)
{
    float4x4 jittered_vp;
    float4x4 cur_vp;
    float4x4 prev_vp;
    float4x4 view_mat;
}

// Wrap in a column_major-qualified struct: see skinned_vert.hlsl for why a bare
// StructuredBuffer<float4x4> mis-packs.
struct ColMat4 { column_major float4x4 m; };
StructuredBuffer<GpuObjectData> objects     : register(t0);
StructuredBuffer<ColMat4>       prev_models : register(t1);

struct VsIn
{
    float3 pos      : POSITION0;
    float3 normal   : NORMAL0;
    float3 color    : COLOR0;
    float3 prev_pos : POSITION1;
};

struct VsOut
{
    float4 sv_pos      : SV_POSITION;
    float3 view_normal : TEXCOORD0;
    float  view_depth  : TEXCOORD1;
    float4 cur_clip    : TEXCOORD2;
    float4 prev_clip   : TEXCOORD3;
    nointerpolation float roughness : TEXCOORD4;
};

VsOut main(VsIn v)
{
    VsOut o;
    GpuObjectData obj = objects[object_id];
    float4x4 model      = obj.model;
    float4x4 prev_model = prev_models[object_id].m;
    float4 cur_world  = mul(model,      float4(v.pos, 1.0));
    float4 prev_world = mul(prev_model, float4(v.prev_pos, 1.0));
    float4 view_pos   = mul(view_mat, cur_world);
    o.sv_pos        = mul(jittered_vp, cur_world);
    float3 world_n  = normalize(mul((float3x3)model, v.normal));
    o.view_normal   = mul((float3x3)view_mat, world_n);
    o.view_depth    = -view_pos.z;
    o.cur_clip      = mul(cur_vp,  cur_world);
    o.prev_clip     = mul(prev_vp, prev_world);
    o.roughness     = obj.roughness;
    // Skybox sentinel: pin to the far plane so sky never occludes scene.
    if (v.color.b > 1.5)
    {
        o.sv_pos.z = o.sv_pos.w * (1.0 - 1e-6);
    }
    return o;
}
