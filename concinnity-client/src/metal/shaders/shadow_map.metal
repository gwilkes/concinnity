// Shadow pass vertex shader.
//
// Renders scene depth from the directional light's perspective into one slice
// of a Depth32Float texture array, one slice per cascade. No fragment function
// is needed - depth writes are automatic. The resulting cascades are sampled
// with PCF in default.metal.
//
// Buffer bindings:
//   buffer(0) -- ShadowUniforms: float4x4 light_vps[NUM_SHADOW_CASCADES],
//                                float4    cascade_splits
//   buffer(1) -- per-vertex data (same layout as default.metal, stride 56 bytes)
//   buffer(2) -- ModelUniforms: float4x4 model
//   buffer(7) -- ShadowPassPush: uint cascade_idx (which cascade VP to use this pass)

#include <metal_stdlib>
using namespace metal;

constant constexpr uint NUM_SHADOW_CASCADES = 4;

struct ShadowVertex {
    float3 pos [[attribute(0)]];
};

struct ShadowUniforms {
    float4x4 light_vps[NUM_SHADOW_CASCADES];
    float4   cascade_splits;
};

struct ModelUniforms {
    float4x4 model;
};

struct ShadowPassPush {
    uint cascade_idx;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

vertex float4 shadow_vertex_main(
    ShadowVertex in                    [[stage_in]],
    constant ShadowUniforms &shadow    [[buffer(0)]],
    constant ModelUniforms  &model_u   [[buffer(2)]],
    constant ShadowPassPush &push      [[buffer(7)]]
) {
    return shadow.light_vps[push.cascade_idx] * model_u.model * float4(in.pos, 1.0);
}

// GPU-driven sibling of shadow_vertex_main. The per-object model
// matrix is read from the bindless GpuObjectData buffer at buffer(9), indexed
// by the object id the shadow cull kernel supplies as each indirect command's
// [[base_instance]] -- the same delivery the main bindless vertex_main uses. No
// per-draw ModelUniforms is bound, which is what lets every cascade's casters
// draw through one compute-encoded indirect command buffer. The deformed
// skinned tail rides this same shader: its vertices are already model-space
// (the pre-skin baked bind-pose -> model space) and its record's model is the
// object's world matrix, so `light_vp * model * deformed_pos` matches static.
struct GpuObjectData {
    float4x4      model;
    packed_float3 tint;
    float         roughness;
    packed_float3 emissive;
    float         metallic;
    uint          albedo_index;
    uint          normal_index;
    float         macro_variation;
    float         terrain_blend;
    packed_float3 bb_min;
    float         cull_distance;
    packed_float3 bb_max;
    float         secondary_blend_sharpness;
    uint          albedo_secondary_index;
    uint          normal_secondary_index;
    uint          _pad2;
    uint          _pad3;
};

vertex float4 shadow_vertex_bindless(
    ShadowVertex in                    [[stage_in]],
    constant ShadowUniforms &shadow    [[buffer(0)]],
    constant ShadowPassPush &push      [[buffer(7)]],
    constant GpuObjectData  *objects   [[buffer(9)]],
    uint                     obj_id    [[base_instance]]
) {
    float4x4 model = objects[obj_id].model;
    return shadow.light_vps[push.cascade_idx] * model * float4(in.pos, 1.0);
}

// Skinned sibling of shadow_vertex_main. Blends the same four joint matrices
// (from buffer(8)) as default.metal's vertex_main_skinned so a skinned mesh
// casts a shadow that matches its deformed silhouette.
struct SkinnedShadowVertex {
    float3  pos     [[attribute(0)]];
    ushort4 joints  [[attribute(5)]];
    float4  weights [[attribute(6)]];
};

vertex float4 shadow_vertex_main_skinned(
    SkinnedShadowVertex in             [[stage_in]],
    constant ShadowUniforms &shadow    [[buffer(0)]],
    constant ModelUniforms  &model_u   [[buffer(2)]],
    constant ShadowPassPush &push      [[buffer(7)]],
    constant float4x4       *joints    [[buffer(8)]]
) {
    float4x4 skin = in.weights.x * joints[in.joints.x]
                  + in.weights.y * joints[in.joints.y]
                  + in.weights.z * joints[in.joints.z]
                  + in.weights.w * joints[in.joints.w];
    return shadow.light_vps[push.cascade_idx] * model_u.model * skin * float4(in.pos, 1.0);
}
