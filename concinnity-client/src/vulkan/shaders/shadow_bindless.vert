#version 450

// Depth-only vertex shader for the GPU-driven shadow pass. Mirrors
// main_bindless.vert: the object id rides gl_InstanceIndex (the cull kernel
// wrote it into first_instance), indexing the per-frame GpuObjectData SSBO for
// the model matrix, so the CPU never pushes a per-draw model. The cascade index
// is a push constant (one indirect draw per cascade), selecting which
// light_vps[i] to project through.

layout(location = 0) in vec3 in_pos;
// Remaining attributes declared to preserve binding locations but not used.
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_tangent;
layout(location = 3) in vec3 in_color;
layout(location = 4) in vec2 in_uv;

layout(std140, set = 0, binding = 0) uniform ShadowGlobal {
    mat4 light_vps[4];
    vec4 cascade_splits;
} sg;

// Layout must match main_bindless.vert's GpuObjectData (std430). The shadow VS
// only reads `model`, but the full layout strides objects[oid] correctly.
struct GpuObjectData {
    mat4  model;
    vec3  tint;      float roughness;
    vec3  emissive;  float metallic;
    uint  albedo_index;
    uint  normal_index;
    float macro_variation;
    float terrain_blend;
    vec3  bb_min;    float cull_distance;
    vec3  bb_max;    float secondary_blend_sharpness;
    uint  albedo_secondary_index;
    uint  normal_secondary_index;
    uint  _pad2;
    uint  _pad3;
};

layout(std430, set = 1, binding = 0) readonly buffer ObjectBlock {
    GpuObjectData objects[];
} obj_buf;

layout(push_constant) uniform CascadePush {
    uint cascade_idx;
} push;

void main() {
    mat4 model = obj_buf.objects[uint(gl_InstanceIndex)].model;
    gl_Position = sg.light_vps[push.cascade_idx] * model * vec4(in_pos, 1.0);
}
