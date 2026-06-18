#version 450

// GPU-driven G-buffer pre-pass vertex shader. Bindless sibling of
// gbuffer_prepass.vert: the object id rides gl_InstanceIndex (the cull kernel
// wrote it into firstInstance), indexing the per-frame GpuObjectData SSBO for
// the model matrix + roughness, so the CPU never pushes a per-draw model. The
// previous-frame model rides a parallel SSBO; the previous-frame vertex position
// rides a second vertex binding (binding 1). The static + instance prefix binds
// the static VB to both bindings (prev_pos == cur_pos, so motion is the model
// delta plus camera); the skinned tail binds the previous-frame deformed buffer
// to binding 1 so per-vertex skin deformation yields the motion vector.

layout(location = 0) in vec3 in_pos;       // binding 0 (current)
layout(location = 1) in vec3 in_normal;    // binding 0
layout(location = 3) in vec3 in_color;     // binding 0 (skybox sentinel)
layout(location = 5) in vec3 in_prev_pos;  // binding 1 (previous)

layout(std140, set = 0, binding = 0) uniform GbView {
    mat4 jittered_vp;
    mat4 cur_vp;
    mat4 prev_vp;
    mat4 view_mat;
} gbview;

// Layout must match main_bindless.vert's GpuObjectData (std430). The VS reads
// `model` + `roughness`; the full layout strides objects[oid] correctly.
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

layout(std430, set = 0, binding = 1) readonly buffer PrevModelBlock {
    mat4 prev_models[];
} pm_buf;

layout(location = 0) out vec3 frag_view_normal;
layout(location = 1) out float frag_view_depth;
layout(location = 2) out vec4 cur_clip;
layout(location = 3) out vec4 prev_clip;
layout(location = 4) flat out float frag_roughness;

void main() {
    uint oid = uint(gl_InstanceIndex);
    GpuObjectData obj = obj_buf.objects[oid];
    mat4 model      = obj.model;
    mat4 prev_model = pm_buf.prev_models[oid];
    vec4 cur_world  = model      * vec4(in_pos, 1.0);
    vec4 prev_world = prev_model * vec4(in_prev_pos, 1.0);
    vec4 view_pos   = gbview.view_mat * cur_world;
    vec3 world_n    = normalize(mat3(model) * in_normal);
    frag_view_normal = mat3(gbview.view_mat) * world_n;
    frag_view_depth  = -view_pos.z;
    cur_clip  = gbview.cur_vp  * cur_world;
    prev_clip = gbview.prev_vp * prev_world;
    frag_roughness = obj.roughness;
    gl_Position = gbview.jittered_vp * cur_world;
    // Skybox sentinel: pin to the far plane so sky never occludes the scene.
    if (in_color.b > 1.5) {
        gl_Position.z = gl_Position.w * (1.0 - 1e-6);
    }
}
