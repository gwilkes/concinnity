#version 450

// GPU-instanced sibling of gbuffer_prepass.vert. Per-instance model matrices
// live in a storage buffer at set=1,binding=0 (same layout the main-pass
// instanced pipeline uses). Instance transforms are immutable, so cur == prev
// (the motion vector is camera-only). The cluster-wide roughness still rides
// the shared push-constant block at offset 128 so all three pre-pass variants
// share one fragment shader. Fuses ssr_prepass_instanced.vert +
// velocity_instanced.vert.

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_tangent;
layout(location = 3) in vec3 in_color;
layout(location = 4) in vec2 in_uv;

layout(std140, set = 0, binding = 0) uniform GbView {
    mat4 jittered_vp;
    mat4 cur_vp;
    mat4 prev_vp;
    mat4 view_mat;
} gbview;

layout(std430, set = 1, binding = 0) readonly buffer InstanceBlock {
    mat4 instances[];
} insts;

// Even though the instanced VS does not read it, the cluster-wide roughness
// still sits in the shared push constant range so the fragment shader's
// offset 128 lookup keeps working.
layout(push_constant) uniform PushBlock {
    mat4 cur_model;
    mat4 prev_model;
    float roughness;
} push;

layout(location = 0) out vec3 frag_view_normal;
layout(location = 1) out float frag_view_depth;
layout(location = 2) out vec4 cur_clip;
layout(location = 3) out vec4 prev_clip;

void main() {
    mat4 model    = insts.instances[gl_InstanceIndex];
    vec4 world    = model * vec4(in_pos, 1.0);
    vec4 view_pos = gbview.view_mat * world;
    vec3 world_n  = normalize(mat3(model) * in_normal);
    frag_view_normal = mat3(gbview.view_mat) * world_n;
    frag_view_depth  = -view_pos.z;
    cur_clip  = gbview.cur_vp  * world;
    prev_clip = gbview.prev_vp * world;
    gl_Position = gbview.jittered_vp * world;
}
