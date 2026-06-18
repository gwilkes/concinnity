#version 450

// Skinned sibling of gbuffer_prepass.vert. 4-influence linear blend skinning
// using the current frame's joint palette (set=1,binding=0) and the previous
// frame's palette (set=2,binding=0), so per-vertex skinned deformation produces
// a correct screen-space motion vector. The two palette sets share one layout
// (the main-pass skinned joint set). Fuses ssr_prepass_skinned.vert +
// velocity_skinned.vert.

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_tangent;
layout(location = 3) in vec3 in_color;
layout(location = 4) in vec2 in_uv;
layout(location = 5) in uvec4 in_joints;
layout(location = 6) in vec4 in_weights;

layout(std140, set = 0, binding = 0) uniform GbView {
    mat4 jittered_vp;
    mat4 cur_vp;
    mat4 prev_vp;
    mat4 view_mat;
} gbview;

layout(push_constant) uniform PushBlock {
    mat4 cur_model;
    mat4 prev_model;
    float roughness;
} push;

layout(std430, set = 1, binding = 0) readonly buffer JointBlock {
    mat4 joints[];
} skin;

layout(std430, set = 2, binding = 0) readonly buffer PrevJointBlock {
    mat4 joints[];
} prev_skin;

layout(location = 0) out vec3 frag_view_normal;
layout(location = 1) out float frag_view_depth;
layout(location = 2) out vec4 cur_clip;
layout(location = 3) out vec4 prev_clip;

void main() {
    mat4 sk = in_weights.x * skin.joints[in_joints.x]
            + in_weights.y * skin.joints[in_joints.y]
            + in_weights.z * skin.joints[in_joints.z]
            + in_weights.w * skin.joints[in_joints.w];
    mat4 prev_sk = in_weights.x * prev_skin.joints[in_joints.x]
                 + in_weights.y * prev_skin.joints[in_joints.y]
                 + in_weights.z * prev_skin.joints[in_joints.z]
                 + in_weights.w * prev_skin.joints[in_joints.w];

    vec4 cur_skinned  = sk * vec4(in_pos, 1.0);
    vec4 prev_skinned = prev_sk * vec4(in_pos, 1.0);
    vec3 skinned_normal = mat3(sk) * in_normal;

    vec4 cur_world  = push.cur_model  * cur_skinned;
    vec4 prev_world = push.prev_model * prev_skinned;
    vec4 view_pos   = gbview.view_mat * cur_world;
    vec3 world_n    = normalize(mat3(push.cur_model) * skinned_normal);
    frag_view_normal = mat3(gbview.view_mat) * world_n;
    frag_view_depth  = -view_pos.z;
    cur_clip  = gbview.cur_vp  * cur_world;
    prev_clip = gbview.prev_vp * prev_world;
    gl_Position = gbview.jittered_vp * cur_world;
}
