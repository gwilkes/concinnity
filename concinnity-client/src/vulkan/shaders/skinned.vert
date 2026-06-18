#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_tangent;
layout(location = 3) in vec3 in_color;
layout(location = 4) in vec2 in_uv;
layout(location = 5) in uvec4 in_joints;
layout(location = 6) in vec4 in_weights;

layout(std140, set = 0, binding = 0) uniform ViewBlock {
    mat4  vp;
    mat4  view_mat;
    float elapsed;
    float _pad0;
    float cam_x; float cam_y; float cam_z;
    float prefilter_mip_count; float _ep0; float _ep1;
} view;

layout(push_constant) uniform PushBlock {
    mat4  model;
    float roughness;
    float metallic;
    float _mpad0; float _mpad1;
    vec3  tint;
    float _mpad2;
    vec3  emissive;
    float _mpad3;
} push;

layout(std430, set = 2, binding = 0) readonly buffer JointBlock {
    mat4 joints[];
} skin;

layout(location = 0) out vec3 frag_world_pos;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_tangent;
layout(location = 3) out vec3 frag_bitangent;
layout(location = 4) out vec2 frag_uv;
layout(location = 5) out float frag_view_depth;
layout(location = 6) out vec3 frag_color;

void main() {
    // Linear blend skinning: weighted sum of the bound joints' matrices.
    mat4 sk = in_weights.x * skin.joints[in_joints.x]
            + in_weights.y * skin.joints[in_joints.y]
            + in_weights.z * skin.joints[in_joints.z]
            + in_weights.w * skin.joints[in_joints.w];

    vec4 skinned_pos     = sk * vec4(in_pos, 1.0);
    vec3 skinned_normal  = mat3(sk) * in_normal;
    vec3 skinned_tangent = mat3(sk) * in_tangent;

    vec4 world = push.model * skinned_pos;
    frag_world_pos = world.xyz;

    mat3 nm = transpose(inverse(mat3(push.model)));
    frag_normal    = normalize(nm * skinned_normal);
    frag_tangent   = normalize(nm * skinned_tangent);
    frag_bitangent = cross(frag_normal, frag_tangent);

    frag_uv    = in_uv;
    frag_color = in_color;
    frag_view_depth = -(view.view_mat * world).z;
    gl_Position = view.vp * world;
}
