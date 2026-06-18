#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_tangent;
layout(location = 3) in vec3 in_color;
layout(location = 4) in vec2 in_uv;

layout(std140, set = 0, binding = 0) uniform ViewBlock {
    mat4  vp;
    mat4  view_mat;
    float elapsed;
    float _pad0;
    float cam_x; float cam_y; float cam_z;
    // prefilter_mip_count = number of mip levels in the IBL prefilter cube. 0 = IBL off.
    float prefilter_mip_count; float _ep0; float _ep1;
} view;

layout(std140, set = 0, binding = 2) uniform ShadowBlock {
    mat4 light_vps[4];
    vec4 cascade_splits;
} shadow_uni;

layout(push_constant) uniform PushBlock {
    mat4  model_unused;
    float roughness;
    float metallic;
    float _mpad0; float _mpad1;
    vec3  tint;
    float _mpad2;
    vec3  emissive;
    float _mpad3;
} push;

layout(std430, set = 2, binding = 0) readonly buffer InstanceBlock {
    mat4 instances[];
} insts;

layout(location = 0) out vec3 frag_world_pos;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_tangent;
layout(location = 3) out vec3 frag_bitangent;
layout(location = 4) out vec2 frag_uv;
layout(location = 5) out float frag_view_depth;
layout(location = 6) out vec3 frag_color;

void main() {
    mat4 model = insts.instances[gl_InstanceIndex];
    vec4 world = model * vec4(in_pos, 1.0);
    frag_world_pos = world.xyz;

    mat3 nm = transpose(inverse(mat3(model)));
    frag_normal    = normalize(nm * in_normal);
    frag_tangent   = normalize(nm * in_tangent);
    frag_bitangent = cross(frag_normal, frag_tangent);

    frag_uv    = in_uv;
    frag_color = in_color;

    frag_view_depth = -(view.view_mat * world).z;
    gl_Position     = view.vp * world;

    // Skybox sentinel (see VERT_GLSL): pin sky vertices to the far plane.
    // Instanced clusters are never skyboxes today, but the contract stays
    // consistent across both vertex paths.
    if (in_color.b > 1.5) {
        gl_Position.z = gl_Position.w * (1.0 - 1e-6);
    }
}
