#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_tangent;
layout(location = 3) in vec3 in_color;
layout(location = 4) in vec2 in_uv;

// std140: mat4(64) + mat4(64) + float(4) + float(4) + 6 floats(24) = 160 bytes,
// matching Rust ViewUniforms.
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
    mat4  model;
    float roughness;
    float metallic;
    float _mpad0; float _mpad1;
    vec3  tint;
    float _mpad2;
    vec3  emissive;
    float _mpad3;
} push;

layout(location = 0) out vec3 frag_world_pos;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_tangent;
layout(location = 3) out vec3 frag_bitangent;
layout(location = 4) out vec2 frag_uv;
layout(location = 5) out float frag_view_depth;
layout(location = 6) out vec3 frag_color;

void main() {
    vec4 world = push.model * vec4(in_pos, 1.0);
    frag_world_pos = world.xyz;

    mat3 nm = transpose(inverse(mat3(push.model)));
    frag_normal    = normalize(nm * in_normal);
    frag_tangent   = normalize(nm * in_tangent);
    frag_bitangent = cross(frag_normal, frag_tangent);

    frag_uv    = in_uv;
    frag_color = in_color;

    // View-space depth (positive in front of camera) for cascade selection.
    frag_view_depth = -(view.view_mat * world).z;

    gl_Position = view.vp * world;

    // Skybox sentinel: vertices tagged with blue > 1.5 are pinned to the far
    // plane (z = w) so the skybox is never clipped by the camera far plane -
    // the sky shell sits well beyond a typical `far` - and always renders
    // behind the scene. Matches default.metal / default_vert.hlsl.
    if (in_color.b > 1.5) {
        gl_Position.z = gl_Position.w * (1.0 - 1e-6);
    }
}
