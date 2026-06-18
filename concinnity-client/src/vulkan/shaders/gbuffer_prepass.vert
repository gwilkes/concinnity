#version 450

// Unified G-buffer pre-pass vertex shader (static geometry). One jittered
// traversal feeds the SSR / SSAO / SSGI / TAA readers: rasterise position via
// the jittered VP (matching the main pass coverage), pass through the
// view-space normal + linear view depth for the normal+depth target, and the
// un-jittered current / previous clip positions so the fragment shader can
// derive a jitter-free motion vector. Fuses ssr_prepass.vert + velocity.vert.

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

// Vertex range covers cur_model + prev_model; the fragment shader sees the
// roughness at offset 128. A single push-constant block is shared between
// stages, each stage references only the fields its range exposes.
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
    vec4 cur_world  = push.cur_model  * vec4(in_pos, 1.0);
    vec4 prev_world = push.prev_model * vec4(in_pos, 1.0);
    vec4 view_pos   = gbview.view_mat * cur_world;
    vec3 world_n    = normalize(mat3(push.cur_model) * in_normal);
    frag_view_normal = mat3(gbview.view_mat) * world_n;
    frag_view_depth  = -view_pos.z;
    cur_clip  = gbview.cur_vp  * cur_world;
    prev_clip = gbview.prev_vp * prev_world;
    gl_Position = gbview.jittered_vp * cur_world;
    // Skybox sentinel: pin to the far plane so sky never occludes the scene.
    if (in_color.b > 1.5) {
        gl_Position.z = gl_Position.w * (1.0 - 1e-6);
    }
}
