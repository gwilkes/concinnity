#version 450

// Glass panel pass - vertex shader. Mirrors vs_main in
// src/directx/shaders/glass.hlsl and glass_vertex in
// src/metal/shaders/glass.metal. Runs in the PassId::Transparent slot after
// SSR resolve and before TAA. The quad vertices are pre-transformed into world
// space at build time (see geometry::glass_quad), so the vertex shader only
// projects them.

layout(std140, set = 0, binding = 0) uniform TransparentViewBlock {
    mat4  vp;          // world -> clip (jittered when TAA is on)
    mat4  inv_vp;      // clip -> world
    vec4  camera_pos;  // world-space camera, .w unused
    vec2  viewport;    // attachment dimensions in pixels
    float time;        // seconds since startup
    float _pad;
} view;

// Only the position is fetched (location 0); the rest of the engine `Vertex`
// rides in the buffer but is unused by the glass pipeline's vertex layout.
layout(location = 0) in vec3 in_pos;

layout(location = 0) out vec3 world_pos;

void main() {
    // Quad vertices are pre-transformed into world space at build time.
    world_pos = in_pos;
    gl_Position = view.vp * vec4(in_pos, 1.0);
}
