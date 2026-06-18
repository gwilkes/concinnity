#version 450
// Particle render pipeline - vertex shader. Mirrors `particle_vertex` in
// src/metal/shaders/particle.metal and `particle_vert.hlsl`. Drawn with
// `vkCmdDraw(4, max_particles, 0, 0)` as a triangle strip: each instance
// reads its `Particle` from the pool by `gl_InstanceIndex`, derives a
// camera-facing axis pair from the bound `ParticleView`, and emits one
// corner of a billboard quad.

struct Particle {
    vec3  position;
    float age;
    vec3  velocity;
    float lifetime;
};

// Per-frame view inputs at set 0 binding 0. 96 bytes: mat4 + two
// (vec3, float) pairs (the trailing scalar is pad in std140 - std140
// packs a vec3 + scalar into 16 bytes).
layout(set = 0, binding = 0, std140) uniform ParticleView {
    mat4 vp;
    vec3 cam_right;
    float _vpad0;
    vec3 cam_up;
    float _vpad1;
} view;

// Per-emitter particle pool at set 1 binding 0. Read-only here - the
// compute pass wrote it through the same SSBO via a transient WAR
// barrier emitted by the encoder.
layout(set = 1, binding = 0, std430) readonly buffer Pool {
    Particle data[];
} pool;

// Per-emitter uniform pushed alongside the draw. Layout matches the
// compute kernel's push-constant block. Vertex stage only reads the
// gradient + size fields; binding the full struct keeps the host-side
// upload single-shot.
layout(push_constant) uniform ParticleParamsBlock {
    vec3  position;
    float spread_cos;
    vec3  direction;
    float speed_min;
    vec3  gravity;
    float speed_max;
    vec4  color_start;
    vec4  color_end;
    float lifetime_min;
    float lifetime_max;
    float size_start;
    float size_end;
    float dt;
    uint  spawn_budget;
    uint  random_seed;
    uint  max_particles;
} params;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;
layout(location = 2) flat out int  v_discard;

void main() {
    Particle pt = pool.data[gl_InstanceIndex];

    // Dead slot → emit a degenerate clip-space point behind the near plane.
    // The fragment shader also discards on `v_discard` so any stray
    // rasterised pixel (numerical edge case at exactly w=0) draws nothing.
    if (pt.lifetime <= 0.0) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        v_uv        = vec2(0.0);
        v_color     = vec4(0.0);
        v_discard   = 1;
        return;
    }

    float t     = clamp(pt.age / pt.lifetime, 0.0, 1.0);
    float size  = mix(params.size_start, params.size_end, t);
    vec4  color = mix(params.color_start, params.color_end, t);

    // 0..3 → (-1,-1), (+1,-1), (-1,+1), (+1,+1) for a triangle strip.
    int   vid    = gl_VertexIndex;
    vec2  corner = vec2(
        (vid & 1) == 0 ? -1.0 : 1.0,
        (vid & 2) == 0 ? -1.0 : 1.0
    );

    vec3 right = view.cam_right * (corner.x * 0.5 * size);
    vec3 up    = view.cam_up    * (corner.y * 0.5 * size);
    vec3 world = pt.position + right + up;
    gl_Position = view.vp * vec4(world, 1.0);

    v_uv      = corner * 0.5 + 0.5;
    v_color   = color;
    v_discard = 0;
}
