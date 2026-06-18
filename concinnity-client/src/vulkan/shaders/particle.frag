#version 450
// Particle render pipeline - fragment shader. Mirrors `particle_fragment` in
// src/metal/shaders/particle.metal and `particle_frag.hlsl`. Samples the
// emitter's albedo texture, multiplies it by the age-interpolated gradient
// colour the vertex stage emitted, and lets the blend state composite the
// result into the resolved HDR target.

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) flat in int v_discard;

// Per-emitter albedo at set 1 binding 1. The fragment shader reads it
// only on the live-particle path; on the dead-slot path the vertex
// shader collapsed the quad to a clip-behind-near-plane degenerate so
// rasterisation should never touch this stage.
layout(set = 1, binding = 1) uniform sampler2D albedo;

layout(location = 0) out vec4 out_color;

void main() {
    if (v_discard != 0) {
        discard;
    }
    // Flip V to match the rest of the engine's texture convention
    // (V = 0 at the top of the image).
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec4 sampled = texture(albedo, uv);
    out_color = vec4(sampled.rgb * v_color.rgb, sampled.a * v_color.a);
}
