#version 450

// Projected (deferred) decal pass - vertex shader. Mirrors
// src/directx/shaders/decal_vert.hlsl and src/metal/shaders/decal.metal.
// Rasterises a per-decal unit cube (positions in [-0.5, 0.5]^3) under the
// decal's local→world model matrix and the camera VP (jittered when TAA
// is on so the rasterised pixel grid matches the main pass exactly).

layout(location = 0) in vec3 in_pos;

layout(std140, set = 0, binding = 0) uniform DecalViewBlock {
    mat4  vp;
    mat4  inv_vp;
    vec2  viewport;
    vec2  _pad;
} view;

// Per-decal params bound via a dynamic-offset uniform buffer - one
// MAX_DECALS-slot ring per frame slot. Each slot is laid out as the
// std140 layout below; the host writes one slot per active decal each
// frame.
layout(std140, set = 0, binding = 1) uniform DecalParamsBlock {
    mat4 model;
    mat4 inv_model;
    vec4 tint;
    vec4 fade;   // .x = fade_pow, .yzw padding
} params;

void main() {
    vec4 world = params.model * vec4(in_pos, 1.0);
    gl_Position = view.vp * world;
}
