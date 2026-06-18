#version 450

// Volumetric fog pass - fullscreen-triangle vertex shader. Mirrors
// src/directx/shaders/fog_vert.hlsl and fog_vertex in src/metal/shaders/fog.metal.
// Three vertices form a single triangle that covers the entire framebuffer;
// the pixel shader runs once per framebuffer pixel and no vertex buffer is
// needed.

void main() {
    vec2 pos = vec2((gl_VertexIndex == 2) ? 3.0 : -1.0,
                    (gl_VertexIndex == 1) ? 3.0 : -1.0);
    gl_Position = vec4(pos, 0.0, 1.0);
}
