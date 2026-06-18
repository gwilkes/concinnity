#version 460

// Fullscreen-triangle vertex shader for the hardware ray-traced reflection pass.
// Identical convention to ssr_fullscreen.vert: the SSR depth + normal pre-pass
// (which RT reuses) wrote its G-buffer with the same negative-height viewport as
// the main HDR target, so a plain [0,1] UV with no Y flip lines the reflection
// taps up with the matching main-pass pixels (and the HDR scene the pass blends
// over). This is the one divergence from the DirectX rt_reflections.hlsl (which
// flips Y for D3D's top-left origin).

layout(location = 0) out vec2 frag_uv;

void main() {
    vec2 pos = vec2((gl_VertexIndex == 2) ?  3.0 : -1.0,
                    (gl_VertexIndex == 1) ?  3.0 : -1.0);
    gl_Position = vec4(pos, 0.0, 1.0);
    frag_uv = (pos + 1.0) * 0.5;
}
