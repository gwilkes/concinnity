#version 450

// Fullscreen-triangle vertex shader shared by the SSGI gather + composite
// passes. Same convention as ssr_fullscreen.vert: the SSR pre-pass wrote its
// G-buffer with a negative-height viewport, so a plain [0,1] UV with no Y flip
// lines the gather taps up with the matching main-pass pixels (and the HDR
// scene the gather samples).

layout(location = 0) out vec2 frag_uv;

void main() {
    vec2 pos = vec2((gl_VertexIndex == 2) ?  3.0 : -1.0,
                    (gl_VertexIndex == 1) ?  3.0 : -1.0);
    gl_Position = vec4(pos, 0.0, 1.0);
    frag_uv = (pos + 1.0) * 0.5;
}
