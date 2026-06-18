#version 450

// Fullscreen-triangle vertex shader for the SSR resolve pass. The SSR pre-pass
// wrote its G-buffer with the same negative-height viewport convention as the
// main HDR target, so a plain [0,1] UV with no Y flip lines the resolve taps
// up with the matching main-pass pixels (and the HDR scene the resolve
// samples). Mirrors the SSAO fullscreen vertex shader.

layout(location = 0) out vec2 frag_uv;

void main() {
    vec2 pos = vec2((gl_VertexIndex == 2) ?  3.0 : -1.0,
                    (gl_VertexIndex == 1) ?  3.0 : -1.0);
    gl_Position = vec4(pos, 0.0, 1.0);
    frag_uv = (pos + 1.0) * 0.5;
}
