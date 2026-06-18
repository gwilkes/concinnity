#version 450

layout(location = 0) out vec2 frag_uv;

void main() {
    // Fullscreen triangle from gl_VertexIndex 0..2 - no vertex buffer. The
    // composite pass uses a standard positive-height viewport, and the HDR
    // image was produced upright by the main pass's negative-height viewport,
    // so the UV is a plain [0,1] map with no Y flip.
    vec2 pos = vec2((gl_VertexIndex == 2) ? 3.0 : -1.0,
                    (gl_VertexIndex == 1) ? 3.0 : -1.0);
    gl_Position = vec4(pos, 0.0, 1.0);
    frag_uv = (pos + 1.0) * 0.5;
}
