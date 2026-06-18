#version 450

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_color;

layout(push_constant) uniform TextPush {
    float win_width;
    float win_height;
    float _pad0;
    float _pad1;
} push;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec3 frag_color;

void main() {
    // in_pos is in pixels, origin top-left. Text renders in the composite
    // pass, which uses a standard positive-height viewport (Y-down NDC), so
    // pixel (0,0) maps to NDC (-1,-1): nx/ny are a plain linear remap.
    float nx = (in_pos.x / push.win_width)  * 2.0 - 1.0;
    float ny = (in_pos.y / push.win_height) * 2.0 - 1.0;
    gl_Position = vec4(nx, ny, 0.0, 1.0);
    frag_uv    = in_uv;
    frag_color = in_color;
}
