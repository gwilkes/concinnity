#version 450

layout(location = 0) in vec3 in_pos;
// Remaining attributes declared to preserve binding locations but not used.
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_tangent;
layout(location = 3) in vec3 in_color;
layout(location = 4) in vec2 in_uv;

layout(std140, set = 0, binding = 0) uniform ShadowGlobal {
    mat4 light_vps[4];
    vec4 cascade_splits;
} sg;

// Push constants: mat4 model (64 bytes) + uint cascade_idx (4) + 12 pad bytes.
// The runtime loops the shadow pass once per cascade and pushes the slot.
layout(push_constant) uniform ShadowPush {
    mat4 model;
    uint cascade_idx;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} push;

void main() {
    gl_Position = sg.light_vps[push.cascade_idx] * push.model * vec4(in_pos, 1.0);
}
