#version 450

// Only the attributes the depth-only skinned shadow VS consumes are declared;
// `skinned_shadow_vertex_input` feeds exactly these three so the pipeline is
// validation-clean (no "attribute not consumed" warnings).
layout(location = 0) in vec3 in_pos;
layout(location = 5) in uvec4 in_joints;
layout(location = 6) in vec4 in_weights;

layout(std140, set = 0, binding = 0) uniform ShadowGlobal {
    mat4 light_vps[4];
    vec4 cascade_splits;
} sg;

layout(push_constant) uniform ShadowPush {
    mat4 model;
    uint cascade_idx;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} push;

layout(std430, set = 1, binding = 0) readonly buffer JointBlock {
    mat4 joints[];
} skin;

void main() {
    mat4 sk = in_weights.x * skin.joints[in_joints.x]
            + in_weights.y * skin.joints[in_joints.y]
            + in_weights.z * skin.joints[in_joints.z]
            + in_weights.w * skin.joints[in_joints.w];
    vec4 skinned_pos = sk * vec4(in_pos, 1.0);
    gl_Position = sg.light_vps[push.cascade_idx] * push.model * skinned_pos;
}
