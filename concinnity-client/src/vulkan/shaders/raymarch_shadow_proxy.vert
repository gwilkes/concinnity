#version 450

// Engine-shipped proxy vertex shader for raymarched SDF shadow casters (Vulkan).
// Rasterises the unit-cube proxy through the active CSM cascade's light
// view-projection (instead of the camera VP), so the depth-only shadow fragment
// marches the SDF from the light side. GLSL port of the shadow vertex in
// directx/shaders/raymarch_shadow.hlsl / metal/shaders/raymarch_shadow.metal.
// The cube VB is the 56-byte engine Vertex; only position (location 0) is read.

layout(location = 0) in vec3 in_pos;

layout(location = 0) out vec3 v_world_pos;

layout(std140, set = 1, binding = 0) uniform SdfVolumeBlock {
    vec3 vol_centre;
    float vol_pad0;
    vec3 vol_extent;
    float vol_pad1;
    float vol_cone_ratio;
    float vol_max_distance;
    int vol_max_steps;
    int vol_receive_shadows;
    vec4 vol_params[8];
} vol;

layout(std140, set = 0, binding = 2) uniform RaymarchShadowBlock {
    mat4 light_vps[4];
    vec4 cascade_splits;
} shadow_uni;

layout(push_constant) uniform ShadowCascade {
    uint cascade_idx;
} pc;

void main() {
    vec3 wp = in_pos * vol.vol_extent + vol.vol_centre;
    v_world_pos = wp;
    gl_Position = shadow_uni.light_vps[pc.cascade_idx] * vec4(wp, 1.0);
}
