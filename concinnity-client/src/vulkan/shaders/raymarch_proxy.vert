#version 450

// Engine-shipped proxy vertex shader for raymarched SDF volumes (Vulkan).
// Rasterises the unit-cube proxy (positions at +/-1, shared VB) scaled by the
// volume's extent and offset by its centre, so the back-face fragments (the
// encoder culls front faces) seed one ray per pixel inside the bounding box.
// GLSL port of raymarch_vertex in directx/shaders/raymarch_template.hlsl. The
// vertex stage only needs position; the cube VB is the 56-byte engine Vertex,
// but the raymarch pipeline's vertex input fetches location 0 only.

layout(location = 0) in vec3 in_pos;

layout(location = 0) out vec3 v_world_pos;

layout(std140, set = 0, binding = 0) uniform RaymarchViewBlock {
    mat4 view_vp;
    mat4 view_inv_vp;
    vec3 view_cam_pos;
    float view_pad0;
    vec2 view_viewport;
    float view_time;
    float view_prefilter_mip_count;
} rmview;

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

void main() {
    vec3 wp = in_pos * vol.vol_extent + vol.vol_centre;
    v_world_pos = wp;
    gl_Position = rmview.view_vp * vec4(wp, 1.0);
}
