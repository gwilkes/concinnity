#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_tangent;
layout(location = 3) in vec3 in_color;
layout(location = 4) in vec2 in_uv;

layout(std140, set = 0, binding = 0) uniform ViewBlock {
    mat4  vp;
    mat4  view_mat;
    float elapsed;
    float _pad0;
    float cam_x; float cam_y; float cam_z;
    float prefilter_mip_count; float _ep0; float _ep1;
} view;

// Layout must match the #[repr(C)] GpuObjectData in gfx::render_types (144
// bytes) under std430: every vec3 is followed by a scalar so no padding drifts.
struct GpuObjectData {
    mat4  model;
    vec3  tint;      float roughness;
    vec3  emissive;  float metallic;
    uint  albedo_index;
    uint  normal_index;
    float macro_variation;
    float terrain_blend;
    vec3  bb_min;    float cull_distance;
    vec3  bb_max;    float secondary_blend_sharpness;
    uint  albedo_secondary_index;
    uint  normal_secondary_index;
    uint  _pad2;
    uint  _pad3;
};

layout(std430, set = 1, binding = 0) readonly buffer ObjectBlock {
    GpuObjectData objects[];
} obj_buf;

layout(location = 0) out vec3 frag_world_pos;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_tangent;
layout(location = 3) out vec3 frag_bitangent;
layout(location = 4) out vec2 frag_uv;
layout(location = 5) out float frag_view_depth;
layout(location = 6) out vec3 frag_color;
// The object id is needed in the fragment stage too (gl_InstanceIndex is a
// vertex-only built-in), so it is forwarded as a flat varying.
layout(location = 7) flat out uint frag_object_id;

void main() {
    uint oid = uint(gl_InstanceIndex);
    frag_object_id = oid;
    mat4 model = obj_buf.objects[oid].model;

    vec4 world = model * vec4(in_pos, 1.0);
    frag_world_pos = world.xyz;

    mat3 nm = transpose(inverse(mat3(model)));
    frag_normal    = normalize(nm * in_normal);
    frag_tangent   = normalize(nm * in_tangent);
    frag_bitangent = cross(frag_normal, frag_tangent);

    frag_uv    = in_uv;
    frag_color = in_color;

    frag_view_depth = -(view.view_mat * world).z;

    gl_Position = view.vp * world;

    if (in_color.b > 1.5) {
        gl_Position.z = gl_Position.w * (1.0 - 1e-6);
    }
}
