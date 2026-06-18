#include <metal_stdlib>
using namespace metal;

// Unified geometry G-buffer pre-pass. One jittered traversal of the visible set
// writes, in a single MRT, everything the screen-space + temporal passes need:
//   color(0) RGBA16F  view-space normal (xyz) + positive linear view depth (a)
//   color(1) R8        perceptual roughness
//   color(2) RG16F     screen-space motion (prev_uv - cur_uv)
// This merges what were three separate pre-passes (SSR / SSAO normal+depth,
// SSR roughness, velocity motion). The rasterised position uses the JITTERED
// VP so coverage matches the main pass; the motion vector is derived from the
// UN-jittered cur/prev VPs so jitter never leaks into it.

struct GbVertex {
    float3 pos     [[attribute(0)]];
    float3 normal  [[attribute(1)]];
    float3 tangent [[attribute(2)]];
    float3 color   [[attribute(3)]];
    float2 uv      [[attribute(4)]];
};

struct GbSkinnedVertex {
    float3  pos     [[attribute(0)]];
    float3  normal  [[attribute(1)]];
    float3  tangent [[attribute(2)]];
    float3  color   [[attribute(3)]];
    float2  uv      [[attribute(4)]];
    ushort4 joints  [[attribute(5)]];
    float4  weights [[attribute(6)]];
};

// buffer(0): the jittered current VP drives the rasterised position (matching
// the main pass exactly); `view` takes the surface normal + position into view
// space; the un-jittered cur/prev VPs derive a jitter-free motion vector.
// Layout matches the Rust GBufferView (4 x float4x4, 256 bytes).
struct GBufferView {
    float4x4 jittered_vp;
    float4x4 cur_vp;
    float4x4 prev_vp;
    float4x4 view;
};

// buffer(2): this draw's current + previous model matrix. Layout matches the
// Rust VelocityModelUniforms (128 bytes). For a static or skinned object with no
// motion the caller passes cur == prev.
struct GbModel {
    float4x4 cur_model;
    float4x4 prev_model;
};

// fragment buffer(0): this draw's roughness, padded to 16 bytes with plain
// floats (a float3 would force 16-byte alignment and bloat the struct). Layout
// matches the Rust SsrPrepassMat push.
struct GbMat {
    float roughness;
    float _pad0;
    float _pad1;
    float _pad2;
};

struct GbVtxOut {
    float4 position    [[position]];
    float3 view_normal;
    // Positive view-space depth (-z); the consumers rebuild view position from
    // it. The cleared background (alpha 0) marks "no geometry".
    float  view_depth;
    float4 cur_clip;
    float4 prev_clip;
};

struct GbFragOut {
    float4 nd    [[color(0)]];  // view normal (xyz) + linear depth (a)
    float  rough [[color(1)]];
    float2 vel   [[color(2)]];  // prev_uv - cur_uv
};

vertex GbVtxOut gbuffer_prepass_vertex(
    GbVertex in             [[stage_in]],
    constant GBufferView &v [[buffer(0)]],
    constant GbModel     &m [[buffer(2)]]
) {
    GbVtxOut out;
    float4 cur_world  = m.cur_model  * float4(in.pos, 1.0);
    float4 prev_world = m.prev_model * float4(in.pos, 1.0);
    float4 view_pos   = v.view * cur_world;
    out.position    = v.jittered_vp * cur_world;
    out.cur_clip    = v.cur_vp  * cur_world;
    out.prev_clip   = v.prev_vp * prev_world;
    float3 world_n  = normalize((m.cur_model * float4(in.normal, 0.0)).xyz);
    out.view_normal = (v.view * float4(world_n, 0.0)).xyz;
    out.view_depth  = -view_pos.z;
    // Skybox sentinel: pin the rasterised depth to the far plane so the sky
    // never occludes scene geometry, matching vertex_main in default.metal.
    if (in.color.b > 1.5) {
        out.position.z = out.position.w * (1.0 - 1e-6);
    }
    return out;
}

// GPU-instanced clusters: per-instance current model at buffer(6), previous at
// buffer(7). Instance transforms never change after init, so the caller binds
// the same buffer at both (cur == prev, zero motion). Clusters never carry the
// sky, so there is no skybox depth pin.
vertex GbVtxOut gbuffer_prepass_vertex_instanced(
    GbVertex in               [[stage_in]],
    constant GBufferView &v   [[buffer(0)]],
    constant float4x4 *cur_i  [[buffer(6)]],
    constant float4x4 *prev_i [[buffer(7)]],
    uint               iid    [[instance_id]]
) {
    float4x4 model    = cur_i[iid];
    float4 cur_world  = model * float4(in.pos, 1.0);
    float4 prev_world = prev_i[iid] * float4(in.pos, 1.0);
    float4 view_pos   = v.view * cur_world;
    GbVtxOut out;
    out.position    = v.jittered_vp * cur_world;
    out.cur_clip    = v.cur_vp  * cur_world;
    out.prev_clip   = v.prev_vp * prev_world;
    float3 world_n  = normalize((model * float4(in.normal, 0.0)).xyz);
    out.view_normal = (v.view * float4(world_n, 0.0)).xyz;
    out.view_depth  = -view_pos.z;
    return out;
}

// Skinned meshes: 4-influence linear-blend skinning with the current and
// previous joint palettes (buffer(8) / buffer(9)) so per-vertex deformation
// produces a correct motion vector. The model matrix is static (cur == prev).
vertex GbVtxOut gbuffer_prepass_vertex_skinned(
    GbSkinnedVertex in         [[stage_in]],
    constant GBufferView &v    [[buffer(0)]],
    constant GbModel     &m    [[buffer(2)]],
    constant float4x4 *cur_j   [[buffer(8)]],
    constant float4x4 *prev_j  [[buffer(9)]]
) {
    float4x4 cur_skin  = in.weights.x * cur_j[in.joints.x]
                       + in.weights.y * cur_j[in.joints.y]
                       + in.weights.z * cur_j[in.joints.z]
                       + in.weights.w * cur_j[in.joints.w];
    float4x4 prev_skin = in.weights.x * prev_j[in.joints.x]
                       + in.weights.y * prev_j[in.joints.y]
                       + in.weights.z * prev_j[in.joints.z]
                       + in.weights.w * prev_j[in.joints.w];
    float4 cur_world  = m.cur_model  * (cur_skin  * float4(in.pos, 1.0));
    float4 prev_world = m.prev_model * (prev_skin * float4(in.pos, 1.0));
    float4 view_pos   = v.view * cur_world;
    GbVtxOut out;
    out.position     = v.jittered_vp * cur_world;
    out.cur_clip     = v.cur_vp  * cur_world;
    out.prev_clip    = v.prev_vp * prev_world;
    float3 skinned_n = (cur_skin * float4(in.normal, 0.0)).xyz;
    float3 world_n   = normalize((m.cur_model * float4(skinned_n, 0.0)).xyz);
    out.view_normal  = (v.view * float4(world_n, 0.0)).xyz;
    out.view_depth   = -view_pos.z;
    return out;
}

fragment GbFragOut gbuffer_prepass_fragment(
    GbVtxOut in         [[stage_in]],
    constant GbMat &mat [[buffer(0)]]
) {
    GbFragOut out;
    out.nd    = float4(normalize(in.view_normal), in.view_depth);
    out.rough = mat.roughness;
    float2 cur_ndc  = in.cur_clip.xy  / in.cur_clip.w;
    float2 prev_ndc = in.prev_clip.xy / in.prev_clip.w;
    float2 cur_uv  = float2(cur_ndc.x  * 0.5 + 0.5, 0.5 - cur_ndc.y  * 0.5);
    float2 prev_uv = float2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    // Stored so the TAA pass can do `prev_uv = uv + motion`.
    out.vel = prev_uv - cur_uv;
    return out;
}

// GPU-driven bindless variant. One unified VS/PS draws the SAME
// per-frame indirect command set the bindless main pass executes, so the
// G-buffer feeder goes fully GPU-driven (no CPU draw loop) for static,
// instanced, chunk, and skinned geometry. The per-object current model +
// roughness come from the GpuObjectData record at buffer(9), indexed by the
// record id the cull kernel baked into each indirect command's [[base_instance]]
// (the same delivery the main + shadow bindless VS use). The previous model
// rides a PARALLEL per-frame buffer at buffer(10), indexed identically. The
// previous vertex position rides a SECOND vertex stream (buffer(2)): for the
// static + instance + chunk prefix the caller binds the same vertex buffer to
// both streams (prev_pos == cur_pos, so the motion is the per-object model
// delta + camera), while the skinned tail binds the current deformed buffer to
// stream 0 and the previous-frame deformed buffer to stream 1, so per-vertex
// skin deformation produces the motion vector. The math is byte-identical to
// gbuffer_prepass_vertex / _skinned above; only the data source differs.

// Mirrors gfx::render_types::GpuObjectData (160 bytes). Only `model` (offset 0)
// and `roughness` (offset 76) are read here, but the full layout must match so
// the [[base_instance]] index lines up with the cull / main buffers.
struct GpuObjectData {
    float4x4      model;
    packed_float3 tint;
    float         roughness;
    packed_float3 emissive;
    float         metallic;
    uint          albedo_index;
    uint          normal_index;
    float         macro_variation;
    float         terrain_blend;
    packed_float3 bb_min;
    float         cull_distance;
    packed_float3 bb_max;
    float         secondary_blend_sharpness;
    uint          albedo_secondary_index;
    uint          normal_secondary_index;
    uint          _pad2;
    uint          _pad3;
};

// buffer(1) = current vertex stream (static VB or current deformed); buffer(2) =
// previous vertex stream (static VB for the prefix, previous-frame deformed for
// the skinned tail). The cull-baked `base_vertex` offsets BOTH streams equally,
// so prefix prev_pos == cur_pos and the skinned tail reads matching deformed
// vertices from each frame's buffer.
struct GbVertexBindless {
    float3 pos      [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float3 color    [[attribute(3)]];
    float3 prev_pos [[attribute(5)]];
};

struct GbVtxOutBindless {
    float4 position    [[position]];
    float3 view_normal;
    float  view_depth;
    float4 cur_clip;
    float4 prev_clip;
    // Sourced from the object record; the FS reads it instead of a per-draw push.
    float  roughness   [[flat]];
};

vertex GbVtxOutBindless gbuffer_prepass_vertex_bindless(
    GbVertexBindless in            [[stage_in]],
    constant GBufferView   &v      [[buffer(0)]],
    constant GpuObjectData *objects [[buffer(9)]],
    constant float4x4      *prev_models [[buffer(10)]],
    uint                    obj_id  [[base_instance]]
) {
    float4x4 model      = objects[obj_id].model;
    float4x4 prev_model = prev_models[obj_id];
    float4 cur_world  = model      * float4(in.pos, 1.0);
    float4 prev_world = prev_model * float4(in.prev_pos, 1.0);
    float4 view_pos   = v.view * cur_world;
    GbVtxOutBindless out;
    out.position    = v.jittered_vp * cur_world;
    out.cur_clip    = v.cur_vp  * cur_world;
    out.prev_clip   = v.prev_vp * prev_world;
    float3 world_n  = normalize((model * float4(in.normal, 0.0)).xyz);
    out.view_normal = (v.view * float4(world_n, 0.0)).xyz;
    out.view_depth  = -view_pos.z;
    out.roughness   = objects[obj_id].roughness;
    // Skybox sentinel: pin to the far plane (matches gbuffer_prepass_vertex).
    if (in.color.b > 1.5) {
        out.position.z = out.position.w * (1.0 - 1e-6);
    }
    return out;
}

fragment GbFragOut gbuffer_prepass_fragment_bindless(GbVtxOutBindless in [[stage_in]]) {
    GbFragOut out;
    out.nd    = float4(normalize(in.view_normal), in.view_depth);
    out.rough = in.roughness;
    float2 cur_ndc  = in.cur_clip.xy  / in.cur_clip.w;
    float2 prev_ndc = in.prev_clip.xy / in.prev_clip.w;
    float2 cur_uv  = float2(cur_ndc.x  * 0.5 + 0.5, 0.5 - cur_ndc.y  * 0.5);
    float2 prev_uv = float2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    out.vel = prev_uv - cur_uv;
    return out;
}
