#include <metal_stdlib>
using namespace metal;

// Compute skinning for ray tracing. The main pass skins in the vertex shader, so
// no deformed-vertex buffer exists for the BVH to trace against. This kernel
// produces one: it reads the bind-pose skinned vertices + a per-object joint
// palette and writes posed (model-space) plain `Vertex`s into a shared deformed
// buffer, which the RT acceleration-structure build then traces. One dispatch
// per skinned object over its vertex range; the deformed buffer mirrors the
// skinned vertex buffer's indexing so the existing (u16) skinned index buffer
// addresses it directly.

// Matches gfx::mesh_payload::SkinnedVertex (repr(C), 80-byte stride). Packed
// types keep the field offsets identical to the Rust struct.
struct SkinnedVtxIn {
    packed_float3 pos;      // 0
    packed_float3 normal;   // 12
    packed_float3 tangent;  // 24
    packed_float3 color;    // 36
    packed_float2 uv;       // 48
    ushort        joints[4];// 56
    packed_float4 weights;  // 64  (..80)
};

// Matches gfx::mesh_payload::Vertex (repr(C), 56-byte stride) - the same layout
// the static RT vertex fetchers read.
struct VtxOut {
    packed_float3 pos;      // 0
    packed_float3 normal;   // 12
    packed_float3 tangent;  // 24
    packed_float3 color;    // 36
    packed_float2 uv;       // 48  (..56)
};

// buffer(3): which slice of the shared buffers this dispatch deforms.
struct SkinParams {
    uint vertex_base;   // first vertex of this object in the shared buffers
    uint vertex_count;  // vertices to deform this dispatch
    uint joint_count;   // palette size (joint indices are clamped below it)
    uint _pad;
};

kernel void rt_skin(
    device const SkinnedVtxIn* src     [[buffer(0)]],
    device VtxOut*             dst     [[buffer(1)]],
    constant float4x4*         palette [[buffer(2)]],
    constant SkinParams&       p       [[buffer(3)]],
    uint                       gid     [[thread_position_in_grid]]
) {
    if (gid >= p.vertex_count) return;
    uint idx = p.vertex_base + gid;
    SkinnedVtxIn v = src[idx];

    uint last = p.joint_count == 0u ? 0u : p.joint_count - 1u;
    float4x4 skin = v.weights.x * palette[min((uint)v.joints[0], last)]
                  + v.weights.y * palette[min((uint)v.joints[1], last)]
                  + v.weights.z * palette[min((uint)v.joints[2], last)]
                  + v.weights.w * palette[min((uint)v.joints[3], last)];
    float3x3 skin3 = float3x3(skin[0].xyz, skin[1].xyz, skin[2].xyz);

    VtxOut o;
    o.pos     = (skin * float4(float3(v.pos), 1.0)).xyz;
    o.normal  = normalize(skin3 * float3(v.normal));
    o.tangent = skin3 * float3(v.tangent);
    o.color   = v.color;
    o.uv      = v.uv;
    dst[idx]  = o;
}
