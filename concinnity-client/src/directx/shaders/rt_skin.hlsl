#pragma pack_matrix(column_major)

// Compute skinning for ray tracing (DXR). The main pass skins in the vertex
// shader, so no deformed-vertex buffer exists for the BVH to trace against.
// This kernel produces one: it reads the bind-pose skinned vertices + a
// per-object joint palette and writes posed (model-space) plain `Vertex`s into
// a shared deformed buffer, which the RT acceleration-structure build then
// traces. One dispatch per skinned object over its vertex range; the deformed
// buffer mirrors the skinned vertex buffer's indexing so the existing (u16)
// skinned index buffer addresses it directly. Ports src/metal/shaders/rt_skin.metal.

// b0: which slice of the shared buffers this dispatch deforms. Matches the
// 16-byte `SkinParams` in directx/raytrace.rs.
cbuffer SkinParams : register(b0)
{
    uint vertex_base;   // first vertex of this object in the shared buffers
    uint vertex_count;  // vertices to deform this dispatch
    uint joint_count;   // palette size (joint indices are clamped below it)
    uint _pad;
};

// t0: bind-pose skinned vertices, raw. The shader fetches the 80-byte
// `SkinnedVertex` fields (gfx::mesh_payload::SkinnedVertex) by byte offset:
// pos@0, normal@12, tangent@24, color@36, uv@48, uint16 joints[4]@56,
// float4 weights@64.
ByteAddressBuffer src : register(t0);
// t1: this object's joint palette (one float4x4 per joint). Wrapped in a
// struct with an explicit `column_major` qualifier to pin the storage layout
// (the pragma's behaviour for raw element-type matrices in a StructuredBuffer
// is ambiguous), matching skinned_vert.hlsl. The Rust upload writes
// `[[f32;4];4]` matrices in column-major order.
struct ColMat4 { column_major float4x4 m; };
StructuredBuffer<ColMat4> palette : register(t1);
// u0: deformed (posed) vertices in the static 56-byte `Vertex` layout
// (gfx::mesh_payload::Vertex): pos@0, normal@12, tangent@24, color@36, uv@48.
RWByteAddressBuffer dst : register(u0);

static const uint SKINNED_VERTEX_STRIDE = 80;
static const uint VERTEX_STRIDE = 56;

// Two u16 joint indices are packed into each 32-bit word the index pair lives
// in; `joints[4]` occupies the 8 bytes at offset 56. Load the two words and
// bit-extract the four 16-bit indices.
uint2 load_joints(uint base)
{
    uint2 words = src.Load2(base + 56);
    // joints[0] = low 16 of word0, joints[1] = high 16 of word0, etc. The
    // unpacked pair (idx0, idx1) lives in words.x; (idx2, idx3) in words.y.
    return words;
}

[numthreads(64, 1, 1)]
void rt_skin(uint3 gid : SV_DispatchThreadID)
{
    if (gid.x >= vertex_count) return;
    uint idx = vertex_base + gid.x;
    uint sbase = idx * SKINNED_VERTEX_STRIDE;

    float3 pos     = asfloat(src.Load3(sbase + 0));
    float3 normal  = asfloat(src.Load3(sbase + 12));
    float3 tangent = asfloat(src.Load3(sbase + 24));
    float3 color   = asfloat(src.Load3(sbase + 36));
    float2 uv      = asfloat(src.Load2(sbase + 48));
    uint2  jw      = load_joints(sbase);
    float4 weights = asfloat(src.Load4(sbase + 64));

    uint j0 = jw.x & 0xFFFFu;
    uint j1 = (jw.x >> 16) & 0xFFFFu;
    uint j2 = jw.y & 0xFFFFu;
    uint j3 = (jw.y >> 16) & 0xFFFFu;

    uint last = joint_count == 0u ? 0u : joint_count - 1u;
    float4x4 skin = weights.x * palette[min(j0, last)].m
                  + weights.y * palette[min(j1, last)].m
                  + weights.z * palette[min(j2, last)].m
                  + weights.w * palette[min(j3, last)].m;
    float3x3 skin3 = (float3x3)skin;

    float3 o_pos     = mul(skin, float4(pos, 1.0)).xyz;
    float3 o_normal  = normalize(mul(skin3, normal));
    float3 o_tangent = mul(skin3, tangent);

    uint dbase = idx * VERTEX_STRIDE;
    dst.Store3(dbase + 0,  asuint(o_pos));
    dst.Store3(dbase + 12, asuint(o_normal));
    dst.Store3(dbase + 24, asuint(o_tangent));
    dst.Store3(dbase + 36, asuint(color));
    dst.Store2(dbase + 48, asuint(uv));
}
