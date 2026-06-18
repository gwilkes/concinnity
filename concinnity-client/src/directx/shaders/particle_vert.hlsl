// Particle render pipeline - vertex shader. Mirrors `particle_vertex` in
// src/metal/shaders/particle.metal. Drawn with `DrawInstanced(4, max_particles)`
// as a triangle strip: each instance reads its `Particle` from the pool by
// `iid`, derives a camera-facing axis pair from the bound `ParticleView`, and
// emits one corner of a billboard quad.

#pragma pack_matrix(column_major)

struct Particle
{
    float3 position;
    float  age;
    float3 velocity;
    float  lifetime;
};

// Per-frame view inputs. 96 bytes: float4x4 + two (float3 + pad) slots.
cbuffer ParticleView : register(b0)
{
    float4x4 vp;
    float3   cam_right;
    float    _vpad0;
    float3   cam_up;
    float    _vpad1;
};

// Per-emitter uniform - same layout as the compute pass. The vertex shader
// only reads the gradient fields, but binding the full struct keeps the
// host-side payload single-pushed.
cbuffer ParticleParams : register(b1)
{
    float3 position;
    float  spread_cos;
    float3 direction;
    float  speed_min;
    float3 gravity;
    float  speed_max;
    float4 color_start;
    float4 color_end;
    float  lifetime_min;
    float  lifetime_max;
    float  size_start;
    float  size_end;
    float  dt;
    uint   spawn_budget;
    uint   random_seed;
    uint   max_particles;
};

// Pool bound as a read-only structured buffer (root SRV at t0). The compute
// pass writes through a separate UAV; a state transition between the dispatch
// and this draw flips the resource into NON_PIXEL_SHADER_RESOURCE.
StructuredBuffer<Particle> pool : register(t0);

struct VsOut
{
    float4 sv_pos       : SV_POSITION;
    float2 uv           : TEXCOORD0;
    float4 color        : TEXCOORD1;
    float  discard_flag : TEXCOORD2;
};

VsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    VsOut o;
    Particle pt = pool[iid];

    // Dead slot → emit a degenerate quad clipped behind the near plane. The
    // fragment shader also discards on `discard_flag` so any stray rasterised
    // pixels (numerical edge case at exactly w=0) draw nothing.
    if (pt.lifetime <= 0.0)
    {
        o.sv_pos = float4(0.0, 0.0, -2.0, 1.0);
        o.uv = float2(0.0, 0.0);
        o.color = float4(0.0, 0.0, 0.0, 0.0);
        o.discard_flag = 1.0;
        return o;
    }

    float t = saturate(pt.age / pt.lifetime);
    float size = lerp(size_start, size_end, t);
    float4 color = lerp(color_start, color_end, t);

    // 0..3 → (-1,-1), (+1,-1), (-1,+1), (+1,+1) for a triangle strip.
    float2 corner = float2(
        (vid & 1u) == 0u ? -1.0 : 1.0,
        (vid & 2u) == 0u ? -1.0 : 1.0);

    float3 right = cam_right * (corner.x * 0.5 * size);
    float3 up    = cam_up    * (corner.y * 0.5 * size);
    float3 world = pt.position + right + up;
    o.sv_pos = mul(vp, float4(world, 1.0));

    // 0..1 in each axis; V is flipped at sample time below to match the rest
    // of the engine's textures (V=0 at the top of the image).
    o.uv = corner * 0.5 + 0.5;
    o.color = color;
    o.discard_flag = 0.0;
    return o;
}
