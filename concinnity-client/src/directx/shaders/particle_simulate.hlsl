// GPU-compute particle simulation kernel for the D3D12 backend. Mirrors the
// `particle_simulate` kernel in src/metal/shaders/particle.metal: one thread
// per slot in the per-emitter particle pool. Each thread ages + integrates
// the slot's current particle; if the slot is dead (lifetime == 0) and the
// per-frame spawn budget still has room, it atomically claims one slot and
// respawns a fresh particle inside a cone of half-angle `acos(spread_cos)`
// around `direction`.

#pragma pack_matrix(column_major)

// One particle slot. Layout must match `GpuParticle` in directx/particle.rs
// (32 bytes per slot - 0/12/16/28).
struct Particle
{
    float3 position;
    float  age;
    float3 velocity;
    float  lifetime;
};

// Per-frame compute uniform. Layout mirrors `ParticleParams` in
// gfx/render_types.rs (112 bytes) field-for-field. HLSL cbuffer packing puts a
// trailing scalar in each float3 + scalar pair into the same 16-byte slot,
// which matches the Rust `[f32; 3]` + `f32` packing exactly.
cbuffer ParticleParams : register(b0)
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

// Particle pool: read-write structured buffer at u0. One Particle per slot.
RWStructuredBuffer<Particle> pool : register(u0);

// One u32 atomic counter at u1, holding the remaining spawn budget for this
// dispatch. Threads racing to spawn decrement it via InterlockedAdd; only
// threads that observed a positive remaining count actually spawn. Using a
// RWByteAddressBuffer so we can InterlockedAdd at byte offset 0 without
// declaring a per-element view.
RWByteAddressBuffer spawn_counter : register(u1);

// Cheap fixed-point hash → unit float in [0, 1). Mirrors the `prng` helper in
// particle.metal. Mutating in-place so a thread that needs several
// uncorrelated samples advances the state between calls.
float prng(inout uint state)
{
    state = state * 1664525u + 1013904223u;
    // Top 24 bits of the hash → mantissa of a float in [0, 1).
    return float(state >> 8) * (1.0 / 16777216.0);
}

// Sample a unit vector inside a cone of half-angle `acos(cone_cos)` centred on
// `axis`. The polar angle is uniform in solid angle so the spawn cloud has no
// axial bunching artefact. Mirrors `sample_cone` in particle.metal.
float3 sample_cone(inout uint rng, float3 axis, float cone_cos)
{
    float u = lerp(cone_cos, 1.0, prng(rng));
    float r = sqrt(max(1.0 - u * u, 0.0));
    float phi = prng(rng) * 6.2831853;
    float3 local = float3(r * cos(phi), r * sin(phi), u);

    // Build any orthonormal basis around `axis`. Pick the world axis least
    // parallel to `axis` so the cross product is well-conditioned.
    float3 up = abs(axis.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 t = normalize(cross(up, axis));
    float3 b = cross(axis, t);
    return normalize(local.x * t + local.y * b + local.z * axis);
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint id = tid.x;
    if (id >= max_particles)
    {
        return;
    }
    Particle pt = pool[id];

    // 1. Age the existing particle in this slot, if any. A `lifetime` of 0
    //    flags a dead slot; everything else is live.
    if (pt.lifetime > 0.0)
    {
        pt.age += dt;
        if (pt.age >= pt.lifetime)
        {
            pt.lifetime = 0.0;
        }
        else
        {
            pt.velocity = pt.velocity + gravity * dt;
            pt.position = pt.position + pt.velocity * dt;
        }
    }

    // 2. If this slot is dead and the per-frame spawn budget still has room,
    //    try to claim one slot from the atomic counter. The counter holds
    //    "remaining" spawns; when it hits zero, no further threads spawn.
    if (pt.lifetime == 0.0 && spawn_budget > 0u)
    {
        // InterlockedAdd with 0xFFFFFFFFu adds -1 in two's-complement,
        // returning the original (pre-subtract) value - exact parity with
        // Metal's `atomic_fetch_sub_explicit(&counter, 1u, relaxed)`.
        uint claimed;
        spawn_counter.InterlockedAdd(0, 0xFFFFFFFFu, claimed);
        if (claimed > 0u && claimed <= spawn_budget)
        {
            uint rng = (id * 747796405u) ^ (random_seed * 2891336453u);
            // Warm up the RNG so adjacent threads decorrelate. `prng` mutates
            // `rng` in-place, so dropping the return value is the actual goal;
            // HLSL has no void-cast form for a function call, so we let the
            // compiler fold the unused result.
            prng(rng);
            float3 dir = sample_cone(rng, normalize(direction), spread_cos);
            float speed = lerp(speed_min, speed_max, prng(rng));
            float life = lerp(lifetime_min, lifetime_max, prng(rng));
            pt.position = position;
            pt.velocity = dir * speed;
            pt.age = 0.0;
            pt.lifetime = max(life, 0.001);
        }
    }

    pool[id] = pt;
}
