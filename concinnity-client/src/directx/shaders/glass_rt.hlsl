#pragma pack_matrix(column_major)

// Glass panel pass, ray-traced reflection variant (DXR 1.1 inline ray tracing).
// Mirrors glass_fragment_rt / glass_fragment_rt_textured in
// src/metal/shaders/glass_rt.metal. Selected over glass.hlsl only while RT is
// live (the scene TLAS is built); otherwise the base glass.hlsl probe/planar
// path runs. Everything outside the reflection itself (two-sided normal, manual
// depth discard, refraction, the Schlick-Fresnel reflection/refraction blend and
// alpha) is identical to glass.hlsl -- toggling RT only sharpens the reflection.
//
// Instead of sampling the box-projected probe cube, the reflection branch traces
// a real reflection ray off the pane against the scene TLAS, so a window mirrors
// actual off-screen geometry. A missed ray falls back to the same probe / sky
// path glass.hlsl uses, so a probe-baked or env-mapped world degrades gracefully.
//
// The trace helpers (RtGeomEntry, the vertex fetchers, rt_shadow, rt_trace,
// rt_shade_hit) are lifted from rt_reflections.hlsl -- keep them in sync. Only
// the register assignments differ: glass already owns t0..t2 (scene/depth/
// prefilter) and the probe cubes are remapped off t7 to t20 (PROBE_CUBES_REGISTER)
// so the RT geometry SRVs fit at t4..t10.
//
// `USE_MSAA` (host #define) matches the depth SRV to the main pass's sample count.
// Compiled through DXC (ps_6_5) because inline RayQuery needs SM 6.5; the base
// glass.hlsl stays on FXC ps_5_1.

cbuffer TransparentView : register(b0)
{
    float4x4 vp;          // world -> clip (jittered when TAA is on)
    float4x4 inv_vp;      // clip -> world
    float4   camera_pos;  // world-space camera, .w unused
    float2   viewport;    // attachment dimensions in pixels
    float    time;        // seconds since startup
    float    prefilter_mip_count; // mips in the sky prefilter cube; 0 = no env map
}

cbuffer GlassParams : register(b1)
{
    float4 centre;  // world-space pane centre, .w unused
    float4 normal;  // unit pane normal (facing direction), .w unused
    float4 tint;    // colour multiplied into the refracted scene, .w unused
    float  opacity;
    float  refraction_strength;
    float  fresnel_power;
    // Unused on the RT path (planar is the RT-off sharp reflection); kept only so
    // the cbuffer byte layout still matches the CPU GlassParamsGpu struct.
    float  planar;
}

// b5: RT tunables + camera + sun. Layout matches render_types::RtParams (144 B).
// Glass uses max_distance / sun_dir / sun_color / prefilter_mip_count (the ray
// origin is the pane surface point, so cam_pos / inv_view are unused here).
cbuffer RtParams : register(b5)
{
    float    rt_intensity;
    float    rt_max_distance;
    float    rt_tan_half_fov_y;
    float    rt_aspect;
    float    rt_prefilter_mip_count;
    float    rt_pad0;
    float    rt_pad1;
    float    rt_pad2;
    float4   rt_cam_pos;
    float4   sun_dir;     // xyz: world unit direction toward the sun (= L)
    float4   sun_color;   // xyz: sun radiance
    float4x4 rt_inv_view;
}

// Per-instance geometry table. Explicit scalar members so the byte offsets match
// the 128-byte #[repr(C)] render_types::RtGeomEntry exactly (independent of how
// the compiler would pack a float3). Lifted from rt_reflections.hlsl.
struct RtGeomEntry
{
    uint  index_offset;
    uint  base_vertex;
    uint  albedo_index;
    uint  normal_index;
    float tint_r;
    float tint_g;
    float tint_b;
    float roughness;
    float metallic;
    float emissive_r;
    float emissive_g;
    float emissive_b;
    float4x4 model;
    uint  emissive_map_index;
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
};

Texture2D<float4> scene_color : register(t0);
#if USE_MSAA
Texture2DMS<float> scene_depth : register(t1);
#else
Texture2D<float>   scene_depth : register(t1);
#endif
// Sky IBL prefilter cube: the reflection fallback where no probe covers the pane.
TextureCube<float4> prefilter_cube : register(t2);

// RT geometry, bound as root SRVs (t4..t10, the slots free once planar (t3) is
// dropped and the probe cubes are remapped to t20). The scene TLAS is reached
// through the t4 root SRV by inline RayQuery.
RaytracingAccelerationStructure scene_tlas : register(t4);
ByteAddressBuffer verts           : register(t5);
ByteAddressBuffer indices         : register(t6); // u32
ByteAddressBuffer skinned_verts   : register(t8);
ByteAddressBuffer skinned_indices : register(t9); // u16
StructuredBuffer<RtGeomEntry> geom : register(t10);

// t0, space1: the bindless albedo + normal-map pool, identical to the main pass.
// Only the textured variant references it.
Texture2D tex_pool[] : register(t0, space1);

SamplerState post_samp   : register(s0); // linear-clamp (scene snapshot)
SamplerState repeat_smp  : register(s1); // linear-repeat (hit albedo / normal map)

// The probe cube array (remapped to t20 via PROBE_CUBES_REGISTER), the ProbeBlock
// cbuffer (b4), and cube_sampler (s2) are declared in probe_common.hlsl,
// concatenated ahead of this shader (no #include handler on DX). The miss path
// reuses probe_set_specular to fall back to the local box-projected scene capture.

static const float RT_F0 = 0.04;            // dielectric base reflectance
static const uint  RT_SKINNED_FLAG = 0x80000000u; // bit 31 of normal_index

struct VsIn
{
    float3 pos     : POSITION;
    float3 normal  : NORMAL;
    float3 tangent : TANGENT;
    float3 color   : COLOR;
    float2 uv      : TEXCOORD;
};

struct VsOut
{
    float4 sv_pos    : SV_POSITION;
    float3 world_pos : TEXCOORD0;
};

VsOut vs_main(VsIn input)
{
    VsOut output;
    // Quad vertices are pre-transformed into world space at build time.
    output.world_pos = input.pos;
    output.sv_pos = mul(vp, float4(input.pos, 1.0));
    return output;
}

// Vertex attribute fetchers into the shared 56-byte Vertex (pos@0, normal@12,
// tangent@24, uv@48). Lifted from rt_reflections.hlsl.
float3 rt_vertex_normal(uint vi)  { return asfloat(verts.Load3(vi * 56 + 12)); }
float3 rt_vertex_tangent(uint vi) { return asfloat(verts.Load3(vi * 56 + 24)); }
float2 rt_vertex_uv(uint vi)      { return asfloat(verts.Load2(vi * 56 + 48)); }
float3 rt_skinned_normal(uint vi)  { return asfloat(skinned_verts.Load3(vi * 56 + 12)); }
float3 rt_skinned_tangent(uint vi) { return asfloat(skinned_verts.Load3(vi * 56 + 24)); }
float2 rt_skinned_uv(uint vi)      { return asfloat(skinned_verts.Load2(vi * 56 + 48)); }

// Per-pixel reflection ray setup. For glass this is the pane surface point + the
// mirror direction (no G-buffer, unlike the fullscreen RT resolve).
struct RtSetup
{
    float3 base;      // miss colour when no IBL is bound (glass: white rim)
    float3 world_pos; // pane surface point (probe box-projection origin)
    float3 origin;    // ray origin (surface point nudged along the normal)
    float3 dir;       // world-space reflection direction
    float  roughness; // 0 -- a pane is a perfect mirror
    float  max_mip;   // prefilter_mip_count - 1
    bool   ibl;       // an EnvironmentMap is bound
};

// Result of tracing the reflection ray. Lifted from rt_reflections.hlsl.
struct RtTrace
{
    bool   hit;
    bool   skin;
    float3 normal;
    float3 tangent;
    float2 uv;
    uint   albedo_index;
    uint   normal_index;
    uint   emissive_map_index;
    float3 tint;
    float  roughness;
    float  metallic;
    float3 emissive;
    float3 env;
    float  shadow;
};

// Trace a shadow ray from `hp` toward the sun; 0 when occluded, 1 when lit.
// TMax is rt_max_distance, matching glass_rt.metal (no separate cap on the pane
// trace -- the GLASS_SHADOW_MAX_DIST cap is a Layer-2 mesh concern only).
float rt_shadow(float3 hp, float3 n)
{
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
    RayDesc sr;
    sr.Origin = hp + n * 0.02;
    sr.Direction = normalize(sun_dir.xyz);
    sr.TMin = 0.001;
    sr.TMax = rt_max_distance;
    sq.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFF, sr);
    sq.Proceed();
    return (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0 : 1.0;
}

// Trace the reflection ray and gather the hit attributes (or the miss colour).
// Lifted from rt_reflections.hlsl::rt_trace; the miss path reuses probe_common's
// probe_set_specular so glass degrades to the same probe / sky fallback.
RtTrace rt_trace(RtSetup s)
{
    RtTrace t;
    t.hit = false;
    t.skin = false;
    t.normal = 0.0.xxx;
    t.tangent = 0.0.xxx;
    t.uv = 0.0.xx;
    t.albedo_index = 0;
    t.normal_index = 0;
    t.emissive_map_index = 0;
    t.tint = 0.0.xxx;
    t.roughness = 0.0;
    t.metallic = 0.0;
    t.emissive = 0.0.xxx;
    t.env = s.base;
    t.shadow = 1.0;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    RayDesc ray;
    ray.Origin = s.origin;
    ray.Direction = s.dir;
    ray.TMin = 0.01;
    ray.TMax = rt_max_distance;
    q.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        RtGeomEntry e = geom[q.CommittedInstanceID()];
        bool skin = (e.normal_index & RT_SKINNED_FLAG) != 0u;
        uint nidx = e.normal_index & ~RT_SKINNED_FLAG;
        uint tri = q.CommittedPrimitiveIndex();
        uint o = e.index_offset + tri * 3;
        float2 b = q.CommittedTriangleBarycentrics();
        float w0 = 1.0 - b.x - b.y;

        float3 nl, tl;
        float2 uv;
        if (skin)
        {
            uint i0 = (skinned_indices.Load((o * 2) & ~3u) >> ((o & 1u) * 16u)) & 0xFFFFu;
            uint i1 = (skinned_indices.Load(((o + 1) * 2) & ~3u) >> (((o + 1) & 1u) * 16u)) & 0xFFFFu;
            uint i2 = (skinned_indices.Load(((o + 2) * 2) & ~3u) >> (((o + 2) & 1u) * 16u)) & 0xFFFFu;
            i0 += e.base_vertex; i1 += e.base_vertex; i2 += e.base_vertex;
            nl = rt_skinned_normal(i0) * w0
               + rt_skinned_normal(i1) * b.x
               + rt_skinned_normal(i2) * b.y;
            tl = rt_skinned_tangent(i0) * w0
               + rt_skinned_tangent(i1) * b.x
               + rt_skinned_tangent(i2) * b.y;
            uv = rt_skinned_uv(i0) * w0
               + rt_skinned_uv(i1) * b.x
               + rt_skinned_uv(i2) * b.y;
        }
        else
        {
            uint i0 = indices.Load(o * 4) + e.base_vertex;
            uint i1 = indices.Load((o + 1) * 4) + e.base_vertex;
            uint i2 = indices.Load((o + 2) * 4) + e.base_vertex;
            nl = rt_vertex_normal(i0) * w0
               + rt_vertex_normal(i1) * b.x
               + rt_vertex_normal(i2) * b.y;
            tl = rt_vertex_tangent(i0) * w0
               + rt_vertex_tangent(i1) * b.x
               + rt_vertex_tangent(i2) * b.y;
            uv = rt_vertex_uv(i0) * w0
               + rt_vertex_uv(i1) * b.x
               + rt_vertex_uv(i2) * b.y;
        }

        float3 nw = normalize(mul(e.model, float4(nl, 0.0)).xyz);
        if (dot(nw, s.dir) > 0.0) nw = -nw;
        float3 tw = mul(e.model, float4(tl, 0.0)).xyz;

        t.hit = true;
        t.skin = skin;
        t.normal = nw;
        t.tangent = tw;
        t.uv = uv;
        t.albedo_index = e.albedo_index;
        t.normal_index = nidx;
        t.emissive_map_index = e.emissive_map_index;
        t.tint = float3(e.tint_r, e.tint_g, e.tint_b);
        t.roughness = e.roughness;
        t.metallic = e.metallic;
        t.emissive = float3(e.emissive_r, e.emissive_g, e.emissive_b);

        float3 hp = s.origin + s.dir * q.CommittedRayT();
        t.shadow = rt_shadow(hp, nw);
    }
    else
    {
        float lod = s.roughness * s.max_mip;
        if (!s.ibl)
        {
            t.env = s.base;
        }
        else if (probes.count > 0u)
        {
            t.env = probe_set_specular(probes, s.world_pos, s.dir, lod);
        }
        else
        {
            t.env = prefilter_cube.SampleLevel(cube_sampler, s.dir, lod).rgb;
        }
    }
    return t;
}

// Metallic/roughness hit shading. Lifted from rt_reflections.hlsl::rt_shade_hit.
float3 rt_shade_hit(RtSetup s, RtTrace t, float3 N, float3 albedo)
{
    float3 F0     = lerp(RT_F0.xxx, albedo, t.metallic);
    float3 diff_a = albedo * (1.0 - t.metallic);
    float  ndl    = saturate(dot(N, sun_dir.xyz));
    float3 sun    = diff_a * sun_color.xyz * ndl * t.shadow;
    if (!s.ibl) return sun + (diff_a + F0) * 0.03 + t.emissive;
    float3 refl = reflect(s.dir, N);
    float3 spec = prefilter_cube.SampleLevel(cube_sampler, refl, t.roughness * s.max_mip).rgb;
    float3 diff = prefilter_cube.SampleLevel(cube_sampler, N, s.max_mip).rgb * diff_a;
    return sun + diff + spec * F0 + t.emissive;
}

// Shared glass surface setup: two-sided normal, manual depth discard, refraction.
// Identical to glass.hlsl's head. `discard`s where opaque geometry occludes the
// pane (the transparent pass binds no depth attachment).
struct GlassSurface
{
    float3 refracted;
    float3 n;
    float3 view_dir;
    float  ndv;
};

GlassSurface glass_surface(VsOut input)
{
    GlassSurface o;
    o.view_dir = normalize(camera_pos.xyz - input.world_pos);
    float3 n = normalize(normal.xyz);
    if (dot(n, o.view_dir) < 0.0) n = -n;
    o.n = n;

    float2 vp_dim = max(viewport, float2(1.0, 1.0));
    float2 frag_uv = float2(input.sv_pos.x / vp_dim.x, input.sv_pos.y / vp_dim.y);

    int2 pixel = min(int2(input.sv_pos.xy), int2(vp_dim) - int2(1, 1));
#if USE_MSAA
    float scene_self_depth = scene_depth.Load(pixel, 0);
#else
    float scene_self_depth = scene_depth.Load(int3(pixel, 0));
#endif
    if (scene_self_depth < input.sv_pos.z)
    {
        discard;
    }

    float2 refract_uv = clamp(frag_uv + n.xy * refraction_strength,
                              float2(0.001, 0.001), float2(0.999, 0.999));
    o.refracted = scene_color.Sample(post_samp, refract_uv).rgb * tint.rgb;
    o.ndv = saturate(dot(n, o.view_dir));
    return o;
}

// Build the reflection ray setup from the pane surface. `base` is white so the
// no-IBL miss yields glass.hlsl's white rim.
RtSetup glass_setup(GlassSurface surf, float3 world_pos)
{
    RtSetup s;
    s.base = float3(1.0, 1.0, 1.0);
    s.world_pos = world_pos;
    s.origin = world_pos + surf.n * 0.02;
    s.dir = reflect(-surf.view_dir, surf.n);
    s.roughness = 0.0;
    s.ibl = prefilter_mip_count > 0.5;
    s.max_mip = prefilter_mip_count - 1.0;
    return s;
}

// Schlick-Fresnel reflection/refraction blend + alpha. Identical to glass.hlsl.
float4 glass_blend(GlassSurface surf, float3 reflection)
{
    float rim = pow(1.0 - surf.ndv, max(fresnel_power, 1e-3));
    float refl_weight = saturate(0.04 + 0.96 * rim);
    float3 colour = lerp(surf.refracted, reflection, refl_weight);
    float alpha = saturate(lerp(opacity, 1.0, rim));
    return float4(colour, alpha);
}

// Flat variant: reflected-hit albedo = per-object material tint (no texture pool).
float4 ps_main_rt(VsOut input) : SV_TARGET
{
    GlassSurface surf = glass_surface(input);
    RtSetup s = glass_setup(surf, input.world_pos);
    RtTrace t = rt_trace(s);
    float3 reflection = t.hit ? rt_shade_hit(s, t, t.normal, t.tint) : t.env;
    return glass_blend(surf, reflection);
}

// Textured variant: reflected-hit albedo / normal / emissive from the bindless
// pool. Mirrors rt_reflections_frag_textured.
float4 ps_main_rt_textured(VsOut input) : SV_TARGET
{
    GlassSurface surf = glass_surface(input);
    RtSetup s = glass_setup(surf, input.world_pos);
    RtTrace t = rt_trace(s);

    float3 reflection;
    if (t.hit)
    {
        // Level 0: a reflected ray's screen-space UV gradients are unstable, so
        // sample the base mip to avoid gradient-driven mip thrash.
        float3 albedo = t.tint
                      * tex_pool[NonUniformResourceIndex(t.albedo_index)]
                            .SampleLevel(repeat_smp, t.uv, 0.0).rgb;
        float3 N = t.normal;
        float tlen = length(t.tangent);
        if (tlen > 1e-4)
        {
            float3 nm = tex_pool[NonUniformResourceIndex(t.normal_index)]
                            .SampleLevel(repeat_smp, t.uv, 0.0).xyz * 2.0 - 1.0;
            float3 T = t.tangent / tlen;
            T = normalize(T - N * dot(N, T));        // Gram-Schmidt
            float3 B = cross(N, T);
            N = normalize(T * nm.x + B * nm.y + N * nm.z);
        }
        if (t.emissive_map_index != 0u)
        {
            t.emissive *= tex_pool[NonUniformResourceIndex(t.emissive_map_index)]
                              .SampleLevel(repeat_smp, t.uv, 0.0).rgb;
        }
        reflection = rt_shade_hit(s, t, N, albedo);
    }
    else
    {
        reflection = t.env;
    }
    return glass_blend(surf, reflection);
}
