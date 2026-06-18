#pragma pack_matrix(column_major)

// Hardware ray-traced reflections (DXR 1.1 inline ray tracing). A fullscreen
// pixel pass that, per glossy pixel, rebuilds a world-space surface point +
// normal from the SSR pre-pass G-buffer, traces a reflection ray against the
// scene's top-level acceleration structure with `RayQuery`, and composites the
// reflected colour over the base scene with the same Fresnel/gloss weighting SSR
// uses. Unlike SSR the ray is a real world-space trace, so reflected geometry
// that is off-screen still appears.
//
// Ports src/metal/shaders/rt_reflections.metal. Two hit-shading variants share
// all the setup + trace logic:
//   * rt_reflections_frag           - flat: material tint only (the fallback
//     used when the bindless texture pool is unavailable).
//   * rt_reflections_frag_textured  - samples the hit's albedo + normal-map
//     textures from the bindless pool, the path standard worlds take.
// Both shade the hit with a metallic/roughness response (sun diffuse for
// dielectrics + split IBL) and fall back to the IBL prefilter cubemap on a miss,
// exactly like SSR. On a hit both trace a second (shadow) ray toward the sun, so
// the sun term is masked where the reflected surface is occluded.

// b0: RT tunables + camera + sun. Layout matches render_types::RtParams (144 B).
cbuffer RtParams : register(b0)
{
    float    intensity;
    float    max_distance;
    float    tan_half_fov_y;
    float    aspect;
    float    prefilter_mip_count;
    float    _pad0;
    float    _pad1;
    float    _pad2;
    float4   cam_pos;     // xyz: world camera position (ray origin)
    float4   sun_dir;     // xyz: world unit direction toward the sun (= L)
    float4   sun_color;   // xyz: sun radiance
    float4x4 inv_view;    // camera-to-world (column-major)
};

// t3: per-instance geometry table. Declared with explicit scalar members so the
// byte offsets match the 128-byte `#[repr(C)]` render_types::RtGeomEntry exactly
// (index_offset@0, base_vertex@4, albedo_index@8, normal_index@12, tint@16,
// roughness@28, metallic@32, emissive@36, model@48, emissive_map_index@112),
// independent of how the compiler would pack a `float3`. The `_pad` tail rounds
// the struct to 128 bytes so its StructuredBuffer stride matches the Rust side.
struct RtGeomEntry
{
    uint  index_offset;  // element offset of this object's first index
    uint  base_vertex;   // added to each fetched index
    uint  albedo_index;  // bindless albedo pool index (textured variant)
    uint  normal_index;  // bindless normal-map pool index (textured variant)
    float tint_r;        // base albedo for hit shading
    float tint_g;
    float tint_b;
    float roughness;     // hit IBL specular mip selection
    float metallic;      // hit PBR response (metals tint the env reflection)
    float emissive_r;    // self-emission added to the hit colour
    float emissive_g;
    float emissive_b;
    float4x4 model;      // object-to-world (column-major)
    uint  emissive_map_index; // bindless emissive-map index (0 = none)
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
};

// t0: the scene top-level acceleration structure (bound as a root SRV).
RaytracingAccelerationStructure scene_tlas : register(t0);
// t1/t2: the shared vertex + (u32) index buffers, raw (bound as root SRVs). The
// shader fetches the hit triangle's attributes from these directly.
ByteAddressBuffer verts   : register(t1);
ByteAddressBuffer indices : register(t2);
// t3: the per-instance geometry table (bound as a StructuredBuffer root SRV).
StructuredBuffer<RtGeomEntry> geom : register(t3);
// t8/t9: the deformed (posed) skinned vertex buffer + the u16 skinned index
// buffer, raw (bound as root SRVs). A skinned hit fetches its triangle from
// these instead of the static u32 buffers; both bind a 1-element dummy when the
// scene carries no skinned geometry, so the binding is always valid.
ByteAddressBuffer skinned_verts   : register(t8);
ByteAddressBuffer skinned_indices : register(t9);

// Screen-space inputs reused from the SSR resolve pass.
Texture2D    scene_tex : register(t4); // hdr_resolve (base scene colour)
Texture2D    gbuffer   : register(t5); // view normal (xyz) + linear depth (a)
Texture2D    rough_tex : register(t6); // roughness (r)
TextureCube  prefilter : register(t7); // IBL prefilter cube (miss fallback)

// t0, space1: the bindless albedo + normal-map pool, identical to the main
// pass. Only the textured variant references it.
Texture2D tex_pool[] : register(t0, space1);

SamplerState smp        : register(s0); // linear-clamp (scene / gbuffer / roughness)
SamplerState cube_smp   : register(s1); // linear-clamp cube mip-linear (prefilter)
SamplerState repeat_smp : register(s2); // linear-repeat (hit albedo / normal map)

struct VsOut
{
    float4 sv_pos : SV_POSITION;
    float2 uv     : TEXCOORD0;
};

// Fullscreen triangle from SV_VertexID; UV flip matches the D3D top-left origin
// so the pass samples the G-buffer + scene at the main pass's pixel coordinates.
VsOut rt_fullscreen_vert(uint vid : SV_VertexID)
{
    float2 pos = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    VsOut o;
    o.sv_pos = float4(pos, 0.0, 1.0);
    o.uv     = float2((pos.x + 1.0) * 0.5, 1.0 - (pos.y + 1.0) * 0.5);
    return o;
}

static const float RT_ROUGH_CUT = 0.6;  // surfaces rougher than this get no reflection
static const float RT_F0        = 0.04; // dielectric base reflectance for the Fresnel

// Skinned objects flag bit 31 of `normal_index`; the trace then fetches the hit
// triangle from the deformed-vertex / u16 skinned index buffers (which mirror
// the static layout) instead of the static u32 ones. Matches
// render_types::RT_SKINNED_FLAG / raytrace.rs.
static const uint RT_SKINNED_FLAG = 0x80000000u;

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// Byte-identical to ssr_view_pos.
float3 rt_view_pos(float2 uv, float depth, float tan_y, float asp)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_y * asp, ndc.y * tan_y, -1.0) * depth;
}

// Vertex attribute fetchers into the shared 14-float Vertex (pos@0, normal@12,
// tangent@24, uv@48), addressed by absolute vertex index `vi`.
float3 rt_vertex_normal(uint vi)  { return asfloat(verts.Load3(vi * 56 + 12)); }
float3 rt_vertex_tangent(uint vi) { return asfloat(verts.Load3(vi * 56 + 24)); }
float2 rt_vertex_uv(uint vi)      { return asfloat(verts.Load2(vi * 56 + 48)); }

// Same fetchers into the deformed (posed) skinned vertex buffer, which carries
// the identical 56-byte `Vertex` layout the skin compute kernel writes.
float3 rt_skinned_normal(uint vi)  { return asfloat(skinned_verts.Load3(vi * 56 + 12)); }
float3 rt_skinned_tangent(uint vi) { return asfloat(skinned_verts.Load3(vi * 56 + 24)); }
float2 rt_skinned_uv(uint vi)      { return asfloat(skinned_verts.Load2(vi * 56 + 48)); }

// Common per-pixel setup shared by both hit-shading variants.
struct RtSetup
{
    bool   reflects;  // false -> sky / too rough, write base unchanged
    float3 base;
    float3 origin;    // ray origin (surface point nudged along the normal)
    float3 dir;       // world-space reflection direction
    float  weight;    // saturate(fresnel * gloss * intensity)
    float  roughness;
    float  max_mip;   // prefilter_mip_count - 1
    bool   ibl;       // an EnvironmentMap is bound
};

RtSetup rt_setup(float2 uv)
{
    RtSetup s;
    s.reflects = false;
    s.base = scene_tex.Sample(smp, uv).rgb;
    s.origin = 0.0.xxx;
    s.dir = 0.0.xxx;
    s.weight = 0.0;
    s.max_mip = 0.0;
    s.ibl = false;
    float4 g = gbuffer.Sample(smp, uv);
    float depth = g.a;
    s.roughness = rough_tex.Sample(smp, uv).r;
    if (depth <= 0.0) return s;                 // background / sky
    float gloss = saturate((RT_ROUGH_CUT - s.roughness) / RT_ROUGH_CUT);
    if (gloss <= 0.0) return s;

    float3 Nv = normalize(g.xyz);
    float3 Pv = rt_view_pos(uv, depth, tan_half_fov_y, aspect);
    float3 Pw = mul(inv_view, float4(Pv, 1.0)).xyz;
    float3 Nw = normalize(mul(inv_view, float4(Nv, 0.0)).xyz);
    float3 V  = normalize(cam_pos.xyz - Pw);

    s.origin = Pw + Nw * 0.01;                  // nudge off the surface
    s.dir    = reflect(-V, Nw);
    s.ibl    = prefilter_mip_count > 0.5;
    s.max_mip = prefilter_mip_count - 1.0;
    float ndv     = saturate(dot(Nw, V));
    float fresnel = RT_F0 + (1.0 - RT_F0) * pow(1.0 - ndv, 5.0);
    s.weight = saturate(fresnel * gloss * intensity);
    s.reflects = true;
    return s;
}

// Result of tracing the reflection ray: the interpolated world normal + uv +
// material on a hit, or the environment colour on a miss. `shadow` is the sun
// visibility at the hit (1 = lit, 0 = a shadow ray to the sun was occluded).
struct RtTrace
{
    bool   hit;
    bool   skin;       // hit was a skinned object: fetch from the deformed / u16 buffers
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

// Trace a shadow ray from `hp` toward the sun; returns 0 when occluded (cast
// shadow inside the reflection), 1 when the sun is visible.
float rt_shadow(float3 hp, float3 n)
{
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
    RayDesc sr;
    sr.Origin = hp + n * 0.02;
    sr.Direction = normalize(sun_dir.xyz);
    sr.TMin = 0.001;
    sr.TMax = max_distance;
    sq.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFF, sr);
    sq.Proceed();
    return (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0 : 1.0;
}

// Trace the reflection ray and gather the hit attributes (or the miss colour).
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
    ray.TMax = max_distance;
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
            // Skinned: u16 indices (2 bytes each) into the deformed buffer; the
            // skinned BLAS bakes absolute indices, so base_vertex is 0. Load2
            // straddles two u16s, so mask out the low/high word per index.
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
        t.env = s.ibl ? prefilter.SampleLevel(cube_smp, s.dir, lod).rgb : s.base;
    }
    return t;
}

// Metallic/roughness hit shading. `N`/`albedo` are the (optionally normal-mapped)
// surface normal and base colour at the hit. A sun diffuse term (dielectric
// only, masked by the shadow ray) plus split IBL: diffuse irradiance along N and
// a specular tap along the onward reflection at a roughness-selected prefilter
// mip, tinted by F0. Metals drop the diffuse term and tint the reflected
// environment by their albedo. The self-emission is added on top.
float3 rt_shade_hit(RtSetup s, RtTrace t, float3 N, float3 albedo)
{
    float3 F0     = lerp(RT_F0.xxx, albedo, t.metallic);
    float3 diff_a = albedo * (1.0 - t.metallic);
    float  ndl    = saturate(dot(N, sun_dir.xyz));
    float3 sun    = diff_a * sun_color.xyz * ndl * t.shadow;
    if (!s.ibl) return sun + (diff_a + F0) * 0.03 + t.emissive;
    float3 refl = reflect(s.dir, N);
    float3 spec = prefilter.SampleLevel(cube_smp, refl, t.roughness * s.max_mip).rgb;
    float3 diff = prefilter.SampleLevel(cube_smp, N, s.max_mip).rgb * diff_a;
    return sun + diff + spec * F0 + t.emissive;
}

// Flat variant: material tint only (no bindless texture pool).
float4 rt_reflections_frag(VsOut p) : SV_TARGET
{
    RtSetup s = rt_setup(p.uv);
    if (!s.reflects) return float4(s.base, 1.0);
    RtTrace t = rt_trace(s);
    float3 reflected = t.hit ? rt_shade_hit(s, t, t.normal, t.tint) : t.env;
    return float4(lerp(s.base, reflected, s.weight), 1.0);
}

// Textured variant: samples the hit's albedo from the bindless pool * tint and
// perturbs the hit normal by the tangent-space normal map.
float4 rt_reflections_frag_textured(VsOut p) : SV_TARGET
{
    RtSetup s = rt_setup(p.uv);
    if (!s.reflects) return float4(s.base, 1.0);
    RtTrace t = rt_trace(s);

    float3 reflected;
    if (t.hit)
    {
        // level 0 - a reflected ray's screen-space UV gradients are unstable, so
        // sample the base mip to avoid gradient-driven mip thrash.
        float3 albedo = t.tint
                      * tex_pool[NonUniformResourceIndex(t.albedo_index)]
                            .SampleLevel(repeat_smp, t.uv, 0.0).rgb;
        // Perturb the geometric normal by the tangent-space normal map. The
        // flat-normal fallback decodes to (0,0,1) so N is unchanged when an
        // object has no map; a degenerate tangent keeps the geometric N.
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
        // Colour the self-emission by the emissive map (e.g. the bistro string
        // lights), mirroring the main bindless pass. Gated on a non-zero pool
        // index; the flat fallback variant has no pool and keeps scalar emissive.
        if (t.emissive_map_index != 0u)
        {
            t.emissive *= tex_pool[NonUniformResourceIndex(t.emissive_map_index)]
                              .SampleLevel(repeat_smp, t.uv, 0.0).rgb;
        }
        reflected = rt_shade_hit(s, t, N, albedo);
    }
    else
    {
        reflected = t.env;
    }
    return float4(lerp(s.base, reflected, s.weight), 1.0);
}
