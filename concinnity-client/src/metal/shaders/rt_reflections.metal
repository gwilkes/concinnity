#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

// Hardware ray-traced reflections. A fullscreen fragment pass that, for each
// glossy pixel, rebuilds a world-space surface point + normal from the SSR
// pre-pass G-buffer, traces a reflection ray against the scene's acceleration
// structure, and composites the reflected colour over the base scene with the
// same Fresnel/gloss weighting SSR uses. Unlike SSR the ray is a real
// world-space trace, so reflected geometry that is off-screen still appears.
//
// Two hit-shading variants share all the setup + trace logic:
//   * rt_reflections_fragment           - flat: material tint only (the
//     fallback used when the bindless texture pool is not available).
//   * rt_reflections_fragment_textured  - samples the hit's albedo + normal-map
//     textures from the bindless pool, the path standard worlds take.
// Both shade the hit with a metallic/roughness response (sun diffuse for
// dielectrics + split IBL: diffuse irradiance along N, specular along the
// onward reflection at a roughness-selected prefilter mip, tinted by F0) and
// fall back to the IBL prefilter cubemap on a miss, exactly like SSR. The
// textured variant additionally perturbs the hit normal by a tangent-space
// normal map. On a hit both trace a second (any-hit) ray toward the sun, so the
// sun term is masked where the reflected surface is occluded - cast shadows
// appear inside the reflection.

constant constexpr uint BINDLESS_TEXTURE_COUNT = 96; // must match default.metal

struct RtVtxOut {
    float4 position [[position]];
    float2 uv;
};

// Fullscreen triangle from vertex_id 0..2 - no vertex buffer.
vertex RtVtxOut rt_fullscreen_vertex(uint vid [[vertex_id]]) {
    float2 pos = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    RtVtxOut out;
    out.position = float4(pos, 0.0, 1.0);
    out.uv = float2((pos.x + 1.0) * 0.5, 1.0 - (pos.y + 1.0) * 0.5);
    return out;
}

// buffer(0): RT tunables + camera + sun. Layout matches render_types::RtParams.
struct RtParams {
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

// buffer(3): per-instance geometry table. Layout MUST match the 128-byte
// `#[repr(C)]` render_types::RtGeomEntry. `tint` and `emissive` are
// `packed_float3` (12 bytes, 4-byte aligned), NOT `float3`: a `float3` would be
// 16-byte aligned and 16 bytes, which would push `roughness` to offset 32 and
// the `float4x4` to offset 64, shifting every later field so the shader reads a
// tightly-packed buffer at the wrong offsets - a GPU address fault in the trace.
// `model` lands at offset 48 (16-aligned, so the float4x4 needs no padding); the
// `_pad` tail then rounds the struct to 128 bytes so the device-array stride
// matches the Rust side (a `float4x4`-bearing struct rounds up to a 16-byte
// multiple, so an unpadded 116-byte struct would mismatch).
struct RtGeomEntry {
    uint     index_offset;  // element offset of this object's first index
    uint     base_vertex;   // added to each fetched index
    uint     albedo_index;  // bindless albedo pool index for the textured hit
    uint     normal_index;  // bindless normal-map index (flat fallback if none)
    packed_float3 tint;     // base albedo for hit shading
    float    roughness;     // hit IBL specular mip selection
    float    metallic;      // hit PBR response (metals tint the env reflection)
    packed_float3 emissive; // self-emission added to the hit colour
    float4x4 model;         // object-to-world (column-major)
    uint     emissive_map_index; // bindless emissive-map index (0 = none)
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
};

// buffer(7): the bindless texture pool, identical layout to default.metal's
// `BindlessTextures` (only `tex_pool` is read here). Bound only by the textured
// variant; the flat variant does not declare it.
struct BindlessTextures {
    array<texture2d<float>, BINDLESS_TEXTURE_COUNT> tex_pool [[id(0)]];
    depth2d_array<float> shadow_map [[id(BINDLESS_TEXTURE_COUNT)]];
    texturecube<float>   irradiance [[id(BINDLESS_TEXTURE_COUNT + 1)]];
    texturecube<float>   prefilter  [[id(BINDLESS_TEXTURE_COUNT + 2)]];
    texture2d<float>     ssao       [[id(BINDLESS_TEXTURE_COUNT + 3)]];
};

// Surfaces rougher than REFLECTION_ROUGHNESS_CUT get no reflection. Injected as
// a shared `constant` (see pipeline.rs) so the SSR / RT / composite gates agree.
constant float RT_F0        = 0.04;  // dielectric base reflectance for the Fresnel
constant float VERTEX_FLOATS = 14.0; // floats per Vertex (stride 56 bytes)

// Reflection-probe set, bound at buffer(8). Layout mirrors `ProbeSet` /
// `ProbeUniforms` in default.metal (and metal::uniforms). A reflection ray that
// misses the scene falls back to the local scene-captured probe (box-projected)
// instead of the foreign sky cube, the same source the forward IBL specular term
// uses. `count` is 0 in worlds with no baked probe, where the miss keeps the sky.
constant constexpr uint  MAX_PROBES         = 8u;
constant constexpr float PROBE_BLEND_MARGIN = 0.2;

struct ProbeUniforms {
    float4 box_min;   // xyz = influence-box min, w = enabled
    float4 box_max;   // xyz = influence-box max
    float4 probe_pos; // xyz = capture position
};

struct ProbeSet {
    uint count;
    // Three scalar uints, NOT a uint3: a uint3 is 16-byte aligned in MSL, which
    // pushes `probes` to offset 32 (struct 416 bytes) and mismatches the CPU-side
    // metal::uniforms::ProbeSet, whose [u32; 3] keeps `probes` at offset 16 (struct
    // 400 bytes). The static_assert locks the 400-byte layout at shader-compile time.
    uint _pad0;
    uint _pad1;
    uint _pad2;
    ProbeUniforms probes[MAX_PROBES];
};
static_assert(sizeof(ProbeSet) == 400,
              "ProbeSet must be 400 bytes to match the CPU-side metal::uniforms::ProbeSet");

// Box-parallax sample of one probe cube: intersect the world-space reflection ray
// with the probe's influence box and re-anchor the sample at that hit relative to
// the capture point, so a static cube tracks the camera. Mirrors default.metal's
// `sample_probe_radiance`; the `dist > 0` guard keeps a blended secondary box the
// surface has already left from sampling backward.
static float3 sample_probe_radiance(
    texturecube<float>      probe_cube,
    constant ProbeUniforms &probe,
    float3                  world_pos,
    float3                  R,
    float                   lod,
    sampler                 cube_sampler
) {
    float3 sample_dir = R;
    if (probe.box_min.w > 0.5) {
        float3 inv_r = 1.0 / R;
        float3 t_max = (probe.box_max.xyz - world_pos) * inv_r;
        float3 t_min = (probe.box_min.xyz - world_pos) * inv_r;
        float3 t_far = max(t_max, t_min);
        float dist = min(min(t_far.x, t_far.y), t_far.z);
        if (dist > 0.0) {
            float3 hit = world_pos + R * dist;
            sample_dir = hit - probe.probe_pos.xyz;
        }
    }
    return probe_cube.sample(cube_sampler, sample_dir, bias(lod)).rgb;
}

// Reflection-probe radiance for `world_pos` along world-space ray `R`. Weights
// every probe whose influence box covers `world_pos` and returns the
// weight-normalised sum of their box-projected samples (partition of unity): each
// probe's weight is `smoothstep(-margin, margin, sd)` of its signed box distance, so
// overlapping boxes cross-fade smoothly (no pop at a 3-way overlap line) and a single
// covering box reduces to one sample. Falls back to the nearest by capture distance
// where no box covers. Matches default.metal's `probe_set_specular`.
static float3 probe_set_specular(
    constant ProbeSet                    &set,
    array<texturecube<float>, MAX_PROBES> probes,
    float3                                world_pos,
    float3                                R,
    float                                 lod,
    sampler                               cube_sampler
) {
    float3 acc = float3(0.0);
    float wsum = 0.0;
    float near_d = 1e30;
    uint near_i = 0u;
    for (uint i = 0u; i < set.count; i++) {
        float3 c = 0.5 * (set.probes[i].box_min.xyz + set.probes[i].box_max.xyz);
        float3 he = 0.5 * (set.probes[i].box_max.xyz - set.probes[i].box_min.xyz);
        float3 q = abs(world_pos - c) - he;
        float sd = -(length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0));
        float margin = max(PROBE_BLEND_MARGIN * min(he.x, min(he.y, he.z)), 1e-4);
        float w = smoothstep(-margin, margin, sd);
        if (w > 0.0) {
            acc += w * sample_probe_radiance(
                           probes[i], set.probes[i], world_pos, R, lod, cube_sampler);
            wsum += w;
        }
        float d = distance(world_pos, set.probes[i].probe_pos.xyz);
        if (d < near_d) {
            near_d = d;
            near_i = i;
        }
    }
    if (wsum > 0.0) {
        return acc / wsum;
    }
    return sample_probe_radiance(
        probes[near_i], set.probes[near_i], world_pos, R, lod, cube_sampler);
}

// Rebuild a view-space position from a UV and its linear (view-space) depth.
static float3 rt_view_pos(float2 uv, float depth, float tan_y, float aspect) {
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_y * aspect, ndc.y * tan_y, -1.0) * depth;
}

// Attribute fetchers into the shared 14-float Vertex (pos@0, normal@12/3,
// uv@48/12).
static float3 rt_vertex_pos(const device float* v, uint vi) {
    uint b = vi * (uint)VERTEX_FLOATS;
    return float3(v[b + 0], v[b + 1], v[b + 2]);
}
static float3 rt_vertex_normal(const device float* v, uint vi) {
    uint b = vi * (uint)VERTEX_FLOATS;
    return float3(v[b + 3], v[b + 4], v[b + 5]);
}
static float3 rt_vertex_tangent(const device float* v, uint vi) {
    uint b = vi * (uint)VERTEX_FLOATS;
    return float3(v[b + 6], v[b + 7], v[b + 8]);
}
static float2 rt_vertex_uv(const device float* v, uint vi) {
    uint b = vi * (uint)VERTEX_FLOATS;
    return float2(v[b + 12], v[b + 13]);
}

// Common per-pixel setup shared by both hit-shading variants. Holds whether the
// pixel reflects at all, the base scene colour, the world-space ray origin +
// reflection direction, and the Fresnel/gloss composite weight. The trace
// itself is kept in each fragment (the acceleration structure is not passed
// through a helper).
struct RtSetup {
    bool   reflects;   // false -> sky / too rough, write base unchanged
    float3 base;
    float3 world_pos;  // world-space surface point (probe box-projection anchor)
    float3 origin;     // ray origin (surface point nudged along the normal)
    float3 dir;        // world-space reflection direction
    float  weight;     // saturate(fresnel * gloss * intensity)
    float  roughness;
    float  max_mip;    // prefilter_mip_count - 1
    bool   ibl;        // an EnvironmentMap is bound
};

static RtSetup rt_setup(
    RtVtxOut in,
    constant RtParams& p,
    texture2d<float> scene,
    texture2d<float> gbuffer,
    texture2d<float> rough_tex,
    sampler smp
) {
    RtSetup s;
    s.reflects = false;
    s.base = scene.sample(smp, in.uv).rgb;
    float4 g = gbuffer.sample(smp, in.uv);
    float depth = g.a;
    if (depth <= 0.0) return s;                 // background / sky
    s.roughness = rough_tex.sample(smp, in.uv).r;
    float gloss = saturate((REFLECTION_ROUGHNESS_CUT - s.roughness) / REFLECTION_ROUGHNESS_CUT);
    if (gloss <= 0.0) return s;

    float3 Nv = normalize(g.xyz);
    float3 Pv = rt_view_pos(in.uv, depth, p.tan_half_fov_y, p.aspect);
    float3 Pw = (p.inv_view * float4(Pv, 1.0)).xyz;
    float3 Nw = normalize((p.inv_view * float4(Nv, 0.0)).xyz);
    float3 V  = normalize(p.cam_pos.xyz - Pw);

    s.world_pos = Pw;
    s.origin = Pw + Nw * 0.01;                  // nudge off the surface
    s.dir    = reflect(-V, Nw);
    s.ibl    = p.prefilter_mip_count > 0.5;
    s.max_mip = p.prefilter_mip_count - 1.0;
    float ndv     = saturate(dot(Nw, V));
    float fresnel = RT_F0 + (1.0 - RT_F0) * pow(1.0 - ndv, 5.0);
    s.weight = saturate(fresnel * gloss * p.intensity);
    s.reflects = true;
    return s;
}

// Result of tracing the reflection ray: the interpolated world normal + uv +
// albedo index on a hit, or the environment colour on a miss. `shadow` is the
// sun visibility at the hit (1 = lit, 0 = a second ray to the sun was occluded),
// so reflected surfaces show cast shadows like the primary image does.
struct RtTrace {
    bool   hit;
    float3 normal;    // interpolated world-space geometric normal
    float3 tangent;   // interpolated world-space tangent (for normal mapping)
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

// Skinned objects flag bit 31 of `normal_index`; the kernel then fetches the
// hit triangle from the deformed-vertex / u16 skinned index buffers (which
// mirror the static layout) instead of the static u32 ones.
constant uint RT_SKINNED_FLAG = 0x80000000u;

// Trace the reflection ray and gather the hit attributes (or the miss colour).
// Inlined into each fragment so the acceleration structure stays a direct
// fragment argument. `verts`/`indices` are the static buffers; `sverts`/`sidx`
// the deformed-vertex / u16 skinned buffers, selected per hit by the skinned
// flag so skinned geometry reflects with its current pose.
#define RT_TRACE(out_trace, s, p, verts, sverts, indices, sidx, geom, accel, prefilter, cube_smp, probe_set, probes) \
    do {                                                                            \
        RtTrace _t; _t.hit = false;                                                 \
        ray _r; _r.origin = (s).origin; _r.direction = (s).dir;                     \
        _r.min_distance = 0.01; _r.max_distance = (p).max_distance;                 \
        intersector<triangle_data, instancing> _isect;                             \
        _isect.assume_geometry_type(geometry_type::triangle);                       \
        _isect.force_opacity(forced_opacity::opaque);                               \
        intersection_result<triangle_data, instancing> _h = _isect.intersect(_r, accel); \
        if (_h.type == intersection_type::triangle) {                               \
            RtGeomEntry _e = (geom)[_h.instance_id];                                \
            bool _skin = (_e.normal_index & RT_SKINNED_FLAG) != 0u;                 \
            uint _nidx = _e.normal_index & ~RT_SKINNED_FLAG;                        \
            const device float* _v = _skin ? (sverts) : (verts);                    \
            uint _tri = _h.primitive_id;                                            \
            uint _o = _e.index_offset + _tri * 3;                                   \
            uint _i0, _i1, _i2;                                                     \
            if (_skin) {                                                            \
                _i0 = (uint)(sidx)[_o + 0] + _e.base_vertex;                        \
                _i1 = (uint)(sidx)[_o + 1] + _e.base_vertex;                        \
                _i2 = (uint)(sidx)[_o + 2] + _e.base_vertex;                        \
            } else {                                                               \
                _i0 = (indices)[_o + 0] + _e.base_vertex;                           \
                _i1 = (indices)[_o + 1] + _e.base_vertex;                           \
                _i2 = (indices)[_o + 2] + _e.base_vertex;                           \
            }                                                                       \
            float2 _b = _h.triangle_barycentric_coord;                              \
            float _w0 = 1.0 - _b.x - _b.y;                                          \
            float3 _nl = rt_vertex_normal(_v, _i0) * _w0                            \
                       + rt_vertex_normal(_v, _i1) * _b.x                           \
                       + rt_vertex_normal(_v, _i2) * _b.y;                          \
            float3 _nw = normalize(((geom)[_h.instance_id].model * float4(_nl, 0.0)).xyz); \
            if (dot(_nw, (s).dir) > 0.0) _nw = -_nw;                                \
            float3 _tl = rt_vertex_tangent(_v, _i0) * _w0                           \
                       + rt_vertex_tangent(_v, _i1) * _b.x                          \
                       + rt_vertex_tangent(_v, _i2) * _b.y;                         \
            float3 _tw = ((geom)[_h.instance_id].model * float4(_tl, 0.0)).xyz;     \
            _t.uv = rt_vertex_uv(_v, _i0) * _w0                                     \
                  + rt_vertex_uv(_v, _i1) * _b.x                                    \
                  + rt_vertex_uv(_v, _i2) * _b.y;                                   \
            _t.hit = true; _t.normal = _nw; _t.tangent = _tw;                       \
            _t.albedo_index = _e.albedo_index; _t.normal_index = _nidx;             \
            _t.emissive_map_index = _e.emissive_map_index;                          \
            _t.tint = _e.tint; _t.roughness = _e.roughness; _t.metallic = _e.metallic; \
            _t.emissive = float3(_e.emissive);                                      \
            /* Shadow ray: from the hit toward the sun, any-hit = occluded. */      \
            float3 _hp = (s).origin + (s).dir * _h.distance;                        \
            ray _sr; _sr.origin = _hp + _nw * 0.02;                                 \
            _sr.direction = normalize((p).sun_dir.xyz);                             \
            _sr.min_distance = 0.001; _sr.max_distance = (p).max_distance;          \
            intersector<triangle_data, instancing> _sisect;                        \
            _sisect.assume_geometry_type(geometry_type::triangle);                  \
            _sisect.force_opacity(forced_opacity::opaque);                          \
            _sisect.accept_any_intersection(true);                                  \
            intersection_result<triangle_data, instancing> _sh =                    \
                _sisect.intersect(_sr, accel);                                      \
            _t.shadow = (_sh.type == intersection_type::triangle) ? 0.0 : 1.0;      \
        } else {                                                                    \
            float _lod = (s).roughness * (s).max_mip;                              \
            if (!(s).ibl) {                                                         \
                _t.env = (s).base;                                                  \
            } else if ((probe_set).count > 0u) {                                    \
                /* Missed ray reflects the local box-projected probe, not the sky. */ \
                _t.env = probe_set_specular((probe_set), (probes),                   \
                                            (s).world_pos, (s).dir, _lod, cube_smp); \
            } else {                                                                \
                _t.env = prefilter.sample(cube_smp, (s).dir, level(_lod)).rgb;      \
            }                                                                       \
        }                                                                           \
        out_trace = _t;                                                             \
    } while (0)

// Metallic/roughness hit shading. `N`/`albedo` are the (optionally normal-mapped)
// surface normal and base colour at the hit. A sun diffuse term (dielectric
// only, masked by the shadow ray) plus split IBL: diffuse irradiance along N,
// and a specular tap along the onward reflection at a roughness-selected
// prefilter mip, tinted by F0. Metals (`metallic`→1) drop the diffuse term and
// tint the reflected environment by their albedo (F0 = albedo). The material's
// self-emission is added on top so glowing surfaces light up in reflections.
// Falls back to a small constant ambient when no EnvironmentMap is bound.
static float3 rt_shade_hit(const RtSetup s, constant RtParams& p, const RtTrace t,
                           float3 N, float3 albedo,
                           texturecube<float> prefilter, sampler cube_smp) {
    float3 F0      = mix(float3(RT_F0), albedo, t.metallic);
    float3 diff_a  = albedo * (1.0 - t.metallic);
    float  ndl     = saturate(dot(N, p.sun_dir.xyz));
    float3 sun     = diff_a * p.sun_color.xyz * ndl * t.shadow;
    if (!s.ibl) return sun + (diff_a + F0) * 0.03 + t.emissive;
    float3 refl    = reflect(s.dir, N);
    float3 spec    = prefilter.sample(cube_smp, refl, level(t.roughness * s.max_mip)).rgb;
    float3 diff    = prefilter.sample(cube_smp, N, level(s.max_mip)).rgb * diff_a;
    return sun + diff + spec * F0 + t.emissive;
}

// Flat variant: material tint only (no bindless texture pool).

fragment float4 rt_reflections_fragment(
    RtVtxOut in                                    [[stage_in]],
    constant RtParams& p                           [[buffer(0)]],
    const device float* verts                      [[buffer(1)]],
    const device uint* indices                     [[buffer(2)]],
    const device RtGeomEntry* geom                 [[buffer(3)]],
    instance_acceleration_structure accel          [[buffer(4)]],
    const device float* sverts                     [[buffer(5)]],
    const device ushort* sidx                      [[buffer(6)]],
    constant ProbeSet& probe_set                   [[buffer(8)]],
    texture2d<float>   scene                       [[texture(0)]],
    texture2d<float>   gbuffer                     [[texture(1)]],
    texture2d<float>   rough_tex                   [[texture(2)]],
    texturecube<float> prefilter                   [[texture(3)]],
    array<texturecube<float>, MAX_PROBES> probes   [[texture(4)]],
    sampler smp                                    [[sampler(0)]],
    sampler cube_smp                               [[sampler(1)]]
) {
    RtSetup s = rt_setup(in, p, scene, gbuffer, rough_tex, smp);
    if (!s.reflects) return float4(0.0);   // no reflection -> composite keeps the scene

    RtTrace t;
    RT_TRACE(t, s, p, verts, sverts, indices, sidx, geom, accel, prefilter, cube_smp, probe_set, probes);

    float3 reflected;
    if (t.hit) {
        // Flat fallback: geometric normal, material tint as albedo (no maps).
        reflected = rt_shade_hit(s, p, t, t.normal, t.tint, prefilter, cube_smp);
    } else {
        reflected = t.env;
    }
    // Reflected radiance (.rgb) + composite weight (.a). The reflection
    // composite blurs this by surface roughness and blends it over the scene.
    return float4(reflected, s.weight);
}

// Textured variant: samples the hit's albedo from the bindless pool × tint.

fragment float4 rt_reflections_fragment_textured(
    RtVtxOut in                                    [[stage_in]],
    constant RtParams& p                           [[buffer(0)]],
    const device float* verts                      [[buffer(1)]],
    const device uint* indices                     [[buffer(2)]],
    const device RtGeomEntry* geom                 [[buffer(3)]],
    instance_acceleration_structure accel          [[buffer(4)]],
    const device float* sverts                     [[buffer(5)]],
    const device ushort* sidx                      [[buffer(6)]],
    constant BindlessTextures& tex                 [[buffer(7)]],
    constant ProbeSet& probe_set                   [[buffer(8)]],
    texture2d<float>   scene                       [[texture(0)]],
    texture2d<float>   gbuffer                     [[texture(1)]],
    texture2d<float>   rough_tex                   [[texture(2)]],
    texturecube<float> prefilter                   [[texture(3)]],
    array<texturecube<float>, MAX_PROBES> probes   [[texture(4)]],
    sampler smp                                    [[sampler(0)]],
    sampler cube_smp                               [[sampler(1)]]
) {
    RtSetup s = rt_setup(in, p, scene, gbuffer, rough_tex, smp);
    if (!s.reflects) return float4(0.0);   // no reflection -> composite keeps the scene

    RtTrace t;
    RT_TRACE(t, s, p, verts, sverts, indices, sidx, geom, accel, prefilter, cube_smp, probe_set, probes);

    float3 reflected;
    if (t.hit) {
        constexpr sampler tsmp(filter::linear, address::repeat);
        // level(0) - a reflected ray's screen-space UV gradients are unstable
        // (neighbouring pixels hit unrelated triangles), so sampling the base
        // mip avoids gradient-driven mip thrash.
        float3 albedo = t.tint
                      * tex.tex_pool[t.albedo_index].sample(tsmp, t.uv, level(0.0)).rgb;
        // Perturb the geometric normal by the tangent-space normal map. The
        // flat-normal fallback decodes to (0,0,1) so N is unchanged when an
        // object has no map; a degenerate tangent skips the frame entirely
        // (procedural meshes may carry no tangents) and keeps the geometric N.
        float3 N    = t.normal;
        float  tlen = length(t.tangent);
        if (tlen > 1e-4) {
            float3 nm = tex.tex_pool[t.normal_index].sample(tsmp, t.uv, level(0.0)).xyz * 2.0 - 1.0;
            float3 T  = t.tangent / tlen;
            T = normalize(T - N * dot(N, T));            // Gram-Schmidt
            float3 B = cross(N, T);
            N = normalize(T * nm.x + B * nm.y + N * nm.z);
        }
        // Colour the self-emission by the emissive map (e.g. the bistro string
        // lights), mirroring the main bindless pass. Gated on a non-zero pool
        // index; the flat fallback variant has no pool and keeps scalar emissive.
        if (t.emissive_map_index != 0u) {
            t.emissive *= float3(tex.tex_pool[t.emissive_map_index].sample(tsmp, t.uv, level(0.0)).rgb);
        }
        reflected = rt_shade_hit(s, p, t, N, albedo, prefilter, cube_smp);
    } else {
        reflected = t.env;
    }
    // Reflected radiance (.rgb) + composite weight (.a). The reflection
    // composite blurs this by surface roughness and blends it over the scene.
    return float4(reflected, s.weight);
}
