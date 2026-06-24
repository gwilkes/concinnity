#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

// --- Glass panel pass, ray-traced reflection variant ---
//
// The same flat pane as glass.metal, but the reflection is a real per-pixel ray
// traced against the scene acceleration structure instead of a box-projected
// probe cube. Built + selected only when ray tracing is on (the device supports
// it and an `EnvironmentMap`/RT toggle has produced `self.rt.accel`); the non-RT
// glass.metal pipeline runs otherwise, so this file is compiled only on RT
// devices (it uses `metal_raytracing`, which a non-RT device cannot compile).
//
// Lives in its own file (not an `#ifdef` in glass.metal) so the always-built
// glass pipeline never sees the raytracing header. Shares the transparent pass's
// argument layout with glass.metal (view @buffer(5), params @buffer(6), ProbeSet
// @buffer(7), scene snapshot @texture(0), depth @texture(1), prefilter cube
// @texture(2), probe cubes @texture(3..)); the RT inputs occupy the otherwise
// free fragment buffers 0..4 + 8..9 (`encode_transparent` binds them when the RT
// glass pipeline is active). This is the FLAT trace (per-object material tint as
// albedo, no bindless texture pool - `BINDLESS_TEXTURE_ARG_BUFFER_INDEX == 7`
// would collide with the ProbeSet); a reflected ray that misses falls back to
// the same probe set / sky cube the non-RT path uses.

struct TransparentView {
    float4x4 vp;          // world -> clip (jittered when TAA is on)
    float4x4 inv_vp;      // clip -> world
    float4   camera_pos;  // world-space camera, .w unused
    float2   viewport;    // attachment dimensions in pixels
    float    time;        // seconds since startup
    float    _pad;
};

struct GlassParams {
    float4 centre;  // world-space pane centre
    float4 normal;  // unit pane normal (facing direction)
    float4 tint;    // colour multiplied into the refracted scene
    float  opacity;
    float  refraction_strength;
    float  fresnel_power;
    float  prefilter_mip_count; // mips in the sky prefilter cube; 0 = none bound
    // Planar reflection control (unused on the RT path, which traces a sharp
    // reflection ray instead). Present so the struct matches the CPU-side
    // metal::uniforms::GlassParams + the non-RT glass.metal byte-for-byte.
    float4 planar;
};

// RT tunables + camera + sun, bound at buffer(0). Layout matches
// render_types::RtParams (shared with rt_reflections.metal); glass uses
// `max_distance`, `sun_dir`, `sun_color`, and `prefilter_mip_count` (the ray
// origin is the pane surface point, so `cam_pos` / `inv_view` are unused here).
struct RtParams {
    float    intensity;
    float    max_distance;
    float    tan_half_fov_y;
    float    aspect;
    float    prefilter_mip_count;
    float    _pad0;
    float    _pad1;
    float    _pad2;
    float4   cam_pos;
    float4   sun_dir;     // xyz: world unit direction toward the sun (= L)
    float4   sun_color;   // xyz: sun radiance
    float4x4 inv_view;
};

// Per-instance geometry table, bound at buffer(3). Layout MUST match the
// 128-byte `#[repr(C)]` render_types::RtGeomEntry. `tint` / `emissive` are
// `packed_float3` (12 bytes, 4-byte aligned), NOT `float3` (which would be
// 16-aligned and shift every later field, faulting the trace) - identical to
// rt_reflections.metal.
struct RtGeomEntry {
    uint     index_offset;
    uint     base_vertex;
    uint     albedo_index;
    uint     normal_index;
    packed_float3 tint;
    float    roughness;
    float    metallic;
    packed_float3 emissive;
    float4x4 model;
    uint     emissive_map_index;
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
};

constant uint  RT_SKINNED_FLAG = 0x80000000u;
constant float VERTEX_FLOATS   = 14.0; // floats per Vertex (stride 56 bytes)
constant float RT_F0           = 0.04; // dielectric base reflectance

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

// The bindless texture pool, bound at buffer(10) by the textured glass RT
// variant (buffer(7) is the ProbeSet, where the main pass keeps its pool, so the
// pool moves to a free slot here). Identical layout to default.metal /
// rt_reflections.metal `BindlessTextures`; only `tex_pool` is read here.
constant constexpr uint BINDLESS_TEXTURE_COUNT = 96; // must match default.metal

struct BindlessTextures {
    array<texture2d<float>, BINDLESS_TEXTURE_COUNT> tex_pool [[id(0)]];
    depth2d_array<float> shadow_map [[id(BINDLESS_TEXTURE_COUNT)]];
    texturecube<float>   irradiance [[id(BINDLESS_TEXTURE_COUNT + 1)]];
    texturecube<float>   prefilter  [[id(BINDLESS_TEXTURE_COUNT + 2)]];
    texture2d<float>     ssao       [[id(BINDLESS_TEXTURE_COUNT + 3)]];
};

// Reflection-probe set + box-projected sampling, used as the ray-miss fallback
// (a reflection ray that leaves the scene reflects the local probe, not black).
// Identical to glass.metal / rt_reflections.metal.
constant constexpr uint  MAX_PROBES         = 8u;
constant constexpr float PROBE_BLEND_MARGIN = 0.2;

struct ProbeUniforms {
    float4 box_min;   // xyz = influence-box min, w = enabled
    float4 box_max;   // xyz = influence-box max
    float4 probe_pos; // xyz = capture position
};

struct ProbeSet {
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    ProbeUniforms probes[MAX_PROBES];
};
static_assert(sizeof(ProbeSet) == 400,
              "ProbeSet must be 400 bytes to match the CPU-side metal::uniforms::ProbeSet");

static float3 sample_probe_radiance(
    texturecube<float>      probe_cube,
    constant ProbeUniforms &probe,
    float3                  world_pos,
    float3                  R,
    float                   lod,
    sampler                 cube_sampler)
{
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

static float3 probe_set_specular(
    constant ProbeSet                    &set,
    array<texturecube<float>, MAX_PROBES> probes,
    float3                                world_pos,
    float3                                R,
    float                                 lod,
    sampler                               cube_sampler)
{
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

struct GlassVtxIn {
    float3 pos     [[attribute(0)]];
    float3 normal  [[attribute(1)]];
    float3 tangent [[attribute(2)]];
    float3 color   [[attribute(3)]];
    float2 uv      [[attribute(4)]];
};

struct GlassVtxOut {
    float4 position [[position]];
    float3 world_pos;
};

vertex GlassVtxOut glass_vertex(
    GlassVtxIn            in [[stage_in]],
    constant TransparentView &v [[buffer(5)]],
    constant GlassParams  &p [[buffer(6)]])
{
    GlassVtxOut out;
    out.world_pos = in.pos;
    out.position = v.vp * float4(in.pos, 1.0);
    return out;
}

fragment float4 glass_fragment_rt(
    GlassVtxOut                        in            [[stage_in]],
    constant RtParams                 &rt            [[buffer(0)]],
    const device float                *verts         [[buffer(1)]],
    const device uint                 *indices       [[buffer(2)]],
    const device RtGeomEntry          *geom          [[buffer(3)]],
    instance_acceleration_structure    accel         [[buffer(4)]],
    constant TransparentView          &v             [[buffer(5)]],
    constant GlassParams              &p             [[buffer(6)]],
    constant ProbeSet                 &probe_set     [[buffer(7)]],
    const device float                *sverts        [[buffer(8)]],
    const device ushort               *sidx          [[buffer(9)]],
    texture2d<float, access::sample>   scene_color   [[texture(0)]],
    depth2d<float>                     scene_depth   [[texture(1)]],
    texturecube<float, access::sample> prefilter     [[texture(2)]],
    array<texturecube<float>, MAX_PROBES> probes     [[texture(3)]],
    sampler                            scene_sampler [[sampler(0)]],
    sampler                            cube_sampler  [[sampler(1)]])
{
    float3 view_dir = normalize(v.camera_pos.xyz - in.world_pos);
    float3 normal = normalize(p.normal.xyz);
    if (dot(normal, view_dir) < 0.0) {
        normal = -normal;
    }

    float2 viewport = max(v.viewport, float2(1.0));
    float2 frag_uv = float2(in.position.x / viewport.x,
                            in.position.y / viewport.y);

    // Manual depth occlusion (the pass has no hardware depth test).
    uint2 self_pixel = min(uint2(in.position.xy), uint2(viewport) - uint2(1));
    if (scene_depth.read(self_pixel) < in.position.z) {
        discard_fragment();
    }

    float2 refract_uv = clamp(frag_uv + normal.xy * p.refraction_strength,
                              float2(0.001), float2(0.999));
    float3 refracted = scene_color.sample(scene_sampler, refract_uv).rgb * p.tint.rgb;

    // Trace a sharp reflection ray from the pane surface against the scene BVH.
    float3 R = reflect(-view_dir, normal);
    float max_mip = max(rt.prefilter_mip_count - 1.0, 0.0);
    float3 reflection;

    ray r;
    r.origin = in.world_pos + normal * 0.02;
    r.direction = R;
    r.min_distance = 0.01;
    r.max_distance = rt.max_distance;
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);
    intersection_result<triangle_data, instancing> hit = isect.intersect(r, accel);

    if (hit.type == intersection_type::triangle) {
        RtGeomEntry e = geom[hit.instance_id];
        bool skin = (e.normal_index & RT_SKINNED_FLAG) != 0u;
        const device float* vbuf = skin ? sverts : verts;
        uint o = e.index_offset + hit.primitive_id * 3;
        uint i0, i1, i2;
        if (skin) {
            i0 = (uint)sidx[o + 0] + e.base_vertex;
            i1 = (uint)sidx[o + 1] + e.base_vertex;
            i2 = (uint)sidx[o + 2] + e.base_vertex;
        } else {
            i0 = indices[o + 0] + e.base_vertex;
            i1 = indices[o + 1] + e.base_vertex;
            i2 = indices[o + 2] + e.base_vertex;
        }
        float2 b = hit.triangle_barycentric_coord;
        float w0 = 1.0 - b.x - b.y;
        float3 nl = rt_vertex_normal(vbuf, i0) * w0
                  + rt_vertex_normal(vbuf, i1) * b.x
                  + rt_vertex_normal(vbuf, i2) * b.y;
        float3 nw = normalize((e.model * float4(nl, 0.0)).xyz);
        if (dot(nw, R) > 0.0) {
            nw = -nw;
        }

        // Shadow ray toward the sun: any-hit = the reflected surface is in
        // shadow, so cast shadows appear inside the reflection.
        float3 hitp = r.origin + R * hit.distance;
        ray sr;
        sr.origin = hitp + nw * 0.02;
        sr.direction = normalize(rt.sun_dir.xyz);
        sr.min_distance = 0.001;
        sr.max_distance = rt.max_distance;
        intersector<triangle_data, instancing> sisect;
        sisect.assume_geometry_type(geometry_type::triangle);
        sisect.force_opacity(forced_opacity::opaque);
        sisect.accept_any_intersection(true);
        float shadow = (sisect.intersect(sr, accel).type == intersection_type::triangle)
                     ? 0.0 : 1.0;

        // Flat metallic/roughness shading: per-object material tint as albedo
        // (no bindless texture pool), sun diffuse (dielectric, shadow-masked) +
        // split IBL, plus self-emission.
        float3 albedo = float3(e.tint);
        float3 F0 = mix(float3(RT_F0), albedo, e.metallic);
        float3 diff_a = albedo * (1.0 - e.metallic);
        float ndl = saturate(dot(nw, rt.sun_dir.xyz));
        float3 sun = diff_a * rt.sun_color.xyz * ndl * shadow;
        if (rt.prefilter_mip_count > 0.5) {
            float3 onward = reflect(R, nw);
            float3 spec = prefilter.sample(cube_sampler, onward, level(e.roughness * max_mip)).rgb;
            float3 diffuse = prefilter.sample(cube_sampler, nw, level(max_mip)).rgb * diff_a;
            reflection = sun + diffuse + spec * F0 + float3(e.emissive);
        } else {
            reflection = sun + (diff_a + F0) * 0.03 + float3(e.emissive);
        }
    } else {
        // Miss: the box-projected probe set (or sky cube), the same fallback the
        // non-RT glass path and the RT-resolve miss use.
        if (probe_set.count > 0u) {
            reflection = probe_set_specular(probe_set, probes, in.world_pos, R, 0.0, cube_sampler);
        } else if (p.prefilter_mip_count > 0.5) {
            reflection = prefilter.sample(cube_sampler, R, level(0.0)).rgb;
        } else {
            reflection = float3(1.0);
        }
    }

    // Schlick Fresnel (F0 = 0.04) reflection/refraction blend, identical to the
    // non-RT path so toggling RT only sharpens the reflection content.
    float n_dot_v = saturate(dot(normal, view_dir));
    float rim = pow(1.0 - n_dot_v, max(p.fresnel_power, 1e-3));
    float refl_weight = saturate(RT_F0 + 0.96 * rim);
    float3 colour = mix(refracted, reflection, refl_weight);
    float alpha = saturate(mix(p.opacity, 1.0, rim));

    return float4(colour, alpha);
}

// Textured variant: identical to glass_fragment_rt but the reflected hit's albedo
// + normal map + emissive map are sampled from the bindless texture pool (bound
// at buffer(10)), so reflected surfaces carry their textures instead of a flat
// per-object tint. Selected only in a bindless world (where the pool exists); the
// flat variant above is the non-bindless RT fallback. The trace setup mirrors
// glass_fragment_rt, adding the per-vertex UV + tangent interpolation the texture
// sampling needs.
fragment float4 glass_fragment_rt_textured(
    GlassVtxOut                        in            [[stage_in]],
    constant RtParams                 &rt            [[buffer(0)]],
    const device float                *verts         [[buffer(1)]],
    const device uint                 *indices       [[buffer(2)]],
    const device RtGeomEntry          *geom          [[buffer(3)]],
    instance_acceleration_structure    accel         [[buffer(4)]],
    constant TransparentView          &v             [[buffer(5)]],
    constant GlassParams              &p             [[buffer(6)]],
    constant ProbeSet                 &probe_set     [[buffer(7)]],
    const device float                *sverts        [[buffer(8)]],
    const device ushort               *sidx          [[buffer(9)]],
    constant BindlessTextures         &tex           [[buffer(10)]],
    texture2d<float, access::sample>   scene_color   [[texture(0)]],
    depth2d<float>                     scene_depth   [[texture(1)]],
    texturecube<float, access::sample> prefilter     [[texture(2)]],
    array<texturecube<float>, MAX_PROBES> probes     [[texture(3)]],
    sampler                            scene_sampler [[sampler(0)]],
    sampler                            cube_sampler  [[sampler(1)]])
{
    float3 view_dir = normalize(v.camera_pos.xyz - in.world_pos);
    float3 normal = normalize(p.normal.xyz);
    if (dot(normal, view_dir) < 0.0) {
        normal = -normal;
    }

    float2 viewport = max(v.viewport, float2(1.0));
    float2 frag_uv = float2(in.position.x / viewport.x,
                            in.position.y / viewport.y);

    uint2 self_pixel = min(uint2(in.position.xy), uint2(viewport) - uint2(1));
    if (scene_depth.read(self_pixel) < in.position.z) {
        discard_fragment();
    }

    float2 refract_uv = clamp(frag_uv + normal.xy * p.refraction_strength,
                              float2(0.001), float2(0.999));
    float3 refracted = scene_color.sample(scene_sampler, refract_uv).rgb * p.tint.rgb;

    float3 R = reflect(-view_dir, normal);
    float max_mip = max(rt.prefilter_mip_count - 1.0, 0.0);
    float3 reflection;

    ray r;
    r.origin = in.world_pos + normal * 0.02;
    r.direction = R;
    r.min_distance = 0.01;
    r.max_distance = rt.max_distance;
    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);
    intersection_result<triangle_data, instancing> hit = isect.intersect(r, accel);

    if (hit.type == intersection_type::triangle) {
        RtGeomEntry e = geom[hit.instance_id];
        bool skin = (e.normal_index & RT_SKINNED_FLAG) != 0u;
        uint nidx = e.normal_index & ~RT_SKINNED_FLAG;
        const device float* vbuf = skin ? sverts : verts;
        uint o = e.index_offset + hit.primitive_id * 3;
        uint i0, i1, i2;
        if (skin) {
            i0 = (uint)sidx[o + 0] + e.base_vertex;
            i1 = (uint)sidx[o + 1] + e.base_vertex;
            i2 = (uint)sidx[o + 2] + e.base_vertex;
        } else {
            i0 = indices[o + 0] + e.base_vertex;
            i1 = indices[o + 1] + e.base_vertex;
            i2 = indices[o + 2] + e.base_vertex;
        }
        float2 b = hit.triangle_barycentric_coord;
        float w0 = 1.0 - b.x - b.y;
        float3 nl = rt_vertex_normal(vbuf, i0) * w0
                  + rt_vertex_normal(vbuf, i1) * b.x
                  + rt_vertex_normal(vbuf, i2) * b.y;
        float3 nw = normalize((e.model * float4(nl, 0.0)).xyz);
        if (dot(nw, R) > 0.0) {
            nw = -nw;
        }
        float3 tl = rt_vertex_tangent(vbuf, i0) * w0
                  + rt_vertex_tangent(vbuf, i1) * b.x
                  + rt_vertex_tangent(vbuf, i2) * b.y;
        float3 tw = (e.model * float4(tl, 0.0)).xyz;
        float2 uv = rt_vertex_uv(vbuf, i0) * w0
                  + rt_vertex_uv(vbuf, i1) * b.x
                  + rt_vertex_uv(vbuf, i2) * b.y;

        // Shadow ray toward the sun (geometric normal, before normal mapping).
        float3 hitp = r.origin + R * hit.distance;
        ray sr;
        sr.origin = hitp + nw * 0.02;
        sr.direction = normalize(rt.sun_dir.xyz);
        sr.min_distance = 0.001;
        sr.max_distance = rt.max_distance;
        intersector<triangle_data, instancing> sisect;
        sisect.assume_geometry_type(geometry_type::triangle);
        sisect.force_opacity(forced_opacity::opaque);
        sisect.accept_any_intersection(true);
        float shadow = (sisect.intersect(sr, accel).type == intersection_type::triangle)
                     ? 0.0 : 1.0;

        // Sample the bindless pool at base mip: a reflected ray's screen-space UV
        // gradients are unstable, so an explicit level(0) avoids mip thrash.
        constexpr sampler tsmp(filter::linear, address::repeat);
        float3 albedo = float3(e.tint) * tex.tex_pool[e.albedo_index].sample(tsmp, uv, level(0.0)).rgb;

        // Perturb the geometric normal by the tangent-space normal map (the
        // flat-normal fallback decodes to (0,0,1); a degenerate tangent keeps N).
        float3 N = nw;
        float tlen = length(tw);
        if (tlen > 1e-4) {
            float3 nm = tex.tex_pool[nidx].sample(tsmp, uv, level(0.0)).xyz * 2.0 - 1.0;
            float3 T = tw / tlen;
            T = normalize(T - N * dot(N, T));
            float3 B = cross(N, T);
            N = normalize(T * nm.x + B * nm.y + N * nm.z);
        }
        float3 emissive = float3(e.emissive);
        if (e.emissive_map_index != 0u) {
            emissive *= tex.tex_pool[e.emissive_map_index].sample(tsmp, uv, level(0.0)).rgb;
        }

        float3 F0 = mix(float3(RT_F0), albedo, e.metallic);
        float3 diff_a = albedo * (1.0 - e.metallic);
        float ndl = saturate(dot(N, rt.sun_dir.xyz));
        float3 sun = diff_a * rt.sun_color.xyz * ndl * shadow;
        if (rt.prefilter_mip_count > 0.5) {
            float3 onward = reflect(R, N);
            float3 spec = prefilter.sample(cube_sampler, onward, level(e.roughness * max_mip)).rgb;
            float3 diffuse = prefilter.sample(cube_sampler, N, level(max_mip)).rgb * diff_a;
            reflection = sun + diffuse + spec * F0 + emissive;
        } else {
            reflection = sun + (diff_a + F0) * 0.03 + emissive;
        }
    } else {
        if (probe_set.count > 0u) {
            reflection = probe_set_specular(probe_set, probes, in.world_pos, R, 0.0, cube_sampler);
        } else if (p.prefilter_mip_count > 0.5) {
            reflection = prefilter.sample(cube_sampler, R, level(0.0)).rgb;
        } else {
            reflection = float3(1.0);
        }
    }

    float n_dot_v = saturate(dot(normal, view_dir));
    float rim = pow(1.0 - n_dot_v, max(p.fresnel_power, 1e-3));
    float refl_weight = saturate(RT_F0 + 0.96 * rim);
    float3 colour = mix(refracted, reflection, refl_weight);
    float alpha = saturate(mix(p.opacity, 1.0, rim));

    return float4(colour, alpha);
}
