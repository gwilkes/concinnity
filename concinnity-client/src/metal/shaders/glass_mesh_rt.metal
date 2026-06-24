#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

// --- Transparent glass MESH pass, ray-traced reflection variant ---
//
// The sibling of glass_rt.metal for IMPORTED transparent meshes (a `Material`
// with `transparent: true` on an RT-capable device), not the flat pre-baked
// `GlassPanel` quad. Two differences from glass_rt.metal:
//   - The geometry is LOCAL-space, so the vertex shader applies the per-draw
//     model matrix (from GlassMeshParams) to position + normal and outputs both
//     interpolated, instead of a pre-transformed world-space quad.
//   - The fragment uses the INTERPOLATED per-vertex world normal (the water_rt
//     trick) instead of a flat pane normal, so a curved glass facade reflects
//     correctly across its surface.
// Everything else -- refraction of the pre-transparent scene snapshot, the
// per-pixel reflection ray traced against the scene BVH, hit shading + sun
// shadow, the probe/sky miss fallback, and the Schlick Fresnel blend -- is
// identical to glass_rt.metal. Shares the transparent pass's RT argument layout
// (RtParams @0, scene verts @1, indices @2, geom @3, TLAS @4, view @5, params
// @6, ProbeSet @7, skinned @8/9, bindless pool @10). The mesh is excluded from
// the BLAS (glass does not reflect glass -- accepted V1), so the trace never
// self-hits.

struct TransparentView {
    float4x4 vp;          // world -> clip (jittered when TAA is on)
    float4x4 inv_vp;      // clip -> world
    float4   camera_pos;  // world-space camera, .w unused
    float2   viewport;    // attachment dimensions in pixels
    float    time;        // seconds since startup
    float    _pad;
};

// Matches metal::uniforms::GlassMeshParams (96 bytes). `model` is first so its
// 16-byte alignment is satisfied at offset 0.
struct GlassMeshParams {
    float4x4 model;             // local -> world
    float4   tint;             // colour multiplied into the refracted scene
    float    opacity;
    float    refraction_strength;
    float    fresnel_power;
    float    prefilter_mip_count; // mips in the sky prefilter cube; 0 = none bound
};

// RT tunables + camera + sun, bound at buffer(0). Layout matches
// render_types::RtParams (shared with rt_reflections.metal / glass_rt.metal).
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
// 128-byte render_types::RtGeomEntry; `tint` / `emissive` are packed_float3.
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
// Sun-shadow rays for a reflected hit only need LOCAL occluders: a contact
// shadow seen in a glass reflection beyond this range is imperceptible. Capping
// the secondary ray here prunes the expensive full-BVH traversal on lit pixels
// (the common case for sky-facing glass), so the per-pixel glass trace cost on a
// large scene drops sharply while local contact shadows are preserved.
constant float GLASS_SHADOW_MAX_DIST = 12.0;

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

// The bindless texture pool, bound at buffer(10) by the textured variant.
constant constexpr uint BINDLESS_TEXTURE_COUNT = 96; // must match default.metal

struct BindlessTextures {
    array<texture2d<float>, BINDLESS_TEXTURE_COUNT> tex_pool [[id(0)]];
    depth2d_array<float> shadow_map [[id(BINDLESS_TEXTURE_COUNT)]];
    texturecube<float>   irradiance [[id(BINDLESS_TEXTURE_COUNT + 1)]];
    texturecube<float>   prefilter  [[id(BINDLESS_TEXTURE_COUNT + 2)]];
    texture2d<float>     ssao       [[id(BINDLESS_TEXTURE_COUNT + 3)]];
};

// Reflection-probe set + box-projected sampling, used as the ray-miss fallback.
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

struct GlassMeshVtxIn {
    float3 pos     [[attribute(0)]];
    float3 normal  [[attribute(1)]];
    float3 tangent [[attribute(2)]];
    float3 color   [[attribute(3)]];
    float2 uv      [[attribute(4)]];
};

struct GlassMeshVtxOut {
    float4 position [[position]];
    float3 world_pos;
    float3 world_normal;
};

vertex GlassMeshVtxOut glass_mesh_vertex(
    GlassMeshVtxIn            in [[stage_in]],
    constant TransparentView &v [[buffer(5)]],
    constant GlassMeshParams &p [[buffer(6)]])
{
    GlassMeshVtxOut out;
    float4 world = p.model * float4(in.pos, 1.0);
    out.world_pos = world.xyz;
    // Rigid / uniform-scale model: M * n is sufficient (no inverse-transpose).
    out.world_normal = (p.model * float4(in.normal, 0.0)).xyz;
    out.position = v.vp * world;
    return out;
}

// Shared refraction + reflection-blend tail. `normal` is the view-facing surface
// normal; `reflection` is the traced (or miss-fallback) radiance. Identical math
// to glass_rt.metal so a mesh and a pane read the same at equal inputs.
static float4 glass_mesh_shade(
    GlassMeshVtxOut                   in,
    constant TransparentView         &v,
    constant GlassMeshParams         &p,
    float3                            normal,
    float3                            view_dir,
    float3                            reflection,
    texture2d<float, access::sample>  scene_color,
    float2                            frag_uv,
    sampler                           scene_sampler)
{
    float2 refract_uv = clamp(frag_uv + normal.xy * p.refraction_strength,
                              float2(0.001), float2(0.999));
    float3 refracted = scene_color.sample(scene_sampler, refract_uv).rgb * p.tint.rgb;

    float n_dot_v = saturate(dot(normal, view_dir));
    float rim = pow(1.0 - n_dot_v, max(p.fresnel_power, 1e-3));
    float refl_weight = saturate(RT_F0 + 0.96 * rim);
    float3 colour = mix(refracted, reflection, refl_weight);
    float alpha = saturate(mix(p.opacity, 1.0, rim));
    return float4(colour, alpha);
}

fragment float4 glass_mesh_fragment_rt(
    GlassMeshVtxOut                    in            [[stage_in]],
    constant RtParams                 &rt            [[buffer(0)]],
    const device float                *verts         [[buffer(1)]],
    const device uint                 *indices       [[buffer(2)]],
    const device RtGeomEntry          *geom          [[buffer(3)]],
    instance_acceleration_structure    accel         [[buffer(4)]],
    constant TransparentView          &v             [[buffer(5)]],
    constant GlassMeshParams          &p             [[buffer(6)]],
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
    // Interpolated per-fragment world normal, oriented toward the viewer.
    float3 normal = normalize(in.world_normal);
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

        float3 hitp = r.origin + R * hit.distance;
        ray sr;
        sr.origin = hitp + nw * 0.02;
        sr.direction = normalize(rt.sun_dir.xyz);
        sr.min_distance = 0.001;
        sr.max_distance = min(GLASS_SHADOW_MAX_DIST, rt.max_distance);
        intersector<triangle_data, instancing> sisect;
        sisect.assume_geometry_type(geometry_type::triangle);
        sisect.force_opacity(forced_opacity::opaque);
        sisect.accept_any_intersection(true);
        float shadow = (sisect.intersect(sr, accel).type == intersection_type::triangle)
                     ? 0.0 : 1.0;

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
        if (probe_set.count > 0u) {
            reflection = probe_set_specular(probe_set, probes, in.world_pos, R, 0.0, cube_sampler);
        } else if (p.prefilter_mip_count > 0.5) {
            reflection = prefilter.sample(cube_sampler, R, level(0.0)).rgb;
        } else {
            reflection = float3(1.0);
        }
    }

    return glass_mesh_shade(in, v, p, normal, view_dir, reflection,
                            scene_color, frag_uv, scene_sampler);
}

// Textured variant: identical trace, but the reflected hit's albedo / normal map
// / emissive map are sampled from the bindless pool (buffer 10), so reflected
// surfaces carry their textures. Selected only in a bindless world.
fragment float4 glass_mesh_fragment_rt_textured(
    GlassMeshVtxOut                    in            [[stage_in]],
    constant RtParams                 &rt            [[buffer(0)]],
    const device float                *verts         [[buffer(1)]],
    const device uint                 *indices       [[buffer(2)]],
    const device RtGeomEntry          *geom          [[buffer(3)]],
    instance_acceleration_structure    accel         [[buffer(4)]],
    constant TransparentView          &v             [[buffer(5)]],
    constant GlassMeshParams          &p             [[buffer(6)]],
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
    float3 normal = normalize(in.world_normal);
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

        float3 hitp = r.origin + R * hit.distance;
        ray sr;
        sr.origin = hitp + nw * 0.02;
        sr.direction = normalize(rt.sun_dir.xyz);
        sr.min_distance = 0.001;
        sr.max_distance = min(GLASS_SHADOW_MAX_DIST, rt.max_distance);
        intersector<triangle_data, instancing> sisect;
        sisect.assume_geometry_type(geometry_type::triangle);
        sisect.force_opacity(forced_opacity::opaque);
        sisect.accept_any_intersection(true);
        float shadow = (sisect.intersect(sr, accel).type == intersection_type::triangle)
                     ? 0.0 : 1.0;

        constexpr sampler tsmp(filter::linear, address::repeat);
        float3 albedo = float3(e.tint) * tex.tex_pool[e.albedo_index].sample(tsmp, uv, level(0.0)).rgb;

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

    return glass_mesh_shade(in, v, p, normal, view_dir, reflection,
                            scene_color, frag_uv, scene_sampler);
}
