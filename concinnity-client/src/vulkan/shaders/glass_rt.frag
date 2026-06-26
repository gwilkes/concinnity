#version 460
#extension GL_EXT_ray_query : require
#extension GL_EXT_nonuniform_qualifier : require

// Glass panel pass, ray-traced reflection variant (Vulkan inline `rayQueryEXT`).
// Mirrors ps_main_rt in src/directx/shaders/glass_rt.hlsl and glass_fragment_rt
// in src/metal/shaders/glass_rt.metal. Selected over glass.frag only while RT is
// live (the scene TLAS is built); otherwise the base glass.frag probe/planar path
// runs. Everything outside the reflection itself (two-sided normal, manual depth
// discard, refraction, the Schlick-Fresnel reflection/refraction blend and alpha)
// is identical to glass.frag -- toggling RT only sharpens the reflection.
//
// Instead of sampling the box-projected probe cube (or the planar mirror), the
// reflection branch traces a real reflection ray off the pane against the scene
// TLAS, so a window mirrors actual off-screen geometry. A missed ray falls back
// to the same probe / sky path glass.frag uses, so a probe-baked or env-mapped
// world degrades gracefully.
//
// The trace helpers (RtGeomEntry, the vertex fetchers, rt_shadow, the trace loop,
// rt_shade_hit) are lifted from rt_reflections.frag -- keep them in sync. Only the
// descriptor layout differs: glass keeps its view / params / global sets (0 / 1 /
// 2) so the surface + probe code is reused unchanged, and the RT geometry rides a
// new set 3 (the bindless pool on set 4 for the textured variant), rather than the
// fullscreen pass's single set 0.
//
// `USE_MSAA` (host define) matches the depth sampler to the main pass's sample
// count. `RT_TEXTURED` selects the bindless-pool hit shading. Compiled through the
// Vulkan-1.2 / SPIR-V-1.4 ray-query target (compile_glsl_rt), unlike the base
// glass.frag (compile_glsl, Vulkan-1.0).

// set 0: the glass view block (identical to glass.frag).
layout(std140, set = 0, binding = 0) uniform TransparentViewBlock {
    mat4  vp;
    mat4  inv_vp;
    vec4  camera_pos;
    vec2  viewport;
    float time;
    // Mips in the sky prefilter cube; 0 = no EnvironmentMap bound (the reflection
    // then keeps the white rim where a ray misses and no probe / env cube exists).
    float prefilter_mip_count;
} view;

// set 1: the per-pane glass params (identical to glass.frag). The `planar` field
// + the set 1 binding 1 planar sampler are unused on the RT path (the trace
// supersedes planar) but kept so the params set + pipeline layout match the base
// pass for descriptor reuse.
layout(std140, set = 1, binding = 0) uniform GlassParamsBlock {
    vec4  centre;
    vec4  normal;
    vec4  tint;
    float opacity;
    float refraction_strength;
    float fresnel_power;
    float planar;
} params;

// Pre-transparent scene snapshot (single-sample HDR) sampled for refraction.
layout(set = 0, binding = 1) uniform sampler2D scene_color;
// Main-pass depth, for the manual occlusion test. Matched to the resource's
// sample count via USE_MSAA.
#if USE_MSAA
layout(set = 0, binding = 2) uniform sampler2DMS scene_depth;
#else
layout(set = 0, binding = 2) uniform sampler2D scene_depth;
#endif

// set 2 binding 5: the sky IBL prefilter cube (identical to glass.frag) -- the
// reflection fallback where a ray misses and no probe covers, plus the hit-shade
// IBL tap.
layout(set = 2, binding = 5) uniform samplerCube prefilter_cube;

// The reflection-probe set (binding 7) + cube array (binding 8) + box-parallax
// sampling, substituted from probe_common.glsl ({PROBE_DESC_SET} = 2). A ray that
// misses the scene falls back to the local probe instead of the foreign sky cube.
{PROBE_COMMON}

// set 3: the ray-tracing scene resources, mirroring rt_reflections.frag set 0 but
// renumbered gap-free -- the fullscreen pass's screen-space scene / gbuffer /
// roughness inputs are dropped (glass traces off the pane surface point, not a
// reconstructed G-buffer pixel), so the static + skinned vertex/index buffers move
// up to bindings 3..6.
layout(set = 3, binding = 0) uniform RtParamsBlock {
    float intensity;
    float max_distance;
    float tan_half_fov_y;
    float aspect;
    float prefilter_mip_count;
    float _pad0;
    float _pad1;
    float _pad2;
    vec4  cam_pos;
    vec4  sun_dir;     // xyz: world unit direction toward the sun (= L)
    vec4  sun_color;   // xyz: sun radiance
    mat4  inv_view;
} rt;
layout(set = 3, binding = 1) uniform accelerationStructureEXT scene_tlas;
// Per-instance geometry table. Scalar members so the std430 offsets match the
// 128-byte #[repr(C)] render_types::RtGeomEntry exactly (lifted from
// rt_reflections.frag).
struct RtGeomEntry {
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
    mat4  model;
    uint  emissive_map_index;
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
};
layout(set = 3, binding = 2, std430) readonly buffer GeomTable { RtGeomEntry geom[]; };
// The shared static vertex (14 floats / 56 B stride) + u32 index buffers.
layout(set = 3, binding = 3, std430) readonly buffer Verts { float verts[]; };
layout(set = 3, binding = 4, std430) readonly buffer Indices { uint indices[]; };
// The deformed (posed) skinned vertex buffer (same 14-float layout the skin
// compute kernel writes) + the u16-packed skinned index buffer, for skinned hits.
// Both bind a 1-element dummy when the scene carries no skinned geometry, so the
// binding is always valid.
layout(set = 3, binding = 5, std430) readonly buffer SVerts { float sverts[]; };
layout(set = 3, binding = 6, std430) readonly buffer SIdx { uint sidx[]; };

#ifdef RT_TEXTURED
// set 4 binding 1: the bindless albedo + normal-map pool, the same set the main
// bindless pass binds (binding 0 there is the object SSBO, unused here). Set 4
// (past the global set 2 + the rt set 3) keeps those at fixed indices across the
// flat + textured variants.
layout(set = 4, binding = 1) uniform sampler2D tex_pool[{POOL_SIZE}];
#endif

const float RT_F0 = 0.04; // dielectric base reflectance
// Skinned objects flag bit 31 of `normal_index`; the trace then fetches the hit
// triangle from the deformed-vertex / u16 skinned index buffers. Matches
// render_types::RT_SKINNED_FLAG / raytrace.rs.
const uint  RT_SKINNED_FLAG = 0x80000000u;

layout(location = 0) in vec3 world_pos;
layout(location = 0) out vec4 out_color;

// Vertex attribute fetchers into the shared 14-float Vertex (normal@float 3,
// tangent@float 6, uv@float 12), addressed by absolute vertex index `vi`.
vec3 rt_vertex_normal(uint vi)  { return vec3(verts[vi * 14u + 3u], verts[vi * 14u + 4u], verts[vi * 14u + 5u]); }
vec3 rt_vertex_tangent(uint vi) { return vec3(verts[vi * 14u + 6u], verts[vi * 14u + 7u], verts[vi * 14u + 8u]); }
vec2 rt_vertex_uv(uint vi)      { return vec2(verts[vi * 14u + 12u], verts[vi * 14u + 13u]); }

vec3 rt_skinned_normal(uint vi)  { return vec3(sverts[vi * 14u + 3u], sverts[vi * 14u + 4u], sverts[vi * 14u + 5u]); }
vec3 rt_skinned_tangent(uint vi) { return vec3(sverts[vi * 14u + 6u], sverts[vi * 14u + 7u], sverts[vi * 14u + 8u]); }
vec2 rt_skinned_uv(uint vi)      { return vec2(sverts[vi * 14u + 12u], sverts[vi * 14u + 13u]); }

// Fetch one u16 index `o` from the packed skinned index buffer (two u16 per uint).
// The skinned BLAS bakes absolute indices, so no base_vertex is added.
uint rt_skinned_index(uint o) {
    return (sidx[o >> 1u] >> ((o & 1u) * 16u)) & 0xFFFFu;
}

// Trace a shadow ray from `hp` toward the sun; 0 when occluded (cast shadow inside
// the reflection), 1 when the sun is visible.
float rt_shadow(vec3 hp, vec3 n) {
    rayQueryEXT sq;
    rayQueryInitializeEXT(
        sq, scene_tlas,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        0xFFu, hp + n * 0.02, 0.001, normalize(rt.sun_dir.xyz), rt.max_distance);
    rayQueryProceedEXT(sq);
    return (rayQueryGetIntersectionTypeEXT(sq, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
        ? 0.0 : 1.0;
}

// Metallic/roughness hit shading. A sun diffuse term (dielectric only, masked by
// the shadow ray) plus split IBL: diffuse irradiance along N and a specular tap
// along the onward reflection at a roughness-selected prefilter mip, tinted by F0.
// Metals drop the diffuse term and tint the reflected environment by their albedo.
// The self-emission is added on top. Lifted from rt_reflections.frag::rt_shade_hit
// (the prefilter cube is glass's set 2 binding 5).
vec3 rt_shade_hit(vec3 N, vec3 albedo, float hit_rough, float metallic, vec3 emissive,
                  vec3 dir, bool ibl, float max_mip, float shadow) {
    vec3 F0     = mix(vec3(RT_F0), albedo, metallic);
    vec3 diff_a = albedo * (1.0 - metallic);
    float ndl   = clamp(dot(N, rt.sun_dir.xyz), 0.0, 1.0);
    vec3 sun    = diff_a * rt.sun_color.xyz * ndl * shadow;
    if (!ibl) return sun + (diff_a + F0) * 0.03 + emissive;
    vec3 refl = reflect(dir, N);
    vec3 spec = textureLod(prefilter_cube, refl, hit_rough * max_mip).rgb;
    vec3 diff = textureLod(prefilter_cube, N, max_mip).rgb * diff_a;
    return sun + diff + spec * F0 + emissive;
}

void main() {
    vec3 view_dir = normalize(view.camera_pos.xyz - world_pos);
    // Two-sided: orient the normal toward the viewer so a pane lit from behind
    // still Fresnels correctly.
    vec3 n = normalize(params.normal.xyz);
    if (dot(n, view_dir) < 0.0) {
        n = -n;
    }

    vec2 vp_dim = max(view.viewport, vec2(1.0, 1.0));
    vec2 frag_uv = gl_FragCoord.xy / vp_dim;

    // Manual depth occlusion: discard where the scene depth at this pixel is nearer
    // than the pane (the pass has no hardware depth test). Identical to glass.frag.
    ivec2 pixel = min(ivec2(gl_FragCoord.xy), ivec2(vp_dim) - ivec2(1, 1));
    float scene_self_depth = texelFetch(scene_depth, pixel, 0).r;
    if (scene_self_depth < gl_FragCoord.z) {
        discard;
    }

    // Refraction: perturb the screen lookup by the pane normal's screen-plane
    // component so the background bends across the pane.
    vec2 refract_uv = clamp(frag_uv + n.xy * params.refraction_strength,
                            vec2(0.001), vec2(0.999));
    vec3 refracted = texture(scene_color, refract_uv).rgb * params.tint.rgb;

    // Per-pixel reflection ray: spawn it off the world-space pane surface point
    // (the interpolated quad vertex, pre-transformed to world space at build time)
    // along the mirror direction. A pane is smooth, so the reflection is sharp.
    vec3 origin = world_pos + n * 0.02;        // nudge off the surface
    vec3 dir    = reflect(-view_dir, n);
    bool ibl    = view.prefilter_mip_count > 0.5;
    float max_mip = view.prefilter_mip_count - 1.0;

    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, scene_tlas, gl_RayFlagsOpaqueEXT, 0xFFu,
                          origin, 0.01, dir, rt.max_distance);
    while (rayQueryProceedEXT(rq)) {}

    vec3 reflection;
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
        uint inst = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));
        uint tri  = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
        vec2 b    = rayQueryGetIntersectionBarycentricsEXT(rq, true);
        float tt  = rayQueryGetIntersectionTEXT(rq, true);

        RtGeomEntry e = geom[inst];
        bool skin = (e.normal_index & RT_SKINNED_FLAG) != 0u;
        uint nidx = e.normal_index & ~RT_SKINNED_FLAG;
        uint o  = e.index_offset + tri * 3u;
        float w0 = 1.0 - b.x - b.y;

        vec3 nl, tl;
        vec2 huv;
        if (skin) {
            uint i0 = rt_skinned_index(o);
            uint i1 = rt_skinned_index(o + 1u);
            uint i2 = rt_skinned_index(o + 2u);
            nl  = rt_skinned_normal(i0) * w0 + rt_skinned_normal(i1) * b.x + rt_skinned_normal(i2) * b.y;
            tl  = rt_skinned_tangent(i0) * w0 + rt_skinned_tangent(i1) * b.x + rt_skinned_tangent(i2) * b.y;
            huv = rt_skinned_uv(i0) * w0 + rt_skinned_uv(i1) * b.x + rt_skinned_uv(i2) * b.y;
        } else {
            uint i0 = indices[o] + e.base_vertex;
            uint i1 = indices[o + 1u] + e.base_vertex;
            uint i2 = indices[o + 2u] + e.base_vertex;
            nl  = rt_vertex_normal(i0) * w0 + rt_vertex_normal(i1) * b.x + rt_vertex_normal(i2) * b.y;
            tl  = rt_vertex_tangent(i0) * w0 + rt_vertex_tangent(i1) * b.x + rt_vertex_tangent(i2) * b.y;
            huv = rt_vertex_uv(i0) * w0 + rt_vertex_uv(i1) * b.x + rt_vertex_uv(i2) * b.y;
        }

        mat3 m3 = mat3(e.model);
        vec3 nw = normalize(m3 * nl);
        if (dot(nw, dir) > 0.0) nw = -nw;
        vec3 tw = m3 * tl;

        vec3 tint     = vec3(e.tint_r, e.tint_g, e.tint_b);
        vec3 emissive = vec3(e.emissive_r, e.emissive_g, e.emissive_b);
        vec3 hp = origin + dir * tt;
        float shadow = rt_shadow(hp, nw);

#ifdef RT_TEXTURED
        // level 0: a reflected ray's screen-space UV gradients are unstable, so
        // sample the base mip to avoid gradient-driven mip thrash.
        vec3 albedo = tint
                    * textureLod(tex_pool[nonuniformEXT(e.albedo_index)], huv, 0.0).rgb;
        vec3 N = nw;
        float tlen = length(tw);
        if (tlen > 1e-4) {
            vec3 nm = textureLod(tex_pool[nonuniformEXT(nidx)], huv, 0.0).xyz * 2.0 - 1.0;
            vec3 T = tw / tlen;
            T = normalize(T - N * dot(N, T));        // Gram-Schmidt
            vec3 B = cross(N, T);
            N = normalize(T * nm.x + B * nm.y + N * nm.z);
        }
        if (e.emissive_map_index != 0u) {
            emissive *= textureLod(tex_pool[nonuniformEXT(e.emissive_map_index)], huv, 0.0).rgb;
        }
        reflection = rt_shade_hit(N, albedo, e.roughness, e.metallic, emissive, dir, ibl, max_mip, shadow);
#else
        reflection = rt_shade_hit(nw, tint, e.roughness, e.metallic, emissive, dir, ibl, max_mip, shadow);
#endif
    } else {
        // The ray escaped the scene: fall back to the local reflection probe (box-
        // parallax, blended across covering probes) when one is baked, else the sky
        // prefilter cube, else a white rim so a probe-less, env-less world still
        // reads as glass. A pane is smooth, so every fallback is sharp (mip 0).
        if (probe_set.count > 0u) {
            reflection = probe_set_specular(world_pos, dir, 0.0);
        } else if (ibl) {
            reflection = textureLod(prefilter_cube, dir, 0.0).rgb;
        } else {
            reflection = vec3(1.0);
        }
    }

    // Schlick Fresnel (F0 = 0.04 dielectric) drives the reflection/refraction
    // blend: ~4% head-on, rising to a full mirror at grazing. Identical to
    // glass.frag.
    float n_dot_v = clamp(dot(n, view_dir), 0.0, 1.0);
    float rim = pow(1.0 - n_dot_v, max(params.fresnel_power, 1e-3));
    float refl_weight = clamp(0.04 + 0.96 * rim, 0.0, 1.0);
    vec3 colour = mix(refracted, reflection, refl_weight);
    float alpha = clamp(mix(params.opacity, 1.0, rim), 0.0, 1.0);

    out_color = vec4(colour, alpha);
}
