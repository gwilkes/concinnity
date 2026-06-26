#version 460
#extension GL_EXT_ray_query : require
#extension GL_EXT_nonuniform_qualifier : require

// Hardware ray-traced reflections (Vulkan inline `rayQueryEXT`). A fullscreen
// fragment pass that, per glossy pixel, rebuilds a world-space surface point +
// normal from the SSR pre-pass G-buffer, traces a reflection ray against the
// scene's top-level acceleration structure, and composites the reflected colour
// over the base scene with the same Fresnel/gloss weighting SSR uses. Unlike SSR
// the ray is a real world-space trace, so reflected geometry that is off-screen
// still appears.
//
// Port of src/directx/shaders/rt_reflections.hlsl (DXR 1.1 inline `RayQuery`).
// `GL_EXT_ray_query` is the direct analog of HLSL `RayQuery`. Two hit-shading
// variants share all the setup + trace logic, selected by `RT_TEXTURED`
// (compiled twice, like the bindless main pass): the flat variant uses the
// material tint, the textured variant samples the hit's albedo + normal map from
// the bindless pool. Both shade the hit with a metallic/roughness response (sun
// diffuse for dielectrics + split IBL), fall back to the IBL prefilter cube on a
// miss, and trace a second shadow ray toward the sun so cast shadows appear
// inside the reflection.

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

// set 0 binding 0: RT tunables + camera + sun. Layout matches render_types::RtParams (144 B).
layout(set = 0, binding = 0) uniform RtParamsBlock {
    float intensity;
    float max_distance;
    float tan_half_fov_y;
    float aspect;
    float prefilter_mip_count;
    float _pad0;
    float _pad1;
    float _pad2;
    vec4  cam_pos;     // xyz: world camera position (ray origin)
    vec4  sun_dir;     // xyz: world unit direction toward the sun (= L)
    vec4  sun_color;   // xyz: sun radiance
    mat4  inv_view;    // camera-to-world (column-major)
} params;

// set 0 binding 1: the scene top-level acceleration structure.
layout(set = 0, binding = 1) uniform accelerationStructureEXT scene_tlas;

// set 0 binding 2: per-instance geometry table. Scalar members so the std430
// offsets match the 128-byte #[repr(C)] render_types::RtGeomEntry exactly
// (index_offset@0, base_vertex@4, albedo_index@8, normal_index@12, tint@16,
// roughness@28, metallic@32, emissive@36, model@48, emissive_map_index@112).
// The _pad tail rounds the struct to 128 bytes so its std430 array stride
// matches the Rust side (a matrix-bearing struct rounds up to a 16-byte multiple).
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
layout(set = 0, binding = 2, std430) readonly buffer GeomTable { RtGeomEntry geom[]; };
// set 0 binding 3/4: the shared vertex (14 floats / 56 B stride) + u32 index
// buffers, bound raw as storage buffers; the shader fetches the hit triangle's
// attributes from these directly.
layout(set = 0, binding = 3, std430) readonly buffer Verts { float verts[]; };
layout(set = 0, binding = 4, std430) readonly buffer Indices { uint indices[]; };
// set 0 binding 9/10: the deformed (posed) skinned vertex buffer (same 14-float
// / 56 B `Vertex` layout the skin compute kernel writes) + the u16 skinned index
// buffer (packed two indices per uint), for skinned hits. A skinned hit fetches
// its triangle from these instead of the static u32 buffers; both bind a
// 1-element dummy when the scene carries no skinned geometry, so the binding is
// always valid.
layout(set = 0, binding = 9, std430)  readonly buffer SVerts { float sverts[]; };
layout(set = 0, binding = 10, std430) readonly buffer SIdx { uint sidx[]; };

// Screen-space inputs reused from the SSR resolve pass.
layout(set = 0, binding = 5) uniform sampler2D   scene_tex; // hdr_resolve (base scene colour)
layout(set = 0, binding = 6) uniform sampler2D   gbuffer;   // view normal (xyz) + linear depth (a)
layout(set = 0, binding = 7) uniform sampler2D   rough_tex; // roughness (r)
layout(set = 0, binding = 8) uniform samplerCube prefilter; // IBL prefilter cube (miss fallback)

// set 1: the forward global set, bound here only for its reflection-probe count +
// per-probe parallax boxes (binding 7) + cube array (binding 8); a ray that misses
// the scene falls back to the local probe instead of the sky. The shared probe
// sampling is substituted in below at set index 1 (the marker must not appear in
// this comment, or it would be substituted here too).
{PROBE_COMMON}

#ifdef RT_TEXTURED
// set 2 binding 1: the bindless albedo + normal-map pool, identical to the main
// bindless pass (binding 0 there is the object SSBO, unused here). Only the
// textured variant references it. Set 2 (not 1) so the global probe set above
// keeps a fixed index (1) across the flat + textured variants.
layout(set = 2, binding = 1) uniform sampler2D tex_pool[{POOL_SIZE}];
#endif

const float RT_ROUGH_CUT = 0.6;  // surfaces rougher than this get no reflection
const float RT_F0        = 0.04; // dielectric base reflectance for the Fresnel

// Skinned objects flag bit 31 of `normal_index`; the trace then fetches the hit
// triangle from the deformed-vertex / u16 skinned index buffers (which mirror
// the static layout) instead of the static u32 ones. The flag is masked off
// before the pool sample, so skinned hits shade textured like static ones.
// Matches render_types::RT_SKINNED_FLAG / raytrace.rs.
const uint RT_SKINNED_FLAG = 0x80000000u;

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// Byte-identical to ssr_view_pos.
vec3 rt_view_pos(vec2 uv, float depth, float tan_y, float asp) {
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return vec3(ndc.x * tan_y * asp, ndc.y * tan_y, -1.0) * depth;
}

// Vertex attribute fetchers into the shared 14-float Vertex (normal@float 3,
// tangent@float 6, uv@float 12), addressed by absolute vertex index `vi`.
vec3 rt_vertex_normal(uint vi)  { return vec3(verts[vi * 14u + 3u], verts[vi * 14u + 4u], verts[vi * 14u + 5u]); }
vec3 rt_vertex_tangent(uint vi) { return vec3(verts[vi * 14u + 6u], verts[vi * 14u + 7u], verts[vi * 14u + 8u]); }
vec2 rt_vertex_uv(uint vi)      { return vec2(verts[vi * 14u + 12u], verts[vi * 14u + 13u]); }

// Same fetchers into the deformed (posed) skinned vertex buffer, which carries
// the identical 56-byte `Vertex` layout the skin compute kernel writes.
vec3 rt_skinned_normal(uint vi)  { return vec3(sverts[vi * 14u + 3u], sverts[vi * 14u + 4u], sverts[vi * 14u + 5u]); }
vec3 rt_skinned_tangent(uint vi) { return vec3(sverts[vi * 14u + 6u], sverts[vi * 14u + 7u], sverts[vi * 14u + 8u]); }
vec2 rt_skinned_uv(uint vi)      { return vec2(sverts[vi * 14u + 12u], sverts[vi * 14u + 13u]); }

// Fetch one u16 index `o` from the packed skinned index buffer (two u16 per
// uint). The skinned BLAS bakes absolute indices, so no base_vertex is added.
uint rt_skinned_index(uint o) {
    return (sidx[o >> 1u] >> ((o & 1u) * 16u)) & 0xFFFFu;
}

// Trace a shadow ray from `hp` toward the sun; returns 0 when occluded (cast
// shadow inside the reflection), 1 when the sun is visible.
float rt_shadow(vec3 hp, vec3 n) {
    rayQueryEXT sq;
    rayQueryInitializeEXT(
        sq, scene_tlas,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        0xFFu, hp + n * 0.02, 0.001, normalize(params.sun_dir.xyz), params.max_distance);
    rayQueryProceedEXT(sq);
    return (rayQueryGetIntersectionTypeEXT(sq, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
        ? 0.0 : 1.0;
}

// Metallic/roughness hit shading. A sun diffuse term (dielectric only, masked by
// the shadow ray) plus split IBL: diffuse irradiance along N and a specular tap
// along the onward reflection at a roughness-selected prefilter mip, tinted by
// F0. Metals drop the diffuse term and tint the reflected environment by their
// albedo. The self-emission is added on top.
vec3 rt_shade_hit(vec3 N, vec3 albedo, float hit_rough, float metallic, vec3 emissive,
                  vec3 dir, bool ibl, float max_mip, float shadow) {
    vec3 F0     = mix(vec3(RT_F0), albedo, metallic);
    vec3 diff_a = albedo * (1.0 - metallic);
    float ndl   = clamp(dot(N, params.sun_dir.xyz), 0.0, 1.0);
    vec3 sun    = diff_a * params.sun_color.xyz * ndl * shadow;
    if (!ibl) return sun + (diff_a + F0) * 0.03 + emissive;
    vec3 refl = reflect(dir, N);
    vec3 spec = textureLod(prefilter, refl, hit_rough * max_mip).rgb;
    vec3 diff = textureLod(prefilter, N, max_mip).rgb * diff_a;
    return sun + diff + spec * F0 + emissive;
}

void main() {
    vec3 base = texture(scene_tex, frag_uv).rgb;
    vec4 g = texture(gbuffer, frag_uv);
    float depth = g.a;
    // Background / sky, or a non-reflecting (too-rough) surface: weight 0 so the
    // reflection composite keeps the scene there. The resolve writes reflected
    // radiance (.rgb) + composite weight (.a), not yet blended.
    if (depth <= 0.0) { out_color = vec4(base, 0.0); return; }

    float roughness = texture(rough_tex, frag_uv).r;
    float gloss = clamp((RT_ROUGH_CUT - roughness) / RT_ROUGH_CUT, 0.0, 1.0);
    if (gloss <= 0.0) { out_color = vec4(base, 0.0); return; }

    vec3 Nv = normalize(g.xyz);
    vec3 Pv = rt_view_pos(frag_uv, depth, params.tan_half_fov_y, params.aspect);
    vec3 Pw = (params.inv_view * vec4(Pv, 1.0)).xyz;
    vec3 Nw = normalize(mat3(params.inv_view) * Nv);
    vec3 V  = normalize(params.cam_pos.xyz - Pw);

    vec3  origin  = Pw + Nw * 0.01;            // nudge off the surface
    vec3  dir     = reflect(-V, Nw);
    bool  ibl     = params.prefilter_mip_count > 0.5;
    float max_mip = params.prefilter_mip_count - 1.0;
    float ndv     = clamp(dot(Nw, V), 0.0, 1.0);
    float fresnel = RT_F0 + (1.0 - RT_F0) * pow(1.0 - ndv, 5.0);
    float weight  = clamp(fresnel * gloss * params.intensity, 0.0, 1.0);

    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, scene_tlas, gl_RayFlagsOpaqueEXT, 0xFFu,
                          origin, 0.01, dir, params.max_distance);
    while (rayQueryProceedEXT(rq)) {}

    vec3 reflected;
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
            // Skinned: u16 indices into the deformed buffer; the skinned BLAS
            // bakes absolute indices, so base_vertex is 0.
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
        // sample the base mip to avoid gradient-driven mip thrash. Skinned hits
        // take the same path: their textures live in the shared bindless pool
        // (the geom entry carries the real pool indices), and `huv` / `tw` come
        // from the deformed buffer, so the textured shading is identical.
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
        // Colour the self-emission by the emissive map (e.g. the bistro string
        // lights), mirroring the main bindless pass. Gated on a non-zero pool
        // index; the flat fallback variant has no pool and keeps scalar emissive.
        if (e.emissive_map_index != 0u) {
            emissive *= textureLod(tex_pool[nonuniformEXT(e.emissive_map_index)], huv, 0.0).rgb;
        }
        reflected = rt_shade_hit(N, albedo, e.roughness, e.metallic, emissive, dir, ibl, max_mip, shadow);
#else
        reflected = rt_shade_hit(nw, tint, e.roughness, e.metallic, emissive, dir, ibl, max_mip, shadow);
#endif
    } else {
        // The ray escaped the scene: fall back to the local reflection probe (box-
        // parallax, blended across covering probes) when one is baked, else the IBL
        // prefilter sky, else the base shading.
        float lod = roughness * max_mip;
        if (probe_set.count > 0u) {
            reflected = probe_set_specular(Pw, dir, lod);
        } else {
            reflected = ibl ? textureLod(prefilter, dir, lod).rgb : base;
        }
    }

    // Reflected radiance + composite weight, not yet blended; the reflection
    // composite pass blurs by roughness and composites this over the scene.
    out_color = vec4(reflected, weight);
}
