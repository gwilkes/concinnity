// Default MSL shader for Concinnity scenes.
//
// Covers the standard FPS/room use case: textured geometry with Cook-Torrance
// GGX PBR lighting (Smith G + Schlick Fresnel, energy-conserving multi-scatter
// compensation via the Karis analytic BRDF fit), tangent-space normal mapping,
// a runtime light array, and PCF directional shadow mapping over N cascades
// (NUM_SHADOW_CASCADES). Use this for most scenes. Only write a custom shader
// when you need lighting models, effects, or buffer layouts this file cannot
// provide.
//
// Buffer bindings (must match metal.rs):
//   buffer(0) -- ViewUniforms: float4x4 vp, float4x4 view, float elapsed,
//                              float _pad, packed_float3 cam_pos,
//                              float prefilter_mip_count (0 = IBL disabled)
//   buffer(1) -- per-vertex data: float3 pos, float3 normal, float3 tangent,
//                                 float3 color, float2 uv
//   buffer(2) -- ModelUniforms: float4x4 model
//   buffer(3) -- MaterialUniforms: roughness, metallic, tint, emissive
//   buffer(4) -- LightUniforms: directional[4], point[8], counts
//   buffer(5) -- ShadowUniforms: float4x4 light_vps[NUM_SHADOW_CASCADES],
//                                float4    cascade_splits
//
// Texture bindings:
//   texture(0) -- 2-D RGBA albedo texture, repeat wrap
//   texture(1) -- 2-D RGBA tangent-space normal map; (128,128,255) = no perturbation
//   texture(2) -- depth2d_array shadow map (one slice per cascade)
//   texture(3) -- texturecube<float> IBL irradiance map (or 1x1 fallback)
//   texture(4) -- texturecube<float> IBL prefiltered radiance map with mip chain
//                 (or 1x1 fallback when IBL is disabled)
//   sampler(0) -- linear filter, repeat wrap (albedo + normal map)
//   sampler(1) -- linear filter, clamp-to-edge, LessEqual compare (shadow PCF)
//   sampler(2) -- linear filter (mip + min + mag), clamp-to-edge (cubemap sampling)
//
// Lighting:
//   Per-vertex normals and tangents are transformed to world space in the vertex
//   shader. The fragment shader builds a TBN matrix and samples the normal map
//   to produce the perturbed surface normal used in Cook-Torrance GGX PBR
//   lighting. All scene lights come from the LightUniforms buffer; if none are
//   declared the runtime supplies a default warm sun (see LightUniforms::DEFAULT
//   in render_types.rs).
//   DirectionalLight.direction is a unit vector pointing TOWARD the light source
//   (same convention as the L vector in the BRDF integrand).
//
// Shadows:
//   The first directional light casts soft cascaded shadows: a depth array
//   with NUM_SHADOW_CASCADES slices, each fitting a slice of the camera
//   frustum. The fragment shader picks a cascade from the fragment's
//   view-space depth, then samples that slice with a 5x5 grid PCF kernel
//   rotated by a per-pixel hash. Cascade math (split fitting, sphere bounds,
//   texel snap) lives in src/gfx/csm.rs.
//
// Sky pass:
//   Skybox vertices carry a blue channel value of 2.0 (sentinel). The vertex
//   shader forces clip-space depth to w so sky always renders behind scene
//   geometry. The fragment shader computes sky colour from the view direction
//   (elevation angle) rather than UV coordinates, giving a seamless gradient
//   across all skybox faces regardless of UV layout.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint NUM_SHADOW_CASCADES = 4;

// Size of the bindless texture pool the static main pass reads through the
// `BindlessTextures` argument buffer. The pool holds every albedo texture
// (including emissive and ORM maps) followed by every normal map; each
// GpuObjectData carries pool indices into it. Must match BINDLESS_TEXTURE_COUNT
// in metal/context.rs. This is an argument-buffer texture array (not a direct
// per-stage binding), so it is not bound by the 128-texture stage limit; large
// scenes like a full city block need room for hundreds of unique textures.
constant constexpr uint BINDLESS_TEXTURE_COUNT = 1024;

struct Vertex {
    float3 pos     [[attribute(0)]];
    float3 normal  [[attribute(1)]];
    float3 tangent [[attribute(2)]];
    float3 color   [[attribute(3)]];
    float2 uv      [[attribute(4)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 world_pos;
    float3 normal;
    float3 tangent;
    float3 color;
    float2 uv;
    // View-space depth (positive in front of camera). Used by the fragment
    // shader to pick a shadow cascade.
    float  view_depth;
    // Object id (index into the GpuObjectData buffer). Set by vertex_main from
    // [[base_instance]]; read by fragment_main_bindless to fetch material +
    // texture indices. The instanced/skinned vertex shaders set it to 0 since
    // their fragment_main pairing does not consult it.
    uint   obj_id [[flat]];
};

struct ViewUniforms {
    float4x4 vp;
    float4x4 view;
    float elapsed;
    // 1.0 when a screen-space / ray-traced reflection resolve composites this
    // frame. Below the reflection roughness cut the forward specular yields to
    // that resolve (whose miss-fallback samples this same probe set), so a
    // glossy surface does not show both the parallax-approximate forward probe
    // reflection and the exact resolved one. 0.0 keeps the full forward probe
    // specular (no resolve: non-RT backends, reflections off, probe-face bake).
    float reflections_enabled;
    packed_float3 cam_pos;
    // Number of mip levels in the bound prefilter cubemap. Set to 0 by the
    // runtime when no EnvironmentMap is bound; the fragment shader uses this
    // as the IBL "enabled" flag and falls back to a flat ambient placeholder.
    float prefilter_mip_count;
};

// Reflection-probe parallax box (buffer(6)). Matches metal::uniforms::ProbeUniforms.
// box_min.w is the enabled flag (0 = no baked probe / parallax off). When on, the
// specular term box-projects the reflection ray against [box_min, box_max] and
// re-anchors the probe cube sample relative to probe_pos (the capture point), so
// a static captured cube tracks a moving camera.
struct ProbeUniforms {
    float4 box_min;   // xyz = influence-box min, w = enabled
    float4 box_max;   // xyz = influence-box max
    float4 probe_pos; // xyz = capture position
};

// Maximum reflection probes bound per frame. Must match metal::uniforms::MAX_PROBES
// and the `BindlessTextures.probes` cube-array length.
constant constexpr uint MAX_PROBES = 8u;

// Probe blend falloff, as a fraction of a box's smallest half-extent. Each probe's
// influence weight ramps across a shell of this half-width centred on the box
// surface (1 deep inside, 0.5 at the surface, 0 a margin outside), so adjacent
// boxes cross-fade through their shared face instead of hard-switching. Larger =
// wider, softer transitions.
constant constexpr float PROBE_BLEND_MARGIN = 0.2;

// The full reflection-probe set (buffer(6)). Matches metal::uniforms::ProbeSet.
// The fragment shader weights every probe by how deep the surface sits inside its
// box and blends the two highest-weighted, falling back to the nearest by
// capture distance where no box covers.
struct ProbeSet {
    uint count;
    // Three scalar uints, NOT a uint3: a uint3 is 16-byte aligned in MSL, which
    // pushes `probes` to offset 32 (struct 416 bytes) and mismatches the CPU-side
    // metal::uniforms::ProbeSet, whose [u32; 3] keeps `probes` at offset 16 (struct
    // 400 bytes). A mismatch reads every probe shifted by one float4. The
    // static_assert locks the 400-byte layout at shader-compile time.
    uint _pad0;
    uint _pad1;
    uint _pad2;
    ProbeUniforms probes[MAX_PROBES];
};
static_assert(sizeof(ProbeSet) == 400,
              "ProbeSet must be 400 bytes to match the CPU-side metal::uniforms::ProbeSet");

struct ModelUniforms {
    float4x4 model;
};

// Per-object record for the bindless static main pass, bound at buffer(9) and
// indexed by the object id delivered through [[base_instance]]. Replaces the
// per-draw ModelUniforms + MaterialUniforms + texture binds, so each static
// draw call carries no state of its own. Layout (144 bytes) must match the
// Rust GpuObjectData in ffi/render_types.rs. The bb_min/bb_max/cull_distance
// fields are unused here, carried for the compute cull kernel.
struct GpuObjectData {
    float4x4      model;
    packed_float3 tint;
    float         roughness;
    packed_float3 emissive;
    float         metallic;
    uint          albedo_index;
    uint          normal_index;
    float         macro_variation;
    float         terrain_blend;
    packed_float3 bb_min;
    float         cull_distance;
    packed_float3 bb_max;
    float         secondary_blend_sharpness;
    uint          albedo_secondary_index;
    uint          normal_secondary_index;
    uint          emissive_map_index;
    uint          orm_map_index;
};

struct MaterialUniforms {
    float  roughness;
    float  metallic;
    float  macro_variation;
    float  terrain_blend;
    packed_float3 tint;
    float  _pad2;
    packed_float3 emissive;
    float  secondary_blend_sharpness;
    uint   albedo_secondary_index;
    uint   normal_secondary_index;
    uint   emissive_map_index;
    uint   orm_map_index;
    // CPU-side routing fields (opacity + transparent + see-through flags).
    // Present so the struct size matches the Rust MaterialUniforms; the opaque
    // main pass never reads them (transparent meshes are skipped CPU-side before
    // this shader).
    float  opacity;
    uint   transparent;
    uint   see_through;
};

// packed_float3 has size=12 align=4 -- matches Rust [f32; 3] so struct offsets
// line up with the DirectionalLightData / PointLightData layouts in render_types.rs.
// Using plain float3 here would give size=16 align=16 in a constant buffer, which
// shifts every following field and makes the color channel read as zeros.
struct DirectionalLightData {
    packed_float3 direction;
    float         intensity;
    packed_float3 color;
    float         _pad;
};

struct PointLightData {
    packed_float3 position;
    float         range;
    packed_float3 color;
    float         intensity;
};

struct LightUniforms {
    DirectionalLightData directional[4];
    PointLightData       point[8];
    int num_directional;
    int num_point;
    // Multiplier on the indirect (IBL / flat-fallback) ambient term. 1.0 leaves
    // the physical ambient untouched; >1 lifts shadow fill. Occupies the first
    // trailing pad word so the layout still matches the Rust LightUniforms.
    float ambient_intensity;
    float _pad;
};

struct ShadowUniforms {
    float4x4 light_vps[NUM_SHADOW_CASCADES];
    // x..w = view-space far depth for cascades 0..3
    float4   cascade_splits;
    // Live cascade count (1..4); slots at or beyond it are unrendered, so the
    // selection + blend below must not reach them.
    uint     active_cascades;
};

// Sky gradient colours matching the procedural sky texture generator.
constant float3 SKY_ZENITH  = float3(0.110, 0.322, 0.726);
constant float3 SKY_HORIZON = float3(0.765, 0.863, 0.941);

// Static main-pass vertex shader. The per-object model matrix is read from
// the bindless GpuObjectData buffer at buffer(9), indexed by the object id the
// renderer supplies as the draw call's [[base_instance]]. No per-draw uniform
// is bound; this is what lets every static draw be a bare
// drawIndexedPrimitives and a compute-encoded indirect command.
vertex VertexOut vertex_main(
    Vertex in                        [[stage_in]],
    constant ViewUniforms   &view    [[buffer(0)]],
    constant GpuObjectData  *objects [[buffer(9)]],
    uint                     obj_id  [[base_instance]]
) {
    VertexOut out;
    float4x4 model    = objects[obj_id].model;
    float4 world_pos  = model * float4(in.pos, 1.0);
    out.position      = view.vp * world_pos;
    out.world_pos     = world_pos.xyz;
    out.normal        = normalize(float3(model * float4(in.normal,  0.0)));
    out.tangent       = normalize(float3(model * float4(in.tangent, 0.0)));
    out.color         = in.color;
    out.uv            = in.uv;
    // view * world_pos -> view-space; positive view depth is -z.
    out.view_depth    = -(view.view * world_pos).z;
    out.obj_id        = obj_id;

    if (in.color.b > 1.5) {
        out.position.z = out.position.w * (1.0 - 1e-6);
    }

    return out;
}

// Skeletally animated sibling of vertex_main. The vertex carries four joint
// indices + blend weights; this shader blends up to four joint matrices from
// the per-object buffer at buffer(8) (linear blend skinning), applies the
// blended matrix to position/normal/tangent, then proceeds exactly like
// vertex_main. Paired with the existing fragment_main.
struct SkinnedVertex {
    float3  pos     [[attribute(0)]];
    float3  normal  [[attribute(1)]];
    float3  tangent [[attribute(2)]];
    float3  color   [[attribute(3)]];
    float2  uv      [[attribute(4)]];
    ushort4 joints  [[attribute(5)]];
    float4  weights [[attribute(6)]];
};

vertex VertexOut vertex_main_skinned(
    SkinnedVertex in                 [[stage_in]],
    constant ViewUniforms  &view     [[buffer(0)]],
    constant ModelUniforms &model_u  [[buffer(2)]],
    constant float4x4      *joints   [[buffer(8)]]
) {
    // Linear blend skinning: weighted sum of the bound joints' matrices.
    float4x4 skin = in.weights.x * joints[in.joints.x]
                  + in.weights.y * joints[in.joints.y]
                  + in.weights.z * joints[in.joints.z]
                  + in.weights.w * joints[in.joints.w];

    float4 skinned_pos     = skin * float4(in.pos, 1.0);
    float3 skinned_normal  = (skin * float4(in.normal,  0.0)).xyz;
    float3 skinned_tangent = (skin * float4(in.tangent, 0.0)).xyz;

    VertexOut out;
    float4 world_pos  = model_u.model * skinned_pos;
    out.position      = view.vp * world_pos;
    out.world_pos     = world_pos.xyz;
    out.normal        = normalize(float3(model_u.model * float4(skinned_normal,  0.0)));
    out.tangent       = normalize(float3(model_u.model * float4(skinned_tangent, 0.0)));
    out.color         = in.color;
    out.uv            = in.uv;
    out.view_depth    = -(view.view * world_pos).z;
    // Unused by the fragment_main pairing of the skinned pipeline.
    out.obj_id        = 0;
    return out;
}

// GPU-instanced sibling of vertex_main. Reads the per-instance world matrix
// from a structured buffer at buffer(6) indexed by [[instance_id]] instead
// of the per-draw ModelUniforms uniform. Paired with the existing fragment_main.
vertex VertexOut vertex_main_instanced(
    Vertex in                            [[stage_in]],
    constant ViewUniforms  &view         [[buffer(0)]],
    constant float4x4      *instances    [[buffer(6)]],
    uint                    iid          [[instance_id]]
) {
    VertexOut out;
    float4x4 model = instances[iid];
    float4 world_pos = model * float4(in.pos, 1.0);
    out.position     = view.vp * world_pos;
    out.world_pos    = world_pos.xyz;
    out.normal       = normalize(float3(model * float4(in.normal,  0.0)));
    out.tangent      = normalize(float3(model * float4(in.tangent, 0.0)));
    out.color        = in.color;
    out.uv           = in.uv;
    out.view_depth   = -(view.view * world_pos).z;
    // Unused by the fragment_main pairing of the instanced pipeline.
    out.obj_id       = 0;
    return out;
}

// Hash a 2D pixel coord to a rotation angle in [0, 2*pi).
static float hash_rotation(float2 p) {
    float h = fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    return h * 6.2831853;
}

// 5x5 PCF of a single cascade. Returns [0, 1] shadow factor (1.0 fully lit),
// or 1.0 when the fragment lies outside this cascade's light frustum.
static float sample_cascade_pcf(
    uint                  cascade,
    float3                world_pos,
    constant ShadowUniforms &shadow,
    depth2d_array<float>  shadow_map,
    sampler               shadow_samp,
    float2                screen_xy
) {
    float4 light_clip = shadow.light_vps[cascade] * float4(world_pos, 1.0);
    float3 ndc = light_clip.xyz / light_clip.w;
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if (any(uv < 0.0f) || any(uv > 1.0f) || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    // World-constant depth bias. The comparison needs a fixed world-space
    // offset along the light direction; as an NDC offset that is the world bias
    // divided by the cascade's depth range. The ortho z scale puts that depth
    // range at 1 / |VP row2 xyz|, so a world bias B becomes B * |row2| in NDC.
    // Deriving it from row2 (depth) rather than row0 (XY) keeps the bias
    // constant across cascades even though csm.rs extends each cascade's near
    // plane to capture tall casters, decoupling the depth range from the XY
    // radius.
    float3 vp_row2 = float3(shadow.light_vps[cascade][0][2],
                            shadow.light_vps[cascade][1][2],
                            shadow.light_vps[cascade][2][2]);
    float bias = 0.03 * length(vp_row2);
    float ref = ndc.z - bias;

    // Per-pixel rotation makes the 5x5 PCF kernel act like a soft random
    // distribution instead of a banded grid.
    float angle = hash_rotation(screen_xy);
    float ca = cos(angle);
    float sa = sin(angle);

    // Texel size in shadow UV space.
    float2 tex_size = float2(1.0) / float2(shadow_map.get_width(), shadow_map.get_height());

    float sum = 0.0;
    constexpr int RADIUS = 2; // 5x5
    constexpr float SAMPLES = float((2 * RADIUS + 1) * (2 * RADIUS + 1));
    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {
            float2 off = float2(dx, dy);
            float2 rot = float2(off.x * ca - off.y * sa, off.x * sa + off.y * ca);
            float2 sample_uv = uv + rot * tex_size;
            sum += shadow_map.sample_compare(shadow_samp, sample_uv, cascade, ref);
        }
    }
    return sum / SAMPLES;
}

// Cascade-aware PCF with cross-cascade blending. Returns [0, 1] shadow factor:
// 1.0 fully lit, 0.0 fully shadowed.
static float shadow_factor_cascaded(
    float3                world_pos,
    float                 view_depth,
    constant ShadowUniforms &shadow,
    depth2d_array<float>  shadow_map,
    sampler               shadow_samp,
    float2                screen_xy
) {
    // Cascade selection: pick the smallest index whose far split exceeds
    // this fragment's view depth. Fragments beyond the last cascade fall
    // through to "fully lit" (no shadow contribution at long range).
    uint cascade = NUM_SHADOW_CASCADES;
    if (view_depth < shadow.cascade_splits[0])      cascade = 0;
    else if (view_depth < shadow.cascade_splits[1]) cascade = 1;
    else if (view_depth < shadow.cascade_splits[2]) cascade = 2;
    else if (view_depth < shadow.cascade_splits[3]) cascade = 3;
    if (cascade >= shadow.active_cascades) return 1.0;

    float shade = sample_cascade_pcf(cascade, world_pos, shadow, shadow_map, shadow_samp, screen_xy);

    // Blend into the next cascade across a band at the far edge of this
    // cascade's depth range. Each cascade places the shadow edge slightly
    // differently (its own texel grid + depth bias), and the split boundary
    // sits a fixed distance ahead of the camera, so under a hard switch that
    // boundary sweeps across the world as the camera moves and the shadow edge
    // appears to glide. Blending the shadow factor over the band turns the
    // jump into a smooth transition that stays anchored to the world.
    if (cascade + 1 < shadow.active_cascades) {
        float split_far  = shadow.cascade_splits[cascade];
        float split_near = (cascade == 0) ? 0.0 : shadow.cascade_splits[cascade - 1];
        float band = (split_far - split_near) * 0.15;
        float t = (view_depth - (split_far - band)) / max(band, 1e-4);
        if (t > 0.0) {
            float next = sample_cascade_pcf(
                cascade + 1, world_pos, shadow, shadow_map, shadow_samp, screen_xy
            );
            shade = mix(shade, next, clamp(t, 0.0, 1.0));
        }
    }
    return shade;
}

constant float PI = 3.14159265359;

// Surfaces rougher than this get no screen-space / ray-traced reflection (the
// resolve and reflection_composite gate at the same value). Below it the
// forward probe specular yields to the resolve to avoid double-counting; at or
// above it the resolve is silent, so the forward probe specular is kept.
// This shader is compiled offline and baked, so unlike the runtime-compiled
// resolve shaders it keeps its own declaration; the value is locked to
// `concinnity_core::gfx::ssr::REFLECTION_ROUGHNESS_CUT` by a unit test
// (`default_metal_reflection_cut_matches_canonical`).
constant float REFL_RESOLVE_CUT = 0.6;

// Trowbridge-Reitz GGX normal distribution.
static float distribution_ggx(float3 N, float3 H, float rough) {
    float a  = rough * rough;
    float a2 = a * a;
    float NdH  = max(dot(N, H), 0.0);
    float NdH2 = NdH * NdH;
    float denom = NdH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

// Schlick-GGX geometry term, Smith remapping for direct lights (k = (rough+1)^2/8).
static float geometry_schlick_ggx(float NdV, float rough) {
    float r = rough + 1.0;
    float k = (r * r) / 8.0;
    return NdV / (NdV * (1.0 - k) + k);
}

// Smith joint masking-shadowing using the Schlick-GGX approximation.
static float geometry_smith(float3 N, float3 V, float3 L, float rough) {
    float NdV = max(dot(N, V), 0.0);
    float NdL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdV, rough) * geometry_schlick_ggx(NdL, rough);
}

// Schlick approximation of the Fresnel term.
static float3 fresnel_schlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Karis 2014 analytic fit of the GGX directional-albedo BRDF LUT. Returns the
// (scale, bias) pair such that single-scatter spec albedo for a given F0 is
// approximately F0 * scale + bias. Used both for direct-light energy
// compensation (here) and, later, for IBL specular when env maps land.
static float2 env_brdf_approx(float NdV, float rough) {
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4( 1.0,  0.0425,  1.040, -0.040);
    float4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdV)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

// Macro variation: a large-scale, world-space brightness modulation that
// hides the obvious repetition of a tiled texture on a big surface (terrain,
// floors). It is two octaves of hash-based value noise sampled in the XZ
// ground plane at wavelengths far longer than the texture's tile, so the eye
// stops locking onto the repeating grid. Strength is `Material.macro_variation`;
// 0 is a no-op so every existing material is unaffected.
static float macro_hash(float2 p) {
    p = fract(p * float2(127.1, 311.7));
    p += dot(p, p + 41.17);
    return fract(p.x * p.y);
}

static float macro_value_noise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);   // smoothstep weights
    float a = macro_hash(i);
    float b = macro_hash(i + float2(1.0, 0.0));
    float c = macro_hash(i + float2(0.0, 1.0));
    float d = macro_hash(i + float2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Albedo multiplier in roughly [1 - strength, 1 + strength]. The noise is
// centred on 0 before scaling so the surface's mean brightness is unchanged
// and only the patch-to-patch spread grows with `strength`.
static float3 macro_variation_factor(float3 world_pos, float strength) {
    if (strength <= 0.0) {
        return float3(1.0);
    }
    float2 p = world_pos.xz;
    // Base octave varies over ~14 world units, detail octave over ~5.
    float n = macro_value_noise(p * 0.07) * 0.65
            + macro_value_noise(p * 0.19) * 0.35;
    return float3(1.0 + (n - 0.5) * 2.0 * strength);
}

// Triplanar projection helper for the terrain shading path. Samples the
// bound texture from each of the three world axes (XZ for top, XY for
// front, YZ for side) and blends by the absolute world-space normal.
// Kills the UV-stretch banding heightfield ground gets when stretched
// across a big mesh, every steep face would otherwise read the same
// row of texels repeated down the slope.
//
// Three layers of variation defeat the obvious tile repetition: each
// projection is sampled at two octaves (coarse + 2.7x fine, summed and
// re-normalised so the mean brightness stays put) and each octave's UV
// is jittered by a hash of the world-space integer cell it lands in.
// The result is that adjacent "tiles" share neither registration nor
// rotation, so the eye stops locking onto the grid.
//
// `tile_scale` is the world-units-per-tile factor (smaller = more
// repetition per metre, larger = less). 0.10 means one tile every 10 m
// in world space, which keeps repeats out of natural-viewing-distance
// frames for hill-scale ground.
static float tile_break_hash(float2 p) {
    p = fract(p * float2(127.1, 311.7));
    p += dot(p, p + 41.17);
    return fract(p.x * p.y);
}

static float4 triplanar_octave(
    texture2d<float> tex,
    sampler          tex_sampler,
    float2           uv
) {
    // Stochastic offset per integer tile keeps adjacent tiles from
    // sharing the same texel pattern at the seam.
    float2 cell = floor(uv);
    float2 jitter = float2(
        tile_break_hash(cell + float2(0.0, 0.0)) - 0.5,
        tile_break_hash(cell + float2(13.0, 7.0)) - 0.5
    );
    return tex.sample(tex_sampler, uv + jitter * 0.3);
}

static float4 triplanar_sample(
    texture2d<float> tex,
    sampler          tex_sampler,
    float3           world_pos,
    float3           world_normal,
    float            tile_scale
) {
    float3 weights = abs(world_normal);
    // Sharpen the blend so transitions between projections are tight;
    // power-of-3 lifts the dominant axis and suppresses the other two,
    // avoiding the "averaged" look that a linear blend produces on
    // 45° faces.
    weights = pow(weights, float3(3.0));
    weights /= max(weights.x + weights.y + weights.z, 1e-4);

    // Two octaves: coarse + fine at an irrational ratio so the second
    // octave's tile boundaries never align with the first's. Each
    // octave runs through the stochastic-jitter sampler above.
    const float octave2 = 2.7183;
    float4 sx = (triplanar_octave(tex, tex_sampler, world_pos.zy * tile_scale)
              + triplanar_octave(tex, tex_sampler, world_pos.zy * (tile_scale * octave2))) * 0.5;
    float4 sy = (triplanar_octave(tex, tex_sampler, world_pos.xz * tile_scale)
              + triplanar_octave(tex, tex_sampler, world_pos.xz * (tile_scale * octave2))) * 0.5;
    float4 sz = (triplanar_octave(tex, tex_sampler, world_pos.xy * tile_scale)
              + triplanar_octave(tex, tex_sampler, world_pos.xy * (tile_scale * octave2))) * 0.5;
    return sx * weights.x + sy * weights.y + sz * weights.z;
}

// Box-parallax sample of one reflection-probe cube. Intersects the reflection ray
// R with the probe's influence box and re-anchors the sample direction at that hit
// relative to the capture point, so a static cube tracks a moving camera; falls
// back to the raw ray when the probe has no baked box (`box_min.w <= 0.5`) or the
// box does not lie ahead of the ray (so a blended secondary box that the surface
// has already left can't sample backward).
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
        // dist > 0 always holds for a point inside the box. A blended secondary
        // probe is sampled from points just outside its box, where the box can lie
        // behind the ray (dist <= 0); re-anchoring then would point the sample
        // backward, so keep the raw ray in that case.
        if (dist > 0.0) {
            float3 hit = world_pos + R * dist;
            sample_dir = hit - probe.probe_pos.xyz;
        }
    }
    // bias() (not level()) so the reflection vector's screen-space footprint widens
    // the mip at grazing or distant angles; a fixed level aliases into sparkle on
    // near mirrors, while flat close-up pixels keep the plain roughness mip.
    return probe_cube.sample(cube_sampler, sample_dir, bias(lod)).rgb;
}

// Reflection-probe radiance for `world_pos` along world-space ray `R`, blended
// across every probe that covers the point (partition of unity). Each probe gets
// `w = smoothstep(-margin, margin, sd)` from the signed box distance (1 deep inside,
// 0.5 on the box surface, 0 a margin outside); the result is the weight-normalised
// sum of each probe's box-projected sample, so a surface inside N overlapping boxes
// cross-fades smoothly across all N (no pop at a 3-way overlap line) and reduces to
// a single sample where only one box covers. Where no box covers (all weights 0),
// falls back to the nearest probe by capture distance.
static float3 probe_set_specular(
    constant ProbeSet                    &set,
    array<texturecube<float>, MAX_PROBES> probe_cubes,
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
        // Signed distance to the box surface: positive inside, negative outside.
        float3 q = abs(world_pos - c) - he;
        float sd = -(length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0));
        // Floor the margin so a zero-extent box axis can't collapse the smoothstep.
        float margin = max(PROBE_BLEND_MARGIN * min(he.x, min(he.y, he.z)), 1e-4);
        float w = smoothstep(-margin, margin, sd);
        if (w > 0.0) {
            acc += w * sample_probe_radiance(
                           probe_cubes[i], set.probes[i], world_pos, R, lod, cube_sampler);
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
    // No box covers the surface within its margin: use the nearest probe.
    return sample_probe_radiance(
        probe_cubes[near_i], set.probes[near_i], world_pos, R, lod, cube_sampler);
}

// Shared Cook-Torrance GGX shading body. Both fragment entry points resolve
// their inputs differently (`fragment_main` from per-draw bindings,
// `fragment_main_bindless` from the GpuObjectData buffer + texture pool), then
// call this. Keeping the lighting in one place means the two binding models
// never drift apart.
static float4 shade_surface(
    VertexOut                in,
    constant ViewUniforms   &view,
    constant ProbeSet       &probes,
    float                    roughness,
    float                    metallic,
    float                    macro_variation,
    float                    terrain_blend,
    float                    secondary_blend_sharpness,
    float3                   tint,
    float3                   emissive,
    constant LightUniforms  &lights,
    constant ShadowUniforms &shadow,
    texture2d<float>         tex,
    texture2d<float>         normal_tex,
    texture2d<float>         tex_secondary,
    texture2d<float>         normal_tex_secondary,
    texture2d<float>         emissive_map,
    texture2d<float>         orm_map,
    bool                     has_emissive_map,
    bool                     has_orm_map,
    depth2d_array<float>     shadow_map,
    texturecube<float>       irradiance_cube,
    texturecube<float>       prefilter_cube,
    array<texturecube<float>, MAX_PROBES> probe_cubes,
    texture2d<float>         ssao_tex,
    sampler                  tex_sampler,
    sampler                  shadow_sampler,
    sampler                  cube_sampler
) {
    bool ibl_enabled = view.prefilter_mip_count > 0.5;
    if (in.color.b > 1.5) {
        float3 view_dir = normalize(in.world_pos - view.cam_pos);
        if (ibl_enabled) {
            // Mip 0 of the prefilter cube is the unfiltered source, use it
            // for the skybox so the sky matches the environment driving IBL.
            return float4(prefilter_cube.sample(cube_sampler, view_dir, level(0.0)).rgb, 1.0);
        }
        float t = max(0.0, view_dir.y);
        return float4(mix(SKY_HORIZON, SKY_ZENITH, t), 1.0);
    }

    // Surface sampling. The default path is a single UV lookup with a
    // tangent-space normal map; when `terrain_blend > 0` the path
    // switches to a triplanar world-space projection (kills the
    // UV-stretch banding stretched heightfields show) and lerps the
    // albedo toward a darker rocky tint on steep slopes (1 - |N.y|
    // gives "0 = flat ground, 1 = vertical cliff face"). Both samples
    // are computed eagerly so the GPU pipeline stays branch-free; the
    // mix selects which result lands in the lit albedo.
    float3 N0 = normalize(in.normal);
    float3 N;
    float4 texel;
    float3 albedo;
    if (terrain_blend > 0.0) {
        // Triplanar at 0.25 tiles/m (one repeat every 4 m world) for the
        // PolyHaven-style PBR terrain set; that's roughly the natural
        // grain scale of a forest-floor / rock-surface texture, and the
        // per-octave hash jitter inside `triplanar_sample` keeps the
        // 4 m tile boundary from reading as a grid. The secondary
        // pair samples at a slightly larger tile so the two layers
        // don't beat against each other at the seam.
        const float tile_scale = 0.25;
        const float tile_scale_secondary = 0.18;

        // Two-layer terrain: primary (flat / leaves) and secondary
        // (slope / rock). Sample both layers triplanar; blend by a
        // slope mask modulated by a per-pixel world-noise tap so the
        // transition reads as natural patches instead of a clean
        // contour line. The blend weight saturates `secondary_blend_sharpness`
        // around 0.5; a `pow` curve sharpens the transition without
        // making it a pure step function.
        float4 primary = triplanar_sample(tex, tex_sampler,
                                          in.world_pos, N0, tile_scale);
        float4 secondary = triplanar_sample(tex_secondary, tex_sampler,
                                            in.world_pos, N0, tile_scale_secondary);

        // Slope mask: 0 on perfectly flat ground, 1 on a vertical
        // face. Modulated by a low-frequency noise so flat tops still
        // get rock patches and cliff faces still get the occasional
        // soil patch, kills the obvious "horizontal-line transition"
        // a pure slope mask gives.
        float slope = saturate(1.0 - abs(N0.y));
        float noise_break = macro_value_noise(in.world_pos.xz * 0.07);
        float blend_in = slope + (noise_break - 0.5) * 0.35;
        // Sharpness control: 0 = wide gradient (smoothstep over a
        // 0.5-wide band); 1 = near-hard cliff edge (band shrinks to
        // ~0.05).
        float band = mix(0.5, 0.05, secondary_blend_sharpness);
        float mid = 0.45;
        float blend = saturate((blend_in - (mid - band * 0.5)) / max(band, 1e-3));
        // Power curve sharpens the centre of the transition so the
        // mid-section reads as a clean rock/leaf boundary rather than
        // a 50/50 wash.
        blend = blend * blend * (3.0 - 2.0 * blend);
        blend *= terrain_blend;

        texel = mix(primary, secondary, blend);
        albedo = texel.rgb * tint;

        // Triplanar tangent-space normal for each layer. Same two-
        // octave jitter as the albedo so per-tile registration
        // doesn't repeat any more obviously than the colour does;
        // each axis's tangent normal is swizzled into world space via
        // the "RNM" trick before the weighted blend, then the two
        // layers' world normals are blended by the same slope mask.
        const float octave2 = 2.7183;
        float3 weights = abs(N0);
        weights = pow(weights, float3(3.0));
        weights /= max(weights.x + weights.y + weights.z, 1e-4);

        // --- primary normal ---
        float3 p_nx_t = ((triplanar_octave(normal_tex, tex_sampler, in.world_pos.zy * tile_scale)
                       + triplanar_octave(normal_tex, tex_sampler, in.world_pos.zy * (tile_scale * octave2))) * 0.5).xyz * 2.0 - 1.0;
        float3 p_ny_t = ((triplanar_octave(normal_tex, tex_sampler, in.world_pos.xz * tile_scale)
                       + triplanar_octave(normal_tex, tex_sampler, in.world_pos.xz * (tile_scale * octave2))) * 0.5).xyz * 2.0 - 1.0;
        float3 p_nz_t = ((triplanar_octave(normal_tex, tex_sampler, in.world_pos.xy * tile_scale)
                       + triplanar_octave(normal_tex, tex_sampler, in.world_pos.xy * (tile_scale * octave2))) * 0.5).xyz * 2.0 - 1.0;
        float3 p_wnx = float3(0.0, p_nx_t.y, p_nx_t.x) + float3(N0.x, 0.0, 0.0);
        float3 p_wny = float3(p_ny_t.x, 0.0, p_ny_t.y) + float3(0.0, N0.y, 0.0);
        float3 p_wnz = float3(p_nz_t.x, p_nz_t.y, 0.0) + float3(0.0, 0.0, N0.z);
        float3 N_primary = normalize(p_wnx * weights.x + p_wny * weights.y + p_wnz * weights.z);

        // --- secondary normal ---
        float3 s_nx_t = ((triplanar_octave(normal_tex_secondary, tex_sampler, in.world_pos.zy * tile_scale_secondary)
                       + triplanar_octave(normal_tex_secondary, tex_sampler, in.world_pos.zy * (tile_scale_secondary * octave2))) * 0.5).xyz * 2.0 - 1.0;
        float3 s_ny_t = ((triplanar_octave(normal_tex_secondary, tex_sampler, in.world_pos.xz * tile_scale_secondary)
                       + triplanar_octave(normal_tex_secondary, tex_sampler, in.world_pos.xz * (tile_scale_secondary * octave2))) * 0.5).xyz * 2.0 - 1.0;
        float3 s_nz_t = ((triplanar_octave(normal_tex_secondary, tex_sampler, in.world_pos.xy * tile_scale_secondary)
                       + triplanar_octave(normal_tex_secondary, tex_sampler, in.world_pos.xy * (tile_scale_secondary * octave2))) * 0.5).xyz * 2.0 - 1.0;
        float3 s_wnx = float3(0.0, s_nx_t.y, s_nx_t.x) + float3(N0.x, 0.0, 0.0);
        float3 s_wny = float3(s_ny_t.x, 0.0, s_ny_t.y) + float3(0.0, N0.y, 0.0);
        float3 s_wnz = float3(s_nz_t.x, s_nz_t.y, 0.0) + float3(0.0, 0.0, N0.z);
        float3 N_secondary = normalize(s_wnx * weights.x + s_wny * weights.y + s_wnz * weights.z);

        N = normalize(mix(N_primary, N_secondary, blend));
    } else {
        texel  = tex.sample(tex_sampler, in.uv);
        albedo = texel.rgb * in.color * tint;
        // Build TBN matrix and apply tangent-space normal map.
        float3 T  = normalize(in.tangent - dot(in.tangent, N0) * N0);
        float3 B  = cross(N0, T);
        float3 ns = normal_tex.sample(tex_sampler, in.uv).xyz * 2.0 - 1.0;
        N  = normalize(float3x3(T, B, N0) * ns);
    }
    // Break up visible tiling on large surfaces (no-op when strength is 0).
    albedo *= macro_variation_factor(in.world_pos, macro_variation);

    // Packed roughness/metalness map overrides the scalar PBR inputs per-texel
    // when bound (G = roughness, B = metalness). The R channel is reserved and
    // NOT read as occlusion: real packed maps (glTF metallic-roughness, the FBX
    // specular maps the importer routes here) leave R empty (0), and reading it
    // as occlusion would multiply the indirect term to black. Ambient occlusion
    // comes from SSAO instead. The `has_orm_map` gate keeps the scalar fallback
    // for materials without one.
    if (has_orm_map) {
        float3 orm = orm_map.sample(tex_sampler, in.uv).rgb;
        roughness = orm.g;
        metallic  = orm.b;
    }
    // Textured emission scales the scalar emissive factor when bound.
    if (has_emissive_map) {
        emissive *= emissive_map.sample(tex_sampler, in.uv).rgb;
    }

    float3 V   = normalize(view.cam_pos - in.world_pos);
    float  NdV = max(dot(N, V), 0.0);

    // PBR base reflectance: dielectrics use a constant 0.04, metals use albedo.
    float3 F0 = mix(float3(0.04), albedo, metallic);

    // Cascaded shadow factor for the first directional light.
    float shad = shadow_factor_cascaded(
        in.world_pos, in.view_depth, shadow, shadow_map, shadow_sampler, in.position.xy
    );

    // Energy-conserving multi-scatter compensation (Fdez-Aguera / Filament).
    // Karis BRDF approximation gives the single-scatter directional albedo Eo;
    // 1 + F0 * (1/Eo - 1) restores the energy that GGX masking-shadowing drops.
    // View-only; reused across every direct light.
    float2 ab        = env_brdf_approx(NdV, roughness);
    float  ess       = ab.x + ab.y;
    float3 energy_ms = 1.0 + F0 * (1.0 / max(ess, 0.001) - 1.0);

    float3 Lo = float3(0.0);

    for (int i = 0; i < lights.num_directional; i++) {
        float3 L        = normalize(lights.directional[i].direction);
        float  intens   = lights.directional[i].intensity;
        float3 radiance = lights.directional[i].color * intens;

        float3 H   = normalize(V + L);
        float  NdL = max(dot(N, L), 0.0);
        float  D   = distribution_ggx(N, H, roughness);
        float  G   = geometry_smith(N, V, L, roughness);
        float3 F   = fresnel_schlick(max(dot(H, V), 0.0), F0);
        float3 kd  = (1.0 - F) * (1.0 - metallic);
        float3 spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001) * energy_ms;
        float3 diff = kd * albedo / PI;

        // Only the first directional light is shadowed.
        float s = (i == 0) ? shad : 1.0;
        Lo += (diff + spec) * radiance * NdL * s;
    }

    for (int j = 0; j < lights.num_point; j++) {
        float3 pos_w  = lights.point[j].position;
        float  range  = lights.point[j].range;
        float3 col    = lights.point[j].color;
        float  intens = lights.point[j].intensity;

        float3 delta = pos_w - in.world_pos;
        float  dist  = length(delta);
        float3 L     = normalize(delta);
        float  atten = clamp(1.0 - dist / range, 0.0, 1.0);
        atten        = atten * atten;
        float3 radiance = col * intens * atten;

        float3 H   = normalize(V + L);
        float  NdL = max(dot(N, L), 0.0);
        float  D   = distribution_ggx(N, H, roughness);
        float  G   = geometry_smith(N, V, L, roughness);
        float3 F   = fresnel_schlick(max(dot(H, V), 0.0), F0);
        float3 kd  = (1.0 - F) * (1.0 - metallic);
        float3 spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001) * energy_ms;
        float3 diff = kd * albedo / PI;
        Lo += (diff + spec) * radiance * NdL;
    }

    // Ambient term: IBL when an EnvironmentMap is bound, otherwise a soft
    // blue-tinted sky bounce so worlds without IBL aren't near-black.
    float3 ambient;
    if (ibl_enabled) {
        float3 F_ibl     = fresnel_schlick(NdV, F0);
        float3 kd_ibl    = (1.0 - F_ibl) * (1.0 - metallic);
        float3 irradiance = irradiance_cube.sample(cube_sampler, N).rgb;
        float3 diffuse_ibl = kd_ibl * albedo * irradiance / PI;

        float3 R = reflect(-V, N);
        float  lod = roughness * (view.prefilter_mip_count - 1.0);
        // Specular reflection samples the local reflection probes (the scene
        // captured into cubes) rather than the sky cube, so glossy surfaces reflect
        // real geometry. Unbaked slots alias the sky `prefilter_cube`, so this
        // matches the sky reflection until a probe is baked. The helper box-parallax
        // corrects each probe (the static cube tracks the moving camera) and blends
        // every box that covers the surface (partition of unity), so reflections
        // cross-fade smoothly where boxes overlap and reduce to one sample where a
        // single box covers.
        float3 prefiltered =
            probe_set_specular(probes, probe_cubes, in.world_pos, R, lod, cube_sampler);
        // Karis split-sum: F0 * scale + bias from env_brdf_approx (already in `ab`).
        float3 specular_ibl = prefiltered * (F0 * ab.x + ab.y);

        // When a reflection-resolve pass is compositing this frame, it owns the
        // sharp specular for glossy DIELECTRICS (its miss-fallback samples this
        // same probe set), so leaving the forward probe specular in place
        // double-counts the reflection: a parallax-approximate probe copy under
        // the exact resolved one. Fade the forward specular out below the
        // resolve's roughness cut (rough surfaces, which the resolve skips, keep
        // it). Gate on dielectrics: the resolve weights its reflection by a
        // dielectric Fresnel (F0 ~ 0.04), so a metal's albedo-tinted probe
        // reflection is NOT something the resolve can stand in for; metals keep
        // their full forward probe specular and the resolve only adds a faint
        // dielectric-strength term on top (no visible double).
        if (view.reflections_enabled > 0.5) {
            float fade = smoothstep(REFL_RESOLVE_CUT * 0.7, REFL_RESOLVE_CUT, roughness);
            specular_ibl *= mix(1.0, fade, 1.0 - metallic);
        }

        ambient = diffuse_ibl + specular_ibl;
    } else {
        ambient = float3(0.35, 0.4, 0.5) * 0.4 * albedo;
    }

    // Authored indirect-fill multiplier (PostProcessConfig.ambient_intensity).
    // Scales the whole ambient term; 1.0 is a no-op. Lit surfaces barely change
    // (the sun dominates), but shadowed areas relying on ambient lift with it.
    ambient *= lights.ambient_intensity;

    // Screen-space ambient occlusion modulates the indirect (ambient / IBL)
    // term only; direct lighting is unaffected. The renderer binds a 1x1
    // white texture when SSAO is disabled, so this samples a constant 1.0 and
    // leaves the ambient term untouched. in.position.xy is the pixel coord;
    // the SSAO target is full drawable resolution.
    constexpr sampler ssao_sampler(filter::linear, address::clamp_to_edge);
    float2 ssao_uv = in.position.xy / float2(ssao_tex.get_width(), ssao_tex.get_height());
    ambient *= ssao_tex.sample(ssao_sampler, ssao_uv).r;

    float3 color = ambient + Lo + emissive;

    // Linear-light HDR output. ACES tonemap + gamma encode + FXAA run in
    // the off-screen composite pass (see metal/pipeline.rs::build_post_pipeline).
    // The HLSL + GLSL fragment shaders likewise write linear HDR; every
    // backend now owns its tonemap in a composite pass.
    return float4(color, texel.a);
}

// Fragment entry point for the GPU-instanced and skinned pipelines. Material
// scalars come from the per-draw MaterialUniforms at buffer(3) and the albedo
// + normal textures from the per-draw texture(0)/texture(1) binds.
fragment float4 fragment_main(
    VertexOut in                       [[stage_in]],
    constant ViewUniforms    &view     [[buffer(0)]],
    constant MaterialUniforms &mat     [[buffer(3)]],
    constant LightUniforms   &lights   [[buffer(4)]],
    constant ShadowUniforms  &shadow   [[buffer(5)]],
    constant ProbeSet        &probes   [[buffer(6)]],
    texture2d<float>     tex            [[texture(0)]],
    texture2d<float>     normal_tex     [[texture(1)]],
    depth2d_array<float> shadow_map     [[texture(2)]],
    texturecube<float>   irradiance_cube[[texture(3)]],
    texturecube<float>   prefilter_cube [[texture(4)]],
    texture2d<float>     ssao_tex       [[texture(5)]],
    // Reflection-probe cube array at texture(6 .. 6+MAX_PROBES). The renderer binds
    // `probe_cube_or_sky(i)` into each slot (the sky prefilter for unbaked slots), so
    // the legacy path selects + blends per-surface from the same set as the bindless
    // path -- no longer limited to probe 0.
    array<texturecube<float>, MAX_PROBES> probe_cubes [[texture(6)]],
    sampler tex_sampler                 [[sampler(0)]],
    sampler shadow_sampler              [[sampler(1)]],
    sampler cube_sampler                [[sampler(2)]]
) {
    // Legacy / per-draw path: the renderer doesn't bind a separate
    // secondary texture pair for the terrain shader (no consumer
    // uses this code path with terrain_blend > 0 today). Re-bind the
    // primary texture as the secondary so the shader compiles + reads
    // valid data; the secondary blend weight is gated on `tex_secondary`
    // resolving to a real layer via the bindless path, which it
    // doesn't here.
    // The specular reflection blends every probe whose box covers the surface
    // (partition of unity), selected + sampled inside `shade_surface`.
    return shade_surface(
        in, view, probes,
        mat.roughness, mat.metallic, mat.macro_variation,
        mat.terrain_blend, mat.secondary_blend_sharpness,
        mat.tint, mat.emissive,
        lights, shadow, tex, normal_tex, tex, normal_tex,
        // No emissive / ORM maps on the legacy per-draw path: pass the albedo
        // as a dummy and gate both off so neither is sampled.
        tex, tex, false, false,
        shadow_map, irradiance_cube,
        prefilter_cube, probe_cubes, ssao_tex, tex_sampler, shadow_sampler,
        cube_sampler);
}

// Every texture the bindless static pass samples, packed into one argument
// buffer bound at buffer(7). Discrete [[texture(n)]] bindings make a fragment
// shader incompatible with indirect command buffers on Apple GPUs, and the
// GPU-driven cull pass issues this shader through an ICB, so its
// textures must arrive through an argument buffer instead. The renderer fills
// `tex_pool` with every albedo texture followed by every normal map (the same
// layout the GpuObjectData pool indices address).
struct BindlessTextures {
    array<texture2d<float>, BINDLESS_TEXTURE_COUNT> tex_pool [[id(0)]];
    depth2d_array<float> shadow_map [[id(BINDLESS_TEXTURE_COUNT)]];
    texturecube<float>   irradiance [[id(BINDLESS_TEXTURE_COUNT + 1)]];
    texturecube<float>   prefilter  [[id(BINDLESS_TEXTURE_COUNT + 2)]];
    // Blurred SSAO occlusion (1x1 white when SSAO is disabled).
    texture2d<float>     ssao       [[id(BINDLESS_TEXTURE_COUNT + 3)]];
    // Local reflection-probe prefiltered radiance: one scene-captured cube per
    // probe. Sampled for the specular reflection term only (the skybox + diffuse
    // keep the sky `prefilter`). Unused slices + the pre-bake state alias the sky
    // `prefilter`, so reflections are unchanged until a probe is baked.
    array<texturecube<float>, MAX_PROBES> probes [[id(BINDLESS_TEXTURE_COUNT + 4)]];
};

// Fragment entry point for the bindless static main pass. Material scalars and
// the albedo/normal texture-pool indices come from the GpuObjectData buffer at
// buffer(9), indexed by the [[base_instance]] object id forwarded through
// VertexOut.obj_id. The texture pool + shadow/IBL maps arrive in the
// `BindlessTextures` argument buffer at buffer(7); samplers are declared inline
// so the shader binds no discrete texture or sampler slots, which is what keeps
// it usable from an indirect command buffer.
fragment float4 fragment_main_bindless(
    VertexOut in                        [[stage_in]],
    constant ViewUniforms     &view     [[buffer(0)]],
    constant LightUniforms    &lights   [[buffer(4)]],
    constant ShadowUniforms   &shadow   [[buffer(5)]],
    constant ProbeSet         &probes   [[buffer(6)]],
    constant GpuObjectData    *objects  [[buffer(9)]],
    constant BindlessTextures &tex      [[buffer(7)]]
) {
    // Static engine samplers, declared inline. Parameters mirror the
    // MTLSamplerDescriptors built in metal/context.rs (albedo / shadow-compare
    // / cubemap), so the bindless pass samples identically to the legacy path.
    constexpr sampler tex_sampler(filter::linear, address::repeat);
    constexpr sampler shadow_sampler(filter::linear, address::clamp_to_edge,
                                     compare_func::less_equal);
    constexpr sampler cube_sampler(filter::linear, mip_filter::linear,
                                   address::clamp_to_edge);
    GpuObjectData obj = objects[in.obj_id];
    // The specular reflection blends every probe whose influence box covers this
    // surface (partition of unity), selected + sampled inside `shade_surface` via
    // `probe_set_specular`. The whole set + cube array are passed through; an unbaked
    // or absent probe (count == 0) leaves the sky fallback in those slots.
    return shade_surface(
        in, view, probes,
        obj.roughness, obj.metallic, obj.macro_variation,
        obj.terrain_blend, obj.secondary_blend_sharpness,
        obj.tint, obj.emissive,
        lights, shadow,
        tex.tex_pool[obj.albedo_index],
        tex.tex_pool[obj.normal_index],
        tex.tex_pool[obj.albedo_secondary_index],
        tex.tex_pool[obj.normal_secondary_index],
        tex.tex_pool[obj.emissive_map_index],
        tex.tex_pool[obj.orm_map_index],
        obj.emissive_map_index != 0,
        obj.orm_map_index != 0,
        tex.shadow_map, tex.irradiance, tex.prefilter,
        tex.probes, tex.ssao,
        tex_sampler, shadow_sampler, cube_sampler);
}
