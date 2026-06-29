#include <metal_stdlib>
using namespace metal;

// --- Volumetric fog: froxel-volume compute + fullscreen sampler ---
//
// Frostbite-style. Each frame the `fog_froxel_kernel` compute pass populates
// a screen-aligned 3D `RGBA16Float` volume of `(scattered_rgb, 1 - T)`
// across the view frustum; the fullscreen `fog_fragment` then samples the
// volume by `(screen_uv, view_z)` instead of marching per pixel.
//
// Why a volume: the scatter integral is the same for every pixel inside a
// froxel column, so we amortise the per-slice work (density + CSM shadow
// tap + Henyey-Greenstein phase) across many pixels. This also lets us add
// per-slice sun shadowing - the inline ray-march can't afford the 32
// shadow taps per pixel.
//
// Z distribution: linear from `z_near` to `z_far` (= fog.max_distance) for
// v1. Log-Z is a follow-up that puts more samples near the camera.
//
// `FogParams` and `FogFroxelParams` layouts must stay in sync with the
// matching Rust structs in `crate::gfx::render_types`.

constant int NUM_SHADOW_CASCADES = 4;

struct FogParams {
    float4x4 inv_vp;
    float4 color;
    packed_float3 cam_pos;
    float _pad0;
    packed_float3 sun_dir;
    float _pad1;
    packed_float3 sun_color;
    float _pad2;
    float density;
    float height_falloff;
    float height_reference;
    float max_distance;
    float phase_g;
    float ambient;
    float2 viewport;
    float inv_max_distance;
    float _pad3a;
    float _pad3b;
    float _pad3c;
};

// Layout note: MSL aligns `uint3` to a 16-byte slot (12 bytes of data + 4
// bytes of trailing pad), so the Rust `FogFroxelParams` mirrors that with
// an explicit `_pad_align` u32 between `froxel_dims` and `z_near`. The
// struct is exactly 96 bytes on both sides.
struct FogFroxelParams {
    float4x4 view;          // world -> view, for view-space depth.
    uint3 froxel_dims;      // (X, Y, Z) volume extents.
    float z_near;           // camera near-plane (view units).
    float z_far;            // fog max_distance - far edge of the volume.
    float _pad0;
    float _pad1;
};

struct ShadowUniforms {
    float4x4 light_vps[NUM_SHADOW_CASCADES];
    float4   cascade_splits;
    uint     active_cascades;
};

// Closed-form Henyey-Greenstein phase function. `cos_theta` is the cosine
// of the angle between the view ray and the direction toward the sun.
// Positive `g` gives forward scattering.
static inline float henyey_greenstein(float cos_theta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265358979 * pow(max(denom, 1e-5), 1.5));
}

// Cascade-aware single-sample shadow tap. No PCF - the froxel volume's
// trilinear sampling at the fragment-shader stage smooths the result.
// Returns `1.0` (fully lit) outside any cascade, matching the main
// shader's fall-through.
static float fog_shadow_factor(
    float3 world_pos,
    float view_depth,
    constant ShadowUniforms &shadow,
    depth2d_array<float> shadow_map,
    sampler shadow_samp
) {
    uint cascade = NUM_SHADOW_CASCADES;
    if (view_depth < shadow.cascade_splits[0])      cascade = 0;
    else if (view_depth < shadow.cascade_splits[1]) cascade = 1;
    else if (view_depth < shadow.cascade_splits[2]) cascade = 2;
    else if (view_depth < shadow.cascade_splits[3]) cascade = 3;
    if (cascade >= shadow.active_cascades) return 1.0;

    float4 light_clip = shadow.light_vps[cascade] * float4(world_pos, 1.0);
    float3 ndc = light_clip.xyz / light_clip.w;
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (any(uv < 0.0f) || any(uv > 1.0f) || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    float bias = 0.0015 * (1.0 + (float)cascade * 0.7);
    float ref = ndc.z - bias;
    return shadow_map.sample_compare(shadow_samp, uv, cascade, ref);
}

// Reconstruct the world-space position at a froxel center.
// `froxel_xyz + 0.5` are sampled at the slab centre for a second-order
// integration; we further jitter by an interleaved-gradient-noise sample
// (see the slab loop) to break the visible Z-banding.
static float3 froxel_to_world(
    uint x, uint y, float z_slice,
    constant FogParams &p,
    constant FogFroxelParams &f
) {
    // Screen-space NDC at the froxel centre.
    float2 uv = (float2(float(x) + 0.5, float(y) + 0.5))
              / float2(float(f.froxel_dims.x), float(f.froxel_dims.y));
    float2 ndc_xy = float2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));

    // Linear-Z distribution across [z_near, z_far].
    float view_z = mix(f.z_near, f.z_far,
                      (z_slice + 0.5) / float(f.froxel_dims.z));

    // Un-project a far-plane direction, then walk that ray to the desired
    // view-space z. Cheaper than inverting a custom per-froxel matrix and
    // works for any perspective projection.
    float4 clip_far = float4(ndc_xy, 1.0, 1.0);
    float4 world_far = p.inv_vp * clip_far;
    world_far /= world_far.w;
    float3 ray = normalize(world_far.xyz - float3(p.cam_pos));

    // The forward distance to the requested view-z. Mirrors the main
    // pass: positive view depth is -z, so the projection of `ray` onto
    // the view-forward axis equals -view.row[2].xyz · ray.
    float3 view_fwd = -float3(f.view[0][2], f.view[1][2], f.view[2][2]);
    float forward = max(dot(ray, view_fwd), 1e-4);
    float t = view_z / forward;

    return float3(p.cam_pos) + ray * t;
}

kernel void fog_froxel_kernel(
    uint2 tid                              [[thread_position_in_grid]],
    constant FogParams       &p            [[buffer(0)]],
    constant FogFroxelParams &f            [[buffer(1)]],
    constant ShadowUniforms  &shadow       [[buffer(2)]],
    depth2d_array<float>      shadow_map   [[texture(0)]],
    texture3d<half, access::write> volume  [[texture(1)]]
) {
    if (tid.x >= f.froxel_dims.x || tid.y >= f.froxel_dims.y) {
        return;
    }

    // Interleaved gradient noise: a per-(x, y) tile offset so neighbouring
    // froxel columns sample density / shadows at slightly different points
    // along Z. Stochastic; trilinear filtering at sample time + TAA later
    // smear the noise into smooth illumination.
    float2 tile_xy = float2(float(tid.x), float(tid.y));
    float ign = fract(52.9829189 *
        fract(dot(tile_xy, float2(0.06711056, 0.00583715))));

    // The view ray direction from the camera to the *(x, y, 0)* froxel.
    // Used for the phase function - within a column the direction is
    // approximately constant across Z (small-FOV approximation), so we
    // sample it once.
    float3 col_world = froxel_to_world(tid.x, tid.y, 0.0, p, f);
    float3 ray_dir = normalize(col_world - float3(p.cam_pos));
    float cos_theta = dot(ray_dir, normalize(float3(p.sun_dir)));
    float phase = henyey_greenstein(cos_theta, p.phase_g);

    // In-scatter contributions per slab. Ambient is isotropic (no phase
    // modulation) so the medium still reads in shaded regions.
    float3 sun_inscatter_unshadowed = float3(p.sun_color) * phase * float3(p.color.rgb);
    float3 ambient_inscatter = float3(p.color.rgb) * p.ambient;

    // Per-slab integration. `transmittance` is the running camera→slab
    // transmittance; `accumulated` is the in-scatter integrated so far.
    // Each slice writes `(accumulated, 1 - transmittance)` so the
    // fragment shader's `over` blend gives `scene*T + scattered` directly.
    float total_z = f.z_far - f.z_near;
    float step_len = total_z / float(f.froxel_dims.z);

    float3 accumulated = float3(0.0);
    float transmittance = 1.0;

    for (uint z = 0; z < f.froxel_dims.z; ++z) {
        // Jittered slab centre (slab integral stays exact in the constant-
        // density limit; only the sample point shifts).
        float z_jittered = float(z) + ign - 0.5;
        float3 pos = froxel_to_world(tid.x, tid.y, z_jittered, p, f);

        // Exponential height falloff (matches the ray-march path).
        float h = pos.y - p.height_reference;
        float local_density = p.density * exp(-max(h, -50.0) * p.height_falloff);

        // CSM shadow tap: pick a cascade by view-space depth at the slab.
        float slab_view_z = mix(f.z_near, f.z_far,
                               (z_jittered + 0.5) / float(f.froxel_dims.z));
        constexpr sampler shadow_samp(
            coord::normalized, address::clamp_to_edge, filter::linear,
            compare_func::less_equal);
        float shad = fog_shadow_factor(pos, slab_view_z, shadow,
                                       shadow_map, shadow_samp);

        // Per-slab Beer-Lambert + analytic energy-conserving in-scatter.
        float tau = local_density * step_len;
        float slab_T = exp(-tau);
        float slab_alpha = 1.0 - slab_T;
        float3 inscatter = sun_inscatter_unshadowed * shad + ambient_inscatter;
        accumulated += transmittance * slab_alpha * inscatter;
        transmittance *= slab_T;

        // Write the running pair into this slice. Convert to half for the
        // RGBA16Float storage. Each slice carries the camera→slice
        // integral so the fragment-shader sample at a scene depth gives
        // the right value without an extra accumulation pass.
        half4 out = half4(half3(accumulated), half(1.0 - transmittance));
        volume.write(out, uint3(tid.x, tid.y, z));

        // Early-out once the medium is almost opaque. The remaining
        // slices keep the saturated value; the fragment sampler will
        // pick any slice past this point with the same output.
        if (transmittance < 0.005) {
            transmittance = 0.0;
            for (uint zz = z + 1; zz < f.froxel_dims.z; ++zz) {
                volume.write(out, uint3(tid.x, tid.y, zz));
            }
            break;
        }
    }
}

// --- Fullscreen sampler ---

struct FogVtxOut {
    float4 position [[position]];
};

vertex FogVtxOut fog_vertex(uint vid [[vertex_id]]) {
    float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    FogVtxOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    return out;
}

fragment float4 fog_fragment(
    FogVtxOut                in           [[stage_in]],
    constant FogParams      &p            [[buffer(0)]],
    constant FogFroxelParams &f           [[buffer(1)]],
    depth2d<float>           scene_depth  [[texture(0)]],
    texture3d<half, access::sample> volume [[texture(1)]]
) {
    uint2 pixel = uint2(in.position.xy);
    if (pixel.x >= uint(p.viewport.x) || pixel.y >= uint(p.viewport.y)) {
        discard_fragment();
    }

    // Reconstruct view-space depth at the pixel. depth == 1.0 (skybox /
    // never-written) maps to the far edge of the volume so the sky takes
    // fog integrated across the whole volume.
    float depth = scene_depth.read(pixel);
    float2 uv = in.position.xy / p.viewport;
    float2 ndc_xy = float2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));
    float view_z;
    if (depth < 1.0) {
        float4 clip = float4(ndc_xy, depth, 1.0);
        float4 world = p.inv_vp * clip;
        world /= world.w;
        view_z = -(f.view * float4(world.xyz, 1.0)).z;
    } else {
        view_z = f.z_far;
    }

    // Map view_z onto a normalised volume W. Clamp into [0, 1] so the
    // skybox + anything past the volume's far edge sample the fully-
    // integrated last slice.
    float z01 = clamp((view_z - f.z_near) / max(f.z_far - f.z_near, 1e-4),
                      0.0, 1.0);

    // Trilinear sample. The volume already stores camera→slice integrated
    // (scattered, 1-T), so the sample IS the output blend pair.
    constexpr sampler vol_samp(
        coord::normalized, address::clamp_to_edge, filter::linear);
    half4 s = volume.sample(vol_samp, float3(uv, z01));
    return float4(float3(s.rgb), float(s.a));
}
