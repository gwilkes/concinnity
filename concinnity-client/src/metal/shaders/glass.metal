#include <metal_stdlib>
using namespace metal;

// --- Glass panel pass ---
//
// The simplest consumer of the engine's transparent pass: a flat, fixed
// rectangular pane. Runs in the same `PassId::Transparent` slot as water,
// after SSR resolve and before TAA. The build-time quad is already in world
// space (see geometry::glass_quad), so the vertex shader only projects it.
//
// The fragment shader:
//   - Discards where nearer opaque geometry occludes the pane (manual depth
//     test against the resolved main depth - the transparent pass binds no
//     depth attachment).
//   - Refracts: samples the pre-transparent scene snapshot at a screen offset
//     perturbed by the (view-facing) pane normal, then tints it.
//   - Adds a Schlick-Fresnel rim that brightens and increases opacity at
//     grazing angles.
//
// Output is straight-alpha blended (SRC_ALPHA / ONE_MINUS_SRC_ALPHA) by the
// pipeline. Shares `TransparentView` (buffer 5) with every other transparent
// draw; `GlassParams` (buffer 6) is per-pane.

struct TransparentView {
    float4x4 vp;          // world -> clip (jittered when TAA is on)
    float4x4 inv_vp;      // clip -> world
    float4   camera_pos;  // world-space camera, .w unused
    float2   viewport;    // attachment dimensions in pixels
    float    time;        // seconds since startup
    float    _pad;
};

struct GlassParams {
    // Vec3 fields stored as float4 (.w unused) so the Rust [f32; 4] layout is
    // byte-identical regardless of MSL float3 packing.
    float4 centre;  // world-space pane centre
    float4 normal;  // unit pane normal (facing direction)
    float4 tint;    // colour multiplied into the refracted scene
    float  opacity;
    float  refraction_strength;
    float  fresnel_power;
    float  _pad;
};

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
    // Quad vertices are pre-transformed into world space at build time.
    out.world_pos = in.pos;
    out.position = v.vp * float4(in.pos, 1.0);
    return out;
}

fragment float4 glass_fragment(
    GlassVtxOut                       in            [[stage_in]],
    constant TransparentView         &v             [[buffer(5)]],
    constant GlassParams             &p             [[buffer(6)]],
    texture2d<float, access::sample>  scene_color   [[texture(0)]],
    depth2d<float>                    scene_depth   [[texture(1)]],
    sampler                           scene_sampler [[sampler(0)]])
{
    float3 view_dir = normalize(v.camera_pos.xyz - in.world_pos);
    // Two-sided: orient the normal toward the viewer so a pane lit from
    // behind still Fresnels correctly.
    float3 normal = normalize(p.normal.xyz);
    if (dot(normal, view_dir) < 0.0) {
        normal = -normal;
    }

    float2 viewport = max(v.viewport, float2(1.0));
    float2 frag_uv = float2(in.position.x / viewport.x,
                            in.position.y / viewport.y);

    // Manual depth occlusion: discard where the resolved scene depth at this
    // pixel is nearer than the pane (the pass has no hardware depth test).
    uint2 self_pixel = min(uint2(in.position.xy), uint2(viewport) - uint2(1));
    float scene_self_depth01 = scene_depth.read(self_pixel);
    if (scene_self_depth01 < in.position.z) {
        discard_fragment();
    }

    // Refraction: perturb the screen lookup by the pane normal's screen-plane
    // component so the background bends across the pane.
    float2 refract_uv = clamp(frag_uv + normal.xy * p.refraction_strength,
                              float2(0.001), float2(0.999));
    float3 refracted = scene_color.sample(scene_sampler, refract_uv).rgb * p.tint.rgb;

    // Schlick-Fresnel rim: brighter + more opaque at grazing angles.
    float n_dot_v = saturate(dot(normal, view_dir));
    float fresnel = pow(1.0 - n_dot_v, max(p.fresnel_power, 1e-3));

    float3 rim = float3(1.0);
    float3 colour = mix(refracted, rim, fresnel * 0.5);
    float alpha = saturate(mix(p.opacity, 1.0, fresnel));

    return float4(colour, alpha);
}
