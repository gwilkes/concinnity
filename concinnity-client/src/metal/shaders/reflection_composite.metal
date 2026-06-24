#include <metal_stdlib>
using namespace metal;

// Roughness-aware reflection composite, split into two passes so the wide glossy
// blur runs at reduced resolution.
//
//   reflection_blur_fragment     (half-res): weight-averages the reflection
//       target over a roughness-scaled cone into a reduced-resolution blur
//       target. This is the expensive part (up to 17 taps), so running it at a
//       fraction of the pixels is the saving.
//   reflection_composite_fragment (full-res): per pixel, lerps the FULL-RES
//       sharp reflection against the upsampled half-res blur by roughness, then
//       composites over the scene. A near-mirror (roughness ~0) reads the sharp
//       full-res tap so it stays razor-sharp; a rough surface reads the cheap
//       upsampled blur, which is low-frequency anyway.

struct ReflCompVtxOut {
    float4 position [[position]];
    float2 uv;
};

// Fullscreen triangle from vertex_id 0..2 - no vertex buffer.
vertex ReflCompVtxOut reflection_composite_vertex(uint vid [[vertex_id]]) {
    float2 pos = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    ReflCompVtxOut out;
    out.position = float4(pos, 0.0, 1.0);
    out.uv = float2((pos.x + 1.0) * 0.5, 1.0 - (pos.y + 1.0) * 0.5);
    return out;
}

// Below REFLECTION_ROUGHNESS_CUT the blur radius (in UV) ramps from a sharp
// mirror at 0 to this maximum; at or above the cut the reflection weight is
// already 0. The cut is the shared `constant` injected by pipeline.rs, so it
// matches the SSR / RT resolve gates exactly.
constant float REFL_BLUR_MAX  = 0.02;

// Two 8-tap rings (45-degree steps) at half and full radius approximate the
// widening glossy reflection cone.
constant float2 REFL_RING[8] = {
    float2( 1.0,         0.0       ), float2( 0.70710678,  0.70710678),
    float2( 0.0,         1.0       ), float2(-0.70710678,  0.70710678),
    float2(-1.0,         0.0       ), float2(-0.70710678, -0.70710678),
    float2( 0.0,        -1.0       ), float2( 0.70710678, -0.70710678),
};

// Pass 1 (reduced resolution): weight-averages the reflection target over the
// roughness cone. `.rgb` is the weight-normalised reflected radiance, `.a` the
// mean coverage weight. Weightless (non-reflecting) taps cannot drag their
// colour in, so a reflective edge fades smoothly.
fragment float4 reflection_blur_fragment(
    ReflCompVtxOut in           [[stage_in]],
    texture2d<float> reflection [[texture(0)]],
    texture2d<float> rough_tex  [[texture(1)]],
    sampler smp                 [[sampler(0)]]
) {
    float4 c        = reflection.sample(smp, in.uv);
    float  roughness = rough_tex.sample(smp, in.uv).r;
    float  radius    = saturate(roughness / REFLECTION_ROUGHNESS_CUT) * REFL_BLUR_MAX;

    float3 sum_rw = c.rgb * c.a;
    float  sum_w  = c.a;
    float  taps   = 1.0;
    if (radius > 1e-5) {
        for (int ring = 0; ring < 2; ring++) {
            float rr = radius * (ring == 0 ? 0.5 : 1.0);
            for (int i = 0; i < 8; i++) {
                float4 t = reflection.sample(smp, in.uv + REFL_RING[i] * rr);
                sum_rw += t.rgb * t.a;
                sum_w  += t.a;
                taps   += 1.0;
            }
        }
    }
    float3 blurred = sum_w > 1e-4 ? sum_rw / sum_w : c.rgb;
    float  weight  = sum_w / taps;
    return float4(blurred, weight);
}

// Pass 2 (full resolution): lerp the sharp full-res reflection against the
// upsampled half-res blur by roughness, then composite over the scene.
fragment float4 reflection_composite_fragment(
    ReflCompVtxOut in           [[stage_in]],
    texture2d<float> reflection [[texture(0)]],
    texture2d<float> scene      [[texture(1)]],
    texture2d<float> gbuffer    [[texture(2)]],
    texture2d<float> rough_tex  [[texture(3)]],
    texture2d<float> blur       [[texture(4)]],
    sampler smp                 [[sampler(0)]]
) {
    float3 base  = scene.sample(smp, in.uv).rgb;
    float  depth = gbuffer.sample(smp, in.uv).a;
    float4 c     = reflection.sample(smp, in.uv);
    // Background, or a pixel that does not reflect (weight 0): keep the scene.
    // A tiny weight is still honoured so the composite is exact at the edges.
    if (depth <= 0.0 || c.a <= 0.0) {
        return float4(mix(base, c.rgb, c.a), 1.0);
    }

    float  roughness = rough_tex.sample(smp, in.uv).r;
    // 0 at a mirror -> use the sharp full-res tap; 1 at the cut -> use the
    // cheap upsampled blur. The blur is low-frequency, so the bilinear upsample
    // is visually free; only the sharp branch needs full-res detail.
    float  t = saturate(roughness / REFLECTION_ROUGHNESS_CUT);
    float4 b = blur.sample(smp, in.uv);

    float3 reflected = mix(c.rgb, b.rgb, t);
    float  weight    = mix(c.a, b.a, t);
    return float4(mix(base, reflected, weight), 1.0);
}
