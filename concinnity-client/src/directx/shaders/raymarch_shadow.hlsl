// src/directx/shaders/raymarch_shadow.hlsl
//
// Engine-shipped depth-only template for raymarched SDF shadow casters
// on D3D12. Appended to the user's `map` + `shade` definitions (after
// `raymarch_helpers.hlsl`) when building the per-volume SHADOW PSO.
// The user's `shade` is dead code here - FXC DCE strips it - so only
// `map` ends up sampled. That's why this template doesn't have to
// re-declare any of the IBL / scene-color resources `raymarch_template`
// binds in the main pass: those resources are only referenced through
// `shade`, which is stripped.
//
// One PSO per `SdfVolume.cast_shadows == true` per backend. The encoder
// draws the proxy unit cube (back faces only - front-face culling) once
// per CSM cascade, writing the SDF hit depth via `SV_DepthLessEqual`
// against the cascade's depth-only DSV. The result composites with
// rasterised casters in the same DSV - the hardware z-test ensures only
// the nearest caster (rasterised or raymarched) survives at each texel.
//
// Resource bindings (matching directx/raymarch.rs's shadow root sig):
//
//   b0 - RaymarchView           (per-frame view; we need view_time so
//                                time-varying SDFs cast time-varying
//                                shadows that line up with the matching
//                                live-pass surface)
//   b1 - SdfVolumeUniforms      (per-volume - vol_centre/extent/params)
//   b2 - RaymarchLights         (light_directional[0].direction)
//   b3 - RaymarchShadowUniforms (shadow_light_vps[cascade])
//   b4 - RaymarchShadowCascade  (cascade_idx, root constant)

cbuffer RaymarchShadowCascade : register(b4)
{
    uint shadow_cascade_idx;
    uint shadow_cascade_pad0;
    uint shadow_cascade_pad1;
    uint shadow_cascade_pad2;
};

struct ShadowVsIn
{
    float3 pos      : POSITION;
    float3 normal   : NORMAL;
    float3 tangent  : TANGENT;
    float3 color    : COLOR;
    float2 uv       : TEXCOORD0;
};

struct ShadowVsOut
{
    // Same `noperspective centroid` qualifier the main raymarch template
    // uses - required by FXC when the pixel shader writes
    // `SV_DepthLessEqual` and is not sample-frequency.
    noperspective centroid float4 sv_pos : SV_POSITION;
    float3 world_pos : WORLD_POS;
};

ShadowVsOut raymarch_shadow_vertex(ShadowVsIn v)
{
    // Unit-cube proxy at ±1; encoder scales by vol_extent + offsets by
    // vol_centre to land at the AABB corners. Identical to the main
    // raymarch vertex - only the projection matrix changes (cascade VP
    // instead of camera VP).
    float3 wp = v.pos * vol_extent + vol_centre;
    ShadowVsOut o;
    o.sv_pos = mul(shadow_light_vps[shadow_cascade_idx], float4(wp, 1.0));
    o.world_pos = wp;
    return o;
}

// Output depth only. Returning `float : SV_DepthLessEqual` is the HLSL
// shorthand for a single SV_DepthLessEqual write; no SV_TARGET attached
// since the shadow pass binds no RTV.
float raymarch_shadow_fragment(ShadowVsOut input) : SV_DepthLessEqual
{
    // For a directional sun, `light_directional[0].direction` is the
    // L vector - from the surface UP to the light. Rays of incoming
    // light travel along -L. Match what shadePbrSun consumes so SDF
    // shadows align with the lit-side reading of the same field.
    float3 ray_dir = -normalize(light_directional[0].direction);

    // `input.world_pos` lies on the bbox face farthest from the light
    // (the encoder culls front faces of the proxy cube). Walk back along
    // `-ray_dir` (i.e. toward the light) by the bbox bounding-sphere
    // diameter, so the slab test below picks up the actual front face
    // entry from outside.
    float diag = length(vol_extent) * 2.5;
    float3 origin = input.world_pos - ray_dir * diag;

    float3 box_min = vol_centre - vol_extent;
    float3 box_max = vol_centre + vol_extent;
    float2 box_t = rayBox(origin, ray_dir, box_min, box_max);
    if (box_t.y < max(box_t.x, 0.0))
    {
        discard;
    }
    float t_enter = max(box_t.x, 0.001);
    float t_max = min(box_t.y, vol_max_distance);
    if (t_enter >= t_max)
    {
        discard;
    }

    RayHit hit = coneRaymarch(origin, ray_dir, t_enter, t_max, view_time);
    if (!hit.hit)
    {
        discard;
    }

    // Project the hit back through the SAME cascade VP the vertex stage
    // rasterised with, so the SV_DepthLessEqual write shares the NDC
    // depth space of rasterised casters in this cascade's DSV. The
    // SV_DepthLessEqual contract is satisfied because the SDF hit is
    // bounded by t_max ≤ box exit, so its NDC.z ≤ the back face's
    // rasterised NDC.z.
    float3 hit_pos = origin + ray_dir * hit.t;
    float4 hit_clip = mul(shadow_light_vps[shadow_cascade_idx],
                          float4(hit_pos, 1.0));
    return hit_clip.z / max(hit_clip.w, 1e-6);
}
