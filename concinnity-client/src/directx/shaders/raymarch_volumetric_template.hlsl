// src/directx/shaders/raymarch_volumetric_template.hlsl
//
// Volumetric variant of the raymarch template. Appended to a user's
// volumetric fragment shader at compile time (after the helpers + the
// user's `sampleVolume` definition). Authoring contract:
//
//     VolumeSample sampleVolume(float3 p, SdfParams params, float time);
//
// where `VolumeSample` is declared in `raymarch_helpers.hlsl`. The user
// shader does NOT need `map` / `shade`; FXC strips the unused engine
// helpers (`sdfNormal`, `coneRaymarch`) along with their `map` calls.
//
// March + integration: same back-face bounding-box rasterisation as the
// surface template, but at the pixel stage we step linearly from box
// entry to exit (`vol_max_steps` samples), accumulate Beer-Lambert
// transmittance front-to-back, and add per-step in-scattered sun light
// plus emission. Output is alpha-blended over the rasterised scene
// (SRC_ALPHA / ONE_MINUS_SRC_ALPHA at the PSO level); no depth write.
//
// V1 limitations:
//   * No self-shadowing - the in-scatter term assumes uniform sun
//     illumination through the medium. Adding a secondary short march
//     toward the sun per step would buy directional shadowing at ~32x
//     density-sample cost; deferred.
//   * No per-pixel scene-depth clamp - the march extends across the
//     full bounding box. Sized so the box does not intersect rasterised
//     geometry the volume is meant to render behind. Adding a
//     scene_depth SRV + per-pixel `t_exit = min(t_exit, scene_t)` is a
//     well-bounded follow-up.

struct VolVsIn
{
    float3 pos      : POSITION;
    float3 normal   : NORMAL;
    float3 tangent  : TANGENT;
    float3 color    : COLOR;
    float2 uv       : TEXCOORD0;
};

struct VolVsOut
{
    float4 sv_pos    : SV_POSITION;
    float3 world_pos : WORLD_POS;
};

VolVsOut raymarch_volumetric_vertex(VolVsIn v)
{
    float3 wp = v.pos * vol_extent + vol_centre;
    VolVsOut o;
    o.sv_pos = mul(view_vp, float4(wp, 1.0));
    o.world_pos = wp;
    return o;
}

float4 raymarch_volumetric_fragment(VolVsOut input) : SV_TARGET0
{
    float3 cam = view_cam_pos;
    float3 ray_dir = normalize(input.world_pos - cam);

    float3 box_min = vol_centre - vol_extent;
    float3 box_max = vol_centre + vol_extent;
    float2 box_t = rayBox(cam, ray_dir, box_min, box_max);
    if (box_t.y < max(box_t.x, 0.0))
    {
        discard;
    }
    float t_enter = max(box_t.x, 0.001);
    float t_exit = min(box_t.y, vol_max_distance);
    if (t_enter >= t_exit)
    {
        discard;
    }

    uint step_count = (uint)max(vol_max_steps, 1);
    float step_size = (t_exit - t_enter) / (float)step_count;

    // Sun radiance for the in-scatter term. No self-shadow march in V1.
    float3 sun_radiance = float3(0.0, 0.0, 0.0);
    if (light_num_directional > 0)
    {
        sun_radiance = light_directional[0].color * light_directional[0].intensity;
    }

    float  transmittance = 1.0;
    float3 luminance = float3(0.0, 0.0, 0.0);

    for (uint i = 0u; i < step_count; ++i)
    {
        float t = t_enter + ((float)i + 0.5) * step_size;
        float3 p = cam + ray_dir * t;

        VolumeSample vs = sampleVolume(p, vol_params, view_time);
        if (vs.density <= 0.0)
        {
            continue;
        }

        float opt_depth = vs.density * step_size;
        float step_T = exp(-opt_depth);

        // Energy absorbed inside this slab = (1 - step_T) * radiance.
        // Radiance contributions: single-scatter from the sun
        // (modulated by per-point `scattering`) plus self-emission.
        float3 step_radiance = sun_radiance * vs.scattering + vs.emission;
        luminance += transmittance * step_radiance * (1.0 - step_T);
        transmittance *= step_T;

        // Early-out once the medium is effectively opaque to further
        // contributions; cheap fast-path for dense clouds.
        if (transmittance < 0.005)
        {
            break;
        }
    }

    float alpha = 1.0 - transmittance;
    if (alpha < 0.005)
    {
        discard;
    }

    return float4(luminance, alpha);
}
