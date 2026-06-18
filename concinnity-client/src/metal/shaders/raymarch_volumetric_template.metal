// src/metal/shaders/raymarch_volumetric_template.metal
//
// Volumetric variant of the raymarch template. Appended to a user's
// volumetric fragment shader at compile time (after the helpers + the
// user's `sampleVolume` definition). Authoring contract:
//
//     VolumeSample sampleVolume(float3 p, constant SdfParams& params, float time);
//
// where `VolumeSample` is declared in `raymarch_helpers.metal`. The user
// shader does NOT need `map` / `shade`; the Metal compiler's dead-code
// elimination strips the unused engine helpers (`sdfNormal`, `coneRaymarch`)
// along with their `map` calls.
//
// March + integration: same back-face bounding-box rasterisation as the
// surface template, but at the pixel stage we step linearly from box
// entry to exit (`max_steps` samples), accumulate Beer-Lambert
// transmittance front-to-back, and add per-step in-scattered sun light
// plus emission. Output is alpha-blended over the rasterised scene
// (SRC_ALPHA / ONE_MINUS_SRC_ALPHA at the pipeline level); no depth write.
//
// V1 limitations:
//   * No self-shadowing - the in-scatter term assumes uniform sun
//     illumination through the medium. Adding a secondary short march
//     toward the sun per step would buy directional shadowing at ~32x
//     density-sample cost; deferred.
//   * No per-pixel scene-depth clamp - the march extends across the
//     full bounding box. Sized so the box does not intersect rasterised
//     geometry the volume is meant to render behind. Adding a
//     scene-depth read + per-pixel `t_exit = min(t_exit, scene_t)` is a
//     well-bounded follow-up.

struct VolVertexIn {
    float3 pos [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float3 tangent [[attribute(2)]];
    float3 color [[attribute(3)]];
    float2 uv [[attribute(4)]];
};

struct VolVertexOut {
    float4 position [[position]];
    float3 world_pos;
};

vertex VolVertexOut raymarch_volumetric_vertex(
    VolVertexIn v [[stage_in]],
    constant RaymarchView& view [[buffer(0)]],
    constant SdfVolumeUniforms& vol [[buffer(1)]]
) {
    float3 wp = v.pos * float3(vol.extent) + float3(vol.centre);
    VolVertexOut o;
    o.position = view.vp * float4(wp, 1.0);
    o.world_pos = wp;
    return o;
}

struct VolFragOut {
    float4 color [[color(0)]];
};

fragment VolFragOut raymarch_volumetric_fragment(
    VolVertexOut in [[stage_in]],
    constant RaymarchView& view [[buffer(0)]],
    constant SdfVolumeUniforms& vol [[buffer(1)]],
    constant RaymarchLights& lights [[buffer(2)]],
    constant RaymarchShadowUniforms& shadow [[buffer(3)]],
    depth2d_ms<float> main_depth [[texture(0)]],
    depth2d_array<float> shadow_map [[texture(1)]],
    texturecube<float> irradiance_cube [[texture(2)]],
    texturecube<float> prefilter_cube [[texture(3)]],
    texture2d<float> scene_color [[texture(4)]],
    sampler depth_samp [[sampler(0)]],
    sampler shadow_samp [[sampler(1)]],
    sampler cube_samp [[sampler(2)]],
    sampler scene_samp [[sampler(3)]]
) {
    float3 cam = float3(view.cam_pos);
    float3 ray_dir = normalize(in.world_pos - cam);

    // Clip the ray to the volume's AABB. The vertex shader rasterised the
    // back faces, so `in.world_pos` is on the far side of the box; the slab
    // test gives both enter + exit and handles the camera-inside-box case.
    float3 box_min = float3(vol.centre) - float3(vol.extent);
    float3 box_max = float3(vol.centre) + float3(vol.extent);
    float2 box_t = rayBox(cam, ray_dir, box_min, box_max);
    if (box_t.y < max(box_t.x, 0.0)) {
        discard_fragment();
    }
    float t_enter = max(box_t.x, 0.001);
    float t_exit = min(box_t.y, vol.max_distance);
    if (t_enter >= t_exit) {
        discard_fragment();
    }

    uint step_count = uint(max(vol.max_steps, 1));
    float step_size = (t_exit - t_enter) / float(step_count);

    // Sun radiance for the in-scatter term. No self-shadow march in V1.
    float3 sun_radiance = float3(0.0, 0.0, 0.0);
    if (lights.num_directional > 0) {
        sun_radiance = float3(lights.directional[0].color)
                     * lights.directional[0].intensity;
    }

    float transmittance = 1.0;
    float3 luminance = float3(0.0, 0.0, 0.0);

    for (uint i = 0u; i < step_count; ++i) {
        float t = t_enter + (float(i) + 0.5) * step_size;
        float3 p = cam + ray_dir * t;

        VolumeSample vs = sampleVolume(p, vol.params, view.time);
        if (vs.density <= 0.0) {
            continue;
        }

        float opt_depth = vs.density * step_size;
        float step_T = exp(-opt_depth);

        // Energy in-scattered inside this slab = (1 - step_T) * radiance.
        // Radiance = single-scatter from the sun (modulated by the per-point
        // scattering albedo) plus self-emission.
        float3 step_radiance = sun_radiance * vs.scattering + vs.emission;
        luminance += transmittance * step_radiance * (1.0 - step_T);
        transmittance *= step_T;

        // Early-out once the medium is effectively opaque.
        if (transmittance < 0.005) {
            break;
        }
    }

    float alpha = 1.0 - transmittance;
    if (alpha < 0.005) {
        discard_fragment();
    }

    VolFragOut o;
    o.color = float4(luminance, alpha);
    return o;
}
