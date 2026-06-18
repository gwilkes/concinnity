// src/vulkan/shaders/raymarch_volumetric_template.glsl
//
// Engine-shipped volumetric fragment template for raymarched SDF volumes on
// Vulkan. Appended to the user's fragment shader at compile time (after the
// helpers + the user's `sampleVolume` definition). GLSL port of
// `raymarch_volumetric_fragment` in directx/shaders/raymarch_volumetric_template.hlsl
// and the Metal counterpart: same march + Beer-Lambert integration, same
// alpha-blend-over-scene compositing.
//
// The user shader defines `sampleVolume` instead of `map` / `shade`; glslang
// prunes the unused surface helpers (`sdfNormal`, `coneRaymarch`) along with
// their forward-declared `map` calls, so the volumetric author needs no surface
// stubs.
//
// March + integration: the proxy vertex shader rasterised the bounding-box back
// faces, so each fragment seeds a ray that pierces the box. We step linearly
// from box entry to exit (`vol_max_steps` samples), accumulate Beer-Lambert
// transmittance front-to-back, and add per-step in-scattered sun light plus
// emission. Output alpha-blends over the rasterised scene (SRC_ALPHA /
// ONE_MINUS_SRC_ALPHA at the pipeline level); no depth write, so the medium
// neither occludes itself nor updates the depth buffer downstream passes read.
//
// V1 limitations (match the DirectX + Metal templates):
//   * No self-shadowing -- the in-scatter term assumes uniform sun illumination
//     through the medium.
//   * No per-pixel scene-depth clamp -- the march extends across the full
//     bounding box. Size the box so it doesn't intersect rasterised geometry the
//     volume is meant to render behind.

layout(location = 0) in vec3 v_world_pos;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 cam = rmview.view_cam_pos;
    vec3 ray_dir = normalize(v_world_pos - cam);

    // Clip the ray to the volume's AABB (slab test; handles camera-inside).
    vec3 box_min = vol.vol_centre - vol.vol_extent;
    vec3 box_max = vol.vol_centre + vol.vol_extent;
    vec2 box_t = rayBox(cam, ray_dir, box_min, box_max);
    if (box_t.y < max(box_t.x, 0.0)) {
        discard;
    }
    float t_enter = max(box_t.x, 0.001);
    float t_exit = min(box_t.y, vol.vol_max_distance);
    if (t_enter >= t_exit) {
        discard;
    }

    int step_count = max(vol.vol_max_steps, 1);
    float step_size = (t_exit - t_enter) / float(step_count);

    // Sun radiance for the in-scatter term. No self-shadow march in V1.
    vec3 sun_radiance = vec3(0.0);
    if (lights.num_dir > 0) {
        sun_radiance = lights.dir[0].col.xyz * lights.dir[0].dir_i.w;
    }

    float transmittance = 1.0;
    vec3 luminance = vec3(0.0);

    for (int i = 0; i < step_count; ++i) {
        float t = t_enter + (float(i) + 0.5) * step_size;
        vec3 p = cam + ray_dir * t;

        VolumeSample vs = sampleVolume(p, vol.vol_params, rmview.view_time);
        if (vs.density <= 0.0) {
            continue;
        }

        float opt_depth = vs.density * step_size;
        float step_T = exp(-opt_depth);

        // Energy in-scattered inside this slab = (1 - step_T) * radiance.
        // Radiance = single-scatter from the sun (modulated by the per-point
        // scattering albedo) plus self-emission.
        vec3 step_radiance = sun_radiance * vs.scattering + vs.emission;
        luminance += transmittance * step_radiance * (1.0 - step_T);
        transmittance *= step_T;

        // Early-out once the medium is effectively opaque.
        if (transmittance < 0.005) {
            break;
        }
    }

    float alpha = 1.0 - transmittance;
    if (alpha < 0.005) {
        discard;
    }

    out_color = vec4(luminance, alpha);
}
