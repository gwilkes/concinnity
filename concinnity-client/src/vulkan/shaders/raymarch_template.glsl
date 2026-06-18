// src/vulkan/shaders/raymarch_template.glsl
//
// Engine-shipped opaque fragment template for raymarched SDF volumes on Vulkan.
// Appended to the user's fragment shader at compile time (after the helpers +
// the user's `map` / `shade` definitions). GLSL port of `raymarch_fragment` in
// directx/shaders/raymarch_template.hlsl: same control flow, same
// depth-compositing rules, same shading sequence.
//
// The proxy vertex shader (raymarch_proxy.vert) rasterised the bounding-box
// back faces, so each fragment seeds a ray that pierces the box. Hit depth flows
// out through `gl_FragDepth` redeclared `depth_less` (the GLSL analogue of
// HLSL's `SV_DepthLessEqual` / Metal's `[[depth(less)]]`): the hardware depth
// test against the existing MSAA depth discards hits behind rasterised geometry,
// and the same write feeds raymarched-surface depth back into the depth buffer
// so downstream passes (decals, fog, SSR) see the SDF. The encoder re-resolves
// the MSAA colour into hdr_resolve after this pass so the single-sample post
// stack picks up the raymarched pixels.

layout(location = 0) in vec3 v_world_pos;
layout(location = 0) out vec4 out_color;

// `depth_less`: preserve early-Z while allowing the shader to write a smaller
// depth, matching the HLSL `SV_DepthLessEqual` contract.
layout(depth_less) out float gl_FragDepth;

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
    float t_max = min(box_t.y, vol.vol_max_distance);
    if (t_enter >= t_max) {
        discard;
    }

    RayHit hit = coneRaymarch(cam, ray_dir, t_enter, t_max, rmview.view_time);
    if (!hit.hit) {
        discard;
    }

    vec3 hit_pos = cam + ray_dir * hit.t;
    vec3 normal = sdfNormal(hit_pos, vol.vol_params, rmview.view_time, 0.001);
    vec2 frag_uv = gl_FragCoord.xy / rmview.view_viewport;
    SdfSurface surf = shade(hit_pos, normal, vol.vol_params, rmview.view_time, frag_uv);

    vec3 view_dir = -ray_dir;
    vec3 color = shadeAmbientIbl(surf, normal, view_dir, rmview.view_prefilter_mip_count);
    if (lights.num_dir > 0) {
        float shadow_factor = 1.0;
        if (vol.vol_receive_shadows != 0) {
            shadow_factor = sampleSunShadow(hit_pos, hit.t, gl_FragCoord.xy);
        }
        color += shadePbrSun(surf, normal, view_dir, lights.dir[0], shadow_factor);
    }
    color += surf.transmitted;

    // Reproject the hit through view_vp for gl_FragDepth -- same matrix the
    // proxy rasterised with, so the depth shares the rasterised NDC depth space.
    vec4 hit_clip = rmview.view_vp * vec4(hit_pos, 1.0);
    out_color = vec4(color, 1.0);
    gl_FragDepth = hit_clip.z / max(hit_clip.w, 1e-6);
}
