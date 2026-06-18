// src/metal/shaders/raymarch_template.metal
//
// Engine-shipped template for raymarched SDF volumes. Appended to the
// user's fragment shader at compile time (after the helpers + the
// user's `map` / `shade` definitions). Provides:
//
//   * `raymarch_vertex` - rasterises the bounding-box proxy (back faces
//     only via the encoder's cull mode). Each output fragment is a
//     candidate for a ray that pierces the box.
//   * `raymarch_fragment` - reconstructs the world-space ray, samples
//     main depth for early-out, calls `coneRaymarch` + the user's
//     `map`, and on hit calls `sdfNormal` + the user's `shade` and
//     applies the engine's PBR + ambient helpers. Writes opaque colour
//     into the bound `hdr_resolve` attachment.
//
// The fragment writes hit depth back into the
// bound depth attachment via `[[depth(less)]]`, so downstream passes
// (water / decal / fog) that sample `hdr_targets.depth_resolve` see the
// raymarched surface and composite correctly. The shader-side cone-
// march early-out still samples the MSAA `hdr_targets.depth` as a
// read-only snapshot, a different texture from the writable
// `depth_resolve` attachment so no aliasing rule applies.

struct VertexIn {
    float3 pos     [[attribute(0)]];
    float3 normal  [[attribute(1)]];
    float3 tangent [[attribute(2)]];
    float3 color   [[attribute(3)]];
    float2 uv      [[attribute(4)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 world_pos;
};

vertex VertexOut raymarch_vertex(
    VertexIn v [[stage_in]],
    constant RaymarchView& view [[buffer(0)]],
    constant SdfVolumeUniforms& vol [[buffer(1)]]
) {
    // The proxy buffer is a unit cube with positions in [-0.5, 0.5]^3.
    // Scale by the volume's extent (which is the AABB half-widths * 2,
    // see raymarch.rs::build_raymarch_cube_buffers) and translate by
    // centre to land in world space.
    float3 wp = v.pos * float3(vol.extent) + float3(vol.centre);
    VertexOut o;
    o.position = view.vp * float4(wp, 1.0);
    o.world_pos = wp;
    return o;
}

struct RaymarchFragOut {
    float4 color [[color(0)]];
    // `depth(less)` lets the hardware discard fragments whose computed
    // hit depth is behind the already-written depth (so overlapping
    // SDF volumes still resolve correctly in z-order), and writes the
    // hit's reprojected NDC depth into the bound `depth_resolve`
    // attachment so downstream passes that sample depth see the
    // raymarched surface, not the rasterised geometry behind it.
    float depth [[depth(less)]];
};

fragment RaymarchFragOut raymarch_fragment(
    VertexOut in [[stage_in]],
    constant RaymarchView& view [[buffer(0)]],
    constant SdfVolumeUniforms& vol [[buffer(1)]],
    constant RaymarchLights& lights [[buffer(2)]],
    constant RaymarchShadowUniforms& shadow [[buffer(3)]],
    depth2d_ms<float> main_depth [[texture(0)]],
    depth2d_array<float> shadow_map [[texture(1)]],
    texturecube<float> irradiance_cube [[texture(2)]],
    texturecube<float> prefilter_cube [[texture(3)]],
    // Pre-raymarch scene snapshot the blit at the head of
    // the pass populated. User shaders sample this through
    // `sampleSceneRefracted` to get the surface-below-water colour.
    texture2d<float> scene_color [[texture(4)]],
    sampler shadow_samp [[sampler(1)]],
    sampler cube_samp [[sampler(2)]],
    sampler scene_samp [[sampler(3)]]
) {
    // Build the world-space ray from camera through this fragment.
    float3 cam = float3(view.cam_pos);
    float3 ray_dir = normalize(in.world_pos - cam);

    // Clip the ray to the volume's AABB. The vertex shader rasterised
    // the back faces, so `in.world_pos` lies on the far side of the
    // box; using the slab test gives both enter + exit and handles the
    // camera-inside-box case uniformly (t_enter clamps to 0 below).
    float3 box_min = float3(vol.centre) - float3(vol.extent);
    float3 box_max = float3(vol.centre) + float3(vol.extent);
    float2 box_t = rayBox(cam, ray_dir, box_min, box_max);
    if (box_t.y < max(box_t.x, 0.0)) {
        discard_fragment();
    }
    float t_enter = max(box_t.x, 0.001);

    // Sample main depth at this pixel. MSAA depth attachment, read
    // sample 0; the raymarch pass is single-sample and the worst-case
    // distance is "the closest pixel of the rasterised geometry behind
    // this fragment", so picking any sample is conservative-enough for
    // the early-out. Convert depth → world-space distance via inv_vp.
    uint2 px = uint2(in.position.xy);
    float depth_ndc = main_depth.read(px, 0);
    float2 ndc_xy = (in.position.xy / view.viewport) * 2.0 - 1.0;
    // Metal NDC has y-down in clip space (after the projection matrix
    // flip the engine applies), so re-mirror Y to match the inv_vp the
    // CPU built (which inverts the un-flipped projection).
    ndc_xy.y = -ndc_xy.y;
    float4 world = view.inv_vp * float4(ndc_xy, depth_ndc, 1.0);
    world /= max(world.w, 1e-6);
    float t_rasterized = length(world.xyz - cam);

    // Clip to the closest of: bounding-box exit, rasterised depth, the
    // per-volume far-clip. Everything past `t_max` is discarded, so a
    // raymarch behind rasterised geometry pays the rasterisation cost
    // only and the shader bails immediately.
    float t_max = min(box_t.y, min(t_rasterized, vol.max_distance));
    if (t_enter >= t_max) {
        discard_fragment();
    }

    RayHit hit = coneRaymarch(cam, ray_dir, t_enter, t_max, vol, view.time);
    if (!hit.hit) {
        discard_fragment();
    }

    float3 hit_pos = cam + ray_dir * hit.t;
    float3 normal = sdfNormal(hit_pos, vol.params, view.time, 0.001);
    // [0, 1] screen UV in Metal's y-down convention, matches the
    // sampler `scene_color.sample` expects directly.
    float2 frag_uv = in.position.xy / view.viewport;
    SdfSurface surf = shade(hit_pos, normal, vol.params, view.time,
                             frag_uv, scene_color, scene_samp);

    // IBL ambient when an EnvironmentMap is bound +
    // CSM cascade-shadowed first directional light. The helpers
    // internally fall back when the gate fails (no IBL → hemispheric
    // ambient; receive_shadows off → no shadow sample).
    float3 view_dir = -ray_dir;
    float3 color = shadeAmbientIbl(
        surf, normal, view_dir,
        view.prefilter_mip_count,
        irradiance_cube, prefilter_cube, cube_samp
    );
    if (lights.num_directional > 0) {
        float shadow_factor = 1.0;
        if (vol.receive_shadows != 0) {
            // `hit.t` is distance along the view ray ~ view-space
            // depth (positive, camera-relative). Good enough for
            // cascade selection without needing the view matrix; the
            // cosine error near screen edges is small for typical
            // SDF volume sizes.
            shadow_factor = sampleSunShadow(
                hit_pos, hit.t, in.position.xy,
                shadow, shadow_map, shadow_samp
            );
        }
        color += shadePbrSun(surf, normal, view_dir,
                              lights.directional[0], shadow_factor);
    }
    // Refraction / transmitted contribution. User shaders
    // that sampled the scene snapshot inside `shade` set this to the
    // colour they want to show through; opaque shaders leave it zero.
    color += surf.transmitted;

    // Reproject the hit position through the camera's view-projection
    // for the NDC depth output. The matrix is the same `view.vp` the
    // vertex shader rasterised the proxy cube with, so the reprojected
    // depth shares the rasterised geometry's depth space exactly.
    float4 hit_clip = view.vp * float4(hit_pos, 1.0);
    RaymarchFragOut o;
    o.color = float4(color, 1.0);
    o.depth = hit_clip.z / max(hit_clip.w, 1e-6);
    return o;
}
