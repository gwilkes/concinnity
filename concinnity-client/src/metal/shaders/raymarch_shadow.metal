// src/metal/shaders/raymarch_shadow.metal
//
// Engine-shipped depth-only template for raymarched SDF shadow casters on
// Metal. Appended to the user's `map` + `shade` definitions (after
// raymarch_helpers.metal) when building the per-volume SHADOW pipeline - note
// the main `raymarch_template.metal` is NOT included, so this library defines
// only the shadow vertex + fragment. The user's `shade` is never called by
// the shadow functions, so it links to nothing and binds no IBL / scene-colour
// resources; only `map` (through `coneRaymarch`) is sampled.
//
// One pipeline per `SdfVolume.cast_shadows == true`. The encoder draws the
// proxy unit cube (back faces only - front-face culling) once per CSM cascade
// into that cascade's `shadow_map` slice, marching the SDF from the light side
// and writing the hit's NDC depth via `[[depth(less)]]`. The cascade slice's
// LESS depth test composites the raymarched caster with the rasterised casters
// already drawn into it - the nearer occluder wins per texel. Mirrors
// src/directx/shaders/raymarch_shadow.hlsl.

// Which cascade this draw targets. Bound at buffer(4); selects
// `shadow.light_vps[idx]` for both the vertex projection and the fragment's
// depth reprojection. Matches `RaymarchShadowCascade` in metal/raymarch.rs.
struct RaymarchShadowCascade {
    uint cascade_idx;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct ShadowVertexIn {
    float3 pos     [[attribute(0)]];
    float3 normal  [[attribute(1)]];
    float3 tangent [[attribute(2)]];
    float3 color   [[attribute(3)]];
    float2 uv      [[attribute(4)]];
};

struct ShadowVertexOut {
    float4 position [[position]];
    float3 world_pos;
};

vertex ShadowVertexOut raymarch_shadow_vertex(
    ShadowVertexIn v [[stage_in]],
    constant SdfVolumeUniforms& vol [[buffer(1)]],
    constant RaymarchShadowUniforms& shadow [[buffer(3)]],
    constant RaymarchShadowCascade& cascade [[buffer(4)]]
) {
    // Unit-cube proxy at ±1; scale by extent + offset by centre to land at
    // the AABB corners. Identical to the main raymarch vertex - only the
    // projection matrix changes to the cascade's light view-projection.
    float3 wp = v.pos * float3(vol.extent) + float3(vol.centre);
    ShadowVertexOut o;
    o.position = shadow.light_vps[cascade.cascade_idx] * float4(wp, 1.0);
    o.world_pos = wp;
    return o;
}

// Depth-only output: the shadow pass binds no colour attachment.
struct ShadowFragOut {
    float depth [[depth(less)]];
};

fragment ShadowFragOut raymarch_shadow_fragment(
    ShadowVertexOut in [[stage_in]],
    constant RaymarchView& view [[buffer(0)]],
    constant SdfVolumeUniforms& vol [[buffer(1)]],
    constant RaymarchLights& lights [[buffer(2)]],
    constant RaymarchShadowUniforms& shadow [[buffer(3)]],
    constant RaymarchShadowCascade& cascade [[buffer(4)]]
) {
    // For a directional sun, `directional[0].direction` is L (surface → light).
    // Incoming light travels along -L, so the shadow ray marches along -L. Match
    // what `shadePbrSun` reads from the same field so SDF shadows line up with
    // the lit-side surface.
    float3 ray_dir = -normalize(float3(lights.directional[0].direction));

    // `in.world_pos` lies on the bbox face farthest from the light (the encoder
    // culls front faces of the proxy cube). Step back toward the light by the
    // bbox bounding-sphere diameter so the slab test below picks up the true
    // front-face entry from outside the box.
    float diag = length(float3(vol.extent)) * 2.5;
    float3 origin = in.world_pos - ray_dir * diag;

    float3 box_min = float3(vol.centre) - float3(vol.extent);
    float3 box_max = float3(vol.centre) + float3(vol.extent);
    float2 box_t = rayBox(origin, ray_dir, box_min, box_max);
    if (box_t.y < max(box_t.x, 0.0)) {
        discard_fragment();
    }
    float t_enter = max(box_t.x, 0.001);
    float t_max = min(box_t.y, vol.max_distance);
    if (t_enter >= t_max) {
        discard_fragment();
    }

    RayHit hit = coneRaymarch(origin, ray_dir, t_enter, t_max, vol, view.time);
    if (!hit.hit) {
        discard_fragment();
    }

    // Reproject the hit through the SAME cascade VP the vertex stage rasterised
    // with, so the depth write shares the rasterised casters' NDC depth space in
    // this slice. The `[[depth(less)]]` contract holds: the hit is bounded by
    // `t_max ≤ box exit`, so its NDC.z ≤ the back face's rasterised NDC.z.
    float3 hit_pos = origin + ray_dir * hit.t;
    float4 hit_clip = shadow.light_vps[cascade.cascade_idx] * float4(hit_pos, 1.0);
    ShadowFragOut o;
    o.depth = hit_clip.z / max(hit_clip.w, 1e-6);
    return o;
}
