// src/directx/shaders/raymarch_template.hlsl
//
// Engine-shipped template for raymarched SDF volumes on D3D12. Appended
// to the user's fragment shader at compile time (after the helpers + the
// user's `map` / `shade` definitions). HLSL port of
// src/metal/shaders/raymarch_template.metal - same control flow, same
// depth-compositing rules, same shading sequence.
//
// `raymarch_vertex` rasterises the back faces of the bounding-box proxy
// (the encoder's cull state culls front faces). Each output fragment is
// a candidate for a ray that pierces the box.
//
// `raymarch_fragment` reconstructs the world-space ray, calls
// `coneRaymarch` + the user's `map` / `shade`, applies the engine PBR +
// ambient helpers, and writes opaque colour into the bound MSAA `hdr_color`
// attachment. Hit depth flows out through `SV_DepthLessEqual` (HLSL's
// analogue of Metal's `[[depth(less)]]`) so the hardware depth test
// against the existing MSAA depth discards hits behind rasterised
// geometry; the same SV_DepthLessEqual write feeds raymarched-surface
// depth back into the MSAA depth buffer so downstream passes that
// sample main depth (decals, fog, SSR) see the SDF, not the geometry
// behind it. The encoder re-resolves `hdr_color → hdr_resolve` after
// this pass so single-sample post-stack passes (Decals, Fog,
// SsrResolve, TaaResolve, Bloom, Composite) pick up the raymarched
// pixels naturally.
//
// Resource bindings (matching `directx/raymarch.rs`):
//
//   b0 - RaymarchView          (per-frame view)
//   b1 - SdfVolumeUniforms     (per-volume, allocated once at init)
//   b2 - RaymarchLights        (lights - shared LightUniforms cbuffer)
//   b3 - RaymarchShadowUniforms (CSM uniforms - shared shadow cbuffer)
//
//   t0 - shadow_map     (Texture2DArray<float>)
//   t1 - irradiance     (TextureCube<float4>)
//   t2 - prefilter      (TextureCube<float4>)
//   t3 - scene_color    (Texture2D<float4>, `hdr_resolve_copy` snapshot
//                        taken at the head of the pass - `sampleSceneRefracted`
//                        in the helpers reads from this for refraction)
//
//   s0 - shadow_samp    (SamplerComparisonState, LESS_EQUAL)
//   s1 - cube_samp      (SamplerState linear-clamp)
//   s2 - scene_samp     (SamplerState linear-clamp)

Texture2DArray<float> shadow_map   : register(t0);
TextureCube<float4>   irradiance_c : register(t1);
TextureCube<float4>   prefilter_c  : register(t2);
Texture2D<float4>     scene_color  : register(t3);

SamplerComparisonState shadow_samp : register(s0);
SamplerState           cube_samp   : register(s1);
SamplerState           scene_samp  : register(s2);

struct VsIn
{
    float3 pos      : POSITION;
    float3 normal   : NORMAL;
    float3 tangent  : TANGENT;
    float3 color    : COLOR;
    float2 uv       : TEXCOORD0;
};

struct VsOut
{
    // `noperspective centroid` is required by the D3D12 spec when the
    // pixel shader writes `SV_DepthLessEqual` (or `SV_DepthGreaterEqual`)
    // and is not running at sample frequency - without it FXC emits
    // "X8000: Interpolation mode for PS input position must be
    // linear_noperspective_centroid or linear_noperspective_sample".
    noperspective centroid float4 sv_pos : SV_POSITION;
    float3 world_pos : WORLD_POS;
};

VsOut raymarch_vertex(VsIn v)
{
    // The proxy buffer is a unit cube with positions at ±1; the encoder
    // scales by `vol_extent` to land at the AABB corners (see
    // `directx/raymarch.rs::build_raymarch_cube_buffers`).
    float3 wp = v.pos * vol_extent + vol_centre;
    VsOut o;
    o.sv_pos = mul(view_vp, float4(wp, 1.0));
    o.world_pos = wp;
    return o;
}

struct PsOut
{
    float4 color : SV_TARGET0;
    // `SV_DepthLessEqual` lets the hardware preserve early-Z while still
    // letting the shader write a smaller depth - analogous to Metal's
    // `[[depth(less)]]`. Hit fragments whose computed depth is behind
    // the rasterised value get discarded; hits in front overwrite both
    // colour and depth so the re-resolve at the tail of the pass
    // propagates the raymarched-surface depth into downstream
    // depth-sampling consumers (decals, fog, SSR).
    float  depth : SV_DepthLessEqual;
};

PsOut raymarch_fragment(VsOut input)
{
    // Build the world-space ray from camera through this fragment.
    float3 cam = view_cam_pos;
    float3 ray_dir = normalize(input.world_pos - cam);

    // Clip the ray to the volume's AABB. The vertex shader rasterised
    // the back faces, so `input.world_pos` lies on the far side of the
    // box; using the slab test gives both enter + exit and handles the
    // camera-inside-box case uniformly (t_enter clamps to 0 below).
    float3 box_min = vol_centre - vol_extent;
    float3 box_max = vol_centre + vol_extent;
    float2 box_t = rayBox(cam, ray_dir, box_min, box_max);
    if (box_t.y < max(box_t.x, 0.0))
    {
        discard;
    }
    float t_enter = max(box_t.x, 0.001);
    // Hardware depth test against the writable DSV handles compositing
    // against rasterised geometry - fragments whose hit lands behind
    // existing depth get discarded by the LESS_EQUAL comparison applied
    // to `SV_DepthLessEqual` output. `t_max` is therefore just the
    // closer of bbox exit and per-volume far-clip.
    float t_max = min(box_t.y, vol_max_distance);
    if (t_enter >= t_max)
    {
        discard;
    }

    RayHit hit = coneRaymarch(cam, ray_dir, t_enter, t_max, view_time);
    if (!hit.hit)
    {
        discard;
    }

    float3 hit_pos = cam + ray_dir * hit.t;
    float3 normal = sdfNormal(hit_pos, vol_params, view_time, 0.001);
    float2 frag_uv = input.sv_pos.xy / view_viewport;
    SdfSurface surf = shade(hit_pos, normal, vol_params, view_time,
                            frag_uv, scene_color, scene_samp);

    // IBL ambient (with hemispheric fallback inside the helper) +
    // CSM-shadowed first directional light.
    float3 view_dir = -ray_dir;
    float3 color = shadeAmbientIbl(
        surf, normal, view_dir,
        view_prefilter_mip_count,
        irradiance_c, prefilter_c, cube_samp);
    if (light_num_directional > 0)
    {
        float shadow_factor = 1.0;
        if (vol_receive_shadows != 0)
        {
            shadow_factor = sampleSunShadow(
                hit_pos, hit.t, input.sv_pos.xy,
                shadow_map, shadow_samp);
        }
        color += shadePbrSun(surf, normal, view_dir,
                             light_directional[0], shadow_factor);
    }
    color += surf.transmitted;

    // Reproject the hit position through view_vp for SV_DepthLessEqual.
    // Same `view.vp` the vertex shader rasterised with, so the depth
    // shares the rasterised geometry's NDC depth space exactly.
    float4 hit_clip = mul(view_vp, float4(hit_pos, 1.0));
    PsOut o;
    o.color = float4(color, 1.0);
    o.depth = hit_clip.z / max(hit_clip.w, 1e-6);
    return o;
}
