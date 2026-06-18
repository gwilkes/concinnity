#include <metal_stdlib>
using namespace metal;

// --- Projected (deferred) decal pass ---
//
// Runs after the main HDR pass. For each decal the vertex shader draws a unit
// cube (positions in [-0.5, 0.5]^3 in local space) transformed by the decal's
// world model matrix and the camera VP. The fragment shader reconstructs the
// world-space sample point of the rasterised pixel from the main pass's MSAA
// depth attachment, transforms it back into the decal's local space, and tests
// whether the point lies inside the unit cube - discarding when it does not.
// Pixels nearer the top / bottom face along local +Y are faded out so the
// stamp does not show a hard edge where it meets a tilted or curved surface.
//
// The composited colour is alpha-blended into the resolved HDR target via the
// pipeline's blend state (Src.A · Src + (1 - Src.A) · Dst).

struct DecalView {
    float4x4 vp;       // jittered or un-jittered view-projection (the main pass)
    float4x4 inv_vp;   // inverse of the same
    float2 viewport;   // attachment width, height in pixels
    float2 _pad;
};

struct DecalParams {
    float4x4 model;      // local → world
    float4x4 inv_model;  // world → local
    float4 tint;         // linear RGB × alpha
    float fade_pow;      // exponent on the soft-fade along local +Y (default 2)
    float _pad0;
    float _pad1;
    float _pad2;
};

struct DecalVtxIn {
    // Unit cube vertex; the bound buffer carries 8 vertices in [-0.5, 0.5]^3.
    float3 pos [[attribute(0)]];
};

struct DecalVtxOut {
    float4 position [[position]];
};

vertex DecalVtxOut decal_vertex(
    DecalVtxIn        in   [[stage_in]],
    constant DecalView   &v [[buffer(0)]],
    constant DecalParams &p [[buffer(1)]]
) {
    DecalVtxOut out;
    float4 world = p.model * float4(in.pos, 1.0);
    out.position = v.vp * world;
    return out;
}

fragment float4 decal_fragment(
    DecalVtxOut             in        [[stage_in]],
    constant DecalView    &v          [[buffer(0)]],
    constant DecalParams  &p          [[buffer(1)]],
    depth2d<float>         scene_depth [[texture(0)]],
    texture2d<float>       decal_tex   [[texture(1)]],
    sampler                samp        [[sampler(0)]]
) {
    // The single-sample `depth_resolve` shares the rasterised pixel grid with
    // the composite target, so we can sample at integer texel coords directly.
    uint2 pixel = uint2(in.position.xy);
    if (pixel.x >= uint(v.viewport.x) || pixel.y >= uint(v.viewport.y)) {
        discard_fragment();
    }
    // 1.0 is the cleared / "no geometry" sentinel - the main pass left this
    // pixel empty (sky was painted by default.metal's skybox sentinel and
    // writes ~far-plane depth instead). Nothing to project onto.
    float depth = scene_depth.read(pixel);
    if (depth >= 1.0) {
        discard_fragment();
    }

    // Reconstruct the world-space point at this pixel via the inverse VP.
    // gl_FragCoord-style position: (px + 0.5) / viewport → NDC [-1, 1].
    float2 ndc_xy = (in.position.xy / v.viewport) * 2.0 - 1.0;
    // Metal flips Y in the framebuffer relative to NDC: a fragment whose y is
    // 0 sits at the top of the screen but at +1 in NDC space.
    ndc_xy.y = -ndc_xy.y;
    float4 clip = float4(ndc_xy, depth, 1.0);
    float4 world = v.inv_vp * clip;
    world /= world.w;

    // Transform into decal-local space and clip against the unit box.
    float4 local = p.inv_model * world;
    float3 ab = abs(local.xyz);
    if (ab.x > 0.5 || ab.y > 0.5 || ab.z > 0.5) {
        discard_fragment();
    }

    // Soft fade along the projection axis (local +Y) so a decal does not show
    // a hard band where the surface tilts away from the projection plane. The
    // fade rolls off the alpha as local |y| approaches 0.5.
    float fade = saturate(1.0 - (ab.y * 2.0));
    fade = pow(fade, max(p.fade_pow, 1.0));

    // Sample the decal texture on local X-Z. UV is in [0, 1].
    float2 uv = local.xz + 0.5;
    // Metal samples with V=0 at the top of the texture, matching the rest of
    // the engine's textures (default.metal flips V for the same reason).
    uv.y = 1.0 - uv.y;
    float4 tex = decal_tex.sample(samp, uv);

    float4 c;
    c.rgb = tex.rgb * p.tint.rgb;
    c.a = tex.a * p.tint.a * fade;
    return c;
}
