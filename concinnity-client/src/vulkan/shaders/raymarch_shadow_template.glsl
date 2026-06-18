// src/vulkan/shaders/raymarch_shadow_template.glsl
//
// Engine-shipped depth-only fragment template for raymarched SDF shadow casters
// on Vulkan. Appended to the user's fragment shader (after the helpers + the
// user's `map` / `shade`) when building the per-volume SHADOW pipeline. The
// user's `shade` and the IBL / scene-colour helpers are dead here (only `map`
// is reached through `coneRaymarch`), so spirv-opt strips them and the shadow
// descriptor set binds just the view / light / shadow UBOs. GLSL port of
// directx/shaders/raymarch_shadow.hlsl / metal/shaders/raymarch_shadow.metal:
// same control flow, same depth-reprojection rule.
//
// One pipeline per `SdfVolume.cast_shadows == true`. The encoder draws the proxy
// back faces once per CSM cascade into that cascade's shadow-map slice, marching
// the SDF from the light side and writing the hit's NDC depth via
// `gl_FragDepth` redeclared `depth_less`. The cascade slice's LESS depth test
// composites the raymarched caster with the rasterised casters already drawn
// into it: the nearer occluder wins per texel.

layout(location = 0) in vec3 v_world_pos;

layout(depth_less) out float gl_FragDepth;

layout(push_constant) uniform ShadowCascade {
    uint cascade_idx;
} pc;

void main() {
    // Directional sun: lights.dir[0].dir_i.xyz is L (surface -> light); incoming
    // light travels along -L, so the shadow ray marches along -L. Matches
    // shadePbrSun so SDF shadows line up with the lit-side surface.
    vec3 ray_dir = -normalize(lights.dir[0].dir_i.xyz);

    // v_world_pos is on the bbox face farthest from the light (front faces are
    // culled). Step back toward the light by the bbox bounding-sphere diameter
    // so the slab test picks up the true front-face entry from outside the box.
    float diag = length(vol.vol_extent) * 2.5;
    vec3 origin = v_world_pos - ray_dir * diag;

    vec3 box_min = vol.vol_centre - vol.vol_extent;
    vec3 box_max = vol.vol_centre + vol.vol_extent;
    vec2 box_t = rayBox(origin, ray_dir, box_min, box_max);
    if (box_t.y < max(box_t.x, 0.0)) {
        discard;
    }
    float t_enter = max(box_t.x, 0.001);
    float t_max = min(box_t.y, vol.vol_max_distance);
    if (t_enter >= t_max) {
        discard;
    }

    RayHit hit = coneRaymarch(origin, ray_dir, t_enter, t_max, rmview.view_time);
    if (!hit.hit) {
        discard;
    }

    // Reproject the hit through the same cascade VP the vertex rasterised with,
    // so the depth write shares the rasterised casters' NDC depth space.
    vec3 hit_pos = origin + ray_dir * hit.t;
    vec4 hit_clip = shadow_uni.light_vps[pc.cascade_idx] * vec4(hit_pos, 1.0);
    gl_FragDepth = hit_clip.z / max(hit_clip.w, 1e-6);
}
