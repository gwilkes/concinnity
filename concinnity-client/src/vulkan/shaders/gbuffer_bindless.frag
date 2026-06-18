#version 450

// GPU-driven G-buffer pre-pass fragment shader. Bindless sibling of
// gbuffer_prepass.frag: identical MRT outputs, but roughness rides a flat
// vertex-shader varying (sourced from GpuObjectData) instead of a push constant.
//   color(0) = RGBA16F: rgb = unit view-space normal, a = positive linear view
//              depth (-view_z). Alpha 0 marks "no geometry" (cleared background).
//   color(1) = R8_UNORM perceptual roughness (1.0 = fully rough background).
//   color(2) = RG16F screen-space motion (prev_uv - cur_uv), derived from the
//              un-jittered clip positions so projection jitter never leaks in.

layout(location = 0) in vec3 frag_view_normal;
layout(location = 1) in float frag_view_depth;
layout(location = 2) in vec4 cur_clip;
layout(location = 3) in vec4 prev_clip;
layout(location = 4) flat in float frag_roughness;

layout(location = 0) out vec4 out_nd;
layout(location = 1) out float out_rough;
layout(location = 2) out vec2 out_vel;

void main() {
    out_nd    = vec4(normalize(frag_view_normal), frag_view_depth);
    out_rough = frag_roughness;
    vec2 cur_ndc  = cur_clip.xy  / cur_clip.w;
    vec2 prev_ndc = prev_clip.xy / prev_clip.w;
    // Image-space UV with 0 = top, matching the negative-height viewport this
    // pass shares with the main pass and the upright resolve the readers sample.
    vec2 cur_uv  = vec2(cur_ndc.x  * 0.5 + 0.5, 0.5 - cur_ndc.y  * 0.5);
    vec2 prev_uv = vec2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    out_vel = prev_uv - cur_uv;
}
