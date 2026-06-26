#version 450

// SSR resolve fragment shader: a fullscreen ray-march over the pre-pass
// G-buffer + roughness texture, compositing the reflected scene colour over
// `scene`. Translated 1:1 from src/directx/shaders/ssr_resolve_frag.hlsl,
// which in turn mirrors src/metal/shaders/ssr.metal::ssr_resolve_fragment.

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform SsrParamsBlock {
    float intensity;
    float max_distance;
    float tan_half_fov_y;
    float aspect;
    float stride;
    float thickness;
    // IBL prefilter cubemap mip count; 0 means no EnvironmentMap is bound and
    // the cube fallback is skipped (missed rays keep the base shading).
    float prefilter_mip_count;
    float _pad;
    // Camera-to-world transform: the 3x3 turns the view-space reflection ray into
    // the world-space direction the prefilter cubemap is sampled with, and the
    // translation column (the world camera position) lifts the view-space surface
    // point to world space for the reflection-probe box projection on a missed ray.
    mat4 inv_view;
} params;

layout(set = 0, binding = 0) uniform sampler2D   scene;
layout(set = 0, binding = 1) uniform sampler2D   gbuffer;
layout(set = 0, binding = 2) uniform sampler2D   rough_tex;
layout(set = 0, binding = 3) uniform samplerCube prefilter;

// set 1: the forward global set, bound here only for its reflection-probe count +
// per-probe parallax boxes + cube array; a screen-space ray that escapes the frame
// falls back to the local probe instead of the foreign sky cube. The shared probe
// sampling is substituted in below at set index 1 (the marker token must not appear
// in this comment, or it would be substituted here too).
{PROBE_COMMON}

const int   SSR_MAX_STEPS = 48;
const int   SSR_REFINE    = 5;
// Surfaces rougher than this get no SSR; glossiness ramps in below it.
const float SSR_ROUGH_CUT = 0.6;
// Dielectric base reflectance (water, glass, polished stone) for the Fresnel.
const float SSR_F0        = 0.04;
// UV margin over which a hit near the screen border fades out.
const float SSR_EDGE_FADE = 0.12;

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// The inverse of ssr_project; matches the SSAO kernel's `ssao_view_pos`.
vec3 ssr_view_pos(vec2 uv, float depth, float tan_y, float asp) {
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return vec3(ndc.x * tan_y * asp, ndc.y * tan_y, -1.0) * depth;
}

// Project a view-space point (z < 0, in front of the camera) to a screen UV.
vec2 ssr_project(vec3 q, float tan_y, float asp) {
    float inv = 1.0 / max(-q.z, 1e-4);
    vec2 ndc = vec2(q.x * inv / (tan_y * asp), q.y * inv / tan_y);
    return vec2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

void main() {
    vec3 base  = texture(scene, frag_uv).rgb;
    vec4 c     = texture(gbuffer, frag_uv);
    float depth = c.a;
    // Background / sky, or a non-reflecting (too-rough) surface: weight 0 so the
    // reflection composite keeps the scene there. The resolve no longer blends
    // inline; it writes reflected radiance (.rgb) + composite weight (.a).
    if (depth <= 0.0) {
        out_color = vec4(base, 0.0);
        return;
    }

    float roughness = texture(rough_tex, frag_uv).r;
    // Glossy surfaces reflect sharply; rough ones get nothing.
    float gloss = clamp((SSR_ROUGH_CUT - roughness) / SSR_ROUGH_CUT, 0.0, 1.0);
    if (gloss <= 0.0) {
        out_color = vec4(base, 0.0);
        return;
    }

    vec3 N = normalize(c.xyz);
    vec3 P = ssr_view_pos(frag_uv, depth, params.tan_half_fov_y, params.aspect);
    vec3 V = normalize(-P);                       // P in view space, camera at origin
    vec3 R = reflect(-V, N);                      // reflected ray direction

    // Environment fallback for a missed (or screen-edge) ray, in the reflected
    // direction at a roughness-keyed mip so a rougher surface reflects a blurrier
    // environment (matching the main pass). With a baked reflection probe this is
    // the local scene capture (box-projected + blended across covering probes), the
    // same source the forward IBL specular term uses, rather than the foreign sky
    // HDR; otherwise it is the IBL prefilter cube. With no EnvironmentMap bound
    // there is nothing to fall back to, so missed rays keep the base shading.
    bool ibl = params.prefilter_mip_count > 0.5;
    vec3 env = base;
    if (ibl) {
        vec3 r_world = mat3(params.inv_view) * R;
        float lod    = roughness * (params.prefilter_mip_count - 1.0);
        if (probe_set.count > 0u) {
            // The full inv_view (its translation column carries the camera
            // position) lifts the view-space surface point P to world space, which
            // the probe box-projection needs.
            vec3 world_pos = (params.inv_view * vec4(P, 1.0)).xyz;
            env = probe_set_specular(world_pos, r_world, lod);
        } else {
            env = textureLod(prefilter, r_world, lod).rgb;
        }
    }

    vec3 step_v = R * params.stride;
    vec3 q = P;
    bool hit = false;
    vec2 hit_uv = frag_uv;
    int  steps_taken = SSR_MAX_STEPS;
    for (int i = 0; i < SSR_MAX_STEPS; i++) {
        q += step_v;
        if (q.z >= 0.0) { steps_taken = i; break; } // crossed the camera plane
        vec2 uv = ssr_project(q, params.tan_half_fov_y, params.aspect);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            steps_taken = i;
            break;
        }
        float scene_depth = texture(gbuffer, uv).a;
        if (scene_depth <= 0.0) continue;          // sky here - keep marching
        float diff = (-q.z) - scene_depth;         // > 0: ray is behind the surface
        if (diff > 0.0 && diff < params.thickness) {
            // Binary-search refine between the last two samples.
            vec3 lo = q - step_v;
            vec3 hi = q;
            for (int r = 0; r < SSR_REFINE; r++) {
                vec3 mid = (lo + hi) * 0.5;
                vec2 muv = ssr_project(mid, params.tan_half_fov_y, params.aspect);
                float sd = texture(gbuffer, muv).a;
                if (sd > 0.0 && (-mid.z) - sd > 0.0) hi = mid;
                else                                 lo = mid;
            }
            hit_uv = ssr_project(hi, params.tan_half_fov_y, params.aspect);
            hit = true;
            steps_taken = i;
            break;
        }
    }

    // The reflected colour: a single sharp screen-space tap (the reflection
    // composite blurs it by roughness), or the environment cube when the ray
    // missed. A hit near the screen border or at the end of its march fades toward
    // the environment rather than snapping flat to the base shading.
    vec3 reflected;
    if (hit) {
        vec3 hit_color = texture(scene, hit_uv).rgb;
        vec2 e = smoothstep(vec2(0.0), vec2(SSR_EDGE_FADE), hit_uv)
               * smoothstep(vec2(0.0), vec2(SSR_EDGE_FADE), vec2(1.0) - hit_uv);
        float edge = e.x * e.y;
        float march = float(steps_taken) / float(SSR_MAX_STEPS);
        float dist_fade = 1.0 - smoothstep(0.7, 1.0, march);
        reflected = mix(env, hit_color, edge * dist_fade);
    } else {
        reflected = env;
    }

    float ndv     = clamp(dot(N, V), 0.0, 1.0);
    float fresnel = SSR_F0 + (1.0 - SSR_F0) * pow(1.0 - ndv, 5.0);
    float w = clamp(fresnel * gloss * params.intensity, 0.0, 1.0);
    // Reflected radiance + composite weight, not yet blended; the reflection
    // composite pass blurs by roughness and composites this over the scene.
    out_color = vec4(reflected, w);
}
