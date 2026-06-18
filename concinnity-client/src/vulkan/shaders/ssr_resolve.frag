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
    // View-space → world-space rotation; turns the view-space reflection ray
    // into the world-space direction the prefilter cubemap is sampled with.
    mat4 inv_view_rot;
} params;

layout(set = 0, binding = 0) uniform sampler2D   scene;
layout(set = 0, binding = 1) uniform sampler2D   gbuffer;
layout(set = 0, binding = 2) uniform sampler2D   rough_tex;
layout(set = 0, binding = 3) uniform samplerCube prefilter;

const int   SSR_MAX_STEPS = 48;
const int   SSR_REFINE    = 5;
// Surfaces rougher than this get no SSR; glossiness ramps in below it.
const float SSR_ROUGH_CUT = 0.6;
// Dielectric base reflectance (water, glass, polished stone) for the Fresnel.
const float SSR_F0        = 0.04;
// UV margin over which a hit near the screen border fades out.
const float SSR_EDGE_FADE = 0.12;
// Largest screen-space (UV) blur radius, reached as roughness approaches the
// cut-off. The reflected scene colour is gathered over a disk this wide so a
// glossy-but-not-mirror surface reflects a blurred image; a near-zero
// roughness shrinks the radius to a single sharp tap.
const float SSR_BLUR_MAX  = 0.018;

// Eight evenly spaced offsets on the unit circle (cos/sin of k * 45 deg).
const vec2 SSR_BLUR_RING[8] = vec2[8](
    vec2( 1.0,         0.0       ), vec2( 0.70710678,  0.70710678),
    vec2( 0.0,         1.0       ), vec2(-0.70710678,  0.70710678),
    vec2(-1.0,         0.0       ), vec2(-0.70710678, -0.70710678),
    vec2( 0.0,        -1.0       ), vec2( 0.70710678, -0.70710678)
);

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

// Gather the reflected scene colour over a roughness-scaled disk. A zero
// radius is one sharp tap (mirror); a wider radius averages an eight-tap ring
// plus the centre, approximating the wider reflection cone of a rough surface.
vec3 ssr_gather(vec2 uv, float radius) {
    vec3 c = texture(scene, uv).rgb;
    if (radius <= 1e-5) {
        return c;
    }
    for (int i = 0; i < 8; i++) {
        c += texture(scene, uv + SSR_BLUR_RING[i] * radius).rgb;
    }
    return c * (1.0 / 9.0);
}

void main() {
    vec3 base  = texture(scene, frag_uv).rgb;
    vec4 c     = texture(gbuffer, frag_uv);
    float depth = c.a;
    if (depth <= 0.0) {
        out_color = vec4(base, 1.0);
        return;
    }

    float roughness = texture(rough_tex, frag_uv).r;
    // Glossy surfaces reflect sharply; rough ones get nothing.
    float gloss = clamp((SSR_ROUGH_CUT - roughness) / SSR_ROUGH_CUT, 0.0, 1.0);
    if (gloss <= 0.0) {
        out_color = vec4(base, 1.0);
        return;
    }

    vec3 N = normalize(c.xyz);
    vec3 P = ssr_view_pos(frag_uv, depth, params.tan_half_fov_y, params.aspect);
    vec3 V = normalize(-P);                       // P in view space, camera at origin
    vec3 R = reflect(-V, N);                      // reflected ray direction

    // Environment fallback: the reflection the IBL prefilter cubemap gives in
    // the reflected direction, sampled at a roughness-keyed mip so a rougher
    // surface reflects a blurrier environment (matching the main pass). With
    // no EnvironmentMap bound there is nothing to fall back to, so the
    // environment stays the base shading and missed rays behave as before.
    bool ibl = params.prefilter_mip_count > 0.5;
    vec3 env = base;
    if (ibl) {
        vec3 r_world = mat3(params.inv_view_rot) * R;
        float lod    = roughness * (params.prefilter_mip_count - 1.0);
        env = textureLod(prefilter, r_world, lod).rgb;
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

    // The reflected colour: a screen-space hit gathered over a roughness-scaled
    // disk, or the environment cube when the ray missed. A hit near the screen
    // border or at the end of its march fades toward the environment rather
    // than snapping flat to the base shading.
    float blur_radius = (roughness / SSR_ROUGH_CUT) * SSR_BLUR_MAX;
    vec3 reflected;
    if (hit) {
        vec3 hit_color = ssr_gather(hit_uv, blur_radius);
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
    out_color = vec4(mix(base, reflected, w), 1.0);
}
