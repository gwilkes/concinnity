#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D scene_tex;
layout(set = 0, binding = 1) uniform sampler2D velocity_tex;
layout(set = 0, binding = 2) uniform sampler2D history_tex;

layout(push_constant) uniform TaaBlock {
    // 0 on the first frame and after a resize - history is then ignored.
    float history_valid;
} taa;

// History blend weight. 0.9 keeps 90% of the accumulated history each frame -
// roughly a 10-frame exponential moving average once converged.
const float TAA_BLEND = 0.9;
// Standard deviations of the neighbourhood the history is allowed to span.
const float TAA_VARIANCE_GAMMA = 1.0;
// Finite ceiling every sampled colour is clamped to (largest finite half).
const float TAA_HDR_CLAMP = 65504.0;

// Scrub a non-finite sample: NaN -> 0, +Inf -> the HDR ceiling, negative -> 0.
vec3 taa_sanitize(vec3 c) {
    c = mix(c, vec3(0.0), bvec3(isnan(c)));
    return clamp(c, vec3(0.0), vec3(TAA_HDR_CLAMP));
}

// RGB <-> YCoCg. The clip box is built in YCoCg so its luma axis aligns with
// perceived error - a tighter, better-oriented box than an RGB AABB.
vec3 rgb_to_ycocg(vec3 c) {
    return vec3(
         0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
         0.5  * c.r            - 0.5  * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
vec3 ycocg_to_rgb(vec3 c) {
    float t = c.x - c.z;
    return vec3(t + c.y, c.x + c.z, t - c.y);
}

// Clip the history sample to the neighbourhood box along the line toward the
// box centre (Karis 2014) - preserves colour direction, so hue shifts less.
vec3 clip_to_aabb(vec3 bmin, vec3 bmax, vec3 hist) {
    vec3 center = 0.5 * (bmax + bmin);
    vec3 extent = 0.5 * (bmax - bmin) + 1e-5;
    vec3 v = hist - center;
    vec3 a = abs(v) / extent;
    float ma = max(a.x, max(a.y, a.z));
    return (ma > 1.0) ? (center + v / ma) : hist;
}

void main() {
    vec2 uv = frag_uv;
    vec2 texel = 1.0 / vec2(textureSize(scene_tex, 0));
    vec3 cur = taa_sanitize(texture(scene_tex, uv).rgb);

    // 3x3 neighbourhood statistics in YCoCg; the reprojected history is clipped
    // to mean +/- gamma*stddev.
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            vec3 s = taa_sanitize(texture(scene_tex, uv + vec2(dx, dy) * texel).rgb);
            vec3 c = rgb_to_ycocg(s);
            m1 += c;
            m2 += c * c;
        }
    }
    vec3 mean  = m1 / 9.0;
    vec3 sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3(0.0)));
    vec3 bmin  = mean - TAA_VARIANCE_GAMMA * sigma;
    vec3 bmax  = mean + TAA_VARIANCE_GAMMA * sigma;

    // The velocity pre-pass stored each surface's screen-space motion as the
    // offset that maps a current-frame UV onto its previous-frame UV.
    vec2 motion  = texture(velocity_tex, uv).rg;
    vec2 prev_uv = uv + motion;
    bool on_screen = all(greaterThanEqual(prev_uv, vec2(0.0)))
                  && all(lessThanEqual(prev_uv, vec2(1.0)));

    vec3 hist = rgb_to_ycocg(taa_sanitize(texture(history_tex, prev_uv).rgb));
    hist = clip_to_aabb(bmin, bmax, hist);

    // Accumulate only when there is valid, on-screen history; otherwise the
    // current frame passes straight through (first frame, resize, off-screen).
    float alpha = (taa.history_valid > 0.5 && on_screen) ? TAA_BLEND : 0.0;
    out_color = vec4(mix(cur, ycocg_to_rgb(hist), alpha), 1.0);
}
