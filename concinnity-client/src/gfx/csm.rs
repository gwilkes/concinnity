// src/gfx/csm.rs
//
// Cascaded shadow map cascade computation. Produces a `ShadowUniforms` carrying
// one orthographic light-view-projection matrix per cascade plus the view-space
// far depth for each cascade (the fragment shader uses these to select which
// cascade slice to sample).
//
// Algorithm (per-frame, called from each backend's draw loop):
//
//   1. Split the camera's [near, shadow_distance] depth range into N cascade
//      sub-ranges using the practical PSSM blend
//      (lambda * logarithmic + (1 - lambda) * linear).
//   2. For each cascade, compute the 8 frustum corners at its near/far depths,
//      then bound the corners with a sphere. The sphere bound makes the
//      orthographic light frustum rotation-invariant, eliminating shimmer
//      when the camera rotates.
//   3. Snap the sphere centre, in world space along the light's right/up axes,
//      to a per-cascade texel grid so the grid stays anchored in the world and
//      individual texels don't crawl as the camera translates.
//   4. Build a RH look_at from outside the sphere along the light direction
//      and an ortho projection that exactly encloses the sphere.
//
// The math is shared across all three backends: Metal, Vulkan, and DirectX
// all use RH view matrices with [0, 1] depth in their orthographic
// projections, so the same VPs are valid for every backend's shadow sampling.

use crate::gfx::render_types::{NUM_SHADOW_CASCADES, ShadowUniforms};

const SPLIT_LAMBDA: f32 = 0.5;

// Default shadow distance when none is supplied. The cascades cover from the
// camera near plane out to this distance.
pub const DEFAULT_SHADOW_DISTANCE: f32 = 80.0;

const IDENTITY4: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

// Fallback uniforms used when no shadow pass is active: identity VPs and a
// single split at +inf so the fragment shader always picks cascade 0 with a
// 1x1 fallback texture (returns "fully lit").
pub fn empty_shadow_uniforms() -> ShadowUniforms {
    ShadowUniforms {
        light_vps: [IDENTITY4; NUM_SHADOW_CASCADES],
        cascade_splits: [f32::INFINITY; NUM_SHADOW_CASCADES],
    }
}

// Compute cascade VPs + split depths from camera + light parameters.
//
// Arguments:
// - `view`: camera view matrix (column-major, RH, same convention as look_at).
// - `cam_pos`: world-space camera position.
// - `fov_y_rad`: vertical FOV in radians.
// - `aspect`: viewport aspect ratio (width / height).
// - `near`: camera near plane.
// - `shadow_distance`: far end of the last cascade. Cascades cover [near, shadow_distance].
// - `light_dir_to_source`: unit vector pointing TOWARD the light. Same convention
//   as `DirectionalLight.direction`; renormalised internally.
// - `shadow_map_size`: per-cascade texture resolution; used for texel snapping.
#[allow(clippy::too_many_arguments)]
pub fn compute_shadow_uniforms(
    view: [[f32; 4]; 4],
    cam_pos: [f32; 3],
    fov_y_rad: f32,
    aspect: f32,
    near: f32,
    shadow_distance: f32,
    light_dir_to_source: [f32; 3],
    shadow_map_size: u32,
) -> ShadowUniforms {
    let shadow_far = shadow_distance.max(near + 1.0);

    // Practical PSSM splits.
    let cascade_count = NUM_SHADOW_CASCADES as f32;
    let mut splits = [0.0_f32; NUM_SHADOW_CASCADES];
    for (i, split) in splits.iter_mut().enumerate() {
        let p = (i + 1) as f32 / cascade_count;
        let log = near * (shadow_far / near).powf(p);
        let lin = near + (shadow_far - near) * p;
        *split = SPLIT_LAMBDA * log + (1.0 - SPLIT_LAMBDA) * lin;
    }

    let l_to = normalize3(light_dir_to_source);

    // Camera basis from view matrix (column-major; look_at fills row 0 = right,
    // row 1 = up, row 2 = -forward into view[*][0], view[*][1], view[*][2]).
    let right = [view[0][0], view[1][0], view[2][0]];
    let up = [view[0][1], view[1][1], view[2][1]];
    let forward = [-view[0][2], -view[1][2], -view[2][2]];

    let tan_half_v = (fov_y_rad * 0.5).tan();
    let tan_half_h = tan_half_v * aspect;

    let mut light_vps = [IDENTITY4; NUM_SHADOW_CASCADES];
    let mut prev_split = near;
    for i in 0..NUM_SHADOW_CASCADES {
        let near_d = prev_split;
        let far_d = splits[i];
        prev_split = far_d;

        // 8 frustum corners at near_d and far_d in world space.
        let h_near = near_d * tan_half_v;
        let w_near = near_d * tan_half_h;
        let h_far = far_d * tan_half_v;
        let w_far = far_d * tan_half_h;
        let cn = add3(cam_pos, scale3(forward, near_d));
        let cf = add3(cam_pos, scale3(forward, far_d));
        let corners: [[f32; 3]; 8] = [
            add3(add3(cn, scale3(right, -w_near)), scale3(up, -h_near)),
            add3(add3(cn, scale3(right, w_near)), scale3(up, -h_near)),
            add3(add3(cn, scale3(right, w_near)), scale3(up, h_near)),
            add3(add3(cn, scale3(right, -w_near)), scale3(up, h_near)),
            add3(add3(cf, scale3(right, -w_far)), scale3(up, -h_far)),
            add3(add3(cf, scale3(right, w_far)), scale3(up, -h_far)),
            add3(add3(cf, scale3(right, w_far)), scale3(up, h_far)),
            add3(add3(cf, scale3(right, -w_far)), scale3(up, h_far)),
        ];

        // Bounding sphere of the corners.
        let mut centre = [0.0_f32; 3];
        for c in &corners {
            centre[0] += c[0];
            centre[1] += c[1];
            centre[2] += c[2];
        }
        centre = scale3(centre, 1.0 / 8.0);
        let mut r2 = 0.0_f32;
        for c in &corners {
            let d = sub3(*c, centre);
            let dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if dd > r2 {
                r2 = dd;
            }
        }
        let radius = r2.sqrt().max(1e-3);

        // Stable up axis: avoid parallel to light direction.
        let up_l = if l_to[1].abs() > 0.95 {
            [1.0_f32, 0.0, 0.0]
        } else {
            [0.0_f32, 1.0, 0.0]
        };

        // Light-space basis matching `look_at` below: f points from the light
        // toward the scene, r and u span the shadow texel grid.
        let f = scale3(l_to, -1.0);
        let r = normalize3(cross3(f, up_l));
        let u = cross3(r, f);

        // Texel-grid snap. Quantise the cascade centre along the light's right
        // and up axes to whole shadow texels so the texel grid stays anchored in
        // world space; the texels then stop crawling under camera translation
        // (the shadow stops chasing the camera). The snap must happen in world
        // space, before look_at: snapping the centre's light-space xy afterwards
        // is a no-op, because look_at always maps the centre onto the optical
        // axis (its light-space xy is identically zero).
        let texel_size = 2.0 * radius / shadow_map_size.max(1) as f32;
        let cx = dot3(r, centre);
        let cy = dot3(u, centre);
        let snap_dx = (cx / texel_size).round() * texel_size - cx;
        let snap_dy = (cy / texel_size).round() * texel_size - cy;
        let centre = add3(add3(centre, scale3(r, snap_dx)), scale3(u, snap_dy));

        // Build the light view from the snapped centre and an ortho projection
        // enclosing the sphere. The eye sits at +radius along the light
        // direction so the sphere centre maps to the middle of the depth range.
        //
        // The near plane is pushed back toward the light by `caster_extent` so
        // casters ABOVE this cascade's volume (tree canopies, tall building
        // tops, anything between the light and the sphere) still render into the
        // shadow map. Without it the near cascades, whose ortho boxes are only
        // `radius` deep along the light, clip any caster taller than the cascade;
        // as the camera moves, which casters fall inside each cascade changes,
        // so elevated shadows pop in and out and their edges slide with the
        // camera. The extension grows only the depth range, not the XY
        // footprint, so shadow-map resolution is unchanged.
        let caster_extent = shadow_far;
        let light_eye = add3(centre, scale3(l_to, radius));
        let light_view = look_at(light_eye, centre, up_l);
        let proj = ortho_rh(
            -radius,
            radius,
            -radius,
            radius,
            -caster_extent,
            2.0 * radius,
        );
        light_vps[i] = mat4_mul(proj, light_view);
    }

    ShadowUniforms {
        light_vps,
        cascade_splits: splits,
    }
}

// Right-handed look-at producing a column-major view matrix matching the
// per-backend look_at helpers (Metal, Vulkan, DX all use the same convention).
fn look_at(eye: [f32; 3], centre: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize3(sub3(centre, eye));
    let r = normalize3(cross3(f, up));
    let u = cross3(r, f);
    [
        [r[0], u[0], -f[0], 0.0],
        [r[1], u[1], -f[1], 0.0],
        [r[2], u[2], -f[2], 0.0],
        [-dot3(r, eye), -dot3(u, eye), dot3(f, eye), 1.0],
    ]
}

// Right-handed orthographic projection with depth mapped to [0, 1] -- shared
// by Metal, Vulkan, and DirectX.
fn ortho_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let rml = right - left;
    let tmb = top - bottom;
    let fmn = far - near;
    [
        [2.0 / rml, 0.0, 0.0, 0.0],
        [0.0, 2.0 / tmb, 0.0, 0.0],
        [0.0, 0.0, -1.0 / fmn, 0.0],
        [
            -(right + left) / rml,
            -(top + bottom) / tmb,
            -near / fmn,
            1.0,
        ],
    ]
}

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            for k in 0..4 {
                out[col][row] += a[k][row] * b[col][k];
            }
        }
    }
    out
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = dot3(v, v).sqrt().max(1e-6);
    [v[0] / len, v[1] / len, v[2] / len]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident_view() -> [[f32; 4]; 4] {
        // Camera at origin looking down -Z, up = +Y.
        look_at([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0])
    }

    #[test]
    fn empty_uniforms_have_infinite_splits() {
        let u = empty_shadow_uniforms();
        for s in &u.cascade_splits {
            assert!(s.is_infinite());
        }
    }

    #[test]
    fn splits_are_strictly_increasing_within_range() {
        let u = compute_shadow_uniforms(
            ident_view(),
            [0.0, 0.0, 0.0],
            std::f32::consts::FRAC_PI_2,
            1.0,
            0.1,
            80.0,
            [0.0, 1.0, 0.0],
            2048,
        );
        for i in 1..NUM_SHADOW_CASCADES {
            assert!(
                u.cascade_splits[i] > u.cascade_splits[i - 1],
                "splits must increase: {:?}",
                u.cascade_splits
            );
        }
        assert!(u.cascade_splits[0] > 0.1);
        assert!((u.cascade_splits[NUM_SHADOW_CASCADES - 1] - 80.0).abs() < 1e-3);
    }

    #[test]
    fn near_clamped_to_avoid_degenerate_log() {
        // shadow_distance smaller than near is clamped to near + 1.0 so the
        // logarithmic split term stays finite.
        let u = compute_shadow_uniforms(
            ident_view(),
            [0.0, 0.0, 0.0],
            std::f32::consts::FRAC_PI_2,
            1.0,
            5.0,
            1.0,
            [0.0, 1.0, 0.0],
            2048,
        );
        for s in &u.cascade_splits {
            assert!(s.is_finite() && *s > 0.0);
        }
    }

    #[test]
    fn cascade_vps_finite_for_typical_inputs() {
        let u = compute_shadow_uniforms(
            ident_view(),
            [10.0, 5.0, -3.0],
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            80.0,
            [-0.4, 0.7, 0.3],
            2048,
        );
        for vp in &u.light_vps {
            for col in vp {
                for v in col {
                    assert!(v.is_finite(), "non-finite element in light_vp");
                }
            }
        }
    }

    #[test]
    fn point_inside_first_cascade_projects_into_unit_box() {
        // A point a few metres in front of the camera should project into the
        // first cascade's light NDC, inside the [-1, 1] xy box (depth [0, 1]).
        let u = compute_shadow_uniforms(
            ident_view(),
            [0.0, 0.0, 0.0],
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            80.0,
            [0.0, 1.0, 0.0],
            2048,
        );
        // World point 2m in front of camera (looking down -Z).
        let p = [0.0_f32, 0.0, -2.0, 1.0];
        let vp = u.light_vps[0];
        let mut clip = [0.0_f32; 4];
        for row in 0..4 {
            clip[row] =
                vp[0][row] * p[0] + vp[1][row] * p[1] + vp[2][row] * p[2] + vp[3][row] * p[3];
        }
        let ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
        assert!(ndc[0].abs() <= 1.0, "x out of range: {}", ndc[0]);
        assert!(ndc[1].abs() <= 1.0, "y out of range: {}", ndc[1]);
        assert!(
            ndc[2] >= -0.05 && ndc[2] <= 1.05,
            "depth out of range: {}",
            ndc[2]
        );
    }

    // Project a fixed world point into cascade 0's shadow map and return its
    // texel coordinates for a camera at `cam` (looking down -Z).
    fn cascade0_texel(cam: [f32; 3], world_p: [f32; 3], light: [f32; 3], size: u32) -> (f32, f32) {
        let view = look_at(cam, [cam[0], cam[1], cam[2] - 1.0], [0.0, 1.0, 0.0]);
        let u = compute_shadow_uniforms(
            view,
            cam,
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            80.0,
            light,
            size,
        );
        let vp = u.light_vps[0];
        let p = [world_p[0], world_p[1], world_p[2], 1.0];
        let mut clip = [0.0_f32; 4];
        for row in 0..4 {
            clip[row] =
                vp[0][row] * p[0] + vp[1][row] * p[1] + vp[2][row] * p[2] + vp[3][row] * p[3];
        }
        let uvx = (clip[0] / clip[3] * 0.5 + 0.5) * size as f32;
        let uvy = (-clip[1] / clip[3] * 0.5 + 0.5) * size as f32;
        (uvx, uvy)
    }

    #[test]
    fn texels_do_not_crawl_under_camera_translation() {
        // The chasing-shadow regression. A fixed world point must land on the
        // SAME sub-texel of the shadow map regardless of camera position: with
        // the texel grid anchored in world space, translating the camera shifts
        // the projected point by a whole number of texels only, so its
        // fractional texel position is invariant. (Before the world-space snap,
        // the centre was snapped after look_at, which is a no-op, and the grid
        // slid continuously with the camera -- the shadow "chased" it.)
        let light = [-0.4, 0.78, 0.5];
        let size = 2048u32;
        let world_p = [3.0_f32, 0.0, -5.0];

        // Two camera positions a fraction of a texel apart in world space.
        let a = cascade0_texel([0.0, 2.0, 0.0], world_p, light, size);
        let b = cascade0_texel([0.137, 2.0, 0.091], world_p, light, size);

        // The texel delta must be (within float error) a whole number of
        // texels: the fractional residual is the per-texel crawl.
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        let rx = dx - dx.round();
        let ry = dy - dy.round();
        assert!(
            rx.abs() < 0.05,
            "shadow x crawls within a texel: residual {rx}"
        );
        assert!(
            ry.abs() < 0.05,
            "shadow y crawls within a texel: residual {ry}"
        );
    }

    // Project a world point through cascade 0's VP and return its light NDC.
    fn cascade0_ndc(cam: [f32; 3], world_p: [f32; 3], light: [f32; 3]) -> [f32; 3] {
        let view = look_at(cam, [cam[0], cam[1], cam[2] - 1.0], [0.0, 1.0, 0.0]);
        let u = compute_shadow_uniforms(
            view,
            cam,
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            80.0,
            light,
            2048,
        );
        let vp = u.light_vps[0];
        let p = [world_p[0], world_p[1], world_p[2], 1.0];
        let mut clip = [0.0_f32; 4];
        for row in 0..4 {
            clip[row] =
                vp[0][row] * p[0] + vp[1][row] * p[1] + vp[2][row] * p[2] + vp[3][row] * p[3];
        }
        [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]]
    }

    #[test]
    fn tall_casters_above_cascade_are_not_clipped() {
        // The disappearing-shadow regression. A caster high above the near
        // cascade (a tree canopy, a building top) must still fall inside the
        // cascade's light frustum so it renders into the shadow map; otherwise
        // its shadow vanishes the moment the receiver drops into a small near
        // cascade, and the clip boundary slides across the world as the camera
        // moves. The ortho near plane is extended toward the light for exactly
        // this. A caster 30m up sits well beyond the few-metre cascade-0 sphere
        // radius, so without the extension it projects to ndc.z < 0 (clipped).
        let cam = [0.0_f32, 0.0, 0.0];
        let light = [0.0_f32, 0.85, 0.3]; // mostly overhead
        let ndc = cascade0_ndc(cam, [0.0, 30.0, -3.0], light);
        assert!(
            ndc[0].abs() <= 1.0,
            "caster x outside footprint: {}",
            ndc[0]
        );
        assert!(
            ndc[1].abs() <= 1.0,
            "caster y outside footprint: {}",
            ndc[1]
        );
        assert!(
            (0.0..=1.0).contains(&ndc[2]),
            "tall caster clipped from shadow map: ndc.z = {}",
            ndc[2]
        );
    }
}
