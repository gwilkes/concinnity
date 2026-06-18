// src/gfx/ssgi.rs
//
// Screen-space global illumination (SSGI) configuration. Backend-agnostic
// resolve of the authored `PostProcessConfig` SSGI fields into clamped
// settings, plus the per-frame GPU uniform. SSGI is a refinement of SSR: it
// reuses the same depth + normal pre-pass G-buffer and screen-space ray-march,
// but integrates bounced radiance over a cosine-weighted hemisphere instead of
// along a single reflection vector, and adds the result on top of the IBL
// ambient term. The hemisphere gather itself lives in each backend's shader;
// this module owns only the parameter math so it can be unit-tested without a
// GPU.

use crate::gfx::render_types::SsgiParams;

// Upper bound on `intensity`. The composite pass adds the gathered indirect
// radiance on top of the existing shading, so this is an additive multiplier
// rather than a `[0, 1]` blend: values above 1 exaggerate the bounce.
const MAX_INTENSITY: f32 = 4.0;

// Smallest usable march distance: a ray shorter than this finds nothing.
const MIN_DISTANCE: f32 = 0.5;
// Largest march distance. SSGI is a near-field effect (the far field is the
// IBL term's job), so the reach is capped well below SSR's.
const MAX_DISTANCE: f32 = 100.0;

// Hemisphere rays cast per pixel, clamped to a sane range. More rays trade
// performance for a smoother, less noisy gather. The default matches the
// historical compile-time count and is the source of truth for the
// `PostProcessConfig` default.
pub const DEFAULT_RAYS: u32 = 8;
const MIN_RAYS: u32 = 1;
const MAX_RAYS: u32 = 32;

// Ray-march samples taken per ray. The step length is `max_distance / steps`,
// so a longer ray spends a longer stride rather than more samples. The default
// matches the historical compile-time count and is the source of truth for the
// `PostProcessConfig` default.
pub const DEFAULT_STEPS: u32 = 12;
const MIN_STEPS: u32 = 1;
const MAX_STEPS: u32 = 64;

// View-space intersection tolerance as a multiple of the march stride. A ray
// point is a hit when it lands behind the scene surface by less than this:
// wide enough to catch a crossing between two samples, tight enough not to
// punch through thin geometry.
const THICKNESS_SCALE: f32 = 2.0;

// Floor on the aspect ratio so a degenerate viewport cannot divide by zero.
const MIN_ASPECT: f32 = 1.0e-3;

// Clamped SSGI tunables resolved from the authored asset fields. Held by the
// backend and turned into a per-frame [`SsgiParams`] once the camera is known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsgiSettings {
    // Indirect-bounce blend strength multiplier in `[0, MAX_INTENSITY]`.
    pub intensity: f32,
    // World-space distance a hemisphere ray marches before giving up.
    pub max_distance: f32,
    // Hemisphere rays cast per pixel, clamped to `[MIN_RAYS, MAX_RAYS]`.
    pub rays: u32,
    // Ray-march samples per ray, clamped to `[MIN_STEPS, MAX_STEPS]`.
    pub steps: u32,
    // Render-resolution divisor for the gather target: 1 is full resolution,
    // 2 is half (a quarter of the pixels), 4 a quarter. The composite pass is a
    // depth-aware bilateral filter, so it upsamples the lower-resolution gather
    // back to full resolution for free. Backends that always allocate the
    // gather at full resolution treat this as 1.
    pub gi_scale: u32,
}

impl SsgiSettings {
    // Clamp the authored tunables into safe ranges.
    pub fn resolve(
        intensity: f32,
        max_distance: f32,
        rays: u32,
        steps: u32,
        gi_scale: u32,
    ) -> Self {
        Self {
            intensity: intensity.clamp(0.0, MAX_INTENSITY),
            max_distance: max_distance.clamp(MIN_DISTANCE, MAX_DISTANCE),
            rays: rays.clamp(MIN_RAYS, MAX_RAYS),
            steps: steps.clamp(MIN_STEPS, MAX_STEPS),
            gi_scale: gi_scale.max(1),
        }
    }

    // Gather-target dimensions for a given render resolution: the render size
    // divided by `gi_scale`, never below 1x1.
    pub fn gi_dimensions(&self, render_w: u32, render_h: u32) -> (u32, u32) {
        (
            (render_w / self.gi_scale).max(1),
            (render_h / self.gi_scale).max(1),
        )
    }

    // Build the per-frame GPU uniform from these settings and the active
    // camera. `fov_y_radians` is the vertical field of view and `aspect` the
    // viewport width / height ratio: together they give the view-ray scale
    // the gather pass needs to project a view-space ray point to a UV.
    pub fn params(&self, fov_y_radians: f32, aspect: f32) -> SsgiParams {
        let stride = self.max_distance / self.steps as f32;
        SsgiParams {
            intensity: self.intensity,
            max_distance: self.max_distance,
            tan_half_fov_y: (fov_y_radians * 0.5).tan(),
            aspect: aspect.max(MIN_ASPECT),
            stride,
            thickness: stride * THICKNESS_SCALE,
            rays: self.rays as f32,
            steps: self.steps as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_clamps_intensity_and_distance() {
        let s = SsgiSettings::resolve(9.0, 1.0e6, DEFAULT_RAYS, DEFAULT_STEPS, 1);
        assert_eq!(s.intensity, MAX_INTENSITY);
        assert_eq!(s.max_distance, MAX_DISTANCE);

        let s = SsgiSettings::resolve(-2.0, -10.0, DEFAULT_RAYS, DEFAULT_STEPS, 1);
        assert_eq!(s.intensity, 0.0);
        assert_eq!(s.max_distance, MIN_DISTANCE);
    }

    #[test]
    fn resolve_passes_through_in_range_values() {
        let s = SsgiSettings::resolve(0.6, 8.0, DEFAULT_RAYS, DEFAULT_STEPS, 2);
        assert_eq!(s.intensity, 0.6);
        assert_eq!(s.max_distance, 8.0);
        assert_eq!(s.rays, DEFAULT_RAYS);
        assert_eq!(s.steps, DEFAULT_STEPS);
        assert_eq!(s.gi_scale, 2);
    }

    #[test]
    fn resolve_clamps_rays_steps_and_scale() {
        // Over-range rays / steps clamp to their maxima; a zero scale floors to
        // full resolution (1).
        let s = SsgiSettings::resolve(0.6, 8.0, 9999, 9999, 0);
        assert_eq!(s.rays, MAX_RAYS);
        assert_eq!(s.steps, MAX_STEPS);
        assert_eq!(s.gi_scale, 1);
        // Under-range rays / steps clamp to their minima.
        let s = SsgiSettings::resolve(0.6, 8.0, 0, 0, 4);
        assert_eq!(s.rays, MIN_RAYS);
        assert_eq!(s.steps, MIN_STEPS);
        assert_eq!(s.gi_scale, 4);
    }

    #[test]
    fn gi_dimensions_divide_by_scale_and_floor_at_one() {
        let full = SsgiSettings::resolve(0.6, 8.0, DEFAULT_RAYS, DEFAULT_STEPS, 1);
        assert_eq!(full.gi_dimensions(1920, 1080), (1920, 1080));
        let half = SsgiSettings::resolve(0.6, 8.0, DEFAULT_RAYS, DEFAULT_STEPS, 2);
        assert_eq!(half.gi_dimensions(1920, 1080), (960, 540));
        // A tiny render target never collapses below 1x1.
        assert_eq!(half.gi_dimensions(1, 1), (1, 1));
    }

    #[test]
    fn params_derive_stride_and_thickness_from_configured_steps() {
        // 12 units over the default 12 steps -> a 1-unit stride.
        let s = SsgiSettings::resolve(0.6, 12.0, DEFAULT_RAYS, DEFAULT_STEPS, 1);
        let p = s.params(std::f32::consts::FRAC_PI_2, 1.6);
        assert!((p.stride - 1.0).abs() < 1.0e-5);
        assert!((p.thickness - THICKNESS_SCALE).abs() < 1.0e-5);
        // A 90-degree vertical FOV has tan(45 deg) == 1.
        assert!((p.tan_half_fov_y - 1.0).abs() < 1.0e-5);
        assert_eq!(p.aspect, 1.6);
        // The ray / step counts ride along in the uniform for the shader loops.
        assert_eq!(p.rays, DEFAULT_RAYS as f32);
        assert_eq!(p.steps, DEFAULT_STEPS as f32);

        // Halving the step count doubles the stride (same reach, fewer samples).
        let s = SsgiSettings::resolve(0.6, 12.0, DEFAULT_RAYS, 6, 1);
        let p = s.params(std::f32::consts::FRAC_PI_2, 1.6);
        assert!((p.stride - 2.0).abs() < 1.0e-5);
    }

    #[test]
    fn params_floor_a_degenerate_aspect() {
        let s = SsgiSettings::resolve(0.6, 8.0, DEFAULT_RAYS, DEFAULT_STEPS, 1);
        let p = s.params(std::f32::consts::FRAC_PI_2, 0.0);
        assert!(p.aspect >= MIN_ASPECT);
    }
}
