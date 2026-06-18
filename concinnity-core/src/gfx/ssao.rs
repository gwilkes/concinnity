// src/gfx/ssao.rs
//
// Screen-space ambient occlusion (GTAO) configuration. Backend-agnostic
// resolve of the authored `PostProcessConfig` SSAO fields into clamped
// settings, plus the per-frame GPU uniform. The horizon-search arc integral
// itself lives in each backend's shader; this module owns only the parameter
// math so it can be unit-tested without a GPU.

use crate::gfx::render_types::SsaoParams;

// Upper bound on `intensity` so a stray asset value cannot drive the ambient
// term fully black across the whole frame.
const MAX_INTENSITY: f32 = 4.0;

// Smallest usable radius. A zero or negative radius would make every horizon
// search degenerate, so the authored value is floored here.
const MIN_RADIUS: f32 = 1.0e-3;

// Clamped SSAO tunables resolved from the authored asset fields. Held by the
// backend and turned into a per-frame [`SsaoParams`] once the camera is known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsaoSettings {
    // World-space hemisphere radius the horizon search covers.
    pub radius: f32,
    // Occlusion strength multiplier applied to the integrated visibility.
    pub intensity: f32,
}

impl SsaoSettings {
    // Clamp the authored radius / intensity into a safe range.
    pub fn resolve(radius: f32, intensity: f32) -> Self {
        Self {
            radius: radius.max(MIN_RADIUS),
            intensity: intensity.clamp(0.0, MAX_INTENSITY),
        }
    }

    // Build the per-frame GPU uniform from these settings and the active
    // camera. `fov_y_radians` is the vertical field of view and `aspect` the
    // viewport width / height ratio: together they give the view-ray scale
    // the kernel needs to rebuild view-space positions from linear depth.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn params(&self, fov_y_radians: f32, aspect: f32) -> SsaoParams {
        SsaoParams {
            radius: self.radius,
            intensity: self.intensity,
            tan_half_fov_y: (fov_y_radians * 0.5).tan(),
            aspect: aspect.max(MIN_RADIUS),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_floors_radius_and_clamps_intensity() {
        let s = SsaoSettings::resolve(-1.0, 100.0);
        assert!(s.radius >= MIN_RADIUS);
        assert_eq!(s.intensity, MAX_INTENSITY);

        let s = SsaoSettings::resolve(0.75, -2.0);
        assert_eq!(s.radius, 0.75);
        assert_eq!(s.intensity, 0.0);
    }

    #[test]
    fn resolve_passes_through_in_range_values() {
        let s = SsaoSettings::resolve(0.5, 1.0);
        assert_eq!(s.radius, 0.5);
        assert_eq!(s.intensity, 1.0);
    }

    #[test]
    fn params_compute_tan_half_fov() {
        let s = SsaoSettings::resolve(0.5, 1.0);
        // A 90-degree vertical FOV has tan(45 deg) == 1.
        let p = s.params(std::f32::consts::FRAC_PI_2, 1.6);
        assert!((p.tan_half_fov_y - 1.0).abs() < 1.0e-5);
        assert_eq!(p.aspect, 1.6);
        assert_eq!(p.radius, 0.5);
        assert_eq!(p.intensity, 1.0);
    }

    #[test]
    fn params_floor_a_degenerate_aspect() {
        let s = SsaoSettings::resolve(0.5, 1.0);
        let p = s.params(std::f32::consts::FRAC_PI_2, 0.0);
        assert!(p.aspect >= MIN_RADIUS);
    }
}
