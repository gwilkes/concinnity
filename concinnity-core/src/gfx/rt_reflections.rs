// src/gfx/rt_reflections.rs
//
// Hardware ray-traced reflection configuration. Backend-agnostic resolve of the
// authored `PostProcessConfig` fields into clamped settings, plus the per-frame
// GPU uniform. The acceleration-structure build and the inline ray-trace itself
// live in the backend (Metal); this module owns only the parameter math so it
// can be unit-tested without a GPU.
//
// RT reflections replace SSR's screen-space resolve: they reuse the same
// authored `ssr_intensity` / `ssr_max_distance` tunables (so a world toggling
// from SSR to RT keeps the same look knobs) but trace a real ray against the
// scene BVH, so reflected geometry that is off-screen still appears.

use crate::gfx::render_types::RtParams;

// Upper bound on `intensity`. The kernel mixes the reflection over the base
// shading by a Fresnel-weighted amount, so a value above 1.0 would just
// over-brighten grazing edges; 1.0 is full physically-weighted reflection.
const MAX_INTENSITY: f32 = 1.0;

// Smallest usable ray reach: a ray shorter than this finds nothing.
const MIN_DISTANCE: f32 = 1.0;
// Largest ray reach. Unlike SSR's screen-march this is a true world-space
// `t_max` on the BVH traversal, so it can reach farther than the SSR cap
// without the per-step cost; still bounded so a stray value can't explode it.
const MAX_DISTANCE: f32 = 1000.0;

// Floor on the aspect ratio so a degenerate viewport cannot divide by zero.
const MIN_ASPECT: f32 = 1.0e-3;

// Clamped RT-reflection tunables resolved from the authored asset fields. Held
// by the backend and turned into a per-frame [`RtParams`] once the camera and
// sun are known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtReflectionSettings {
    // Reflection blend strength multiplier in `[0, 1]`.
    pub intensity: f32,
    // World-space distance the reflection ray travels before it misses.
    pub max_distance: f32,
}

impl RtReflectionSettings {
    // Clamp the authored intensity / distance into a safe range.
    pub fn resolve(intensity: f32, max_distance: f32) -> Self {
        Self {
            intensity: intensity.clamp(0.0, MAX_INTENSITY),
            max_distance: max_distance.clamp(MIN_DISTANCE, MAX_DISTANCE),
        }
    }

    // Build the per-frame GPU uniform from these settings, the active camera,
    // and the sun. `fov_y_radians` / `aspect` give the view-ray scale used to
    // rebuild a view-space position from the SSR pre-pass G-buffer.
    // `inv_view_rot` is the view-to-world rotation (the transpose of the view
    // matrix's orthonormal 3x3) and `cam_pos` the world camera position;
    // together they form the camera-to-world transform that lifts the
    // reconstructed hit point + normal into the BVH's world space.
    // `sun_dir` is the world-space unit direction toward the sun and
    // `sun_color` its radiance; `prefilter_mip_count` is the IBL cubemap mip
    // count (0 = no IBL) for the miss fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn params(
        &self,
        fov_y_radians: f32,
        aspect: f32,
        inv_view_rot: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        sun_dir: [f32; 3],
        sun_color: [f32; 3],
        prefilter_mip_count: f32,
    ) -> RtParams {
        // Camera-to-world: the rotation already lives in `inv_view_rot`'s 3x3
        // (its translation column is identity); set the translation column to
        // the world camera position to complete the rigid inverse of the view.
        let mut inv_view = inv_view_rot;
        inv_view[3] = [cam_pos[0], cam_pos[1], cam_pos[2], 1.0];
        RtParams {
            intensity: self.intensity,
            max_distance: self.max_distance,
            tan_half_fov_y: (fov_y_radians * 0.5).tan(),
            aspect: aspect.max(MIN_ASPECT),
            prefilter_mip_count,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            cam_pos: [cam_pos[0], cam_pos[1], cam_pos[2], 0.0],
            sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
            sun_color: [sun_color[0], sun_color[1], sun_color[2], 0.0],
            inv_view,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    #[test]
    fn resolve_clamps_intensity_and_distance() {
        let s = RtReflectionSettings::resolve(5.0, 1.0e6);
        assert_eq!(s.intensity, MAX_INTENSITY);
        assert_eq!(s.max_distance, MAX_DISTANCE);

        let s = RtReflectionSettings::resolve(-2.0, -10.0);
        assert_eq!(s.intensity, 0.0);
        assert_eq!(s.max_distance, MIN_DISTANCE);
    }

    #[test]
    fn resolve_passes_through_in_range_values() {
        let s = RtReflectionSettings::resolve(0.7, 60.0);
        assert_eq!(s.intensity, 0.7);
        assert_eq!(s.max_distance, 60.0);
    }

    #[test]
    fn params_carry_camera_and_sun_inputs() {
        let s = RtReflectionSettings::resolve(0.8, 40.0);
        let p = s.params(
            std::f32::consts::FRAC_PI_2,
            1.6,
            IDENTITY,
            [3.0, 4.0, 5.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.9, 0.8],
            6.0,
        );
        assert_eq!(p.intensity, 0.8);
        assert_eq!(p.max_distance, 40.0);
        // A 90-degree vertical FOV has tan(45 deg) == 1.
        assert!((p.tan_half_fov_y - 1.0).abs() < 1.0e-5);
        assert_eq!(p.aspect, 1.6);
        assert_eq!(p.prefilter_mip_count, 6.0);
        assert_eq!(p.cam_pos, [3.0, 4.0, 5.0, 0.0]);
        assert_eq!(p.sun_dir, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(p.sun_color, [1.0, 0.9, 0.8, 0.0]);
    }

    #[test]
    fn params_assemble_camera_to_world_translation_column() {
        // inv_view's translation column must be the world camera position so the
        // reconstructed view-space hit point lifts to the right world point.
        let s = RtReflectionSettings::resolve(0.8, 40.0);
        let p = s.params(
            std::f32::consts::FRAC_PI_2,
            1.6,
            IDENTITY,
            [3.0, 4.0, 5.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            6.0,
        );
        assert_eq!(p.inv_view[3], [3.0, 4.0, 5.0, 1.0]);
        // The rotation columns are untouched.
        assert_eq!(p.inv_view[0], [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn params_floor_a_degenerate_aspect() {
        let s = RtReflectionSettings::resolve(0.7, 40.0);
        let p = s.params(
            std::f32::consts::FRAC_PI_2,
            0.0,
            IDENTITY,
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            0.0,
        );
        assert!(p.aspect >= MIN_ASPECT);
    }
}
