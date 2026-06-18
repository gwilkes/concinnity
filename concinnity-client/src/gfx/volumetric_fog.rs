// src/gfx/volumetric_fog.rs
//
// Backend-agnostic resolution of the authored `VolumetricFog` asset into a
// clamped settings struct plus the per-frame `FogParams` uniform the Metal
// fog fragment shader consumes. Pure CPU; unit-testable without a GPU.

use crate::gfx::render_types::FogParams;

// Upper bound on the volumetric density. The integral
// `1 - exp(-density * step)` saturates near 1.0 well before this cap, so
// anything higher just wastes precision and risks numeric blowups for the
// Henyey-Greenstein factor. 10/world-unit is already pea-soup territory.
const MAX_DENSITY: f32 = 10.0;
// Largest sensible height-falloff rate. Beyond this the density drops to
// nothing within centimetres above the reference height, which is not
// useful (and is rounding-error fragile in the shader's `exp`).
const MAX_HEIGHT_FALLOFF: f32 = 4.0;
// Cap on the ray-march distance. The marcher takes a fixed number of steps,
// so a longer ray spends more world units per step rather than more samples.
// Going past this trades shadow / phase accuracy for distance with no
// real visual win.
const MAX_DISTANCE_CAP: f32 = 2_000.0;
// Floor on the ray-march distance. The shader divides by it, and the
// per-step length collapses to zero past about a millimetre.
const MIN_DISTANCE: f32 = 1.0;
// Floor on the viewport short edge so a zero-sized swapchain (initial layout)
// cannot poison the reciprocal the shader uses to convert screen to NDC.
const MIN_VIEWPORT: f32 = 1.0;

// Resolved and clamped fog tunables, threaded into the backend at init.
// `None` from `FogSettings::resolve_optional` means the world declared no
// `VolumetricFog`: the renderer then skips the fog pass entirely.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogSettings {
    pub color: [f32; 3],
    pub density: f32,
    pub height_falloff: f32,
    pub height_reference: f32,
    pub max_distance: f32,
    pub phase_g: f32,
    pub ambient: f32,
}

impl FogSettings {
    // Clamp the authored fields into a safe range. Mirrors `VolumetricFog::from_args`;
    // those clamps are the asset-side floor; this is the gfx-side ceiling.
    pub fn resolve(
        color: [f32; 3],
        density: f32,
        height_falloff: f32,
        height_reference: f32,
        max_distance: f32,
        phase_g: f32,
        ambient: f32,
    ) -> Self {
        let max_distance = if max_distance.is_finite() {
            max_distance.clamp(MIN_DISTANCE, MAX_DISTANCE_CAP)
        } else {
            MIN_DISTANCE
        };
        let color = [color[0].max(0.0), color[1].max(0.0), color[2].max(0.0)];
        Self {
            color,
            density: density.clamp(0.0, MAX_DENSITY),
            height_falloff: height_falloff.clamp(0.0, MAX_HEIGHT_FALLOFF),
            height_reference,
            max_distance,
            // Mirror the asset clamp so a settings built from out-of-range
            // raw floats (e.g. in tests) still produces stable HG output.
            phase_g: phase_g.clamp(-0.95, 0.95),
            ambient: ambient.clamp(0.0, MAX_DENSITY),
        }
    }

    // Build the per-frame GPU uniform from these settings and the active
    // camera. `inv_vp` is the inverse view-projection used to reconstruct
    // world positions from depth; `cam_pos` is the camera origin; `sun_dir`
    // and `sun_color` are the first directional light's direction (toward
    // the light) and `intensity * colour`. `viewport` is the HDR resolve
    // target's pixel dimensions.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn params(
        &self,
        inv_vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        sun_dir: [f32; 3],
        sun_color: [f32; 3],
        viewport: [f32; 2],
    ) -> FogParams {
        let viewport = [viewport[0].max(MIN_VIEWPORT), viewport[1].max(MIN_VIEWPORT)];
        FogParams {
            inv_vp,
            color: [self.color[0], self.color[1], self.color[2], 1.0],
            cam_pos,
            _pad0: 0.0,
            sun_dir,
            _pad1: 0.0,
            sun_color,
            _pad2: 0.0,
            density: self.density,
            height_falloff: self.height_falloff,
            height_reference: self.height_reference,
            max_distance: self.max_distance,
            phase_g: self.phase_g,
            ambient: self.ambient,
            viewport,
            inv_max_distance: 1.0 / self.max_distance,
            _pad3: [0.0; 3],
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
    fn resolve_clamps_density_falloff_and_distance() {
        let s = FogSettings::resolve([1.0, 1.0, 1.0], 100.0, 20.0, 0.0, 1e9, 1.5, -1.0);
        assert_eq!(s.density, MAX_DENSITY);
        assert_eq!(s.height_falloff, MAX_HEIGHT_FALLOFF);
        assert_eq!(s.max_distance, MAX_DISTANCE_CAP);
        assert!(s.phase_g <= 0.95 && s.phase_g > 0.0);
        assert_eq!(s.ambient, 0.0);
    }

    #[test]
    fn resolve_passes_through_in_range_values() {
        let s = FogSettings::resolve([0.6, 0.7, 0.8], 0.08, 0.25, 1.5, 120.0, 0.4, 0.2);
        assert_eq!(s.color, [0.6, 0.7, 0.8]);
        assert!((s.density - 0.08).abs() < 1e-6);
        assert!((s.phase_g - 0.4).abs() < 1e-6);
        assert!((s.max_distance - 120.0).abs() < 1e-6);
    }

    #[test]
    fn resolve_handles_non_finite_distance() {
        let s = FogSettings::resolve([0.6; 3], 0.05, 0.2, 0.0, f32::NAN, 0.4, 0.15);
        assert!(s.max_distance.is_finite());
        assert!(s.max_distance >= MIN_DISTANCE);
    }

    #[test]
    fn params_derive_inverse_max_distance() {
        let s = FogSettings::resolve([0.7; 3], 0.05, 0.2, 0.0, 50.0, 0.4, 0.15);
        let p = s.params(
            IDENTITY,
            [0.0; 3],
            [0.0, 1.0, 0.0],
            [1.0; 3],
            [1280.0, 720.0],
        );
        assert!((p.inv_max_distance - (1.0 / 50.0)).abs() < 1e-6);
        assert_eq!(p.viewport, [1280.0, 720.0]);
    }

    #[test]
    fn params_floor_a_degenerate_viewport() {
        let s = FogSettings::resolve([0.7; 3], 0.05, 0.2, 0.0, 50.0, 0.4, 0.15);
        let p = s.params(IDENTITY, [0.0; 3], [0.0, 1.0, 0.0], [1.0; 3], [0.0, 0.0]);
        assert!(p.viewport[0] >= MIN_VIEWPORT);
        assert!(p.viewport[1] >= MIN_VIEWPORT);
    }

    #[test]
    fn params_zero_padding_words_are_zero() {
        let s = FogSettings::resolve([0.7; 3], 0.05, 0.2, 0.0, 50.0, 0.4, 0.15);
        let p = s.params(IDENTITY, [0.0; 3], [0.0, 1.0, 0.0], [1.0; 3], [1.0, 1.0]);
        assert_eq!(p._pad0, 0.0);
        assert_eq!(p._pad1, 0.0);
        assert_eq!(p._pad2, 0.0);
        assert_eq!(p._pad3, [0.0; 3]);
    }
}
