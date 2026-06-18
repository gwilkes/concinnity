// src/gfx/lights.rs
//
// Converts drained DirectionalLight and PointLight asset components into the
// packed GPU uniform struct consumed by the renderer.

use crate::assets::{DirectionalLight, PointLight};
use crate::gfx::render_types::{
    DirectionalLightData, LightUniforms, MAX_DIRECTIONAL_LIGHTS, MAX_POINT_LIGHTS, PointLightData,
};

pub(crate) fn build_light_uniforms(
    dir_lights: Vec<DirectionalLight>,
    pt_lights: Vec<PointLight>,
    ambient_intensity: f32,
) -> LightUniforms {
    if dir_lights.is_empty() && pt_lights.is_empty() {
        return LightUniforms {
            ambient_intensity,
            ..LightUniforms::DEFAULT
        };
    }

    const ZERO_DIR: DirectionalLightData = DirectionalLightData {
        direction: [0.0; 3],
        intensity: 0.0,
        color: [0.0; 3],
        _pad: 0.0,
    };
    const ZERO_PT: PointLightData = PointLightData {
        position: [0.0; 3],
        range: 0.0,
        color: [0.0; 3],
        intensity: 0.0,
    };

    let mut directional = [ZERO_DIR; MAX_DIRECTIONAL_LIGHTS];
    let mut point = [ZERO_PT; MAX_POINT_LIGHTS];
    let num_directional = dir_lights.len().min(MAX_DIRECTIONAL_LIGHTS);
    let num_point = pt_lights.len().min(MAX_POINT_LIGHTS);

    if dir_lights.len() > MAX_DIRECTIONAL_LIGHTS {
        tracing::warn!(
            "GraphicsSystem: {} directional lights declared; only {} are supported -- extras ignored",
            dir_lights.len(),
            MAX_DIRECTIONAL_LIGHTS
        );
    }
    if pt_lights.len() > MAX_POINT_LIGHTS {
        tracing::warn!(
            "GraphicsSystem: {} point lights declared; only {} are supported -- extras ignored",
            pt_lights.len(),
            MAX_POINT_LIGHTS
        );
    }

    for (i, l) in dir_lights
        .into_iter()
        .take(MAX_DIRECTIONAL_LIGHTS)
        .enumerate()
    {
        directional[i] = DirectionalLightData {
            direction: l.direction,
            intensity: l.intensity,
            color: l.color,
            _pad: 0.0,
        };
    }
    for (i, l) in pt_lights.into_iter().take(MAX_POINT_LIGHTS).enumerate() {
        point[i] = PointLightData {
            position: l.position,
            range: l.range,
            color: l.color,
            intensity: l.intensity,
        };
    }

    LightUniforms {
        directional,
        point,
        num_directional: num_directional as i32,
        num_point: num_point as i32,
        ambient_intensity,
        _pad: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(direction: [f32; 3], color: [f32; 3], intensity: f32) -> DirectionalLight {
        DirectionalLight {
            direction,
            color,
            intensity,
        }
    }

    fn pt(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> PointLight {
        PointLight {
            position,
            color,
            intensity,
            range,
        }
    }

    #[test]
    fn empty_inputs_return_default() {
        let u = build_light_uniforms(vec![], vec![], 1.0);
        assert_eq!(u.num_directional, LightUniforms::DEFAULT.num_directional);
        assert_eq!(u.num_point, LightUniforms::DEFAULT.num_point);
    }

    #[test]
    fn ambient_intensity_carried_in_both_branches() {
        // Empty (DEFAULT) branch and the populated branch both honour the
        // authored multiplier.
        let empty = build_light_uniforms(vec![], vec![], 2.5);
        assert!((empty.ambient_intensity - 2.5).abs() < 1e-6);
        let populated =
            build_light_uniforms(vec![dir([-0.3, 0.85, 0.4], [1.0; 3], 1.0)], vec![], 3.0);
        assert!((populated.ambient_intensity - 3.0).abs() < 1e-6);
    }

    #[test]
    fn single_directional_light_fields_mapped() {
        let u = build_light_uniforms(
            vec![dir([-0.3, 0.85, 0.4], [1.0, 0.95, 0.8], 1.5)],
            vec![],
            1.0,
        );
        assert_eq!(u.num_directional, 1);
        assert_eq!(u.num_point, 0);
        assert_eq!(u.directional[0].direction, [-0.3, 0.85, 0.4]);
        assert_eq!(u.directional[0].color, [1.0, 0.95, 0.8]);
        assert!((u.directional[0].intensity - 1.5).abs() < 1e-6);
    }

    #[test]
    fn single_point_light_fields_mapped() {
        let u = build_light_uniforms(
            vec![],
            vec![pt([2.0, 3.0, 4.0], [1.0, 0.8, 0.5], 8.0, 6.0)],
            1.0,
        );
        assert_eq!(u.num_directional, 0);
        assert_eq!(u.num_point, 1);
        assert_eq!(u.point[0].position, [2.0, 3.0, 4.0]);
        assert_eq!(u.point[0].color, [1.0, 0.8, 0.5]);
        assert!((u.point[0].intensity - 8.0).abs() < 1e-6);
        assert!((u.point[0].range - 6.0).abs() < 1e-6);
    }

    #[test]
    fn excess_directional_lights_clamped_to_max() {
        let lights: Vec<DirectionalLight> = (0..MAX_DIRECTIONAL_LIGHTS + 2)
            .map(|i| dir([i as f32, 0.0, 0.0], [1.0; 3], 1.0))
            .collect();
        let u = build_light_uniforms(lights, vec![], 1.0);
        assert_eq!(u.num_directional, MAX_DIRECTIONAL_LIGHTS as i32);
    }

    #[test]
    fn excess_point_lights_clamped_to_max() {
        let lights: Vec<PointLight> = (0..MAX_POINT_LIGHTS + 2)
            .map(|i| pt([i as f32, 0.0, 0.0], [1.0; 3], 1.0, 5.0))
            .collect();
        let u = build_light_uniforms(vec![], lights, 1.0);
        assert_eq!(u.num_point, MAX_POINT_LIGHTS as i32);
    }
}
