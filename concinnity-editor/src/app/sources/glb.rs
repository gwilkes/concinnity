// src/app/sources/glb.rs
//
// Renderer scaffold presets that `cn add` injects when a 3D scene target
// (`.glb` / `.fbx`) lands in a world that has no renderer trigger yet: the
// "bootstrap a viewable scene from one command" UX. The scene file itself is
// expanded at build time from a single `SceneImport` asset (see
// `concinnity_core::build::import`), so no per-format entry generation lives
// here anymore.
//
// To tweak the defaults a fresh scene add produces, edit this file.

// Renderer scaffold injected when a scene add lands in a fresh (renderer-less)
// world. GraphicsConfig triggers companion injection of the GraphicsSystem,
// default ShaderStages (vertex + fragment + shadow), and Window. No Camera3D is
// scaffolded so the import's framed Camera3D wins the runtime's
// first-Camera3D query.
pub fn scaffold() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "name": "scaffold_graphics",
            "type": "GraphicsConfig",
            "args": {
                "clear_color": [0.05, 0.05, 0.08, 1.0],
                "shadow_map_size": 2048,
            }
        }),
        serde_json::json!({
            "name": "ambient_light",
            "type": "DirectionalLight",
            "args": {
                "color": [1.0, 0.95, 0.8],
                "direction": [-0.3, 0.85, 0.4],
                "intensity": 0.8,
            }
        }),
    ]
}

// Aesthetic-defaults scaffold for `cn add foo.glb --template showcase`.
// Same renderer-trigger role as `scaffold()` but layers visual polish so a
// dropped scene lands in a richer setting: PostProcessConfig (bloom + tonemap),
// an IBL EnvironmentMap (procedural sky generator, no .hdr needed),
// and VolumetricFog. Shipped hardcoded for now; will move to an infra-fetched
// template registry once there are 2+ templates that justify the indirection.
pub fn template_showcase() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "name": "scaffold_graphics",
            "type": "GraphicsConfig",
            "args": {
                "clear_color": [0.53, 0.71, 0.87, 1.0],
                "shadow_map_size": 2048,
            }
        }),
        serde_json::json!({
            "name": "ambient_light",
            "type": "DirectionalLight",
            "args": {
                "color": [1.0, 0.95, 0.8],
                "direction": [-0.3, 0.85, 0.4],
                "intensity": 1.1,
            }
        }),
        serde_json::json!({
            "name": "showcase_post",
            "type": "PostProcessConfig",
            "args": {
                "bloom_intensity": 0.7,
                "bloom_threshold": 1.1,
                "exposure_ev": 0.0,
            }
        }),
        serde_json::json!({
            "name": "showcase_env",
            "type": "EnvironmentMap",
            "args": {
                "generator": "sky",
            }
        }),
        serde_json::json!({
            "name": "showcase_fog",
            "type": "VolumetricFog",
            "args": {
                "density": 0.02,
                "color": [0.75, 0.82, 0.95],
                "height_falloff": 0.18,
                "max_distance": 180.0,
                "phase_g": 0.5,
            }
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_emits_graphics_config_and_directional_light() {
        let entries = scaffold();
        let types: Vec<&str> = entries
            .iter()
            .filter_map(|e| e.get("type").and_then(|v| v.as_str()))
            .collect();
        assert_eq!(types, vec!["GraphicsConfig", "DirectionalLight"]);
    }

    #[test]
    fn template_showcase_includes_polish_assets_on_top_of_scaffold() {
        let entries = template_showcase();
        let types: Vec<&str> = entries
            .iter()
            .filter_map(|e| e.get("type").and_then(|v| v.as_str()))
            .collect();
        // Order isn't behaviourally significant but pinning it catches drift.
        assert_eq!(
            types,
            vec![
                "GraphicsConfig",
                "DirectionalLight",
                "PostProcessConfig",
                "EnvironmentMap",
                "VolumetricFog",
            ]
        );
    }

    #[test]
    fn template_showcase_environment_map_uses_procedural_sky() {
        // The whole point of using the "sky" generator is that the template
        // works without shipping an .hdr file. If someone swaps to `source`
        // the template silently breaks on any fresh project.
        let entries = template_showcase();
        let env = entries
            .iter()
            .find(|e| e.get("type").and_then(|v| v.as_str()) == Some("EnvironmentMap"))
            .expect("template should declare an EnvironmentMap");
        assert_eq!(
            env["args"]["generator"],
            serde_json::json!("sky"),
            "EnvironmentMap must use the procedural sky generator (no .hdr file dependency)"
        );
    }
}
