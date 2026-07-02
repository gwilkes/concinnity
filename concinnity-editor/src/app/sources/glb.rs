// src/app/sources/glb.rs
//
// Content template presets for `cn add <scene> --template <name>`. The
// renderer stack itself (GraphicsConfig, Window, shader stages, menu, HUDs,
// sky) is injected at build time from the scene's own assets, so a plain
// scene add writes no extra lines; a template layers optional visual polish
// on top as real, user-owned world.jsonl entries.
//
// To tweak what a template produces, edit this file.

// Aesthetic-defaults template for `cn add foo.glb --template showcase`.
// Layers visual polish so a dropped scene lands in a richer setting: a warm
// key light, PostProcessConfig (bloom + tonemap), an IBL EnvironmentMap
// (procedural sky generator, no .hdr needed; the sky mesh that displays it is
// injected at build time), and VolumetricFog. Shipped hardcoded for now; will
// move to an infra-fetched template registry once there are 2+ templates that
// justify the indirection.
pub fn template_showcase() -> Vec<serde_json::Value> {
    vec![
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
    fn template_showcase_layers_polish_assets() {
        let entries = template_showcase();
        let types: Vec<&str> = entries
            .iter()
            .filter_map(|e| e.get("type").and_then(|v| v.as_str()))
            .collect();
        // Order isn't behaviourally significant but pinning it catches drift.
        assert_eq!(
            types,
            vec![
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
