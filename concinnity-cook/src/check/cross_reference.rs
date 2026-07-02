// src/check/cross_reference.rs
//
// Cross-asset name reference validation. Runs on the fully expanded world and
// checks that every named reference (Prop -> Mesh, Material -> Texture, etc.)
// resolves to an asset of the right kind, and that Prop parent chains are
// acyclic. Every problem found is collected: validation never stops at the
// first error, so the caller can report them all in one pass.
//
// Each referencing asset declares its own references by implementing
// `CrossReferenced` in its asset file; `cross_refs_for` dispatches to those
// impls by type. The validator here only resolves a `RefKind` to the matching
// set of asset names and detects Prop parent cycles. Adding a new referencing
// asset means writing one impl plus one arm in `cross_refs_for`.

use crate::assets::FileKind;
use crate::world::WorldJsonlAsset;
use concinnity_core::check::cross_reference::{CrossRef, CrossReferenced, RefKind};
use std::collections::HashSet;

// Dispatch reference extraction by normalized asset type. Every arm delegates
// to a `CrossReferenced` impl in the named asset's file.
fn cross_refs_for(type_norm: &str, name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
    use crate::assets::{
        DebugHud, Decal, FpsCounter, InstancedProp, Joint, Material, Model, ParticleEmitter, Prop,
        Scene, SceneReel, StatHud, VoxelChunk, VoxelWorld,
    };
    match type_norm {
        "prop" => Prop::cross_refs(name, args),
        "model" => Model::cross_refs(name, args),
        "scenereel" | "scenreel" => SceneReel::cross_refs(name, args),
        "scene" => Scene::cross_refs(name, args),
        "instancedprop" | "instanced" => InstancedProp::cross_refs(name, args),
        "voxelchunk" | "chunk" => VoxelChunk::cross_refs(name, args),
        "voxelworld" => VoxelWorld::cross_refs(name, args),
        "material" => Material::cross_refs(name, args),
        "decal" => Decal::cross_refs(name, args),
        "joint" => Joint::cross_refs(name, args),
        "particleemitter" | "particles" => ParticleEmitter::cross_refs(name, args),
        "stathud" => StatHud::cross_refs(name, args),
        "debughud" => DebugHud::cross_refs(name, args),
        "fpscounter" => FpsCounter::cross_refs(name, args),
        _ => Vec::new(),
    }
}

// The name-sets of an expanded world, one per `RefKind`. Built once per
// validation pass; `contains` answers whether a reference resolves.
struct RefScope<'a> {
    mesh_sources: HashSet<&'a str>,
    textures: HashSet<&'a str>,
    materials: HashSet<&'a str>,
    models: HashSet<&'a str>,
    props: HashSet<&'a str>,
    scenes: HashSet<&'a str>,
    camera3ds: HashSet<&'a str>,
    block_types: HashSet<&'a str>,
    text_labels: HashSet<&'a str>,
}

impl<'a> RefScope<'a> {
    fn build(assets: &'a [WorldJsonlAsset]) -> Self {
        let norm = |t: &str| t.to_lowercase().replace('_', "");

        // Names of every asset whose normalized type satisfies the predicate.
        let by_type = |is_match: &dyn Fn(&str) -> bool| -> HashSet<&'a str> {
            assets
                .iter()
                .filter(|a| is_match(&norm(&a.asset_type)))
                .map(|a| a.name.as_str())
                .collect()
        };

        // Mesh, ProceduralMesh, VoxelChunk, and mesh-kind File are all valid
        // mesh sources.
        let mesh_sources = assets
            .iter()
            .filter(|a| match norm(&a.asset_type).as_str() {
                "mesh" | "proceduralmesh" | "voxelchunk" | "chunk" => true,
                "file" => a
                    .args
                    .get("kind")
                    .and_then(|k| k.as_str())
                    .and_then(FileKind::from_ext)
                    .map(|fk| fk.is_mesh())
                    .unwrap_or(false),
                _ => false,
            })
            .map(|a| a.name.as_str())
            .collect();

        RefScope {
            mesh_sources,
            textures: by_type(&|t| t == "texture"),
            materials: by_type(&|t| t == "material"),
            models: by_type(&|t| t == "model"),
            props: by_type(&|t| t == "prop"),
            scenes: by_type(&|t| t == "scene"),
            camera3ds: by_type(&|t| t == "camera3d"),
            block_types: by_type(&|t| t == "blocktype" || t == "block"),
            text_labels: by_type(&|t| t == "textlabel"),
        }
    }

    // True when `name` is satisfied by an asset of the given reference kind.
    fn contains(&self, kind: RefKind, name: &str) -> bool {
        match kind {
            RefKind::MeshSource => self.mesh_sources.contains(name),
            RefKind::Texture => self.textures.contains(name),
            RefKind::Material => self.materials.contains(name),
            RefKind::Model => self.models.contains(name),
            RefKind::Prop => self.props.contains(name),
            RefKind::Scene => self.scenes.contains(name),
            RefKind::CameraShot => self.camera3ds.contains(name),
            RefKind::BlockType => self.block_types.contains(name),
            RefKind::TextLabel => self.text_labels.contains(name),
        }
    }
}

// Validate cross-asset name references on the expanded world.
pub(crate) fn validate_cross_references(assets: &[WorldJsonlAsset]) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();
    let scope = RefScope::build(assets);

    for asset in assets {
        let type_norm = asset.asset_type.to_lowercase().replace('_', "");
        for cross_ref in cross_refs_for(&type_norm, &asset.name, &asset.args) {
            match cross_ref {
                CrossRef::Resolve {
                    kind,
                    target,
                    error,
                } => {
                    if !scope.contains(kind, &target) {
                        errors.push(error);
                    }
                }
                CrossRef::Issue(msg) => errors.push(msg),
            }
        }
    }

    // Detect cycles in the Prop parent chain. This is a graph-global pass, so
    // it stays in the validator rather than the per-asset trait.
    let prop_parent_map: std::collections::HashMap<&str, &str> = assets
        .iter()
        .filter(|a| a.asset_type.to_lowercase().replace('_', "") == "prop")
        .filter_map(|a| {
            let parent = a
                .args
                .get("parent")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())?;
            Some((a.name.as_str(), parent))
        })
        .collect();

    for &start in prop_parent_map.keys() {
        let mut visited = std::collections::HashSet::new();
        let mut current = start;
        visited.insert(current);
        while let Some(&parent) = prop_parent_map.get(current) {
            if !visited.insert(parent) {
                errors.push(format!(
                    "Prop '{}': parent chain contains a cycle (via '{}')",
                    start, parent
                ));
                break;
            }
            current = parent;
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str, asset_type: &str, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            name: name.to_string(),
            asset_type: asset_type.to_string(),
            args,
        }
    }

    // Joins every cross-reference error into one string so a test can assert
    // on a substring regardless of how many problems were reported.
    fn err_text(assets: &[WorldJsonlAsset]) -> String {
        validate_cross_references(assets).unwrap_err().join("\n")
    }

    #[test]
    fn valid_references_pass() {
        let assets = vec![
            asset(
                "my_mesh",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "my_tex",
                "Texture",
                serde_json::json!({"generator":"brick"}),
            ),
            asset(
                "my_prop",
                "Prop",
                serde_json::json!({"mesh":"my_mesh","texture":"my_tex"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn prop_missing_mesh_fails() {
        let assets = vec![asset(
            "my_prop",
            "Prop",
            serde_json::json!({"mesh":"missing_mesh"}),
        )];
        assert!(err_text(&assets).contains("missing_mesh"));
    }

    #[test]
    fn prop_missing_texture_fails() {
        let assets = vec![
            asset(
                "my_mesh",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "my_prop",
                "Prop",
                serde_json::json!({"mesh":"my_mesh","texture":"no_tex"}),
            ),
        ];
        assert!(err_text(&assets).contains("no_tex"));
    }

    #[test]
    fn prop_missing_material_fails() {
        let assets = vec![
            asset(
                "my_mesh",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "my_prop",
                "Prop",
                serde_json::json!({"mesh":"my_mesh","material":"no_mat"}),
            ),
        ];
        assert!(err_text(&assets).contains("no_mat"));
    }

    #[test]
    fn prop_model_valid_passes() {
        let assets = vec![
            asset(
                "body",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[0.4,0.4,0.4]}),
            ),
            asset("mat_wood", "Material", serde_json::json!({"roughness":0.7})),
            asset(
                "crate_model",
                "Model",
                serde_json::json!({"meshes":[{"mesh":"body","material":"mat_wood"}]}),
            ),
            asset(
                "crate_a",
                "Prop",
                serde_json::json!({"model":"crate_model"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn model_missing_mesh_fails() {
        let assets = vec![asset(
            "my_model",
            "Model",
            serde_json::json!({"meshes":[{"mesh":"ghost_mesh"}]}),
        )];
        assert!(err_text(&assets).contains("ghost_mesh"));
    }

    #[test]
    fn prop_missing_model_fails() {
        let assets = vec![asset(
            "my_prop",
            "Prop",
            serde_json::json!({"model":"ghost_model"}),
        )];
        assert!(err_text(&assets).contains("ghost_model"));
    }

    #[test]
    fn file_mesh_counts_as_mesh_source() {
        let assets = vec![
            asset(
                "room_obj",
                "File",
                serde_json::json!({"path":"assets/room.obj","kind":"obj"}),
            ),
            asset("my_prop", "Prop", serde_json::json!({"mesh":"room_obj"})),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn file_non_mesh_does_not_count_as_mesh_source() {
        let assets = vec![
            asset(
                "wall_png",
                "File",
                serde_json::json!({"path":"assets/wall.png","kind":"png"}),
            ),
            asset("my_prop", "Prop", serde_json::json!({"mesh":"wall_png"})),
        ];
        assert!(err_text(&assets).contains("wall_png"));
    }

    #[test]
    fn raw_mesh_counts_as_mesh_source() {
        let assets = vec![
            asset(
                "inline_mesh",
                "Mesh",
                serde_json::json!({"vertices":[],"indices":[]}),
            ),
            asset("my_prop", "Prop", serde_json::json!({"mesh":"inline_mesh"})),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn material_missing_albedo_fails() {
        let assets = vec![asset(
            "my_mat",
            "Material",
            serde_json::json!({"albedo":"no_tex"}),
        )];
        assert!(err_text(&assets).contains("no_tex"));
    }

    #[test]
    fn material_missing_normal_map_fails() {
        let assets = vec![asset(
            "my_mat",
            "Material",
            serde_json::json!({"normal_map":"no_nrm"}),
        )];
        assert!(err_text(&assets).contains("no_nrm"));
    }

    #[test]
    fn material_valid_normal_map_passes() {
        let assets = vec![
            asset(
                "nrm_tex",
                "Texture",
                serde_json::json!({"generator":"solid","color":[128,128,255,255]}),
            ),
            asset(
                "my_mat",
                "Material",
                serde_json::json!({"normal_map":"nrm_tex"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn empty_optional_references_pass() {
        let assets = vec![
            asset(
                "my_mesh",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset("my_prop", "Prop", serde_json::json!({"mesh":"my_mesh"})),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn prop_valid_parent_passes() {
        let assets = vec![
            asset(
                "box",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "frame",
                "Prop",
                serde_json::json!({"mesh":"box","position":[0,0,0]}),
            ),
            asset(
                "panel",
                "Prop",
                serde_json::json!({"mesh":"box","parent":"frame"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn prop_missing_parent_fails() {
        let assets = vec![
            asset(
                "box",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "panel",
                "Prop",
                serde_json::json!({"mesh":"box","parent":"ghost_prop"}),
            ),
        ];
        assert!(err_text(&assets).contains("ghost_prop"));
    }

    #[test]
    fn prop_parent_cycle_fails() {
        let assets = vec![
            asset(
                "box",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset("a", "Prop", serde_json::json!({"mesh":"box","parent":"b"})),
            asset("b", "Prop", serde_json::json!({"mesh":"box","parent":"a"})),
        ];
        assert!(err_text(&assets).contains("cycle"));
    }

    #[test]
    fn prop_parent_chain_no_cycle_passes() {
        let assets = vec![
            asset(
                "box",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset("root", "Prop", serde_json::json!({"mesh":"box"})),
            asset(
                "mid",
                "Prop",
                serde_json::json!({"mesh":"box","parent":"root"}),
            ),
            asset(
                "leaf",
                "Prop",
                serde_json::json!({"mesh":"box","parent":"mid"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn scene_reel_valid_scenes_pass() {
        let assets = vec![
            asset(
                "day",
                "Scene",
                serde_json::json!({"duration_secs":3.0,"transition":"FadeBlack"}),
            ),
            asset(
                "night",
                "Scene",
                serde_json::json!({"duration_secs":3.0,"transition":"FadeBlack"}),
            ),
            asset(
                "reel",
                "SceneReel",
                serde_json::json!({"scenes":["day","night"]}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn scene_reel_missing_scene_fails() {
        let assets = vec![
            asset("day", "Scene", serde_json::json!({})),
            asset(
                "reel",
                "SceneReel",
                serde_json::json!({"scenes":["day","ghost_scene"]}),
            ),
        ];
        assert!(err_text(&assets).contains("ghost_scene"));
    }

    #[test]
    fn scene_reel_empty_scenes_fails() {
        let assets = vec![asset("reel", "SceneReel", serde_json::json!({"scenes":[]}))];
        assert!(err_text(&assets).contains("empty"));
    }

    #[test]
    fn voxel_chunk_counts_as_mesh_source() {
        let assets = vec![
            asset("air", "BlockType", serde_json::json!({"solid":false})),
            asset("stone", "BlockType", serde_json::json!({})),
            asset(
                "chunk",
                "VoxelChunk",
                serde_json::json!({
                    "palette":["air","stone"],
                    "dim":[1,1,1],
                    "blocks":[1],
                }),
            ),
            asset("p", "Prop", serde_json::json!({"mesh":"chunk"})),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn voxel_chunk_unknown_block_type_fails() {
        let assets = vec![
            asset("air", "BlockType", serde_json::json!({"solid":false})),
            asset(
                "chunk",
                "VoxelChunk",
                serde_json::json!({
                    "palette":["air","ghost_block"],
                    "dim":[1,1,1],
                    "blocks":[0],
                }),
            ),
        ];
        assert!(err_text(&assets).contains("ghost_block"));
    }

    #[test]
    fn instanced_prop_valid_references_pass() {
        let assets = vec![
            asset(
                "rock_mesh",
                "ProceduralMesh",
                serde_json::json!({"generator":"sphere","radius":0.4,"rings":6,"segments":8}),
            ),
            asset(
                "tex_rock",
                "Texture",
                serde_json::json!({"generator":"stone"}),
            ),
            asset(
                "mat_rock",
                "Material",
                serde_json::json!({"albedo":"tex_rock"}),
            ),
            asset(
                "rocks",
                "InstancedProp",
                serde_json::json!({
                    "mesh":"rock_mesh",
                    "material":"mat_rock",
                    "instances":[
                        {"position":[1.0, 0.0, 2.0]},
                        {"position":[3.0, 0.0, -1.0]}
                    ]
                }),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn instanced_prop_missing_mesh_fails() {
        let assets = vec![asset(
            "rocks",
            "InstancedProp",
            serde_json::json!({"mesh":"ghost_mesh","instances":[]}),
        )];
        assert!(err_text(&assets).contains("ghost_mesh"));
    }

    #[test]
    fn instanced_prop_empty_mesh_field_fails() {
        let assets = vec![asset(
            "rocks",
            "InstancedProp",
            serde_json::json!({"instances":[]}),
        )];
        assert!(err_text(&assets).contains("`mesh` field is required"));
    }

    #[test]
    fn instanced_prop_missing_material_fails() {
        let assets = vec![
            asset(
                "rock_mesh",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[0.5,0.5,0.5]}),
            ),
            asset(
                "rocks",
                "InstancedProp",
                serde_json::json!({"mesh":"rock_mesh","material":"ghost_mat","instances":[]}),
            ),
        ];
        assert!(err_text(&assets).contains("ghost_mat"));
    }

    #[test]
    fn instanced_prop_voxel_chunk_mesh_passes() {
        let assets = vec![
            asset("air", "BlockType", serde_json::json!({"solid":false})),
            asset("stone", "BlockType", serde_json::json!({})),
            asset(
                "chunk",
                "VoxelChunk",
                serde_json::json!({
                    "palette":["air","stone"],
                    "dim":[1,1,1],
                    "blocks":[1],
                }),
            ),
            asset(
                "rocks",
                "InstancedProp",
                serde_json::json!({"mesh":"chunk","instances":[]}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn joint_valid_two_body_passes() {
        let assets = vec![
            asset(
                "box",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "a",
                "Prop",
                serde_json::json!({"mesh":"box","collider":{"shape":"cuboid","half_extents":[1,1,1]}}),
            ),
            asset(
                "b",
                "Prop",
                serde_json::json!({"mesh":"box","collider":{"shape":"cuboid","half_extents":[1,1,1]}}),
            ),
            asset(
                "j",
                "Joint",
                serde_json::json!({"kind":"revolute","body_a":"a","body_b":"b"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn joint_world_anchor_passes_without_body_b() {
        let assets = vec![
            asset(
                "box",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "bob",
                "Prop",
                serde_json::json!({"mesh":"box","collider":{"shape":"ball","radius":0.3}}),
            ),
            asset(
                "pendulum",
                "Joint",
                serde_json::json!({"kind":"revolute","body_a":"bob","anchor_b":[0,5,0]}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn joint_missing_body_a_fails() {
        let assets = vec![asset(
            "j",
            "Joint",
            serde_json::json!({"kind":"fixed","body_b":"b"}),
        )];
        assert!(err_text(&assets).contains("body_a"));
    }

    #[test]
    fn joint_unknown_body_b_fails() {
        let assets = vec![
            asset(
                "box",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset("a", "Prop", serde_json::json!({"mesh":"box"})),
            asset(
                "j",
                "Joint",
                serde_json::json!({"kind":"fixed","body_a":"a","body_b":"ghost"}),
            ),
        ];
        assert!(err_text(&assets).contains("ghost"));
    }

    #[test]
    fn joint_unknown_kind_fails() {
        let assets = vec![
            asset(
                "box",
                "ProceduralMesh",
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset("a", "Prop", serde_json::json!({"mesh":"box"})),
            asset(
                "j",
                "Joint",
                serde_json::json!({"kind":"frumpus","body_a":"a"}),
            ),
        ];
        assert!(err_text(&assets).contains("frumpus"));
    }

    #[test]
    fn all_errors_are_collected() {
        // A prop with three independent bad references should report all three.
        let assets = vec![asset(
            "broken",
            "Prop",
            serde_json::json!({"mesh":"no_mesh","material":"no_mat","texture":"no_tex"}),
        )];
        let errs = validate_cross_references(&assets).unwrap_err();
        assert_eq!(errs.len(), 3);
    }
}
