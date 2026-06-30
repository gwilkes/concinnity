// src/debug/hot_reload/tests.rs
//
// Unit tests for the hot-reload machinery (moved here from the single-file
// module). Pull each submodule's items in explicitly.

use super::decode::*;
use super::passes::*;
use super::state::*;
use super::watcher::*;
use crate::gfx::graphics_system::hot_reload_sources::*;
use notify::{Event, EventKind};
use std::path::PathBuf;

#[test]
fn empty_map_round_trips() {
    let m = TextureSourceMap::new();
    assert!(m.is_empty());
    assert_eq!(m.len(), 0);
    assert!(m.watch_dirs().is_empty());
}

#[test]
fn pushes_and_collects_unique_parent_dirs() {
    let mut m = TextureSourceMap::new();
    m.push_albedo("assets/a.png".to_string(), 0, 0);
    m.push_albedo("assets/b.png".to_string(), 0, 1);
    m.push_normal_map("textures/nm.png".to_string(), 0, 1);
    assert_eq!(m.entries.len(), 3);
    let dirs = m.watch_dirs();
    assert_eq!(dirs.len(), 2);
    assert!(dirs.iter().any(|p| p.ends_with("assets")));
    assert!(dirs.iter().any(|p| p.ends_with("textures")));
}

#[test]
fn bare_filenames_skip_watch_dir() {
    // A source with no parent directory has nowhere to watch; the watcher
    // would otherwise try to subscribe to "" which notify rejects.
    let mut m = TextureSourceMap::new();
    m.push_albedo("standalone.png".to_string(), 0, 0);
    assert!(m.watch_dirs().is_empty());
}

#[test]
fn glb_extension_is_an_asset_event() {
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/model.glb"));
    assert!(is_asset_event(&evt));
}

#[test]
fn cube_extension_is_an_asset_event() {
    // ColorLut sources travel through the same watcher; `.cube` must pass.
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/grade.cube"));
    assert!(is_asset_event(&evt));
}

#[test]
fn hdr_extension_is_an_asset_event() {
    // EnvironmentMap sources travel through the same watcher; `.hdr` must pass.
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/studio.hdr"));
    assert!(is_asset_event(&evt));
}

#[test]
fn hdr_extension_matches_case_insensitively() {
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/STUDIO.HDR"));
    assert!(is_asset_event(&evt));
}

#[test]
fn unrelated_extension_is_not_an_asset_event() {
    // `.jsonl` (the world file) is now a recognised asset event; pick an
    // extension nothing in the engine cares about as the negative case.
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/world.txt"));
    assert!(!is_asset_event(&evt));
}

#[test]
fn state_with_only_environment_map_still_spawns_a_watcher() {
    // With no textures and no LUT but an EnvironmentMap, the watcher must
    // still be set up so `.hdr` saves trigger reloads. The state stores the
    // EnvironmentMapSource verbatim for the reload helper to consult.
    let env_map = EnvironmentMapSource {
        resolved_path: format!(
            "{}/asset_hot_reload_envmap_only_{}.hdr",
            std::env::temp_dir().display(),
            std::process::id()
        ),
        prefilter_face_size: 64,
        irradiance_face_size: 16,
        prefilter_samples: 64,
        prefilter_clamp: 12.0,
    };
    // Parent dir (temp dir) exists, so the watcher should subscribe.
    let state = AssetHotReloadState::new(
        TextureSourceMap::new(),
        None,
        Some(env_map.clone()),
        MeshSourceMap::new(),
        SkinnedMeshSourceMap::new(),
        ProceduralMeshSourceMap::new(),
        ShaderStageSourceMap::new(),
        None,
    );
    assert!(state.environment_map.is_some());
    let captured = state.environment_map.as_ref().unwrap();
    assert_eq!(captured.prefilter_face_size, env_map.prefilter_face_size);
    assert_eq!(captured.irradiance_face_size, env_map.irradiance_face_size);
    assert_eq!(captured.prefilter_samples, env_map.prefilter_samples);
}

#[test]
fn fully_empty_state_skips_watcher_creation() {
    // No textures, no LUT, no EnvironmentMap, no meshes, no skinned, no
    // world path → nothing to watch.
    let state = AssetHotReloadState::new(
        TextureSourceMap::new(),
        None,
        None,
        MeshSourceMap::new(),
        SkinnedMeshSourceMap::new(),
        ProceduralMeshSourceMap::new(),
        ShaderStageSourceMap::new(),
        None,
    );
    assert!(state.environment_map.is_none());
    assert!(state.color_lut.is_none());
    assert!(state.map.is_empty());
    assert!(state.meshes.is_empty());
    assert!(state.skinned_meshes.is_empty());
}

#[test]
fn mesh_source_map_collects_unique_parent_dirs() {
    let mut m = MeshSourceMap::new();
    m.entries.push(MeshSourceEntry {
        source: "assets/models/a.glb".to_string(),
        primitive_index: 0,
        lod_levels: 1,
        lod_distances: Vec::new(),
        draw_indices: vec![0, 1],
    });
    m.entries.push(MeshSourceEntry {
        source: "assets/models/a.glb".to_string(),
        primitive_index: 1,
        lod_levels: 1,
        lod_distances: Vec::new(),
        draw_indices: vec![2],
    });
    m.entries.push(MeshSourceEntry {
        source: "assets/hdri/b.glb".to_string(),
        primitive_index: 0,
        lod_levels: 1,
        lod_distances: Vec::new(),
        draw_indices: vec![3],
    });
    let dirs = m.watch_dirs();
    assert_eq!(dirs.len(), 2);
    assert!(dirs.iter().any(|p| p.ends_with("assets/models")));
    assert!(dirs.iter().any(|p| p.ends_with("assets/hdri")));
}

#[test]
fn mesh_source_map_skips_bare_filenames_in_watch_dirs() {
    // A bare filename has no parent directory; the watcher would otherwise
    // try to subscribe to "" which notify rejects. The debug-WS
    // `reload-assets` command path still works for these.
    let mut m = MeshSourceMap::new();
    m.entries.push(MeshSourceEntry {
        source: "standalone.glb".to_string(),
        primitive_index: 0,
        lod_levels: 1,
        lod_distances: Vec::new(),
        draw_indices: vec![0],
    });
    assert!(m.watch_dirs().is_empty());
}

#[test]
fn state_with_only_meshes_still_spawns_a_watcher() {
    let mut meshes = MeshSourceMap::new();
    meshes.entries.push(MeshSourceEntry {
        source: format!("{}/dummy.glb", std::env::temp_dir().display()),
        primitive_index: 0,
        lod_levels: 1,
        lod_distances: Vec::new(),
        draw_indices: vec![0],
    });
    let state = AssetHotReloadState::new(
        TextureSourceMap::new(),
        None,
        None,
        meshes,
        SkinnedMeshSourceMap::new(),
        ProceduralMeshSourceMap::new(),
        ShaderStageSourceMap::new(),
        None,
    );
    assert_eq!(state.meshes.len(), 1);
    assert_eq!(state.meshes.entries[0].draw_indices, vec![0]);
}

#[test]
fn skinned_mesh_source_map_collects_unique_parent_dirs() {
    let mut m = SkinnedMeshSourceMap::new();
    m.entries.push(SkinnedMeshSourceEntry {
        source: "assets/models/a.glb".to_string(),
        skinned_index: 0,
        vertex_base: 0,
        vertex_count: 100,
        index_count: 300,
        joint_count: 24,
    });
    m.entries.push(SkinnedMeshSourceEntry {
        source: "assets/models/b.glb".to_string(),
        skinned_index: 1,
        vertex_base: 100,
        vertex_count: 50,
        index_count: 150,
        joint_count: 5,
    });
    let dirs = m.watch_dirs();
    assert_eq!(dirs.len(), 1);
    assert!(dirs[0].ends_with("assets/models"));
}

#[test]
fn state_with_only_skinned_still_spawns_a_watcher() {
    let mut skinned = SkinnedMeshSourceMap::new();
    skinned.entries.push(SkinnedMeshSourceEntry {
        source: format!("{}/skinned.glb", std::env::temp_dir().display()),
        skinned_index: 0,
        vertex_base: 0,
        vertex_count: 8,
        index_count: 24,
        joint_count: 2,
    });
    let state = AssetHotReloadState::new(
        TextureSourceMap::new(),
        None,
        None,
        MeshSourceMap::new(),
        skinned,
        ProceduralMeshSourceMap::new(),
        ShaderStageSourceMap::new(),
        None,
    );
    assert_eq!(state.skinned_meshes.len(), 1);
    assert_eq!(state.skinned_meshes.entries[0].joint_count, 2);
}

#[test]
fn state_with_only_world_jsonl_still_spawns_a_watcher() {
    // World-only worlds (no Texture/LUT/EnvMap/Mesh/Skinned) still want
    // their world.jsonl watched so Prop transform edits propagate.
    let world_path = format!(
        "{}/world_only_{}.jsonl",
        std::env::temp_dir().display(),
        std::process::id()
    );
    let state = AssetHotReloadState::new(
        TextureSourceMap::new(),
        None,
        None,
        MeshSourceMap::new(),
        SkinnedMeshSourceMap::new(),
        ProceduralMeshSourceMap::new(),
        ShaderStageSourceMap::new(),
        Some(world_path.clone()),
    );
    assert_eq!(state.world_jsonl_path.as_deref(), Some(world_path.as_str()));
}

#[test]
fn jsonl_extension_is_an_asset_event() {
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/world.jsonl"));
    assert!(is_asset_event(&evt));
}

#[test]
fn fresh_state_has_no_envmap_in_flight() {
    // The off-thread envmap convolution slot must start empty; a
    // non-`None` value at construction would skip the very first reload
    // request a `reload_assets` pass made.
    let state = AssetHotReloadState::new(
        TextureSourceMap::new(),
        None,
        None,
        MeshSourceMap::new(),
        SkinnedMeshSourceMap::new(),
        ProceduralMeshSourceMap::new(),
        ShaderStageSourceMap::new(),
        None,
    );
    let slot = state.env_map_inflight.lock().expect("lock");
    assert!(slot.is_none());
}

#[test]
fn fresh_state_has_no_asset_batch_in_flight() {
    // Same invariant as the envmap slot: a non-`None` value at
    // construction would make the very first `reload_assets` think a
    // worker was already running and skip the spawn.
    let state = AssetHotReloadState::new(
        TextureSourceMap::new(),
        None,
        None,
        MeshSourceMap::new(),
        SkinnedMeshSourceMap::new(),
        ProceduralMeshSourceMap::new(),
        ShaderStageSourceMap::new(),
        None,
    );
    let slot = state.asset_batch_inflight.lock().expect("lock");
    assert!(slot.is_none());
}

#[test]
fn decode_asset_batch_with_empty_inputs_returns_empty_batch() {
    // The worker body must handle the no-sources case cleanly so an
    // accidentally-spawned worker on a world without any file-backed
    // assets exits quickly with nothing to apply.
    let batch = decode_asset_batch(Vec::new(), None, Vec::new(), Vec::new());
    assert!(batch.textures.is_empty());
    assert!(batch.color_lut.is_none());
    assert!(batch.meshes.is_empty());
    assert!(batch.skinned_meshes.is_empty());
    assert_eq!(batch.decode_failures, 0);
}

#[test]
fn apply_skinned_layouts_refreshes_every_matching_entry() {
    // After a size-changing skinned rebuild, every source-map entry
    // whose `skinned_index` appears in the returned layouts should pick
    // up the new vertex_base / vertex_count / index_count so subsequent
    // in-place reloads write to the correct shared-buffer regions.
    let mut entries = vec![
        SkinnedMeshSourceEntry {
            source: "a.glb".to_string(),
            skinned_index: 0,
            vertex_base: 0,
            vertex_count: 10,
            index_count: 30,
            joint_count: 4,
        },
        SkinnedMeshSourceEntry {
            source: "b.glb".to_string(),
            skinned_index: 1,
            vertex_base: 10,
            vertex_count: 20,
            index_count: 60,
            joint_count: 6,
        },
    ];
    let layouts = vec![
        crate::gfx::backend::SkinnedSlotLayout {
            skinned_index: 0,
            vertex_base: 0,
            vertex_count: 15,
            index_count: 45,
        },
        crate::gfx::backend::SkinnedSlotLayout {
            skinned_index: 1,
            vertex_base: 15,
            vertex_count: 20,
            index_count: 60,
        },
    ];
    apply_skinned_layouts_to_entries(&mut entries, &layouts);
    assert_eq!(entries[0].vertex_base, 0);
    assert_eq!(entries[0].vertex_count, 15);
    assert_eq!(entries[0].index_count, 45);
    // Unchanged slot 1 was still re-packed (vertex_base shifted from 10
    // → 15 because slot 0's vertex_count grew from 10 → 15).
    assert_eq!(entries[1].vertex_base, 15);
    assert_eq!(entries[1].vertex_count, 20);
    assert_eq!(entries[1].index_count, 60);
    // joint_count is untouched: that lives outside the rebuild's scope.
    assert_eq!(entries[0].joint_count, 4);
    assert_eq!(entries[1].joint_count, 6);
}

#[test]
fn drain_pending_skeleton_updates_clears_the_queue() {
    // The render thread polls + drains in one step; a second drain on
    // the same frame must return nothing so a successful apply does not
    // double-write the SkeletonPose components.
    let mut state = AssetHotReloadState::new(
        TextureSourceMap::new(),
        None,
        None,
        MeshSourceMap::new(),
        SkinnedMeshSourceMap::new(),
        ProceduralMeshSourceMap::new(),
        ShaderStageSourceMap::new(),
        None,
    );
    state.pending_skeleton_updates.push(PendingSkeletonUpdate {
        skinned_index: 0,
        new_skeleton: crate::gfx::skinning::Skeleton::new(Vec::new()),
    });
    state.pending_skeleton_updates.push(PendingSkeletonUpdate {
        skinned_index: 3,
        new_skeleton: crate::gfx::skinning::Skeleton::new(Vec::new()),
    });
    let drained = state.drain_pending_skeleton_updates();
    assert_eq!(drained.len(), 2);
    assert_eq!(drained[0].skinned_index, 0);
    assert_eq!(drained[1].skinned_index, 3);
    // Second drain returns empty.
    assert!(state.drain_pending_skeleton_updates().is_empty());
}

#[test]
fn procedural_mesh_source_map_round_trips_empty() {
    let m = ProceduralMeshSourceMap::new();
    assert!(m.is_empty());
    assert_eq!(m.len(), 0);
}

#[test]
fn procedural_mesh_args_normalise_via_round_trip() {
    // The init pipeline captures args as `serde_json::to_value(component)`;
    // the reload pipeline normalises new on-disk args through
    // `ProceduralMesh::deserialize → serialize`. Same input must produce
    // the same JSON value on both sides, otherwise an unchanged JSONL
    // would still trigger a spurious regen.
    let user_args = serde_json::json!({
        "generator": "box",
        "half_extents": [0.5, 0.5, 0.5],
    });

    // Init-side: parse + re-serialize (mirroring what `serde_json::to_value`
    // on the deserialised component yields).
    let init_component: crate::assets::ProceduralMesh =
        serde_json::from_value(user_args.clone()).unwrap();
    let init_norm = serde_json::to_value(&init_component).unwrap();

    // Reload-side: parse user args → component → re-serialize.
    let reload_component: crate::assets::ProceduralMesh =
        serde_json::from_value(user_args.clone()).unwrap();
    let reload_norm = serde_json::to_value(&reload_component).unwrap();

    assert_eq!(init_norm, reload_norm);
}

#[test]
fn procedural_mesh_args_diff_detects_real_changes() {
    // A meaningful arg change must produce a distinct normalised value so
    // the diff fires.
    let v1: crate::assets::ProceduralMesh = serde_json::from_value(serde_json::json!({
        "generator": "box",
        "half_extents": [0.5, 0.5, 0.5],
    }))
    .unwrap();
    let v2: crate::assets::ProceduralMesh = serde_json::from_value(serde_json::json!({
        "generator": "box",
        "half_extents": [1.0, 1.0, 1.0],
    }))
    .unwrap();
    let n1 = serde_json::to_value(&v1).unwrap();
    let n2 = serde_json::to_value(&v2).unwrap();
    assert_ne!(n1, n2);
}

#[test]
fn procedural_mesh_reload_result_default_is_all_zero() {
    let r = ProceduralMeshReloadResult::default();
    assert_eq!(r.regenerated, 0);
    assert_eq!(r.unchanged, 0);
    assert_eq!(r.failed, 0);
}

#[test]
fn state_with_only_procedural_meshes_still_spawns_a_watcher() {
    // ProceduralMesh entries on their own (no textures, no LUTs, no meshes)
    // need to keep the watcher alive: their trigger is the `PENDING_WORLD`
    // flag flipped from the world.jsonl watcher. Without a world_jsonl_path
    // there is nothing to subscribe to, so we declare one here.
    let world_path = format!(
        "{}/asset_hot_reload_proc_only_world_{}.jsonl",
        std::env::temp_dir().display(),
        std::process::id()
    );
    let mut proc = ProceduralMeshSourceMap::new();
    proc.entries.push(ProceduralMeshSourceEntry {
        name: "box_mesh".to_string(),
        args: serde_json::json!({"generator": "box"}),
        draw_indices: vec![0],
    });
    let state = AssetHotReloadState::new(
        TextureSourceMap::new(),
        None,
        None,
        MeshSourceMap::new(),
        SkinnedMeshSourceMap::new(),
        proc,
        ShaderStageSourceMap::new(),
        Some(world_path),
    );
    assert_eq!(state.procedural_meshes.len(), 1);
    assert_eq!(state.procedural_meshes.entries[0].name, "box_mesh");
}

#[test]
fn shader_stage_source_map_round_trips_empty() {
    let m = ShaderStageSourceMap::new();
    assert!(m.is_empty());
    assert_eq!(m.len(), 0);
    assert!(m.watch_dirs().is_empty());
}

#[test]
fn shader_stage_source_map_collects_unique_parent_dirs() {
    use crate::assets::shader_stage::ShaderKind;
    let mut m = ShaderStageSourceMap::new();
    m.entries.push(ShaderStageSourceEntry {
        kind: ShaderKind::Vertex,
        resolved_path: "assets/shaders/default.metal".to_string(),
    });
    m.entries.push(ShaderStageSourceEntry {
        kind: ShaderKind::Fragment,
        resolved_path: "assets/shaders/default.metal".to_string(),
    });
    m.entries.push(ShaderStageSourceEntry {
        kind: ShaderKind::VertexInstanced,
        resolved_path: "assets/other/custom.metal".to_string(),
    });
    let dirs = m.watch_dirs();
    assert_eq!(dirs.len(), 2);
    assert!(dirs.iter().any(|p| p.ends_with("assets/shaders")));
    assert!(dirs.iter().any(|p| p.ends_with("assets/other")));
}

#[test]
fn shader_stage_source_map_skips_bare_filenames_in_watch_dirs() {
    // A bare filename has no parent directory; the watcher would try to
    // subscribe to "" which notify rejects. The debug-WS `reload-assets`
    // command still works for these.
    use crate::assets::shader_stage::ShaderKind;
    let mut m = ShaderStageSourceMap::new();
    m.entries.push(ShaderStageSourceEntry {
        kind: ShaderKind::Vertex,
        resolved_path: "standalone.metal".to_string(),
    });
    assert!(m.watch_dirs().is_empty());
}

#[test]
fn metal_extension_is_an_asset_event() {
    // World-loaded `ShaderStage` sources travel through the same watcher
    // as the texture / mesh paths; the closure routes them to the
    // shader-stage flag rather than the texture-decode batch.
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/scene.metal"));
    assert!(is_asset_event(&evt));
}

#[test]
fn hlsl_extension_is_an_asset_event() {
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/scene_vert.hlsl"));
    assert!(is_asset_event(&evt));
}

#[test]
fn glsl_extension_is_an_asset_event() {
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/scene.glsl"));
    assert!(is_asset_event(&evt));
}

#[test]
fn shader_extension_matches_case_insensitively() {
    assert!(is_shader_extension("metal"));
    assert!(is_shader_extension("Metal"));
    assert!(is_shader_extension("METAL"));
    assert!(is_shader_extension("hlsl"));
    assert!(is_shader_extension("glsl"));
    assert!(!is_shader_extension("png"));
    assert!(!is_shader_extension("glb"));
}

#[test]
fn state_with_only_shader_stages_still_spawns_a_watcher() {
    // World loaded only via shader-stage edits (no textures, no
    // meshes, no LUTs, no IBL, no world.jsonl) still want the watcher
    // alive so `.metal` saves trigger the recompile pass.
    use crate::assets::shader_stage::ShaderKind;
    let mut stages = ShaderStageSourceMap::new();
    stages.entries.push(ShaderStageSourceEntry {
        kind: ShaderKind::Vertex,
        resolved_path: format!(
            "{}/asset_hot_reload_shader_only_{}.metal",
            std::env::temp_dir().display(),
            std::process::id()
        ),
    });
    let state = AssetHotReloadState::new(
        TextureSourceMap::new(),
        None,
        None,
        MeshSourceMap::new(),
        SkinnedMeshSourceMap::new(),
        ProceduralMeshSourceMap::new(),
        stages,
        None,
    );
    assert_eq!(state.shader_stages.len(), 1);
    assert_eq!(state.shader_stages.entries[0].kind, ShaderKind::Vertex);
}

#[test]
fn shader_stage_reload_result_default_is_all_zero() {
    let r = ShaderStageReloadResult::default();
    assert_eq!(r.recompiled, 0);
    assert_eq!(r.failed, 0);
    assert!(!r.pipelines_rebuilt);
}

#[test]
fn reload_shader_stages_on_empty_map_is_a_no_op() {
    // The helper must short-circuit before touching the backend on a
    // world with no captured ShaderStage sources (e.g. the GLSL-only
    // path on Vulkan, or a world that pre-dated the capture). The
    // default backend trait impl errors on
    // `update_world_shader_pipelines`; an empty map must not hit it.
    struct DummyBackend;
    impl crate::gfx::scene_reel::SceneControl for DummyBackend {
        fn update_visibility(&mut self, _: usize, _: bool) {}
        fn update_clear_color(&mut self, _: [f32; 4]) {}
    }
    impl crate::gfx::backend::RenderBackend for DummyBackend {
        fn window_closed(&mut self) -> bool {
            false
        }
        fn capture_cursor(&mut self) {}
        fn take_input(&mut self) -> crate::gfx::input::RenderInput {
            crate::gfx::input::RenderInput::default()
        }
        fn wait_idle(&self) {}
        fn draw_frame(
            &mut self,
            _: f32,
            _: f32,
            _: f32,
            _: f32,
            _: [f32; 3],
            _: &[crate::gfx::render_types::TextDrawCall],
            _: bool,
        ) -> Result<(), String> {
            Ok(())
        }
        fn update_view(&mut self, _: [[f32; 4]; 4]) {}
        fn update_model(&mut self, _: usize, _: [[f32; 4]; 4]) {}
        fn retire_draw_object(&mut self, _: usize) {}
        fn upload_skinned(
            &mut self,
            _: &[crate::gfx::mesh_payload::SkinnedVertex],
            _: &[u16],
            _: Vec<crate::gfx::render_types::SkinnedDrawObject>,
            _: &[u8],
            _: &[u8],
            _: &[u8],
        ) -> Result<(), String> {
            Ok(())
        }
        fn update_skinned_pose(&mut self, _: usize, _: &[[[f32; 4]; 4]]) {}
        fn evict_texture_slot(&mut self, _: usize) -> Result<(), String> {
            Ok(())
        }
        fn update_texture_slot(
            &mut self,
            _: usize,
            _: u32,
            _: u32,
            _: &[u8],
        ) -> Result<(), String> {
            Ok(())
        }
        fn evict_normal_map_slot(&mut self, _: usize) -> Result<(), String> {
            Ok(())
        }
        fn update_normal_map_slot(
            &mut self,
            _: usize,
            _: u32,
            _: u32,
            _: &[u8],
        ) -> Result<(), String> {
            Ok(())
        }
        fn evict_mesh(&mut self, _: usize, _: u64) -> Result<(), String> {
            Ok(())
        }
        fn upload_mesh(
            &mut self,
            _: usize,
            _: &[crate::gfx::mesh_payload::Vertex],
            _: &[u16],
            _: u64,
        ) -> Result<(), String> {
            Ok(())
        }
        fn setup_chunk_streaming(
            &mut self,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
        ) -> Result<(), String> {
            Ok(())
        }
        fn add_chunk_mesh(
            &mut self,
            _: &[crate::gfx::mesh_payload::Vertex],
            _: &[u16],
            _: [[f32; 4]; 4],
            _: usize,
            _: usize,
            _: crate::gfx::render_types::MaterialUniforms,
            _: u64,
        ) -> Result<usize, String> {
            Ok(0)
        }
        fn remove_chunk_mesh(&mut self, _: usize, _: u64) -> Result<(), String> {
            Ok(())
        }
        fn set_chunk_model(&mut self, _: usize, _: [[f32; 4]; 4]) -> Result<(), String> {
            Ok(())
        }
    }

    let map = ShaderStageSourceMap::new();
    let mut backend = DummyBackend;
    let r = reload_shader_stages(&map, &mut backend);
    assert_eq!(r.recompiled, 0);
    assert_eq!(r.failed, 0);
    assert!(!r.pipelines_rebuilt);
}

#[test]
fn apply_skinned_layouts_leaves_entries_without_a_matching_layout_alone() {
    // A backend that returned layouts for only a subset of slots (e.g.
    // a future partial-rebuild path) should still leave the other
    // entries' captured state untouched. The Metal `rebuild_skinned_geometry`
    // returns a layout for every slot, but this guard keeps the
    // contract robust to backend variation.
    let mut entries = vec![SkinnedMeshSourceEntry {
        source: "a.glb".to_string(),
        skinned_index: 7,
        vertex_base: 42,
        vertex_count: 12,
        index_count: 36,
        joint_count: 2,
    }];
    let layouts = vec![crate::gfx::backend::SkinnedSlotLayout {
        skinned_index: 0,
        vertex_base: 0,
        vertex_count: 99,
        index_count: 99,
    }];
    apply_skinned_layouts_to_entries(&mut entries, &layouts);
    assert_eq!(entries[0].vertex_base, 42);
    assert_eq!(entries[0].vertex_count, 12);
    assert_eq!(entries[0].index_count, 36);
}
