// GraphicsSystem one-time setup: backend creation, draw-list build, and the
// shader / texture / streaming wiring performed on the first tick.

use crate::assets::{
    BlockType, Camera3D, ColorLut, Decal, DirectionalLight, EnvironmentMap, Font, GlassPanel,
    GraphicsConfig, HitRegion, Material, Model, ParticleEmitter, PointLight, PostProcessConfig,
    Prop, SdfVolume, ShaderKind, ShaderStage, StreamingConfig, TextLabel, Texture, VolumetricFog,
    VoxelWorld, WaterSurface, Window,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Component, PipelineContext};
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::{
    draw_list::{self, MaterialEntry},
    lights, skinning, text,
};
use std::time::Instant;

use super::helpers::*;
use super::*;

impl GraphicsSystem {
    pub(super) fn run_init(&mut self, ctx: &mut PipelineContext) {
        // Persisted settings-menu choices override the world's authored defaults
        // below (each field is None when the user never changed that setting).
        let user_graphics = crate::config::Settings::load().graphics;

        if let Some(w) = ctx.drain::<Window>().into_iter().next() {
            self.window_args = w.to_args();
        }
        if let Some(m) = user_graphics.window_mode {
            self.window_args.mode = m;
        }
        if let Some([w, h]) = user_graphics.window_size {
            self.window_args.width = w;
            self.window_args.height = h;
        }

        if let Some(c) = ctx.drain::<GraphicsConfig>().into_iter().next() {
            let args = c.to_args();
            self.frames_in_flight = args.frames_in_flight as usize;
            self.vsync = args.vsync;
            self.clear_color = args.clear_color;
            self.max_frames = args.max_frames;
            self.shadow_map_size = args.shadow_map_size;
            self.shadow_update = args.shadow_update;
        }
        // A persisted vsync choice overrides the world's value. Applied outside
        // the GraphicsConfig block (unconditional), matching window_mode /
        // window_size, so it wins over both the authored value and the default.
        if let Some(v) = user_graphics.vsync {
            self.vsync = v;
        }

        // Resolve post-process tunables. The first declared PostProcessConfig
        // wins; with none declared the renderer uses the stack defaults. The
        // `taa` toggle is a plain bool threaded alongside the resolved params.
        let post_config = ctx.drain::<PostProcessConfig>().into_iter().next();
        let mut post_process = post_config
            .as_ref()
            .map(|c| c.resolve())
            .unwrap_or(crate::gfx::render_types::PostProcessParams::DEFAULT);
        // A persisted exposure choice (the Exposure slider) overrides the
        // world's `exposure_ev`. Stored as EV (stops); convert to the linear
        // multiplier the shaders use, clamped like `PostProcessConfig::resolve`.
        // Applied live at runtime and re-applied here each launch so a persisted
        // value survives a restart.
        // Persisted slider choices override the world's values, re-applied here
        // each launch so they survive a restart. The transform / clamp is shared
        // with the live drag-apply via `settings::slider_apply_value`, so the
        // value re-applied at launch matches the value applied at drag time.
        use crate::gfx::settings::slider_apply_value;
        if let Some(ev) = user_graphics.exposure_ev {
            post_process.exposure = slider_apply_value("exposure", ev);
        }
        if let Some(v) = user_graphics.bloom_intensity {
            post_process.bloom_intensity = slider_apply_value("bloom_intensity", v);
        }
        if let Some(v) = user_graphics.bloom_threshold {
            post_process.bloom_threshold = slider_apply_value("bloom_threshold", v);
        }
        if let Some(v) = user_graphics.vignette {
            post_process.vignette = slider_apply_value("vignette", v);
        }
        if let Some(v) = user_graphics.lut_strength {
            post_process.lut_strength = slider_apply_value("lut_strength", v);
        }
        // Keep a copy as the live source of truth for the slider settings to
        // read at init and mutate at runtime (PostProcessParams is Copy, so the
        // value is still passed into the backend below).
        self.post_process = post_process;
        // Ambient (IBL) scale: the world's `PostProcessConfig.ambient_intensity`
        // overridden by any persisted choice. It rides `LightUniforms`, not
        // `PostProcessParams`, so it is held here and pushed to the backend once
        // after it is built (the world value is already seeded at backend init,
        // so this only matters for a persisted override). Clamped like
        // `PostProcessConfig::ambient_intensity`.
        let world_ambient = post_config
            .as_ref()
            .map(|c| c.ambient_intensity())
            .unwrap_or(1.0);
        self.ambient_intensity = slider_apply_value(
            "ambient_intensity",
            user_graphics.ambient_intensity.unwrap_or(world_ambient),
        );
        // Quality-feature toggles: the world's config overlaid with the user's
        // persisted choices, stored as the source of truth for the Quality-group
        // rows. A runtime toggle flips a field here, re-derives the per-feature
        // settings, and rebuilds the affected GPU resources. The overlay applies
        // only when the world declares a config: overriding a feature is
        // meaningless without its tunables, and the upscaler / ambient resolution
        // below intentionally keys off `post_config.is_some()` (synthesizing a
        // config would wrongly engage the upscaler). The stored copy is still
        // defaulted when absent so a runtime toggle has a config to flip.
        self.post_config = post_config.clone().unwrap_or_default();
        if post_config.is_some() {
            if let Some(v) = user_graphics.taa {
                super::set_quality_toggle(&mut self.post_config, "taa", v);
            }
            if let Some(v) = user_graphics.ssao {
                super::set_quality_toggle(&mut self.post_config, "ssao", v);
            }
            if let Some(v) = user_graphics.ssr {
                super::set_quality_toggle(&mut self.post_config, "ssr", v);
            }
            if let Some(v) = user_graphics.ray_traced_reflections {
                super::set_quality_toggle(&mut self.post_config, "ray_traced_reflections", v);
            }
            if let Some(v) = user_graphics.ssgi {
                super::set_quality_toggle(&mut self.post_config, "ssgi", v);
            }
            if let Some(v) = user_graphics.auto_exposure {
                super::set_quality_toggle(&mut self.post_config, "auto_exposure", v);
            }
        }
        // Per-feature settings, derived from the overlaid config. Each is the
        // init-time gate the backend builds against; the same derivation feeds a
        // live rebuild (`derive_quality_settings`). RT reflections need an
        // RT-capable GPU, falling back to SSR where ray tracing is unavailable.
        // RT takes precedence over SSR where both are on (the graph builder picks
        // `RtReflections`), reusing the same SSR pre-pass G-buffer + resolve
        // target.
        let taa_enabled = self.post_config.taa;
        let ssao_settings = self.post_config.ssao_settings();
        let ssr_settings = self.post_config.ssr_settings();
        let rt_reflection_settings = self.post_config.rt_reflection_settings();
        let reflection_blur_scale = self.post_config.reflection_blur_divisor();
        let ssgi_settings = self.post_config.ssgi_settings();
        // The authored `exposure_ev` becomes an additive bias on the adapted EV
        // when auto-exposure is on; otherwise the static path bakes it into
        // `post_process.exposure` (resolve()) and the bias here is unused.
        let auto_exposure_settings = self.post_config.auto_exposure_settings();
        let auto_exposure_bias_ev = self.post_config.exposure_ev;
        // HDR display output toggle, gated on the platform advertising an
        // HDR-capable surface (else it warns and falls back to the SDR composite
        // path).
        let hdr_display = post_config.as_ref().map(|c| c.hdr_display).unwrap_or(false);
        let hdr_pq = post_config.as_ref().map(|c| c.hdr_pq).unwrap_or(false);
        // Temporal upscaling toggle + per-axis render scale, resolved from
        // `PostProcessConfig.upscale_quality`.
        let temporal_upscaling = post_config
            .as_ref()
            .map(|c| c.temporal_upscaling)
            .unwrap_or(false);
        // Render-scale (upscaling quality): the world's choice overridden by any
        // persisted settings-menu choice. Restart-required -- the upscaler and
        // render targets are sized from this once, here. `self.render_scale` is
        // kept for the settings row to display and cycle.
        let world_quality = post_config
            .as_ref()
            .map(|c| c.upscale_quality)
            .unwrap_or_default();
        self.render_scale = user_graphics.render_scale.unwrap_or(world_quality);
        let upscale_scale = if post_config.is_some() {
            self.render_scale.scale()
        } else {
            1.0
        };

        // Set each settings value label to its live value before the first
        // render, so a persisted/authored choice shows instead of the build's
        // placeholder. HitRegions are still present here: GraphicsSystem.init
        // runs before UiInputSystem.init, which drains them.
        let (vsync, mode, win_w, win_h, scale) = (
            self.vsync,
            self.window_args.mode,
            self.window_args.width,
            self.window_args.height,
            self.render_scale,
        );
        // Audio / controls value labels read from the persisted settings store
        // (with the baseline default when unset); their owning systems apply the
        // value at their own init.
        let user_settings = crate::config::Settings::load();
        let master_volume = user_settings
            .audio
            .master_volume
            .unwrap_or(crate::gfx::settings::DEFAULT_MASTER_VOLUME);
        // Movement key map: a persisted rebind set overrides the engine default.
        // Pushed to the backend after it is built (below) and used to sync the
        // Controls-tab rebind row labels (`init_rebind_rows`).
        self.keymap = user_settings.controls.keymap.unwrap_or_default();
        // Snapshot of the resolved quality toggles for the value-label arm below
        // (a copy, matching the other snapshot locals, so the closure does not
        // borrow self while ctx is borrowed mutably).
        let quality_cfg = self.post_config.clone();
        sync_setting_value_labels(ctx, |key| match key {
            "vsync" => Some(vsync as usize),
            "window_mode" => Some(crate::gfx::settings::window_mode_index(mode)),
            "window_size" => Some(crate::gfx::settings::window_size_index(win_w, win_h)),
            "render_scale" => Some(crate::gfx::settings::render_scale_index(scale)),
            "master_volume" => Some(crate::gfx::settings::master_volume_index(master_volume)),
            // mouse_sensitivity is a slider now, synced by `init_sliders`.
            // Quality toggles: index 0 = Off, 1 = On, matching OFF_ON_OPTIONS.
            key if crate::gfx::settings::is_quality_toggle(key) => {
                super::quality_toggle_on(&quality_cfg, key).map(|on| on as usize)
            }
            _ => None,
        });
        // Capture the slider rows and sync each handle + value label to its live
        // value (e.g. the persisted/authored exposure). Like the cycle-row sync
        // above, this runs before UiInputSystem drains the HitRegions.
        self.init_sliders(ctx);
        // Capture the rebind rows and sync each value label to the live bound
        // key (persisted or default). Like the slider sync, before UiInputSystem
        // drains the HitRegions.
        self.init_rebind_rows(ctx);
        // Capture each ScrollPanel's per-element clip bands for the draw path,
        // before UiInputSystem drains the panels (init order: graphics first).
        self.init_clip_rects(ctx);
        // Upscaler backend selector from `PostProcessConfig.upscale_backend`.
        // Honoured by the DirectX and Vulkan backends (FSR3 / DLSS / XeSS);
        // Metal always uses MetalFX, so it ignores the selector.
        let upscale_backend = post_config
            .as_ref()
            .map(|c| c.upscale_backend)
            .unwrap_or_default();
        // Two-pass Hi-Z occlusion toggle, gated on the bindless GPU-cull path
        // being active (the phase-2 cull pipeline must exist).
        let occlusion_two_pass = post_config
            .as_ref()
            .map(|c| c.occlusion_two_pass)
            .unwrap_or(false);

        // Resolve the asset-streaming config. The first declared StreamingConfig
        // wins; with none declared, streaming stays off and every texture is
        // uploaded eagerly at init as before.
        let streaming_config = ctx.drain::<StreamingConfig>().into_iter().next();

        // Infinite-world chunk streaming. The first declared VoxelWorld wins;
        // with none declared, no chunks stream. BlockTypes are drained here so
        // the runtime can resolve the VoxelWorld palette to chunk-mesh data.
        let voxel_world = ctx.drain::<VoxelWorld>().into_iter().next();
        let block_types: std::collections::HashMap<AssetId, BlockType> = ctx
            .drain::<BlockType>()
            .into_iter()
            .map(|bt| (bt.asset_id, bt))
            .collect();

        // Whether the blob payloads came from files on disk (`cn run`) rather
        // than an in-memory build (`cn debug`). Captured before the blobs are
        // released; the streaming subsystem uses it to pick a disk-backed
        // payload source so streamed bytes need not stay RAM-resident.
        let blob_disk_backed = ctx.blob.disk_backed();

        // Snapshot each ProceduralMesh's args before `load_mesh_geometry`
        // drains them, so the world.jsonl hot-reload pass can diff a fresh
        // on-disk args object against the init state and re-run the generator
        // when they differ. Captured as a `serde_json::Value` keyed by AssetId;
        // a `None` here (hot-reload off) keeps the captured set empty so the
        // reload pass has nothing to inspect on `cn run`. Names come from the
        // interner so the reload log can read "regenerated 'box_mesh'" instead
        // of an opaque id.
        let proc_mesh_args_snapshot: std::collections::HashMap<
            AssetId,
            (String, serde_json::Value),
        > = if crate::app::dev_flags::enabled() {
            let name_table = crate::ecs::asset_id::name_table();
            ctx.query::<crate::assets::ProceduralMesh>()
                .filter_map(|pm| {
                    let name = name_table.get(pm.asset_id.0 as usize).cloned()?;
                    let v = serde_json::to_value(pm).ok()?;
                    Some((pm.asset_id, (name, v)))
                })
                .collect()
        } else {
            std::collections::HashMap::new()
        };

        let (mesh_geometry, mesh_sources, always_resident_meshes) =
            match draw_list::load_mesh_geometry(ctx) {
                Some(m) => m,
                None => {
                    self.failed = true;
                    return;
                }
            };

        // Drain SkinnedMesh assets and decode their geometry payloads now,
        // before the shared blob is released. The skeleton, world transform,
        // and material references travel in the component args; only the
        // vertex/index geometry lives in the compiled blob.
        // Decoded geometry for one SkinnedMesh: the asset, its vertices, LOD0
        // indices, the bind-pose skeleton, and LOD alternates.
        type SkinnedGeometry = (
            crate::assets::SkinnedMesh,
            Vec<crate::gfx::mesh_payload::SkinnedVertex>,
            Vec<u16>,
            Vec<crate::assets::JointDef>,
            Vec<(f32, Vec<u16>)>,
        );
        let mut skinned_geometry: Vec<SkinnedGeometry> = Vec::new();
        let mut skinned_blob_indices: Vec<u32> = Vec::new();
        for sm in ctx.drain::<crate::assets::SkinnedMesh>() {
            let locator = match &sm.locator {
                Some(l) => l.clone(),
                None => {
                    tracing::error!(
                        "GraphicsSystem: SkinnedMesh '{}' has no compiled payload",
                        sm.asset_id
                    );
                    self.failed = true;
                    return;
                }
            };
            skinned_blob_indices.push(locator.blob_index);
            let bytes = match ctx.read_payload(&locator) {
                Ok(b) => b.to_vec(),
                Err(e) => {
                    tracing::error!(
                        "GraphicsSystem: failed to read SkinnedMesh '{}' payload: {:?}",
                        sm.asset_id,
                        e
                    );
                    self.failed = true;
                    return;
                }
            };
            match crate::gfx::mesh_payload::deserialise_skinned_with_lods(&bytes) {
                Ok((v, idx, payload_joints, alternates)) => {
                    let skeleton = crate::geometry::payload_joints_to_defs(payload_joints);
                    skinned_geometry.push((sm, v, idx, skeleton, alternates));
                }
                Err(e) => {
                    tracing::error!("GraphicsSystem: malformed SkinnedMesh payload: {}", e);
                    self.failed = true;
                    return;
                }
            }
        }

        // drain Model components into a name-keyed map for Prop lookup
        let models = ctx.drain::<Model>();
        let model_map: std::collections::HashMap<AssetId, Vec<crate::assets::SubMeshRef>> =
            models.into_iter().map(|m| (m.asset_id, m.meshes)).collect();

        // decode Room payloads before shaders/textures are read; all payloads
        // live in the same blob and must be consumed before it is released
        let (room_geometry, room_blob_indices) = match draw_list::load_room_geometry(ctx) {
            Some(r) => r,
            None => {
                self.failed = true;
                return;
            }
        };

        let mut shaders = ctx.drain::<ShaderStage>();
        let find_shader = |shaders: &mut Vec<ShaderStage>, kind: ShaderKind| {
            shaders
                .iter()
                .position(|s| s.kind == kind)
                .map(|i| shaders.remove(i))
        };

        let vert_instanced_shader = find_shader(&mut shaders, ShaderKind::VertexInstanced);

        let vert_shader = match find_shader(&mut shaders, ShaderKind::Vertex) {
            Some(s) => s,
            None => {
                tracing::error!(
                    "GraphicsSystem: no vertex ShaderStage found -- add one to world.jsonl"
                );
                self.failed = true;
                return;
            }
        };
        let frag_shader = match find_shader(&mut shaders, ShaderKind::Fragment) {
            Some(s) => s,
            None => {
                tracing::error!(
                    "GraphicsSystem: no fragment ShaderStage found -- add one to world.jsonl"
                );
                self.failed = true;
                return;
            }
        };

        let vert_locator = match &vert_shader.locator {
            Some(l) => l.clone(),
            None => {
                tracing::error!(
                    "GraphicsSystem: vertex ShaderStage '{}' has no compiled payload",
                    vert_shader.source
                );
                self.failed = true;
                return;
            }
        };
        let frag_locator = match &frag_shader.locator {
            Some(l) => l.clone(),
            None => {
                tracing::error!(
                    "GraphicsSystem: fragment ShaderStage '{}' has no compiled payload",
                    frag_shader.source
                );
                self.failed = true;
                return;
            }
        };

        // instanced vertex shader is optional; required only when at least
        // one InstancedProp is in the world (which we don't know yet).
        let vert_instanced_locator = vert_instanced_shader
            .as_ref()
            .and_then(|s| s.locator.clone());

        // Capture every world-loaded ShaderStage's resolved on-disk source
        // path so the asset hot-reload watcher can recompile + rebuild
        // pipelines on a `.metal` / `.hlsl` / `.glsl` save. Stages whose
        // current-platform source is the embedded GLSL fallback (or whose
        // declaration uses a non-platform-compatible extension) carry no
        // file to watch and are skipped; the inline GLSL path keeps
        // rendering at whatever was baked in.
        let mut shader_stage_source_map = super::hot_reload_sources::ShaderStageSourceMap::new();
        if crate::app::dev_flags::enabled() {
            let mut capture = |stage_opt: Option<&ShaderStage>, kind: ShaderKind| {
                let Some(stage) = stage_opt else {
                    return;
                };
                let Some(raw) = stage.current_platform_source() else {
                    return;
                };
                // Engine-bundled built-ins are served from `include_str!`-baked
                // source by `concinnity_cook::shader::compile_shader`, not from
                // disk, and a separate watcher in `crate::metal::hot_reload`
                // already covers them via `src/metal/shaders/`. Skip them
                // here so the asset watcher does not redundantly subscribe to
                // a path it cannot meaningfully reload.
                if crate::build::shader::builtin_shader_source(&raw).is_some() {
                    return;
                }
                let resolved = crate::assets::shader_stage::resolve_runtime_source_path(&raw);
                shader_stage_source_map.entries.push(
                    super::hot_reload_sources::ShaderStageSourceEntry {
                        kind,
                        resolved_path: resolved,
                    },
                );
            };
            capture(Some(&vert_shader), ShaderKind::Vertex);
            capture(Some(&frag_shader), ShaderKind::Fragment);
            capture(vert_instanced_shader.as_ref(), ShaderKind::VertexInstanced);
        }

        // drain textures and record a name->slot mapping for Prop texture lookup.
        // Under `cn debug` we also capture the file-backed source paths into
        // `asset_source_map` so the hot-reload watcher can re-decode + re-upload
        // when the user saves a texture on disk. Procedural textures (generator
        // non-empty) and source-less assets carry no source file and are
        // omitted from the map.
        let textures = ctx.drain::<Texture>();
        let mut texture_name_to_slot: std::collections::HashMap<AssetId, usize> =
            std::collections::HashMap::new();
        let mut texture_locators = Vec::new();
        let mut asset_source_map = super::hot_reload_sources::TextureSourceMap::new();
        let capture_sources = crate::app::dev_flags::enabled();
        for (slot, tex) in textures.iter().enumerate() {
            match &tex.locator {
                Some(l) => {
                    // key by asset id (injected via inject_name) so Prop.texture
                    // references match regardless of generator or source path
                    texture_name_to_slot.insert(tex.asset_id, slot);
                    texture_locators.push(l.clone());
                    if capture_sources && tex.generator.is_empty() && !tex.source.is_empty() {
                        asset_source_map.push_albedo(tex.source.clone(), tex.image_index, slot);
                    }
                }
                None => {
                    tracing::error!(
                        "GraphicsSystem: Texture has no compiled payload -- did the build succeed?"
                    );
                    self.failed = true;
                    return;
                }
            }
        }

        // drain Materials and build a name -> (albedo_slot, normal_map_slot, gpu uniforms) map.
        // Materials have no payload; all data lives in their args.
        // normal_map_slot 0 is always the flat-normal fallback; named maps start at 1.
        let mut normal_map_name_to_slot: std::collections::HashMap<AssetId, usize> =
            std::collections::HashMap::new();
        let mut normal_map_locators: Vec<crate::ecs::PayloadLocator> = Vec::new();

        let mut material_map: std::collections::HashMap<AssetId, MaterialEntry> =
            std::collections::HashMap::new();
        for mat in ctx.drain::<Material>() {
            let albedo_slot = match mat.albedo {
                None => 0,
                Some(albedo_id) => match texture_name_to_slot.get(&albedo_id) {
                    Some(&slot) => slot,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Material {} references unknown texture {} -- add a Texture asset with that id",
                            mat.asset_id,
                            albedo_id
                        );
                        self.failed = true;
                        return;
                    }
                },
            };

            // normal_map_slot 0 = flat-normal fallback; real maps get slot >= 1
            let normal_map_slot = match mat.normal_map {
                None => 0,
                Some(nm_id) => {
                    if let Some(&slot) = normal_map_name_to_slot.get(&nm_id) {
                        slot
                    } else {
                        match textures.iter().find(|t| t.asset_id == nm_id) {
                            Some(tex) => match &tex.locator {
                                Some(l) => {
                                    let slot = normal_map_locators.len() + 1;
                                    normal_map_name_to_slot.insert(nm_id, slot);
                                    normal_map_locators.push(l.clone());
                                    if capture_sources
                                        && tex.generator.is_empty()
                                        && !tex.source.is_empty()
                                    {
                                        asset_source_map.push_normal_map(
                                            tex.source.clone(),
                                            tex.image_index,
                                            slot,
                                        );
                                    }
                                    slot
                                }
                                None => {
                                    tracing::error!(
                                        "GraphicsSystem: Material {} normal_map {} has no compiled payload",
                                        mat.asset_id,
                                        nm_id
                                    );
                                    self.failed = true;
                                    return;
                                }
                            },
                            None => {
                                tracing::error!(
                                    "GraphicsSystem: Material {} references unknown normal_map {} -- add a Texture asset with that id",
                                    mat.asset_id,
                                    nm_id
                                );
                                self.failed = true;
                                return;
                            }
                        }
                    }
                }
            };

            // Optional secondary albedo/normal pair for the terrain
            // shader's slope-blending mode. Same resolution as the
            // primary pair (albedo into `texture_name_to_slot`, normal
            // into `normal_map_name_to_slot`); slot 0 + slot 0 fall
            // through when either is unset and the shader's
            // `terrain_blend > 0` gate is what controls whether the
            // secondary actually gets sampled.
            let albedo_secondary_slot: u32 = match mat.albedo_secondary {
                None => 0,
                Some(id) => match texture_name_to_slot.get(&id) {
                    Some(&slot) => slot as u32,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Material {} references unknown albedo_secondary texture {} -- add a Texture asset with that id",
                            mat.asset_id,
                            id
                        );
                        self.failed = true;
                        return;
                    }
                },
            };
            let normal_secondary_slot: u32 = match mat.normal_secondary {
                None => 0,
                Some(nm_id) => {
                    if let Some(&slot) = normal_map_name_to_slot.get(&nm_id) {
                        slot as u32
                    } else {
                        match textures.iter().find(|t| t.asset_id == nm_id) {
                            Some(tex) => match &tex.locator {
                                Some(l) => {
                                    let slot = normal_map_locators.len() + 1;
                                    normal_map_name_to_slot.insert(nm_id, slot);
                                    normal_map_locators.push(l.clone());
                                    if capture_sources
                                        && tex.generator.is_empty()
                                        && !tex.source.is_empty()
                                    {
                                        asset_source_map.push_normal_map(
                                            tex.source.clone(),
                                            tex.image_index,
                                            slot,
                                        );
                                    }
                                    slot as u32
                                }
                                None => {
                                    tracing::error!(
                                        "GraphicsSystem: Material {} normal_secondary {} has no compiled payload",
                                        mat.asset_id,
                                        nm_id
                                    );
                                    self.failed = true;
                                    return;
                                }
                            },
                            None => {
                                tracing::error!(
                                    "GraphicsSystem: Material {} references unknown normal_secondary {} -- add a Texture asset with that id",
                                    mat.asset_id,
                                    nm_id
                                );
                                self.failed = true;
                                return;
                            }
                        }
                    }
                }
            };

            // Emissive + packed-ORM maps live in the albedo region of the
            // bindless pool, so they resolve through `texture_name_to_slot`
            // exactly like the primary/secondary albedo. Slot 0 (unset) is the
            // sentinel the shader gates on to keep the scalar fallback.
            let emissive_map_slot: u32 = match mat.emissive_map {
                None => 0,
                Some(id) => match texture_name_to_slot.get(&id) {
                    Some(&slot) => slot as u32,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Material {} references unknown emissive_map texture {} -- add a Texture asset with that id",
                            mat.asset_id,
                            id
                        );
                        self.failed = true;
                        return;
                    }
                },
            };
            let orm_map_slot: u32 = match mat.orm_map {
                None => 0,
                Some(id) => match texture_name_to_slot.get(&id) {
                    Some(&slot) => slot as u32,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Material {} references unknown orm_map texture {} -- add a Texture asset with that id",
                            mat.asset_id,
                            id
                        );
                        self.failed = true;
                        return;
                    }
                },
            };

            let uniforms = crate::gfx::render_types::MaterialUniforms {
                roughness: mat.roughness,
                metallic: mat.metallic,
                macro_variation: mat.macro_variation,
                terrain_blend: mat.terrain_blend,
                tint: mat.tint,
                _pad2: 0.0,
                emissive: mat.emissive_factor,
                secondary_blend_sharpness: mat.secondary_blend_sharpness,
                albedo_secondary_index: albedo_secondary_slot,
                normal_secondary_index: normal_secondary_slot,
                emissive_map_index: emissive_map_slot,
                orm_map_index: orm_map_slot,
                opacity: mat.opacity,
                transparent: u32::from(mat.transparent),
                see_through: u32::from(mat.see_through),
            };
            material_map.insert(mat.asset_id, (albedo_slot, normal_map_slot, uniforms));
        }

        // Build skinned draw objects, the shared skinned vertex/index buffers,
        // and bind-pose skeletons from the decoded SkinnedMesh geometry. Runs
        // after the material map so SkinnedMesh material references resolve.
        let mut skinned_vertices: Vec<crate::gfx::mesh_payload::SkinnedVertex> = Vec::new();
        let mut skinned_indices: Vec<u16> = Vec::new();
        let mut skinned_draw_objects: Vec<crate::gfx::render_types::SkinnedDrawObject> = Vec::new();
        let mut skinned_skeletons: Vec<(AssetId, skinning::Skeleton)> = Vec::new();
        // Asset hot-reload (`cn debug` only) needs the per-slot vertex region
        // + joint count so it can reject size + shape changes before pushing
        // to the backend. SkinnedMesh is 1:1 with its draw slot (no Prop
        // fan-out), so one entry per asset.
        let mut skinned_mesh_source_map = super::hot_reload_sources::SkinnedMeshSourceMap::new();
        for (sm, verts, idxs, joint_defs, lod_alts) in &skinned_geometry {
            let (texture_slot, normal_map_slot, material) = if let Some(mat_id) = sm.material {
                match material_map.get(&mat_id) {
                    Some(&(s, n, u)) => (s, n, u),
                    None => {
                        tracing::error!(
                            "GraphicsSystem: SkinnedMesh '{}' references unknown material {}",
                            sm.asset_id,
                            mat_id
                        );
                        self.failed = true;
                        return;
                    }
                }
            } else if let Some(tex_id) = sm.texture {
                (
                    *texture_name_to_slot.get(&tex_id).unwrap_or(&0),
                    0,
                    crate::gfx::render_types::MaterialUniforms::DEFAULT,
                )
            } else {
                (0, 0, crate::gfx::render_types::MaterialUniforms::DEFAULT)
            };

            let base = skinned_vertices.len() as u16;
            let index_offset = skinned_indices.len();
            skinned_vertices.extend_from_slice(verts);
            skinned_indices.extend(idxs.iter().map(|i| i + base));

            // LOD alternates share this slot's vertex region. The runtime
            // skinned IB is u16, so each alternate's mesh-relative indices
            // are rebased onto the same `base` as LOD0, identical to how
            // the shadow / velocity / SSAO / SSR pre-passes already consume
            // the IB.
            let mut lod_slices: Vec<crate::gfx::render_types::LodSlice> =
                Vec::with_capacity(lod_alts.len());
            for (switch_distance, alt_idx) in lod_alts {
                let alt_offset = skinned_indices.len();
                skinned_indices.extend(alt_idx.iter().map(|i| i + base));
                lod_slices.push(crate::gfx::render_types::LodSlice {
                    index_offset: alt_offset,
                    index_count: alt_idx.len(),
                    switch_distance: *switch_distance,
                });
            }

            let skeleton = crate::assets::build_skeleton_from_joint_defs(joint_defs);
            let joint_count = skeleton.len().min(crate::gfx::render_types::MAX_JOINTS);

            // Bind-pose (object-space) AABB over this mesh's vertices. The
            // GPU-driven skinned fold pads + transforms it per frame for culling.
            let (local_bb_min, local_bb_max) = if verts.is_empty() {
                ([0.0; 3], [0.0; 3])
            } else {
                let mut lo = [f32::INFINITY; 3];
                let mut hi = [f32::NEG_INFINITY; 3];
                for v in verts.iter() {
                    for a in 0..3 {
                        lo[a] = lo[a].min(v.pos[a]);
                        hi[a] = hi[a].max(v.pos[a]);
                    }
                }
                (lo, hi)
            };

            let skinned_index = skinned_draw_objects.len();
            skinned_draw_objects.push(crate::gfx::render_types::SkinnedDrawObject {
                vertex_base: base,
                vertex_count: verts.len(),
                index_offset,
                index_count: idxs.len(),
                model: sm.model_matrix(),
                texture_slot,
                normal_map_slot,
                material,
                visible: true,
                joint_count,
                local_bb_min,
                local_bb_max,
                lod_alternates: lod_slices,
            });
            if capture_sources && !sm.source.is_empty() {
                skinned_mesh_source_map.entries.push(
                    super::hot_reload_sources::SkinnedMeshSourceEntry {
                        source: sm.source.clone(),
                        skinned_index,
                        vertex_base: base,
                        vertex_count: verts.len(),
                        index_count: idxs.len(),
                        joint_count,
                    },
                );
            }
            skinned_skeletons.push((sm.asset_id, skeleton));
        }

        // read all payloads before releasing -- they may share a blob
        let vert_bytes = match ctx.read_payload(&vert_locator) {
            Ok(b) => b.to_vec(),
            Err(e) => {
                tracing::error!("GraphicsSystem: failed to read vertex shader: {:?}", e);
                self.failed = true;
                return;
            }
        };
        let frag_bytes = match ctx.read_payload(&frag_locator) {
            Ok(b) => b.to_vec(),
            Err(e) => {
                tracing::error!("GraphicsSystem: failed to read fragment shader: {:?}", e);
                self.failed = true;
                return;
            }
        };

        // The DirectX backend engages its bindless main pass only when the
        // main-shader override is empty (it then uses its embedded bindless
        // pipeline + the embedded default for any legacy/streamed fallback). The
        // built-in default ShaderStage compiles to non-empty DXBC, which would
        // pin every built-in world to the legacy per-draw path. When the world's
        // main shader IS the built-in default, hand DX empty bytes so it takes
        // the bindless path, matching Vulkan (whose default payload is already
        // empty) and Metal (whose `default.metal` drives its own bindless pass).
        // Custom-shader worlds keep their compiled bytes and the legacy path.
        // Metal loads its metallib from these bytes, so it is left untouched.
        #[cfg(backend_dx)]
        let (vert_bytes, frag_bytes) = {
            let is_builtin_main = |src: Option<String>| {
                matches!(
                    src.as_deref(),
                    Some("default_vert.hlsl") | Some("default_frag.hlsl") | Some("default.metal")
                )
            };
            if is_builtin_main(vert_shader.current_platform_source())
                && is_builtin_main(frag_shader.current_platform_source())
            {
                (Vec::new(), Vec::new())
            } else {
                (vert_bytes, frag_bytes)
            }
        };
        // The shadow shader is engine-internal now (compiled from
        // `shadow_map.metal`), so there is no per-world shadow payload. The
        // DX / Vulkan constructors still take a shadow byte slice pending their
        // own internal-shadow migration; Metal ignores it.
        let shadow_bytes: Vec<u8> = Vec::new();
        let vert_instanced_bytes = if let Some(ref locator) = vert_instanced_locator {
            match ctx.read_payload(locator) {
                Ok(b) => b.to_vec(),
                Err(e) => {
                    tracing::error!(
                        "GraphicsSystem: failed to read instanced vertex shader: {:?}",
                        e
                    );
                    self.failed = true;
                    return;
                }
            }
        } else {
            Vec::new()
        };

        let mut texture_data: Vec<(u32, u32, Vec<u8>)> = Vec::new();
        // Raw compiled texture payloads, kept past blob release so the
        // asset-streaming subsystem can re-decode them off the main thread.
        // Left empty when the blobs are disk-backed: the streamer then re-reads
        // each payload from its blob file instead of holding a RAM copy.
        let mut texture_payloads: Vec<Vec<u8>> = Vec::new();
        for locator in &texture_locators {
            let tex_bytes = match ctx.read_payload(locator) {
                Ok(b) => b.to_vec(),
                Err(e) => {
                    tracing::error!("GraphicsSystem: failed to read texture payload: {:?}", e);
                    self.failed = true;
                    return;
                }
            };
            match crate::build::texture::deserialise(&tex_bytes) {
                Ok(t) => texture_data.push(t),
                Err(e) => {
                    tracing::error!("GraphicsSystem: malformed texture payload: {}", e);
                    self.failed = true;
                    return;
                }
            }
            if !blob_disk_backed {
                texture_payloads.push(tex_bytes);
            }
        }

        // Drain the first EnvironmentMap asset and capture its payload.
        // The runtime supports at most one IBL environment per world;
        // additional declarations are logged and ignored. Under `cn debug` we also
        // capture the resolved HDR source path + sizing knobs into
        // `environment_map_source` so the hot-reload watcher knows what to
        // subscribe to and the reload helper can re-run the convolutions with
        // matching dimensions. Procedural `generator` declarations have no
        // file to watch and are skipped.
        let env_maps = ctx.drain::<EnvironmentMap>();
        let mut env_map_bytes: Option<Vec<u8>> = None;
        let mut environment_map_source: Option<super::hot_reload_sources::EnvironmentMapSource> =
            None;
        for (idx, em) in env_maps.iter().enumerate() {
            if idx > 0 {
                tracing::warn!(
                    "GraphicsSystem: ignoring extra EnvironmentMap '{}' (only the first is used)",
                    em.asset_id
                );
                continue;
            }
            if capture_sources && em.generator.is_empty() && !em.source.is_empty() {
                let resolved = crate::build::environment_map::resolve_source_path(&em.source);
                environment_map_source = Some(super::hot_reload_sources::EnvironmentMapSource {
                    resolved_path: resolved,
                    prefilter_face_size: em.prefilter_face_size,
                    irradiance_face_size: em.irradiance_face_size,
                    prefilter_samples: em.prefilter_samples,
                    prefilter_clamp: em.prefilter_clamp,
                });
            }
            match &em.locator {
                Some(l) => match ctx.read_payload(l) {
                    Ok(b) => env_map_bytes = Some(b.to_vec()),
                    Err(e) => {
                        tracing::error!(
                            "GraphicsSystem: failed to read EnvironmentMap '{}' payload: {:?}",
                            em.asset_id,
                            e
                        );
                        self.failed = true;
                        return;
                    }
                },
                None => {
                    tracing::error!(
                        "GraphicsSystem: EnvironmentMap '{}' has no compiled payload -- did the build succeed?",
                        em.asset_id
                    );
                    self.failed = true;
                    return;
                }
            }
        }

        // Drain the first ColorLut asset and capture its payload. At most one
        // colour-grading LUT per world; extras are logged and ignored. Under
        // `cn debug` we also capture the resolved source path into
        // `color_lut_source` so the hot-reload watcher knows what to subscribe
        // to and the reload helper knows where to re-read the LUT.
        let color_luts = ctx.drain::<ColorLut>();
        let mut color_lut_bytes: Option<Vec<u8>> = None;
        let mut color_lut_source: Option<super::hot_reload_sources::ColorLutSource> = None;
        for (idx, cl) in color_luts.iter().enumerate() {
            if idx > 0 {
                tracing::warn!(
                    "GraphicsSystem: ignoring extra ColorLut '{}' (only the first is used)",
                    cl.asset_id
                );
                continue;
            }
            if capture_sources && !cl.source.is_empty() {
                let resolved = crate::build::color_lut::resolve_source_path(&cl.source);
                color_lut_source = Some(super::hot_reload_sources::ColorLutSource {
                    resolved_path: resolved,
                });
            }
            match &cl.locator {
                Some(l) => match ctx.read_payload(l) {
                    Ok(b) => color_lut_bytes = Some(b.to_vec()),
                    Err(e) => {
                        tracing::error!(
                            "GraphicsSystem: failed to read ColorLut '{}' payload: {:?}",
                            cl.asset_id,
                            e
                        );
                        self.failed = true;
                        return;
                    }
                },
                None => {
                    tracing::error!(
                        "GraphicsSystem: ColorLut '{}' has no compiled payload -- did the build succeed?",
                        cl.asset_id
                    );
                    self.failed = true;
                    return;
                }
            }
        }

        let mut normal_map_data: Vec<(u32, u32, Vec<u8>)> = Vec::new();
        // Raw compiled normal-map payloads, kept past blob release so the
        // asset-streaming subsystem can re-decode them off the main thread.
        // Left empty when the blobs are disk-backed: the streamer then re-reads
        // each payload from its blob file instead of holding a RAM copy.
        let mut normal_map_payloads: Vec<Vec<u8>> = Vec::new();
        for locator in &normal_map_locators {
            let nm_bytes = match ctx.read_payload(locator) {
                Ok(b) => b.to_vec(),
                Err(e) => {
                    tracing::error!("GraphicsSystem: failed to read normal_map payload: {:?}", e);
                    self.failed = true;
                    return;
                }
            };
            match crate::build::texture::deserialise(&nm_bytes) {
                Ok(t) => normal_map_data.push(t),
                Err(e) => {
                    tracing::error!("GraphicsSystem: malformed normal_map payload: {}", e);
                    self.failed = true;
                    return;
                }
            }
            if !blob_disk_backed {
                normal_map_payloads.push(nm_bytes);
            }
        }

        // drain Font components; deserialise atlas + metrics for text rendering
        let fonts = ctx.drain::<Font>();
        let mut text_atlas_data: Vec<(u32, u32, Vec<u8>)> = Vec::new();
        for (slot, font) in fonts.iter().enumerate() {
            let locator = match &font.locator {
                Some(l) => l.clone(),
                None => {
                    tracing::error!(
                        "GraphicsSystem: Font '{}' has no compiled payload -- did the build succeed?",
                        font.asset_id
                    );
                    self.failed = true;
                    return;
                }
            };
            let bytes = match ctx.read_payload(&locator) {
                Ok(b) => b.to_vec(),
                Err(e) => {
                    tracing::error!(
                        "GraphicsSystem: failed to read Font '{}' payload: {:?}",
                        font.asset_id,
                        e
                    );
                    self.failed = true;
                    return;
                }
            };
            match crate::build::font::deserialise(&bytes) {
                Ok((aw, ah, supersample, rgba, metrics)) => {
                    let metrics_map: std::collections::HashMap<
                        u32,
                        crate::build::font::GlyphMetrics,
                    > = metrics.into_iter().map(|m| (m.char_code, m)).collect();
                    self.loaded_fonts.insert(
                        font.asset_id,
                        text::LoadedFont {
                            atlas_slot: slot,
                            cap_px: text::derive_cap_px(&metrics_map, font.size_px as f32),
                            metrics: metrics_map,
                            atlas_w: aw,
                            atlas_h: ah,
                            size_px: font.size_px as f32,
                            supersample: (supersample.max(1)) as f32,
                        },
                    );
                    text_atlas_data.push((aw, ah, rgba));
                }
                Err(e) => {
                    tracing::error!("GraphicsSystem: malformed Font payload: {}", e);
                    self.failed = true;
                    return;
                }
            }
        }

        // Indirect-ambient multiplier from PostProcessConfig, folded into the
        // shared LightUniforms so every backend's main pass scales its IBL /
        // flat-fallback ambient by it. 1.0 (the default) is a no-op.
        let ambient_intensity = post_config
            .as_ref()
            .map(|c| c.ambient_intensity())
            .unwrap_or(1.0);
        let light_uniforms = lights::build_light_uniforms(
            ctx.drain::<DirectionalLight>(),
            ctx.drain::<PointLight>(),
            ambient_intensity,
        );

        let font_blob_indices: Vec<u32> = fonts
            .iter()
            .filter_map(|f| f.locator.as_ref().map(|l| l.blob_index))
            .collect();

        // AudioSystem inits after GraphicsSystem and reads AudioClip payloads,
        // so any blob an AudioClip lives in must survive this release sweep.
        let audio_blobs = crate::assets::audio_clip::audio_clip_blob_indices(ctx);
        // SdfVolume payloads are drained later in this same init pass (see
        // the `sdf_volumes` block below), so the release sweep here must
        // also leave their blobs resident. Without this gate, any world
        // whose SDF shader bytes happen to land alone in a blob shows
        // "failed to read fragment shader payload: FileIo; skipping" at
        // runtime and the SDF surface never draws.
        let sdf_blobs = crate::assets::sdf_volume::sdf_volume_blob_indices(ctx);
        // PhysicsSystem inits after GraphicsSystem and reads the baked
        // heightfield collider grid from a heightfield ProceduralMesh's
        // payload, so those blobs must also survive this sweep.
        let terrain_blobs = crate::assets::procedural_mesh::heightfield_blob_indices(ctx);
        let mut released = std::collections::HashSet::new();
        for idx in std::iter::once(vert_locator.blob_index)
            .chain(std::iter::once(frag_locator.blob_index))
            .chain(vert_instanced_locator.iter().map(|l| l.blob_index))
            .chain(texture_locators.iter().map(|l| l.blob_index))
            .chain(normal_map_locators.iter().map(|l| l.blob_index))
            .chain(room_blob_indices)
            .chain(font_blob_indices)
            .chain(skinned_blob_indices)
        {
            if !audio_blobs.contains(&idx)
                && !sdf_blobs.contains(&idx)
                && !terrain_blobs.contains(&idx)
                && released.insert(idx)
            {
                ctx.release_blob(idx);
            }
        }

        // InstancedProp components are drained because every instance becomes a
        // baked DrawObject; there is no per-frame update path yet. Drain before
        // taking Prop references because drain shifts the underlying Vec.
        let instanced_props = ctx.drain::<crate::assets::InstancedProp>();

        // query rather than drain so Props remain in the world for Camera3DSystem
        let props: Vec<_> = ctx.query::<Prop>().collect();

        // Entities owning each Prop, in the same column order as `props`, so the
        // decomposed render path can attach RenderHandle + GlobalTransform to
        // them below. Empty unless the decomposed path is enabled.
        let prop_entities: Vec<crate::ecs::Entity> = if self.decomposed_render {
            ctx.query_with_entity::<Prop>()
                .map(|(entity, _)| entity)
                .collect()
        } else {
            Vec::new()
        };

        // Snapshot the Props now (as owned clones) for the world.jsonl
        // hot-reload pass; we need a same-order `Vec<Prop>` later, but the
        // `props: Vec<&Prop>` borrow above must not survive into the next
        // `ctx.drain` call (which mutably re-borrows the same world). Stored
        // unconditionally; the field is only consulted under `cn debug`.
        let init_props_snapshot: Vec<Prop> = props.iter().map(|p| (*p).clone()).collect();

        // build parent-index table: for each prop, record the index of its parent
        // prop (if any) so world matrices can be resolved top-down each frame
        {
            let prop_id_to_idx: std::collections::HashMap<AssetId, usize> = props
                .iter()
                .enumerate()
                .map(|(i, p)| (p.asset_id, i))
                .collect();
            self.prop_parents = props
                .iter()
                .map(|p| {
                    p.parent
                        .and_then(|parent_id| prop_id_to_idx.get(&parent_id).copied())
                })
                .collect();
        }

        let world_mats = draw_list::compute_world_matrices(props.as_slice(), &self.prop_parents);

        let (
            all_vertices,
            all_indices,
            mut draw_objects,
            instanced_clusters,
            prop_draw_indices,
            mesh_id_to_draws,
        ) = match draw_list::build_draw_list(
            &props,
            &instanced_props,
            &world_mats,
            &model_map,
            &mesh_geometry,
            &room_geometry,
            &texture_name_to_slot,
            &material_map,
            &always_resident_meshes,
        ) {
            Some(d) => d,
            None => {
                self.failed = true;
                return;
            }
        };
        self.prop_draw_indices = prop_draw_indices;

        // Decomposed render path: give each prop entity a RenderHandle (its GPU
        // draw slots) and a GlobalTransform (seeded to its init world matrix), so
        // the per-frame push reads these instead of the positional side-tables.
        // prop_entities is column-aligned with prop_draw_indices and world_mats.
        if self.decomposed_render {
            for (i, &entity) in prop_entities.iter().enumerate() {
                let draws: Vec<u32> = self.prop_draw_indices[i]
                    .iter()
                    .map(|&slot| slot as u32)
                    .collect();
                ctx.insert(entity, crate::assets::RenderHandle { draws });
                ctx.insert(entity, crate::assets::GlobalTransform(world_mats[i]));
            }
        }

        // Asset hot-reload mesh map: cross-reference the file-backed source
        // metadata captured at drain time with the per-Mesh draw indices
        // build_draw_list just produced. A Mesh without any draws (referenced
        // by nothing) carries no entry; the watcher would still fire on the
        // .glb change but the reload helper has nothing to push to.
        let mut mesh_source_map = super::hot_reload_sources::MeshSourceMap::new();
        if capture_sources {
            for (asset_id, meta) in &mesh_sources {
                if let Some(draws) = mesh_id_to_draws.get(asset_id) {
                    if draws.is_empty() {
                        continue;
                    }
                    mesh_source_map
                        .entries
                        .push(super::hot_reload_sources::MeshSourceEntry {
                            source: meta.source.clone(),
                            primitive_index: meta.primitive_index,
                            lod_levels: meta.lod_levels,
                            lod_distances: meta.lod_distances.clone(),
                            draw_indices: draws.clone(),
                        });
                }
            }
        }

        // Procedural-mesh hot-reload map: same cross-reference, but the
        // "source" is the JSONL `args` object captured pre-drain rather than
        // a file path. A procedural mesh that no Prop references carries no
        // draws and is omitted; a JSONL save changing its args would be
        // observable only through a future system that introspects the args
        // map directly, which we deliberately do not maintain.
        let mut procedural_mesh_source_map =
            super::hot_reload_sources::ProceduralMeshSourceMap::new();
        if capture_sources {
            for (asset_id, (name, args)) in &proc_mesh_args_snapshot {
                if let Some(draws) = mesh_id_to_draws.get(asset_id) {
                    if draws.is_empty() {
                        continue;
                    }
                    procedural_mesh_source_map.entries.push(
                        super::hot_reload_sources::ProceduralMeshSourceEntry {
                            name: name.clone(),
                            args: args.clone(),
                            draw_indices: draws.clone(),
                        },
                    );
                }
            }
        }

        // A geometry-less world (e.g. text-only) is valid: the backend is
        // initialised with empty geometry buffers and only the text path runs.

        // Per-texture-slot draw positions, captured before `draw_objects` is
        // moved into the backend. The streaming subsystem scores each texture
        // by the camera's distance to the nearest draw that samples it.
        let texture_centers: Vec<Vec<[f32; 3]>> = {
            let mut centers = vec![Vec::new(); texture_data.len()];
            for obj in &draw_objects {
                if let Some(slot) = centers.get_mut(obj.texture_slot) {
                    slot.push(draw_object_position(obj));
                }
            }
            centers
        };

        // Per-normal-map draw positions. Streamed item `i` is normal-map pool
        // slot `i + 1` -- slot 0 is the flat-normal fallback and never streams,
        // so a draw sampling slot 0 contributes to no streamed item.
        let normal_map_centers: Vec<Vec<[f32; 3]>> = {
            let mut centers = vec![Vec::new(); normal_map_data.len()];
            for obj in &draw_objects {
                if obj.normal_map_slot >= 1
                    && let Some(slot) = centers.get_mut(obj.normal_map_slot - 1)
                {
                    slot.push(draw_object_position(obj));
                }
            }
            centers
        };

        // Per-streamed-mesh data, also captured before `draw_objects` moves
        // into the backend. Only static, frustum-cullable draws stream; skybox,
        // rooms, and dynamic props (sentinel AABB) stay resident so structural
        // geometry never pops in. Each payload is a copy of the draw's region
        // of the shared vertex/index buffers, scored by its AABB centre.
        let (mesh_stream_draw_indices, mesh_centers, mesh_payloads) = {
            let mut draw_indices: Vec<usize> = Vec::new();
            let mut centers: Vec<Vec<[f32; 3]>> = Vec::new();
            let mut payloads: Vec<crate::app::mesh_stream::DecodedMesh> = Vec::new();
            for (draw_idx, obj) in draw_objects.iter().enumerate() {
                if !obj.cullable() {
                    continue;
                }
                let vstart = obj.vertex_offset / std::mem::size_of::<Vertex>();
                let vend = vstart + obj.vertex_count;
                let iend = obj.index_offset + obj.index_count;
                if vend > all_vertices.len() || iend > all_indices.len() {
                    // Build-time offsets should always be in range; skip
                    // defensively rather than risk an out-of-bounds slice.
                    continue;
                }
                draw_indices.push(draw_idx);
                centers.push(vec![draw_object_position(obj)]);
                // Indices are stored mesh-relative (0-based): the sub-allocator
                // places the mesh's vertices anywhere on upload, and upload_mesh
                // rebases the indices onto whatever vertex region it chose.
                // mesh-relative index is global - vbase; each per-mesh region
                // fits in u16 (the build-time splitter enforces this), so we
                // narrow back here for DecodedMesh's per-mesh u16 indices.
                let vbase = vstart as u32;
                payloads.push(crate::app::mesh_stream::DecodedMesh {
                    vertices: all_vertices[vstart..vend].to_vec(),
                    indices: all_indices[obj.index_offset..iend]
                        .iter()
                        .map(|&i| (i - vbase) as u16)
                        .collect(),
                });
            }
            (draw_indices, centers, payloads)
        };

        // Mesh streaming and LOD alternates don't yet cooperate: upload_mesh
        // writes only LOD0 to its newly-allocated region, but obj.lod_alternates
        // still carries the build-time offsets for LOD1..N. Once another stream
        // upload reuses those byte ranges, active_lod() returns offsets that
        // point at unrelated geometry and the draw renders garbage / nothing
        // (the obelisks vanish past their first LOD switch_distance). Until
        // upload_mesh learns to stream every LOD, strip the alternates from
        // every streamable draw so active_lod() always returns LOD0.
        if streaming_config.is_some() && !mesh_payloads.is_empty() {
            for &draw_idx in &mesh_stream_draw_indices {
                if let Some(obj) = draw_objects.get_mut(draw_idx) {
                    obj.lod_alternates.clear();
                }
            }
        }

        // Shrinkable seed VRAM (Metal + DirectX + Vulkan). By default
        // `build_draw_list` bakes every streamed mesh into the shared
        // vertex/index buffers, sizing them for the whole streamed set, so
        // streaming reuses space but never shrinks GPU memory. When the residency
        // cap is smaller than the streamed set, compact the resident geometry and
        // reserve a smaller seed headroom -- sized to the cap-many largest meshes
        // -- for the streamed meshes, which are placed into it on upload
        // (tolerating a transient alloc miss while freed regions await their
        // retire frame). Done before `init_backend` so the GPU buffers are born
        // small and the RT acceleration structure (built over resident draws
        // inside init) sees the final offsets. Gated to the backends whose
        // `seed_mesh_streaming` seeds the sub-allocators with the headroom block.
        #[cfg(any(backend_metal, backend_dx, backend_vk))]
        let mut all_vertices = all_vertices;
        #[cfg(any(backend_metal, backend_dx, backend_vk))]
        let mut all_indices = all_indices;
        #[cfg(any(backend_metal, backend_dx, backend_vk))]
        let mut instanced_clusters = instanced_clusters;
        let mesh_seed_region: Option<crate::gfx::mesh_seed::MeshSeedRegion> = {
            #[cfg(any(backend_metal, backend_dx, backend_vk))]
            {
                match streaming_config.as_ref() {
                    Some(cfg) if !mesh_payloads.is_empty() => {
                        let sizes: Vec<(u64, u64)> = mesh_payloads
                            .iter()
                            .map(|m| {
                                (
                                    (m.vertices.len() * std::mem::size_of::<Vertex>()) as u64,
                                    (m.indices.len() * std::mem::size_of::<u32>()) as u64,
                                )
                            })
                            .collect();
                        match crate::gfx::mesh_seed::plan_seed_bytes(&sizes, cfg.mesh_cap()) {
                            Some((seed_vtx, seed_idx)) => {
                                let mut streamed = vec![false; draw_objects.len()];
                                for &idx in &mesh_stream_draw_indices {
                                    if let Some(s) = streamed.get_mut(idx) {
                                        *s = true;
                                    }
                                }
                                let region = crate::gfx::mesh_seed::compact_for_streaming(
                                    &mut all_vertices,
                                    &mut all_indices,
                                    &mut draw_objects,
                                    &mut instanced_clusters,
                                    &streamed,
                                    seed_vtx,
                                    seed_idx,
                                );
                                tracing::info!(
                                    "GraphicsSystem: shrinkable seed VRAM -- {} streamed mesh(es), cap {}, seed headroom {} KiB vtx + {} KiB idx",
                                    mesh_stream_draw_indices.len(),
                                    cfg.mesh_cap(),
                                    seed_vtx / 1024,
                                    seed_idx / 1024,
                                );
                                Some(region)
                            }
                            None => None,
                        }
                    }
                    _ => None,
                }
            }
            #[cfg(not(any(backend_metal, backend_dx, backend_vk)))]
            {
                None
            }
        };

        let draw_object_count = draw_objects.len();
        let cluster_count = instanced_clusters.len();
        let total_instances: usize = instanced_clusters.iter().map(|c| c.instances.len()).sum();

        // Build projected-decal records from the world's `Decal` components.
        // Resolved here (rather than per-frame) because the decal set is built
        // at init and never grows: each record carries the resolved texture
        // slot and pre-inverted model matrix the fragment shader needs. The
        // Decal components are drained because the runtime keeps no per-frame
        // update path for them.
        let decal_records = {
            let decals: Vec<Decal> = ctx.drain::<Decal>();
            let refs: Vec<&Decal> = decals.iter().collect();
            crate::gfx::decal::build_decal_records(&refs, &texture_name_to_slot)
        };
        let decal_count = decal_records.len();

        // Build particle-emitter records from the world's `ParticleEmitter`
        // components. Same drain-at-init pattern as decals: each record carries
        // the clamped emitter tunables and the resolved texture slot. The
        // backend allocates one persistent GPU pool per record at init.
        let particle_records = {
            let emitters: Vec<ParticleEmitter> = ctx.drain::<ParticleEmitter>();
            let refs: Vec<&ParticleEmitter> = emitters.iter().collect();
            crate::gfx::particles::build_particle_records(&refs, &texture_name_to_slot)
        };
        let particle_count = particle_records.len();

        // Drain transparent water surfaces. The Metal backend builds a
        // tessellated grid + per-surface uniforms per record at init;
        // DirectX / Vulkan ignore the slice for now.
        let water_surfaces: Vec<WaterSurface> = ctx.drain::<WaterSurface>();

        // Drain translucent glass panels. Every backend builds a world-space
        // quad + per-panel uniforms per record at init and draws them in the
        // shared transparent pass (`metal/glass.rs`, `directx/glass.rs`,
        // `vulkan/glass.rs`).
        let glass_panels: Vec<GlassPanel> = ctx.drain::<GlassPanel>();

        // Drain raymarched SDF volumes and pull the compiled-payload
        // fragment-shader source bytes for each. Each backend wraps the bytes
        // with the engine-shipped helpers + template and compiles a per-volume
        // pipeline at init. Volumes whose payload read fails are dropped with a
        // logged warning rather than failing the whole world build.
        let sdf_volumes: Vec<(SdfVolume, Vec<u8>, String)> = {
            let raw: Vec<SdfVolume> = ctx.drain::<SdfVolume>();
            let name_table = crate::ecs::asset_id::name_table();
            let mut out = Vec::with_capacity(raw.len());
            for v in raw {
                let asset_id = v.asset_id;
                let label = name_table
                    .get(asset_id.0 as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("sdf_volume_{}", asset_id.0));
                let locator = match v.locator.as_ref() {
                    Some(l) => l.clone(),
                    None => {
                        tracing::warn!(
                            "SdfVolume '{}': no payload locator (fragment shader \
                             never compiled); skipping",
                            label
                        );
                        continue;
                    }
                };
                match ctx.read_payload(&locator) {
                    Ok(bytes) => {
                        let owned = bytes.to_vec();
                        out.push((v, owned, label));
                    }
                    Err(e) => {
                        tracing::warn!(
                            "SdfVolume '{}': failed to read fragment shader \
                             payload: {:?}; skipping",
                            label,
                            e
                        );
                    }
                }
            }
            out
        };

        // Resolve the world's `VolumetricFog`. The first declared instance
        // wins; later ones are silently dropped (one homogeneous medium is
        // all the fog pass models). `None` means the renderer skips the
        // fog pass; an asset with `enabled = false` also yields `None`.
        let fog_settings = {
            let fogs: Vec<VolumetricFog> = ctx.drain::<VolumetricFog>();
            fogs.into_iter().find(|f| f.enabled).map(|f| {
                crate::gfx::volumetric_fog::FogSettings::resolve(
                    f.color,
                    f.density,
                    f.height_falloff,
                    f.height_reference,
                    f.max_distance,
                    f.phase_g,
                    f.ambient,
                )
            })
        };
        let fog_enabled = fog_settings.is_some();
        // Seed the hot-reload dedupe state. Subsequent reload_volumetric_fog
        // calls compare resolved JSONL settings against this and only push
        // (and log) on a real change.
        self.last_fog_settings = fog_settings;

        // The CLI `--validation` flag (via `dev_flags`) drives the DirectX /
        // Vulkan debug layers; unset falls back to the build profile. Metal is
        // unaffected here: its validation layer is enabled by the CLI re-execing
        // with `MTL_DEBUG_LAYER`, not through this flag.
        let validation = crate::app::dev_flags::validation().unwrap_or(cfg!(debug_assertions));
        // Shader hot-reload is opted in by `cn debug` (sets the static flag
        // in `crate::app::dev_flags` before world build). Production `cn run`
        // leaves it off; the backend then never spawns the filesystem watcher
        // and shader sources stay strictly include_str!-baked.
        let hot_reload = crate::app::dev_flags::enabled();
        // Worst-case resident chunk count for the streaming VoxelWorld (0 for a
        // non-voxel world). Threaded into the backend so its GPU-cull buffers
        // reserve a chunk record region at init; resident chunks fold into the
        // indirect path each frame. The VoxelWorld is consumed later by
        // `setup_voxel_world_streaming`, so borrow it here.
        let n_chunk_max = voxel_world.as_ref().map_or(
            0,
            crate::gfx::graphics_system::streaming::chunk_reserve_count,
        );

        // Reflection-probe auto-seed geometry. Computed here, before `draw_objects`
        // moves into the backend: when the world declares no `ReflectionProbe`, surface-
        // voxelise the static geometry so a watertight single-mesh interior is detected
        // (object AABBs alone would read it as a solid block). Budget-gated, so a heavy
        // import keeps coarse AABB occupancy. `None` -> the backend's own AABB auto-seed.
        let auto_seed_geometry_probes = if ctx
            .query::<crate::assets::ReflectionProbe>()
            .next()
            .is_some()
        {
            None
        } else {
            gather_auto_seed_triangles(&draw_objects, &all_vertices, &all_indices).and_then(
                |tris| {
                    let occupancy: Vec<([f32; 3], [f32; 3])> = draw_objects
                        .iter()
                        .map(|o| (o.bb_min, o.bb_max))
                        .filter(|(mn, mx)| mn.iter().chain(mx).all(|c| c.is_finite()))
                        .collect();
                    let (mn, mx) =
                        crate::gfx::reflection_probe::fold_world_bounds(occupancy.iter().copied())?;
                    Some(
                        crate::gfx::reflection_probe::auto_seed_probes_with_geometry(
                            mn, mx, &occupancy, &tris,
                        ),
                    )
                },
            )
        };

        self.backend = init_backend(
            &self.window_args,
            validation,
            self.frames_in_flight,
            self.vsync,
            self.clear_color,
            &all_vertices,
            &all_indices,
            draw_objects,
            instanced_clusters,
            // Skinned draw-object count, to size the backend's GPU-cull buffers
            // for the merged total at init. The skinned geometry is uploaded later
            // by `upload_skinned` (which consumes `skinned_draw_objects`).
            skinned_draw_objects.len(),
            // Worst-case resident chunk count, to reserve a chunk record region in
            // the backend's GPU-cull buffers at init.
            n_chunk_max,
            &vert_bytes,
            &frag_bytes,
            &shadow_bytes,
            &vert_instanced_bytes,
            &texture_data,
            &normal_map_data,
            light_uniforms,
            self.shadow_map_size,
            self.shadow_update,
            text_atlas_data,
            env_map_bytes.as_deref(),
            post_process,
            color_lut_bytes.as_deref(),
            taa_enabled,
            ssao_settings,
            ssr_settings,
            ssgi_settings,
            rt_reflection_settings,
            reflection_blur_scale,
            decal_records,
            particle_records,
            fog_settings,
            auto_exposure_settings,
            auto_exposure_bias_ev,
            hdr_display,
            hdr_pq,
            temporal_upscaling,
            upscale_scale,
            upscale_backend,
            occlusion_two_pass,
            water_surfaces,
            glass_panels,
            sdf_volumes,
            hot_reload,
        );

        if self.backend.is_none() {
            self.failed = true;
            return;
        }

        // Reflection probes: hand the backend the declared `ReflectionProbe`
        // placements (Metal bakes a cube per probe; an empty list auto-seeds from
        // the scene bounds). Pushed once here, after construction; DX/VK no-op.
        if let Some(backend) = self.backend.as_deref_mut() {
            let declared: Vec<crate::gfx::reflection_probe::ProbePlacement> = ctx
                .drain::<crate::assets::ReflectionProbe>()
                .into_iter()
                .map(|p| {
                    crate::gfx::reflection_probe::ProbePlacement::from_center_extents(
                        p.position,
                        p.half_extents,
                    )
                })
                .collect();
            // Declared probes win; otherwise the geometry-aware auto-seed (when the scene
            // was small enough to gather); otherwise an empty list, which lets the backend
            // run its own coarse-AABB auto-seed (the unchanged path for heavy imports).
            let placements = if !declared.is_empty() {
                declared
            } else {
                auto_seed_geometry_probes.unwrap_or_default()
            };
            backend.set_reflection_probes(&placements);
        }

        // World.jsonl path for the Prop transform reload pass. Best-effort:
        // `cn debug` runs from the client checkout so the lookup succeeds in
        // practice; embedded preview / WS-driven runs leave it None and the
        // file watcher simply has no `.jsonl` to subscribe to.
        let world_jsonl_path: Option<String> = if capture_sources {
            crate::world::find_world_jsonl(None).ok()
        } else {
            None
        };

        // Asset hot-reload state. Built only when `cn debug` opted in
        // (`capture_sources`) and the world declared at least one file-backed
        // asset (texture, ColorLut, EnvironmentMap, Mesh, SkinnedMesh, or
        // world.jsonl). The constructor spawns a `notify` watcher over the
        // parent directories of every captured source path; `step` polls the
        // shared atomic at frame start.
        if capture_sources
            && (!asset_source_map.is_empty()
                || color_lut_source.is_some()
                || environment_map_source.is_some()
                || !mesh_source_map.is_empty()
                || !skinned_mesh_source_map.is_empty()
                || !procedural_mesh_source_map.is_empty()
                || !shader_stage_source_map.is_empty()
                || world_jsonl_path.is_some())
        {
            tracing::info!(
                "asset hot-reload: captured {} file-backed texture source(s), {} \
                 ColorLut source(s), {} EnvironmentMap source(s), {} Mesh \
                 source(s), {} SkinnedMesh source(s), {} ProceduralMesh source(s), \
                 {} ShaderStage source(s), and world.jsonl path = {:?}",
                asset_source_map.len(),
                color_lut_source.as_ref().map(|_| 1).unwrap_or(0),
                environment_map_source.as_ref().map(|_| 1).unwrap_or(0),
                mesh_source_map.len(),
                skinned_mesh_source_map.len(),
                procedural_mesh_source_map.len(),
                shader_stage_source_map.len(),
                world_jsonl_path
            );
            self.pending_hot_reload_sources = Some(super::hot_reload_sources::HotReloadSources {
                map: asset_source_map,
                color_lut: color_lut_source,
                environment_map: environment_map_source,
                meshes: mesh_source_map,
                skinned_meshes: skinned_mesh_source_map,
                procedural_meshes: procedural_mesh_source_map,
                shader_stages: shader_stage_source_map,
                world_jsonl_path,
            });
            // Snapshot the init-order Prop list so the world.jsonl reload
            // pass can rebuild a same-order `Vec<Prop>` with the new
            // transforms. The clone was taken earlier (before the
            // `ctx.drain` mutations) so it doesn't tangle the borrow
            // checker here.
            self.init_props = Some(init_props_snapshot);
            // Auxiliary asset-resolution tables for the world.jsonl reload
            // pass, populated only when hot-reload is on, so a `cn run` does
            // not pay the clone cost. `mesh_id_to_draw` keeps one example
            // draw_idx per mesh so the clone-static-draw-object path has a
            // geometry template to copy from. `scene_names` snapshots every
            // declared Scene's raw name (reverse-looked-up from the interner)
            // so the reload pass can replay the build-time `<scene>_*` prefix
            // rule. We query (not drain) here so `setup_scene_reel` can still
            // drain the Scene components on the next pass.
            let mesh_id_to_draw: std::collections::HashMap<AssetId, usize> = mesh_id_to_draws
                .iter()
                .filter_map(|(id, draws)| draws.first().map(|&d| (*id, d)))
                .collect();
            let name_table = crate::ecs::asset_id::name_table();
            let scene_names: Vec<String> = ctx
                .query::<crate::assets::Scene>()
                .filter_map(|s| name_table.get(s.asset_id.0 as usize).cloned())
                .collect();
            self.world_reload = Some(super::WorldReloadState {
                material_map: material_map.clone(),
                texture_name_to_slot: texture_name_to_slot.clone(),
                model_map: model_map.clone(),
                mesh_id_to_draw,
                scene_names,
            });
        }

        // Upload skinned geometry to the backend and publish one SkeletonPose
        // per skinned mesh for AnimationSystem to drive. The poses are published
        // regardless of backend so the system graph is identical.
        if !skinned_skeletons.is_empty() {
            if let Some(backend) = self.backend.as_deref_mut() {
                // Metal uses `vert_bytes` + `frag_bytes` and sources the shadow
                // shader internally; `shadow_bytes` is empty (engine-internal
                // shadow). DX/VK compile their vertex/shadow paths inline.
                if let Err(e) = backend.upload_skinned(
                    &skinned_vertices,
                    &skinned_indices,
                    std::mem::take(&mut skinned_draw_objects),
                    &vert_bytes,
                    &frag_bytes,
                    &shadow_bytes,
                ) {
                    tracing::error!("GraphicsSystem: skinned geometry upload failed: {}", e);
                    self.failed = true;
                    return;
                }
            }
            let skinned_count = skinned_skeletons.len();
            for (idx, (mesh_id, skeleton)) in skinned_skeletons.into_iter().enumerate() {
                ctx.push(crate::assets::SkeletonPose::new(mesh_id, idx, skeleton));
            }
            tracing::info!("GraphicsSystem: {} skinned mesh(es) ready", skinned_count);
        }

        self.setup_texture_streaming(
            streaming_config.clone(),
            texture_payloads,
            &texture_locators,
            blob_disk_backed,
            texture_centers,
        );
        self.setup_normal_map_streaming(
            streaming_config.clone(),
            normal_map_payloads,
            &normal_map_locators,
            blob_disk_backed,
            normal_map_centers,
        );
        self.setup_mesh_streaming(
            streaming_config,
            mesh_payloads,
            mesh_centers,
            mesh_stream_draw_indices,
            blob_disk_backed,
            mesh_seed_region,
        );
        self.setup_voxel_world_streaming(voxel_world, &block_types, &material_map);

        // Decide cursor handling. A plain first-person world (Camera3D, no UI)
        // captures the cursor at startup as before. A Camera3D world that also
        // has UI (a MainMenu's HitRegion / KeyBinding) is "menu mode": capture
        // is driven per-frame in `run_step` by whether a menu view is active,
        // so Escape can pause the camera into the menu and back. A UI-only
        // world (no camera) stays free-cursor.
        let has_ui = ctx.query::<HitRegion>().next().is_some()
            || ctx.query::<crate::assets::KeyBinding>().next().is_some();
        let has_camera = ctx.query::<Camera3D>().next().is_some();
        self.menu_mode = has_camera && has_ui;
        let mut device_caps = crate::gfx::backend::DeviceCapabilities::ALL;
        if let Some(backend) = self.backend.as_deref_mut() {
            // Capability flags drive the settings-menu gating below.
            device_caps = backend.capabilities();
            backend.set_menu_mode(self.menu_mode);
            // Push the effective ambient scale (world value or persisted
            // override). The backend already seeds the world value at its own
            // init, so this is the path that applies a persisted Ambient-slider
            // choice; idempotent when there is no override. No-op on DX/VK.
            backend.set_ambient_intensity(self.ambient_intensity);
            // Push the movement key map (the persisted rebinds, or the default).
            // The backend decodes physical keys through it; idempotent with its
            // own default seed when there is no override.
            backend.set_keymap(&self.keymap);
            if has_camera && !has_ui {
                backend.capture_cursor();
            }
        }
        self.caps = device_caps;
        // Gray out + disable settings rows whose feature the device cannot
        // provide (e.g. ray-traced reflections on a GPU without hardware ray
        // tracing). Runs while the menu HitRegions / TextLabels / ScrollPanels
        // are still present (GraphicsSystem.init runs before UiInputSystem drains
        // them); the value-label sync above already set each row's live value.
        self.apply_capability_gating(ctx);

        self.setup_scene_reel(ctx);

        self.start_time = Some(Instant::now());
        tracing::info!(
            "GraphicsSystem: ready ({}x{} \"{}\", {} frames in flight, {} draw objects, {} instanced clusters ({} instances total), {} decals, {} particle emitter(s), fog={})",
            self.window_args.width,
            self.window_args.height,
            self.window_args.title,
            self.frames_in_flight,
            draw_object_count,
            cluster_count,
            total_instances,
            decal_count,
            particle_count,
            if fog_enabled { "on" } else { "off" },
        );
    }
}

// Set the value TextLabel of every `setting:<key>` HitRegion to the live value
// of that setting. `current_index` maps a setting key to the index of its
// active option (None for an unknown key). Runs once at init, before any
// system drains the HitRegions.
fn sync_setting_value_labels(
    ctx: &mut PipelineContext,
    current_index: impl Fn(&str) -> Option<usize>,
) {
    // (setting key, value-label id) for each settings row.
    let rows: Vec<(String, AssetId)> = ctx
        .query::<HitRegion>()
        .filter_map(|r| {
            let rest = r.action.strip_prefix("setting:")?;
            let key = rest.split(':').next()?;
            Some((key.to_string(), r.label?))
        })
        .collect();

    for (key, label_id) in rows {
        let (Some(opts), Some(idx)) = (crate::gfx::settings::options(&key), current_index(&key))
        else {
            continue;
        };
        if let Some(text) = opts.get(idx).copied() {
            for l in ctx.query_mut::<TextLabel>() {
                if l.asset_id == label_id {
                    l.content = text.to_string();
                    break;
                }
            }
        }
    }
}
