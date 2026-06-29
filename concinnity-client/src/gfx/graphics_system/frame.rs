// GraphicsSystem per-frame step: input polling, streaming updates, scene-reel
// ticking, and the backend draw call.

use crate::assets::{
    Camera3D, DespawnRequest, FrameInput, HitRegion, LabelBox, LayoutContainer, ReparentRequest,
    SceneCommand, SettingCommand, SettingOp, SpawnRequest, Sprite, TextLabel, WindowMode,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult};
use crate::gfx::{
    draw_list::{self},
    scene_reel, settings, sprite as gfx_sprite, text,
};

use super::helpers::*;
use super::*;

// Muted gray applied to the labels of a capability-disabled settings row, so it
// reads as unavailable next to the live rows.
const DISABLED_ROW_COLOR: [f32; 3] = [0.42, 0.42, 0.47];

// The full set of label ids to gray for a set of capability-gated rows: the
// gated value labels themselves (the fallback when a row is not in a scroll
// panel), plus every element of any scroll row that contains one of them, so a
// row dims as a whole (its name + value + stepper glyphs) rather than only its
// value. `rows` is each scroll row's element id list.
fn expand_dim_set(
    gated: &std::collections::HashSet<AssetId>,
    rows: &[Vec<AssetId>],
) -> std::collections::HashSet<AssetId> {
    let mut dim = gated.clone();
    for row in rows {
        if row.iter().any(|id| gated.contains(id)) {
            dim.extend(row.iter().copied());
        }
    }
    dim
}

// Reposition the labels owned by every visible `LayoutContainer`. This runs in
// the frame step because measuring a label needs the loaded font metrics,
// which live on the GraphicsSystem; the resolved origin is written back into
// each label so `build_text_calls` then draws it in place.
fn apply_label_layout(
    ctx: &mut PipelineContext,
    loaded_fonts: &std::collections::HashMap<AssetId, text::LoadedFont>,
) {
    let containers: Vec<LayoutContainer> = ctx
        .query::<LayoutContainer>()
        .filter(|c| c.visible)
        .cloned()
        .collect();
    if containers.is_empty() {
        return;
    }
    // Measure every label once, keyed by id.
    let mut boxes: std::collections::HashMap<AssetId, LabelBox> = std::collections::HashMap::new();
    for label in ctx.query::<TextLabel>() {
        if let Some(b) = text::measure_label_box(label, loaded_fonts) {
            boxes.insert(label.asset_id, b);
        }
    }
    // Resolve placements, then write them back into the labels.
    let placements: Vec<_> = containers
        .iter()
        .flat_map(|c| c.layout(|id| boxes.get(&id).copied()))
        .collect();
    for p in placements {
        for label in ctx.query_mut::<TextLabel>() {
            if label.asset_id == p.id {
                label.x = p.x;
                label.y = p.y;
                break;
            }
        }
    }
}

// Overwrite the text of the TextLabel with the given id, if present.
fn set_label_content(ctx: &mut PipelineContext, id: AssetId, text: &str) {
    for l in ctx.query_mut::<TextLabel>() {
        if l.asset_id == id {
            l.content = text.to_string();
            break;
        }
    }
}

// Set a cycle row's value label from its init-captured id. Used to update a
// row other than the one that was clicked (the master preset relabels the
// quality toggles + render scale; a quality-toggle change relabels the master
// row). The menu's HitRegions are drained after init, so the row -> label map
// is captured once (`init_cycle_value_labels`) rather than re-queried here.
fn set_cached_row_label(
    labels: &std::collections::HashMap<String, AssetId>,
    ctx: &mut PipelineContext,
    key: &str,
    text: &str,
) {
    if let Some(&id) = labels.get(key) {
        set_label_content(ctx, id, text);
    }
}

// Move the Sprite with the given id to `x` (its left edge), if present. Used to
// slide a slider's handle along its track.
fn set_sprite_x(ctx: &mut PipelineContext, id: AssetId, x: f32) {
    for s in ctx.query_mut::<Sprite>() {
        if s.asset_id == id {
            s.x = x;
            break;
        }
    }
}

// The setting key of a slider drag action (`setting:<key>:drag`), or `None`.
fn slider_key_of(action: &str) -> Option<&str> {
    action
        .strip_prefix("setting:")?
        .strip_suffix(":drag")
        .filter(|k| !k.is_empty())
}

// The setting key of a key-rebind action (`setting:<key>:rebind`), or `None`.
fn rebind_key_of(action: &str) -> Option<&str> {
    action
        .strip_prefix("setting:")?
        .strip_suffix(":rebind")
        .filter(|k| !k.is_empty())
}

// The setting key of a cycle row's forward stepper (`setting:<key>:next`), or
// `None`. A cycle row emits a matching `:prev` region with the same value label,
// so capturing the `:next` region alone maps each cycle key once.
fn cycle_next_key_of(action: &str) -> Option<&str> {
    action
        .strip_prefix("setting:")?
        .strip_suffix(":next")
        .filter(|k| !k.is_empty())
}

impl GraphicsSystem {
    pub(super) fn run_step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        if self.failed {
            return StepResult::Done;
        }

        let elapsed = self
            .start_time
            .map(|t| t.elapsed().as_secs_f32())
            .unwrap_or(0.0);
        // Per-frame delta for time-based countdowns (Lifetime). Clamped to
        // non-negative so a clock reset never rushes an expiry.
        let dt = (elapsed - self.prev_elapsed).max(0.0);
        self.prev_elapsed = elapsed;

        // read projection and view from Camera3D; view_matrix was written by
        // Camera3DSystem on the previous tick
        let (fov_y_radians, near, far, view_matrix, cam_pos) = ctx
            .query::<Camera3D>()
            .next()
            .map(|c| {
                (
                    c.fov_y_degrees.to_radians(),
                    c.near,
                    c.far,
                    c.view_matrix,
                    c.position,
                )
            })
            .unwrap_or((
                std::f32::consts::FRAC_PI_4,
                0.05,
                200.0,
                IDENTITY4,
                [0.0; 3],
            ));

        // build text + sprite draw calls. Sprites render as solid-coloured
        // quads through the same UI pass as TextLabel (sentinel-UV path), so
        // they share the text pipeline and require no new render state.
        // Backdrop / HUD sprites are emitted first so labels composite on top;
        // `follow_cursor` sprites are emitted last so the cursor sits on top of
        // everything. `want_ui_cursor` is true when a cursor sprite is shown,
        // so the backend can hide the system cursor under it.
        let (text_calls, want_ui_cursor, menu_active): (
            Vec<crate::gfx::render_types::TextDrawCall>,
            bool,
            bool,
        ) = {
            let (win_w, win_h) = self.viewport_size();
            // Reposition LayoutContainer-managed labels before measuring them
            // for draw, so a HUD reflows to its live text each frame.
            apply_label_layout(ctx, &self.loaded_fonts);
            let default_atlas_slot = self.loaded_fonts.values().next().map(|f| f.atlas_slot);
            let sprites: Vec<&Sprite> = ctx.query::<Sprite>().collect();
            let (cursor_sprites, scene_sprites): (Vec<&Sprite>, Vec<&Sprite>) =
                sprites.into_iter().partition(|s| s.follow_cursor);

            let mut calls = gfx_sprite::build_sprite_calls(
                &scene_sprites,
                default_atlas_slot,
                [win_w, win_h],
                &self.clip_rects,
            );
            let labels: Vec<&TextLabel> = ctx.query::<TextLabel>().collect();
            calls.extend(text::build_text_calls(
                &labels,
                &self.loaded_fonts,
                win_w,
                win_h,
                &self.clip_rects,
            ));

            // Draw each cursor sprite as an arrow pointer at the latest mouse
            // position (tip on the pointer), after the text so it sits on top.
            calls.extend(crate::gfx::cursor::build_cursor_calls(
                &cursor_sprites,
                self.cursor_pos,
                default_atlas_slot,
                [win_w, win_h],
            ));

            let want_cursor = cursor_sprites.iter().any(|s| s.visible && s.tint[3] > 0.0);
            // A menu is "active" when any view-owned UI element is visible; used
            // to drive cursor capture and to freeze gameplay input below.
            let menu_active = labels.iter().any(|l| l.visible && l.view.is_some())
                || scene_sprites.iter().any(|s| s.visible && s.view.is_some())
                || cursor_sprites.iter().any(|s| s.visible && s.view.is_some());
            (calls, want_cursor, menu_active)
        };

        let backend = match self.backend.as_deref_mut() {
            Some(b) => b,
            None => return StepResult::Done,
        };

        // Hide the system cursor while an in-engine cursor sprite is shown
        // (edge-triggered in the backend, so this is cheap every frame).
        backend.set_ui_cursor_hidden(want_ui_cursor);

        // In menu mode (a MainMenu over a controlled camera), capture the cursor
        // for the camera unless a menu view is open. Edge-triggered in the
        // backend, so this is cheap every frame and a no-op in other worlds.
        if self.menu_mode {
            backend.set_camera_capture(!menu_active);
        }

        // Runtime decal / emitter spawn (`cn debug` only) is drained + dispatched
        // from the binary's `DebugHook::tick` (see `crate::debug::runtime_spawn`),
        // not here. `cn run` has no debug hook, so this step never touches it.

        // Asset / shader / world.jsonl hot-reload (`cn debug` only) is driven
        // from the binary's `DebugHook::tick` (see the `debug` module), not
        // here: it reaches the reload passes through
        // `GraphicsSystem::hot_reload_drive`. `cn run` has no debug hook, so
        // this per-frame path is reload-free.

        let result = {
            {
                if backend.window_closed() {
                    tracing::info!("GraphicsSystem: window closed");
                    backend.wait_idle();
                    return StepResult::Stop;
                }

                // Timed despawn: decrement every Lifetime by this frame's dt and
                // despawn the entities whose countdown reached zero, through the
                // same cascade a DespawnRequest uses. This is the churn that
                // returns draw slots to the free list for the spawn drain below
                // to recycle.
                let expired = super::spawn::tick_lifetimes(ctx, dt);
                for entity in expired {
                    super::despawn::despawn_subtree(ctx, backend, entity);
                }

                // Runtime entity despawn: drain DespawnRequest events, resolve
                // each name to its entity, hide that entity's draw slots, and
                // remove it (and its descendants) from the ECS. Done before the
                // transform push so a despawned entity is already gone from the
                // GlobalTransform x RenderHandle join this frame and contributes
                // nothing to any pass.
                let despawn_names: Vec<AssetId> = match ctx.events::<DespawnRequest>() {
                    Some(events) => events
                        .read(&mut self.despawn_cmd_cursor)
                        .into_iter()
                        .map(|r| r.name)
                        .collect(),
                    None => Vec::new(),
                };
                if !despawn_names.is_empty() {
                    // Clone the name index out so the ctx borrow ends before the
                    // despawns, which take &mut ctx.
                    let by_name = ctx
                        .resource::<crate::ecs::decompose::EntityByName>()
                        .map(|n| n.0.clone())
                        .unwrap_or_default();
                    for name in despawn_names {
                        if let Some(&entity) = by_name.get(&name) {
                            super::despawn::despawn_subtree(ctx, backend, entity);
                        }
                    }
                }

                // Runtime re-parenting: drain ReparentRequest events, resolve the
                // child + parent names to entities, and re-point the child's
                // Parent edge (recomposing world matrices). After the despawn
                // drain so a reparent naming a just-removed entity simply finds
                // nothing to move.
                let reparents: Vec<ReparentRequest> = match ctx.events::<ReparentRequest>() {
                    Some(events) => events
                        .read(&mut self.reparent_cmd_cursor)
                        .into_iter()
                        .copied()
                        .collect(),
                    None => Vec::new(),
                };
                if !reparents.is_empty() {
                    let by_name = ctx
                        .resource::<crate::ecs::decompose::EntityByName>()
                        .map(|n| n.0.clone())
                        .unwrap_or_default();
                    for req in reparents {
                        let Some(&child) = by_name.get(&req.child) else {
                            continue;
                        };
                        let parent = req.parent.and_then(|p| by_name.get(&p).copied());
                        // A named-but-unresolved parent skips, so a typo never
                        // silently detaches the child to a root.
                        if req.parent.is_some() && parent.is_none() {
                            continue;
                        }
                        draw_list::reparent(ctx, child, parent);
                    }
                }

                // Runtime entity spawn: drain SpawnRequest events, resolve each
                // template name to its entity, and instantiate a copy at the
                // requested transform. Each cloned draw slot reuses one freed by
                // an earlier despawn / Lifetime expiry before the backend grows
                // its draw_objects, so steady spawn/despawn churn does not leak
                // slots. After the despawn / reparent drains so a spawn can reuse
                // slots freed this same frame.
                let spawn_reqs: Vec<SpawnRequest> = match ctx.events::<SpawnRequest>() {
                    Some(events) => events
                        .read(&mut self.spawn_cmd_cursor)
                        .into_iter()
                        .copied()
                        .collect(),
                    None => Vec::new(),
                };
                if !spawn_reqs.is_empty() {
                    let by_name = ctx
                        .resource::<crate::ecs::decompose::EntityByName>()
                        .map(|n| n.0.clone())
                        .unwrap_or_default();
                    for req in spawn_reqs {
                        let Some(&template) = by_name.get(&req.template) else {
                            continue;
                        };
                        // A skinned template (a SkeletonPose entity) claims a
                        // pre-reserved instance slot; a static one clones a draw
                        // slot. Dispatch on which the template carries.
                        if ctx.get::<crate::assets::SkeletonPose>(template).is_some() {
                            super::spawn::spawn_skinned_from_template(
                                ctx,
                                template,
                                Some(req.name),
                                req.transform,
                                req.lifetime_secs,
                                |tmpl, model| backend.spawn_skinned_instance(tmpl, model),
                            );
                        } else {
                            super::spawn::spawn_from_template(
                                ctx,
                                template,
                                Some(req.name),
                                req.transform,
                                req.lifetime_secs,
                                |src, model| backend.clone_static_draw_object(src, model).ok(),
                            );
                        }
                    }
                }

                // Cadence-driven spawn: advance every Spawner's clock and
                // instantiate the copies now due, at the spawner's position.
                // Transient (unnamed) and Lifetime-bounded, so a steady spawner
                // churns through recycled draw slots. After the SpawnRequest
                // drain so both spawn paths reuse slots freed this frame.
                let due_spawns = super::spawn::tick_spawners(ctx, dt);
                if !due_spawns.is_empty() {
                    let by_name = ctx
                        .resource::<crate::ecs::decompose::EntityByName>()
                        .map(|n| n.0.clone())
                        .unwrap_or_default();
                    for due in due_spawns {
                        let Some(&template) = by_name.get(&due.template) else {
                            continue;
                        };
                        if ctx.get::<crate::assets::SkeletonPose>(template).is_some() {
                            super::spawn::spawn_skinned_from_template(
                                ctx,
                                template,
                                None,
                                due.transform,
                                due.lifetime,
                                |tmpl, model| backend.spawn_skinned_instance(tmpl, model),
                            );
                        } else {
                            super::spawn::spawn_from_template(
                                ctx,
                                template,
                                None,
                                due.transform,
                                due.lifetime,
                                |src, model| backend.clone_static_draw_object(src, model).ok(),
                            );
                        }
                    }
                }

                // Push updated model matrices for any entity whose transform
                // changed since last frame (physics, camera interact, reparent):
                // resolve each entity's GlobalTransform from Transform + Parent
                // (top-down so parents propagate to children), then push it to the
                // entity's GPU draw slots.
                draw_list::propagate_transforms(ctx);
                for (_entity, global, handle) in
                    ctx.join2::<crate::assets::GlobalTransform, crate::assets::RenderHandle>()
                {
                    for &slot in &handle.draws {
                        backend.update_model(slot as usize, global.0);
                    }
                }

                // Push the latest skinned poses to the GPU. AnimationSystem
                // wrote them into the SkeletonPose components on the previous
                // tick; the one-frame lag is invisible at animation rates.
                for pose in ctx.query::<crate::assets::SkeletonPose>() {
                    backend.update_skinned_pose(pose.skinned_index, &pose.joint_matrices);
                }
                // Push the model matrix for skinned instances that carry a
                // Transform (the runtime-spawned ones), so a moved instance
                // follows it. The authored templates have no Transform and keep
                // the model baked into their draw object at load.
                for (_entity, pose, transform) in
                    ctx.join2::<crate::assets::SkeletonPose, crate::assets::Transform>()
                {
                    backend.update_skinned_model(pose.skinned_index, transform.model_matrix());
                }

                // apply any imperative scene jumps sent by UiInputSystem this
                // tick, copied out of the event queue so the borrow is released
                // before the jump touches the backend
                let scene_cmds: Vec<SceneCommand> = match ctx.events::<SceneCommand>() {
                    Some(events) => events
                        .read(&mut self.scene_cmd_cursor)
                        .into_iter()
                        .cloned()
                        .collect(),
                    None => Vec::new(),
                };
                // Source scene-jump visibility from the per-entity components,
                // snapshotting once for the whole command batch.
                let (draws, scenes) = if scene_cmds.is_empty() {
                    (Vec::new(), Vec::new())
                } else {
                    super::scene::decomposed_visibility_snapshot(ctx)
                };
                for cmd in scene_cmds {
                    scene_reel::jump_to_scene(
                        &mut self.reel,
                        &draws,
                        &scenes,
                        elapsed,
                        cmd.scene,
                        &cmd.transition,
                        backend,
                    );
                }

                // apply graphics settings changes UiInputSystem sent last tick:
                // cycle the setting, apply it to the backend, refresh the value
                // label, and persist the new value. Clone the commands out of
                // the queue so the ctx borrow is released before the loop body,
                // which needs &mut ctx (label/sprite updates, ControlsCommand /
                // AudioCommand sends).
                let setting_cmds: Vec<SettingCommand> = match ctx.events::<SettingCommand>() {
                    Some(events) => events
                        .read(&mut self.setting_cmd_cursor)
                        .into_iter()
                        .cloned()
                        .collect(),
                    None => Vec::new(),
                };
                for cmd in setting_cmds {
                    // Key-rebind settings (Controls tab) take a Rebind op: bind
                    // the named action to the captured key, swapping with whatever
                    // action held it, push the map to the backend, persist, and
                    // refresh the affected row label(s). Handled first; the
                    // slider + cycle settings below take SetFraction / Next / Prev.
                    if let SettingOp::Rebind(key) = cmd.op {
                        let Some(action) =
                            crate::gfx::keymap::Bindable::from_setting_key(&cmd.setting)
                        else {
                            tracing::warn!("GraphicsSystem: unknown rebind '{}'", cmd.setting);
                            continue;
                        };
                        // The action (if any) that currently holds the new key,
                        // captured before the swap so its label is refreshed too.
                        let victim = self.keymap.action_for_key(key).filter(|&a| a != action);
                        self.keymap.rebind(action, key);
                        backend.set_keymap(&self.keymap);
                        let mut cfg = crate::config::Settings::load();
                        cfg.controls.keymap = Some(self.keymap);
                        if let Err(e) = cfg.save() {
                            tracing::warn!("GraphicsSystem: persist keymap: {}", e);
                        }
                        // Refresh the rebound row label and any swap victim's,
                        // reading the registry by direct field access (disjoint
                        // from the `backend` borrow).
                        for act in [Some(action), victim].into_iter().flatten() {
                            if let Some(value_id) = self
                                .rebind_rows
                                .iter()
                                .find(|r| r.action == act)
                                .map(|r| r.value_id)
                            {
                                let name = self.keymap.get(act).display_name();
                                set_label_content(ctx, value_id, name);
                            }
                        }
                        continue;
                    }
                    // Slider settings (continuous) take a SetFraction op: apply
                    // the value live to the post-process params, move the handle,
                    // refresh the value label, and persist only on the commit
                    // frame (drag release). Handled here; the cycle settings
                    // below take Next/Prev.
                    if let SettingOp::SetFraction(frac) = cmd.op {
                        let Some(value) = settings::slider_value_at(&cmd.setting, frac) else {
                            tracing::warn!("GraphicsSystem: unknown slider '{}'", cmd.setting);
                            continue;
                        };
                        // Track geometry for the handle, copied out so the
                        // `self.sliders` borrow ends before mutating self below.
                        let geom = self
                            .sliders
                            .iter()
                            .find(|s| s.key == cmd.setting)
                            .map(|s| (s.handle_id, s.track_x, s.track_w, s.handle_w));
                        // Apply the value to the live render param. The clamp /
                        // EV-to-multiplier transform lives in
                        // `settings::slider_apply_value` (shared with the
                        // persisted re-apply at init, so they cannot diverge).
                        let stored = settings::slider_apply_value(&cmd.setting, value);
                        let is_qparam = settings::is_quality_param_slider(&cmd.setting);
                        match cmd.setting.as_str() {
                            "exposure" => self.post_process.exposure = stored,
                            "bloom_intensity" => self.post_process.bloom_intensity = stored,
                            "bloom_threshold" => self.post_process.bloom_threshold = stored,
                            "bloom_knee" => self.post_process.bloom_knee = stored,
                            "vignette" => self.post_process.vignette = stored,
                            "lut_strength" => self.post_process.lut_strength = stored,
                            // Per-feature sub-quality sliders live on the stored
                            // PostProcessConfig (the source of truth a later rebuild
                            // re-derives from); the live apply below mutates the
                            // backend's stored settings without a rebuild.
                            "ssao_radius" => self.post_config.ssao_radius = stored,
                            "ssao_intensity" => self.post_config.ssao_intensity = stored,
                            "ssr_intensity" => self.post_config.ssr_intensity = stored,
                            "ssr_max_distance" => self.post_config.ssr_max_distance = stored,
                            "ssgi_intensity" => self.post_config.ssgi_intensity = stored,
                            "ssgi_max_distance" => self.post_config.ssgi_max_distance = stored,
                            "auto_exposure_min_ev" => {
                                self.post_config.auto_exposure_min_ev = stored
                            }
                            "auto_exposure_max_ev" => {
                                self.post_config.auto_exposure_max_ev = stored
                            }
                            "auto_exposure_speed" => self.post_config.auto_exposure_speed = stored,
                            _ => {}
                        }
                        // Apply live. The sub-quality sliders mutate the backend's
                        // stored *Settings via update_quality_params (re-read into a
                        // per-frame uniform, no pass rebuild). The post-process
                        // sliders push PostProcessParams. Mouse sensitivity is not a
                        // render param, so it skips both (handled below); the ambient
                        // re-push through update_post_process is harmless.
                        if is_qparam {
                            backend.update_quality_params(super::derive_quality_settings(
                                &self.post_config,
                            ));
                        } else if cmd.setting != "mouse_sensitivity" {
                            backend.update_post_process(self.post_process);
                        }
                        // Ambient (IBL) scale lives in LightUniforms, not
                        // PostProcessParams, so it takes a dedicated setter.
                        if cmd.setting == "ambient_intensity" {
                            self.ambient_intensity = stored;
                            backend.set_ambient_intensity(stored);
                        }
                        // Mouse sensitivity is owned by the camera controller,
                        // not the renderer: hand the new radians/pixel value
                        // across as a ControlsCommand the camera reads this tick
                        // (live, no restart).
                        if cmd.setting == "mouse_sensitivity" {
                            ctx.events_mut::<crate::assets::ControlsCommand>().send(
                                crate::assets::ControlsCommand {
                                    mouse_sensitivity: stored,
                                },
                            );
                        }
                        // Move the handle to the new fraction.
                        if let Some((handle_id, track_x, track_w, handle_w)) = geom {
                            let hx = track_x + frac.clamp(0.0, 1.0) * (track_w - handle_w).max(0.0);
                            set_sprite_x(ctx, handle_id, hx);
                        }
                        // Refresh the value label.
                        if let Some(label_id) = cmd.value_label {
                            set_label_content(
                                ctx,
                                label_id,
                                &settings::format_slider_value(&cmd.setting, value),
                            );
                        }
                        // Persist only on release (the in-progress frames apply
                        // live but skip the disk write).
                        if cmd.persist {
                            let mut cfg = crate::config::Settings::load();
                            match cmd.setting.as_str() {
                                "exposure" => cfg.graphics.exposure_ev = Some(value),
                                "bloom_intensity" => cfg.graphics.bloom_intensity = Some(value),
                                "bloom_threshold" => cfg.graphics.bloom_threshold = Some(value),
                                "bloom_knee" => cfg.graphics.bloom_knee = Some(value),
                                "vignette" => cfg.graphics.vignette = Some(value),
                                "lut_strength" => cfg.graphics.lut_strength = Some(value),
                                "ambient_intensity" => cfg.graphics.ambient_intensity = Some(value),
                                "ssao_radius" => cfg.graphics.ssao_radius = Some(value),
                                "ssao_intensity" => cfg.graphics.ssao_intensity = Some(value),
                                "ssr_intensity" => cfg.graphics.ssr_intensity = Some(value),
                                "ssr_max_distance" => cfg.graphics.ssr_max_distance = Some(value),
                                "ssgi_intensity" => cfg.graphics.ssgi_intensity = Some(value),
                                "ssgi_max_distance" => cfg.graphics.ssgi_max_distance = Some(value),
                                "auto_exposure_min_ev" => {
                                    cfg.graphics.auto_exposure_min_ev = Some(value)
                                }
                                "auto_exposure_max_ev" => {
                                    cfg.graphics.auto_exposure_max_ev = Some(value)
                                }
                                "auto_exposure_speed" => {
                                    cfg.graphics.auto_exposure_speed = Some(value)
                                }
                                // Persist the radians/pixel value (what the
                                // camera reads), not the 1..100 UI value.
                                "mouse_sensitivity" => {
                                    cfg.controls.mouse_sensitivity = Some(stored)
                                }
                                _ => {}
                            }
                            if let Err(e) = cfg.save() {
                                tracing::warn!(
                                    "GraphicsSystem: persist setting '{}': {}",
                                    cmd.setting,
                                    e
                                );
                            }
                        }
                        continue;
                    }
                    // Master "Graphics Quality" preset row. A preset is a
                    // performance ceiling over the world's authored look (it never
                    // enables a feature the world did not author), so picking a
                    // tier / Auto clears the per-row quality overrides and
                    // re-derives the toggles + render scale from the world's
                    // authored config under the new ceiling: raising a preset
                    // restores the world's features, lowering it clamps them off.
                    // Custom resolves to the no-op ceiling (the world's look).
                    // Render scale is restart-required, so it only persists +
                    // relabels here. See gfx/quality_preset.rs.
                    if cmd.setting == "graphics_quality" {
                        use crate::gfx::quality_preset;
                        let opts = settings::options("graphics_quality").unwrap_or(&[]);
                        let cur = quality_preset::preset_index(self.quality_preset);
                        let next = settings::cycle(cur, opts.len(), cmd.op);
                        let preset = quality_preset::preset_at(next);
                        self.quality_preset = preset;
                        let ceiling = quality_preset::resolve_ceiling(preset, &self.gpu_profile);

                        // Re-derive the live quality toggles from the world
                        // baseline under the new ceiling (force off where
                        // disallowed; never turn on).
                        self.post_config = self.authored_post_config.clone();
                        for (key, allowed) in [
                            ("taa", ceiling.taa),
                            ("ssao", ceiling.ssao),
                            ("ssr", ceiling.ssr),
                            ("ray_traced_reflections", ceiling.ray_traced_reflections),
                            ("ssgi", ceiling.ssgi),
                            ("auto_exposure", ceiling.auto_exposure),
                        ] {
                            if !allowed {
                                super::set_quality_toggle(&mut self.post_config, key, false);
                            }
                        }
                        // And clamp the cycle quality knobs (SSGI gather +
                        // reflection blur) under the ceiling (overrides cleared,
                        // so clamp every one).
                        for key in settings::QUALITY_CYCLE_KEYS {
                            super::clamp_quality_cycle(&mut self.post_config, key, &ceiling, false);
                        }
                        backend.apply_quality_settings(super::derive_quality_settings(
                            &self.post_config,
                        ));
                        // Auto-exposure may have flipped off; re-push the static
                        // post-process params so exposure reverts (mirrors the
                        // quality-toggle arm below).
                        backend.update_post_process(self.post_process);
                        // Restart-required: update the live render scale for the
                        // row label only (the upscaler + targets are sized at init,
                        // so it takes effect at the next launch).
                        self.render_scale = quality_preset::more_aggressive_upscale(
                            self.authored_post_config.upscale_quality,
                            ceiling.min_upscale,
                        );
                        // Re-derive the shadow knobs from the authored baselines
                        // under the new ceiling. The cadence is live (the scheduler
                        // reads it each frame); the resolution is restart-required,
                        // so it only updates the row label below.
                        self.shadow_map_size = quality_preset::clamp_shadow_map_size(
                            self.authored_shadow_map_size,
                            &ceiling,
                        );
                        self.shadow_update = quality_preset::clamp_shadow_update(
                            self.authored_shadow_update,
                            &ceiling,
                        );
                        backend.set_shadow_update(self.shadow_update);

                        // Persist the preset and drop the per-row quality overrides,
                        // so the next launch re-resolves them from the world +
                        // ceiling exactly as this live re-derive did.
                        let mut cfg = crate::config::Settings::load();
                        cfg.graphics.quality_preset = Some(preset);
                        cfg.graphics.taa = None;
                        cfg.graphics.ssao = None;
                        cfg.graphics.ssr = None;
                        cfg.graphics.ray_traced_reflections = None;
                        cfg.graphics.ssgi = None;
                        cfg.graphics.auto_exposure = None;
                        cfg.graphics.ssgi_resolution = None;
                        cfg.graphics.ssgi_rays = None;
                        cfg.graphics.ssgi_steps = None;
                        cfg.graphics.reflection_blur_resolution = None;
                        cfg.graphics.shadow_map_size = None;
                        cfg.graphics.shadow_update = None;
                        cfg.graphics.render_scale = None;
                        if let Err(e) = cfg.save() {
                            tracing::warn!("GraphicsSystem: persist preset: {e}");
                        }

                        // Refresh the dependent rows (quality toggles + render
                        // scale) from the init-captured value-label ids -- the
                        // menu's HitRegions are drained by UiInputSystem after
                        // init, so they cannot be re-queried here.
                        for key in settings::QUALITY_TOGGLE_KEYS {
                            let on =
                                super::quality_toggle_on(&self.post_config, key).unwrap_or(false);
                            if let Some(text) =
                                settings::options(key).and_then(|o| o.get(on as usize).copied())
                            {
                                set_cached_row_label(&self.cycle_value_labels, ctx, key, text);
                            }
                        }
                        if let Some(text) = settings::options("render_scale").and_then(|o| {
                            o.get(settings::render_scale_index(self.render_scale))
                                .copied()
                        }) {
                            set_cached_row_label(
                                &self.cycle_value_labels,
                                ctx,
                                "render_scale",
                                text,
                            );
                        }
                        for key in settings::QUALITY_CYCLE_KEYS {
                            if let Some(text) = super::quality_cycle_index(&self.post_config, key)
                                .and_then(|idx| {
                                    settings::options(key).and_then(|o| o.get(idx).copied())
                                })
                            {
                                set_cached_row_label(&self.cycle_value_labels, ctx, key, text);
                            }
                        }
                        // And the shadow rows (their state lives on self, not the
                        // post_config, so they relabel from the live fields).
                        for key in ["shadow_map_size", "shadow_update"] {
                            let idx = match key {
                                "shadow_map_size" => {
                                    settings::shadow_resolution_index(self.shadow_map_size)
                                }
                                _ => settings::shadow_update_index(self.shadow_update),
                            };
                            if let Some(text) =
                                settings::options(key).and_then(|o| o.get(idx).copied())
                            {
                                set_cached_row_label(&self.cycle_value_labels, ctx, key, text);
                            }
                        }
                        // The master row's own label carries the Auto(tier) suffix
                        // and is updated through the event-carried value-label id.
                        let label = quality_preset::preset_label(preset, &self.gpu_profile);
                        if let Some(id) = cmd.value_label {
                            set_label_content(ctx, id, &label);
                        }
                        continue;
                    }
                    let Some(opts) = settings::options(&cmd.setting) else {
                        tracing::warn!("GraphicsSystem: unknown setting '{}'", cmd.setting);
                        continue;
                    };
                    // Apply per setting: cycle the value, apply it (live for
                    // window/vsync; render_scale is restart-required so it only
                    // persists), then persist and refresh the value label.
                    let mut cfg = crate::config::Settings::load();
                    let new_text: Option<&str> = match cmd.setting.as_str() {
                        "vsync" => {
                            let next = settings::cycle(self.vsync as usize, opts.len(), cmd.op);
                            self.vsync = next == 1;
                            backend.set_vsync(self.vsync);
                            cfg.graphics.vsync = Some(self.vsync);
                            Some(opts[next])
                        }
                        "window_mode" => {
                            let cur = settings::window_mode_index(self.window_args.mode);
                            let next = settings::cycle(cur, opts.len(), cmd.op);
                            let mode = settings::window_mode_at(next);
                            self.window_args.mode = mode;
                            backend.set_window_mode(mode);
                            // Returning to windowed: re-apply the remembered
                            // windowed size, since borderless/fullscreen left the
                            // window at the display size (no-op while fullscreen
                            // is still animating; each backend guards that).
                            if mode == WindowMode::Windowed {
                                backend.set_window_size(
                                    self.window_args.width,
                                    self.window_args.height,
                                );
                            }
                            cfg.graphics.window_mode = Some(mode);
                            Some(opts[next])
                        }
                        "window_size" => {
                            let cur = settings::window_size_index(
                                self.window_args.width,
                                self.window_args.height,
                            );
                            let next = settings::cycle(cur, opts.len(), cmd.op);
                            let (w, h) = settings::window_size_at(next);
                            self.window_args.width = w;
                            self.window_args.height = h;
                            // Resizing only applies in windowed mode; the preset
                            // is still remembered for the return to windowed.
                            if self.window_args.mode == WindowMode::Windowed {
                                backend.set_window_size(w, h);
                            }
                            cfg.graphics.window_size = Some([w, h]);
                            Some(opts[next])
                        }
                        "render_scale" => {
                            // Restart-required: persist + display only; the
                            // upscaler and render targets are sized once at init.
                            let cur = settings::render_scale_index(self.render_scale);
                            let next = settings::cycle(cur, opts.len(), cmd.op);
                            self.render_scale = settings::render_scale_at(next);
                            cfg.graphics.render_scale = Some(self.render_scale);
                            // Render scale is ceiling-governed, so an explicit
                            // choice opts the master preset out to Custom.
                            self.quality_preset = crate::gfx::quality_preset::QualityPreset::Custom;
                            cfg.graphics.quality_preset = Some(self.quality_preset);
                            set_cached_row_label(
                                &self.cycle_value_labels,
                                ctx,
                                "graphics_quality",
                                self.quality_preset.name(),
                            );
                            Some(opts[next])
                        }
                        "master_volume" => {
                            // Live: cycle the gain, persist it, and hand it to
                            // AudioSystem (which owns the audio engine) as an
                            // AudioCommand it drains this same tick -- GraphicsSystem
                            // runs first, so the change applies this frame. A world
                            // with no audio simply has no AudioSystem to drain it;
                            // the persisted value then applies at the next audio init.
                            let cur = settings::master_volume_index(
                                cfg.audio
                                    .master_volume
                                    .unwrap_or(settings::DEFAULT_MASTER_VOLUME),
                            );
                            let next = settings::cycle(cur, opts.len(), cmd.op);
                            let gain = settings::master_volume_at(next);
                            cfg.audio.master_volume = Some(gain);
                            ctx.events_mut::<crate::assets::AudioCommand>().send(
                                crate::assets::AudioCommand {
                                    master_volume: gain,
                                },
                            );
                            Some(opts[next])
                        }
                        // Quality-feature toggles: flip the matching field on the
                        // stored config, persist the bool, then apply live by
                        // rebuilding the affected render resources (Metal; a
                        // no-op backend keeps the choice for the next launch).
                        key if settings::is_quality_toggle(key) => {
                            let cur =
                                super::quality_toggle_on(&self.post_config, key).unwrap_or(false);
                            let next = settings::cycle(cur as usize, opts.len(), cmd.op);
                            let on = next == 1;
                            super::set_quality_toggle(&mut self.post_config, key, on);
                            match key {
                                "taa" => cfg.graphics.taa = Some(on),
                                "ssao" => cfg.graphics.ssao = Some(on),
                                "ssr" => cfg.graphics.ssr = Some(on),
                                "ray_traced_reflections" => {
                                    cfg.graphics.ray_traced_reflections = Some(on)
                                }
                                "ssgi" => cfg.graphics.ssgi = Some(on),
                                "auto_exposure" => cfg.graphics.auto_exposure = Some(on),
                                _ => {}
                            }
                            // An explicit per-row quality change opts the master
                            // preset out to Custom (no ceiling clamps the user's
                            // choice), and updates the master row's label to match.
                            self.quality_preset = crate::gfx::quality_preset::QualityPreset::Custom;
                            cfg.graphics.quality_preset = Some(self.quality_preset);
                            set_cached_row_label(
                                &self.cycle_value_labels,
                                ctx,
                                "graphics_quality",
                                self.quality_preset.name(),
                            );
                            backend.apply_quality_settings(super::derive_quality_settings(
                                &self.post_config,
                            ));
                            // Auto-exposure overwrites the backend's live exposure
                            // each frame while it runs; once it is toggled off, the
                            // backend's copy is frozen at the last adapted value.
                            // Re-push the static post-process params (this side's
                            // `post_process.exposure` is the authored / slider EV,
                            // untouched by auto-exposure) so exposure reverts. A
                            // toggle-on is harmless: the AE loop overwrites it next
                            // frame.
                            if key == "auto_exposure" {
                                backend.update_post_process(self.post_process);
                            }
                            Some(opts[next])
                        }
                        // Cycle quality knobs (SSGI gather sub-quality dropdowns):
                        // cycle the value on the stored config, persist it, flip
                        // the preset to Custom, then rebuild the affected effect
                        // live (Metal; a no-op backend keeps the choice for the
                        // next launch). Rides the same apply_quality_settings path
                        // as the toggles -- the sub-tunable travels in the feature's
                        // settings payload, so no new backend method is needed.
                        key if super::is_quality_cycle(key) => {
                            let cur =
                                super::quality_cycle_index(&self.post_config, key).unwrap_or(0);
                            let next = settings::cycle(cur, opts.len(), cmd.op);
                            super::set_quality_cycle(&mut self.post_config, key, next);
                            match key {
                                "ssgi_resolution" => {
                                    cfg.graphics.ssgi_resolution =
                                        Some(self.post_config.ssgi_resolution)
                                }
                                "ssgi_rays" => {
                                    cfg.graphics.ssgi_rays = Some(self.post_config.ssgi_rays)
                                }
                                "ssgi_steps" => {
                                    cfg.graphics.ssgi_steps = Some(self.post_config.ssgi_steps)
                                }
                                "reflection_blur_resolution" => {
                                    cfg.graphics.reflection_blur_resolution =
                                        Some(self.post_config.reflection_blur_resolution)
                                }
                                _ => {}
                            }
                            self.quality_preset = crate::gfx::quality_preset::QualityPreset::Custom;
                            cfg.graphics.quality_preset = Some(self.quality_preset);
                            set_cached_row_label(
                                &self.cycle_value_labels,
                                ctx,
                                "graphics_quality",
                                self.quality_preset.name(),
                            );
                            backend.apply_quality_settings(super::derive_quality_settings(
                                &self.post_config,
                            ));
                            Some(opts[next])
                        }
                        // Display-output / upscaling preference toggles. Restart-
                        // required: persist + display only (the swapchain format /
                        // render targets are sized once at init, so it applies at
                        // the next launch). Independent of the quality preset, so
                        // no Custom-flip and no live backend call.
                        key @ ("temporal_upscaling" | "hdr_display" | "hdr_pq") => {
                            let cur = match key {
                                "temporal_upscaling" => self.temporal_upscaling,
                                "hdr_display" => self.hdr_display,
                                _ => self.hdr_pq,
                            };
                            let next = settings::cycle(cur as usize, opts.len(), cmd.op);
                            let on = next == 1;
                            match key {
                                "temporal_upscaling" => {
                                    self.temporal_upscaling = on;
                                    cfg.graphics.temporal_upscaling = Some(on);
                                }
                                "hdr_display" => {
                                    self.hdr_display = on;
                                    cfg.graphics.hdr_display = Some(on);
                                }
                                _ => {
                                    self.hdr_pq = on;
                                    cfg.graphics.hdr_pq = Some(on);
                                }
                            }
                            Some(opts[next])
                        }
                        // Shadow resolution: restart-required (the shadow map array
                        // is sized once at init), so persist + display only; the
                        // new size takes effect at the next launch. Preset-governed,
                        // so an explicit choice opts the master preset out to Custom.
                        "shadow_map_size" => {
                            let cur = settings::shadow_resolution_index(self.shadow_map_size);
                            let next = settings::cycle(cur, opts.len(), cmd.op);
                            self.shadow_map_size = settings::shadow_resolution_at(next);
                            cfg.graphics.shadow_map_size = Some(self.shadow_map_size);
                            self.quality_preset = crate::gfx::quality_preset::QualityPreset::Custom;
                            cfg.graphics.quality_preset = Some(self.quality_preset);
                            set_cached_row_label(
                                &self.cycle_value_labels,
                                ctx,
                                "graphics_quality",
                                self.quality_preset.name(),
                            );
                            Some(opts[next])
                        }
                        // Shadow re-render cadence: live -- the cascade scheduler
                        // reads the policy each frame, so it applies on the next
                        // draw. Preset-governed, so an explicit choice flips the
                        // master preset to Custom.
                        "shadow_update" => {
                            let cur = settings::shadow_update_index(self.shadow_update);
                            let next = settings::cycle(cur, opts.len(), cmd.op);
                            self.shadow_update = settings::shadow_update_at(next);
                            backend.set_shadow_update(self.shadow_update);
                            cfg.graphics.shadow_update = Some(self.shadow_update);
                            self.quality_preset = crate::gfx::quality_preset::QualityPreset::Custom;
                            cfg.graphics.quality_preset = Some(self.quality_preset);
                            set_cached_row_label(
                                &self.cycle_value_labels,
                                ctx,
                                "graphics_quality",
                                self.quality_preset.name(),
                            );
                            Some(opts[next])
                        }
                        // System / streaming restart rows. Restart-required (the
                        // ring buffers / cull pipeline / streaming pool are sized
                        // once at init), so persist + display only; independent of
                        // the quality preset, so no Custom-flip and no live call.
                        "frames_in_flight" => {
                            let cur =
                                settings::frames_in_flight_index(self.frames_in_flight as u32);
                            let next = settings::cycle(cur, opts.len(), cmd.op);
                            self.frames_in_flight = settings::frames_in_flight_at(next) as usize;
                            cfg.graphics.frames_in_flight = Some(self.frames_in_flight as u32);
                            Some(opts[next])
                        }
                        "occlusion_two_pass" => {
                            let next = settings::cycle(
                                self.occlusion_two_pass as usize,
                                opts.len(),
                                cmd.op,
                            );
                            self.occlusion_two_pass = next == 1;
                            cfg.graphics.occlusion_two_pass = Some(self.occlusion_two_pass);
                            Some(opts[next])
                        }
                        // One row drives both the streaming pool cap and the
                        // per-frame upload budget.
                        "texture_quality" => {
                            let cur = settings::texture_quality_index(self.texture_cap);
                            let next = settings::cycle(cur, opts.len(), cmd.op);
                            let (cap, budget) = settings::texture_quality_at(next);
                            self.texture_cap = cap;
                            self.texture_budget = budget;
                            cfg.graphics.texture_cap = Some(cap);
                            cfg.graphics.texture_budget = Some(budget);
                            Some(opts[next])
                        }
                        _ => None,
                    };
                    if let Some(text) = new_text {
                        if let Err(e) = cfg.save() {
                            tracing::warn!(
                                "GraphicsSystem: persist setting '{}': {}",
                                cmd.setting,
                                e
                            );
                        }
                        if let Some(label_id) = cmd.value_label {
                            set_label_content(ctx, label_id, text);
                        }
                    }
                }

                // advance SceneReel and apply fade / visibility changes, sourcing
                // visibility from the live per-entity components; the snapshot is
                // rebuilt each frame the reel exists.
                if self.reel.is_some() {
                    let (draws, scenes) = super::scene::decomposed_visibility_snapshot(ctx);
                    scene_reel::tick_reel(&mut self.reel, &draws, &scenes, elapsed, backend);
                }

                // Drive albedo-texture streaming: re-score every slot by
                // camera distance, dispatch this frame's background loads
                // within budget, then apply completed uploads + evictions.
                // Each backend's update_texture_slot rewrites whichever
                // descriptors / argument-buffers sample that slot so it
                // takes effect on this same draw_frame.
                if let Some(streamer) = &mut self.texture_streamer {
                    streamer.update_scores(cam_pos, self.frame_count);
                    for slot in streamer.plan_and_dispatch() {
                        if let Err(e) = backend.evict_texture_slot(slot) {
                            tracing::warn!("GraphicsSystem: texture evict slot {}: {}", slot, e);
                        }
                    }
                    streamer.drain_completed(self.frame_count, |slot, w, h, px| {
                        if let Err(e) = backend.update_texture_slot(slot, w, h, px) {
                            tracing::warn!("GraphicsSystem: texture upload slot {}: {}", slot, e);
                        }
                    });
                    // Surface streaming progress periodically so a headless
                    // run can confirm textures are coming resident.
                    if self.frame_count.is_multiple_of(120) {
                        let (resident, pending, unloaded) = streamer.stats();
                        tracing::info!(
                            "GraphicsSystem: texture streaming -- {} resident, {} pending, {} unloaded",
                            resident,
                            pending,
                            unloaded
                        );
                    }
                }

                // Drive normal-map streaming: identical to the albedo path
                // above, but streamed item `i` maps to normal-map pool slot
                // `i + 1` (slot 0 is the flat-normal fallback).
                if let Some(streamer) = &mut self.normal_map_streamer {
                    streamer.update_scores(cam_pos, self.frame_count);
                    for item in streamer.plan_and_dispatch() {
                        if let Err(e) = backend.evict_normal_map_slot(item + 1) {
                            tracing::warn!(
                                "GraphicsSystem: normal-map evict slot {}: {}",
                                item + 1,
                                e
                            );
                        }
                    }
                    streamer.drain_completed(self.frame_count, |item, w, h, px| {
                        if let Err(e) = backend.update_normal_map_slot(item + 1, w, h, px) {
                            tracing::warn!(
                                "GraphicsSystem: normal-map upload slot {}: {}",
                                item + 1,
                                e
                            );
                        }
                    });
                    if self.frame_count.is_multiple_of(120) {
                        let (resident, pending, unloaded) = streamer.stats();
                        tracing::info!(
                            "GraphicsSystem: normal-map streaming -- {} resident, {} pending, {} unloaded",
                            resident,
                            pending,
                            unloaded
                        );
                    }
                }

                // Drive mesh-geometry streaming: re-score each streamed mesh
                // by camera distance, dispatch this frame's background loads,
                // then apply completed geometry uploads + evictions. A mesh is
                // skipped in every pass until its geometry region is resident.
                if let Some(streamer) = &mut self.mesh_streamer {
                    streamer.update_scores(cam_pos, self.frame_count);
                    // A runtime eviction's freed space must not be reused
                    // until the in-flight command buffers that drew it retire.
                    let retire_frame = self.frame_count + self.frames_in_flight as u64;
                    for stream_id in streamer.plan_and_dispatch() {
                        if let Some(&draw_idx) = self.mesh_stream_draw_indices.get(stream_id)
                            && let Err(e) = backend.evict_mesh(draw_idx, retire_frame)
                        {
                            tracing::warn!("GraphicsSystem: mesh evict draw {}: {}", draw_idx, e);
                        }
                    }
                    let draw_indices = &self.mesh_stream_draw_indices;
                    let frame = self.frame_count;
                    streamer.drain_completed(self.frame_count, |stream_id, verts, idxs| {
                        match draw_indices.get(stream_id) {
                            // Return the upload result so the streamer can roll
                            // a transient seed-full miss back to Unloaded and
                            // retry it once freed regions reclaim, rather than
                            // marking the mesh resident with no GPU geometry.
                            Some(&draw_idx) => backend.upload_mesh(draw_idx, verts, idxs, frame),
                            None => Ok(()),
                        }
                    });
                    if self.frame_count.is_multiple_of(120) {
                        let (resident, pending, unloaded) = streamer.stats();
                        tracing::info!(
                            "GraphicsSystem: mesh streaming -- {} resident, {} pending, {} unloaded",
                            resident,
                            pending,
                            unloaded
                        );
                    }
                }

                // Drive infinite-world chunk streaming: generate + upload the
                // chunks entering the camera's view window and remove those
                // that have left it. None unless a VoxelWorld was declared.
                //
                // Camera-relative rendering: chunk geometry is placed
                // relative to a render origin that follows the camera's chunk,
                // and the view + camera position handed to the backend are
                // rebased onto the same origin. The world transform is
                // unchanged -- it is just evaluated from small coordinates, so
                // an unbounded world renders without large-coordinate jitter.
                // `final_view` / `final_cam_pos` stay absolute when no
                // VoxelWorld is streaming, leaving a non-voxel world
                // byte-for-byte unchanged.
                let mut final_view = view_matrix;
                let mut final_cam_pos = cam_pos;
                if let Some(cs) = &mut self.chunk_stream {
                    let camera_chunk = cs.streamer.camera_chunk(cam_pos);
                    let retire_frame = self.frame_count + self.frames_in_flight as u64;
                    for coord in cs.streamer.plan_and_dispatch(camera_chunk) {
                        if let Some(draw_idx) = cs.draws.remove(&coord)
                            && let Err(e) = backend.remove_chunk_mesh(draw_idx, retire_frame)
                        {
                            tracing::warn!(
                                "GraphicsSystem: chunk remove ({},{}): {}",
                                coord.x,
                                coord.z,
                                e
                            );
                        }
                    }
                    // The camera crossed into a new chunk: move the render
                    // origin to it and rebase every resident chunk's model
                    // matrix. `prev_draw_models` is deliberately left alone --
                    // the rebase is exact, so a stationary chunk shows zero TAA
                    // velocity across the shift.
                    if camera_chunk != cs.origin_chunk {
                        for (&coord, &draw_idx) in &cs.draws {
                            let model =
                                chunk_model_matrix(coord, camera_chunk, cs.chunk_w, cs.chunk_d);
                            if let Err(e) = backend.set_chunk_model(draw_idx, model) {
                                tracing::warn!(
                                    "GraphicsSystem: chunk rebase ({},{}): {}",
                                    coord.x,
                                    coord.z,
                                    e
                                );
                            }
                        }
                        cs.origin_chunk = camera_chunk;
                    }
                    let frame = self.frame_count;
                    let (chunk_w, chunk_d) = (cs.chunk_w, cs.chunk_d);
                    let (tex, nm, mat) = (cs.texture_slot, cs.normal_map_slot, cs.material);
                    let mut added: Vec<(crate::gfx::chunk_coord::ChunkCoord, usize)> = Vec::new();
                    cs.streamer.drain_completed(|coord, verts, idxs| {
                        let model = chunk_model_matrix(coord, camera_chunk, chunk_w, chunk_d);
                        match backend.add_chunk_mesh(verts, idxs, model, tex, nm, mat, frame) {
                            Ok(draw_idx) => added.push((coord, draw_idx)),
                            Err(e) => tracing::warn!(
                                "GraphicsSystem: chunk add ({},{}): {}",
                                coord.x,
                                coord.z,
                                e
                            ),
                        }
                    });
                    for (coord, draw_idx) in added {
                        cs.draws.insert(coord, draw_idx);
                    }
                    // Rebase the view + camera onto the render origin so the
                    // origin-relative chunk geometry above transforms exactly.
                    let (ox, oz) = camera_chunk.origin_world(cs.chunk_w, cs.chunk_d);
                    let origin = [ox, 0.0, oz];
                    final_view =
                        crate::gfx::chunk_coord::camera_relative_view(view_matrix, cam_pos, origin);
                    final_cam_pos = [cam_pos[0] - ox, cam_pos[1], cam_pos[2] - oz];
                    if self.frame_count.is_multiple_of(120) {
                        let (resident, pending) = cs.streamer.stats();
                        let (near, far) = cs.streamer.detail_counts();
                        tracing::info!(
                            "GraphicsSystem: chunk streaming -- {} resident ({} full, {} impostor), {} pending",
                            resident,
                            near,
                            far,
                            pending
                        );
                    }
                }

                // On Metal, pump_ns_events runs inside draw_frame, so update_view
                // is called first so any key/mouse events that arrived since the
                // last tick are in InputState before take_input() snapshots and
                // clears it.
                backend.update_view(final_view);
                match backend.draw_frame(
                    elapsed,
                    fov_y_radians,
                    near,
                    far,
                    final_cam_pos,
                    &text_calls,
                ) {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::error!("GraphicsSystem: draw_frame: {}", e);
                        backend.wait_idle();
                        return StepResult::Stop;
                    }
                }

                // Publish this frame's render stats for the profiler overlay.
                // Backends without GPU-timed stats return the trait's default
                // (all zeros), which the HUD displays as "--".
                ctx.profile.render = backend.render_stats();

                // deposit input for Camera3DSystem / UiInputSystem to read this
                // tick. Both query (not drain) it, so clear the previous frame's
                // snapshot first.
                let raw = backend.take_input();
                // Cache the cursor position for next frame's follow_cursor
                // sprites (the draw list is built before this poll).
                self.cursor_pos = (raw.mouse_x, raw.mouse_y);
                // Live viewport for UiInputSystem's overlay hit-testing, so a
                // scaled menu's HitRegions map back to the cursor consistently.
                let (vp_w, vp_h) = backend.logical_size();
                let _ = ctx.drain::<FrameInput>();
                // While a menu view is open, freeze gameplay input so the camera
                // does not drift behind the menu; the UI still gets the cursor
                // position, clicks, and Escape.
                let gameplay = !menu_active;
                let frame_input = FrameInput {
                    forward: raw.forward && gameplay,
                    backward: raw.backward && gameplay,
                    left: raw.left && gameplay,
                    right: raw.right && gameplay,
                    sprint: raw.sprint && gameplay,
                    interact: raw.interact && gameplay,
                    jump: raw.jump && gameplay,
                    mouse_dx: if gameplay { raw.mouse_dx } else { 0.0 },
                    mouse_dy: if gameplay { raw.mouse_dy } else { 0.0 },
                    // Not gated by `gameplay`: a scrollable menu still scrolls
                    // while it is open (the camera is what freezes behind it).
                    scroll_delta: raw.scroll_delta,
                    mouse_x: raw.mouse_x,
                    mouse_y: raw.mouse_y,
                    left_click: raw.left_click,
                    left_button_down: raw.left_button_down,
                    viewport: [vp_w, vp_h],
                    hud_toggle: raw.hud_toggle,
                    escape: raw.escape,
                    // Not gated by `gameplay`: the rebind capture works while the
                    // settings menu is open (the camera is what freezes behind it).
                    captured_key: raw.captured_key,
                };
                // Publish the same snapshot two ways: the resource readers can
                // fetch by type, and the component column the camera and UI
                // systems still drain/query.
                ctx.insert_resource(frame_input.clone());
                ctx.push(frame_input);

                StepResult::Continue
            }
        };

        if result == StepResult::Continue {
            self.frame_count += 1;
            if let Some(max) = self.max_frames
                && self.frame_count >= max
            {
                tracing::info!("GraphicsSystem: max_frames ({}) reached", max);
                self.wait_idle();
                return StepResult::Done;
            }
        }

        result
    }

    // Capture each slider row's runtime bookkeeping from its drag HitRegion +
    // handle Sprite, then sync the handle position and value label to the live
    // value. Runs once at init, before UiInputSystem drains the HitRegions and
    // hides the view elements. The HitRegions / Sprites are still present here.
    pub(super) fn init_sliders(&mut self, ctx: &mut PipelineContext) {
        let sprite_w: std::collections::HashMap<AssetId, f32> = ctx
            .query::<Sprite>()
            .map(|s| (s.asset_id, s.width))
            .collect();
        let mut sliders: Vec<SliderViz> = Vec::new();
        for r in ctx.query::<HitRegion>() {
            let Some(key) = slider_key_of(&r.action) else {
                continue;
            };
            let (Some(handle_id), Some(value_id)) = (r.drag_handle, r.label) else {
                continue;
            };
            let handle_w = sprite_w.get(&handle_id).copied().unwrap_or(0.0);
            sliders.push(SliderViz {
                key: key.to_string(),
                track_x: r.x,
                track_w: r.width,
                handle_w,
                handle_id,
                value_id,
            });
        }
        // Sync each slider's handle + value label to its live value.
        for s in &sliders {
            let Some(value) = self.slider_current_value(&s.key) else {
                continue;
            };
            let frac = settings::slider_fraction(&s.key, value).unwrap_or(0.0);
            let hx = s.track_x + frac.clamp(0.0, 1.0) * (s.track_w - s.handle_w).max(0.0);
            set_sprite_x(ctx, s.handle_id, hx);
            set_label_content(
                ctx,
                s.value_id,
                &settings::format_slider_value(&s.key, value),
            );
        }
        self.sliders = sliders;
    }

    // Capture each key-rebind row's bookkeeping from its `setting:key_*:rebind`
    // HitRegion, then sync each value label to the live bound key. Runs once at
    // init (after the keymap is seeded), before UiInputSystem drains the
    // HitRegions; they are still present here.
    pub(super) fn init_rebind_rows(&mut self, ctx: &mut PipelineContext) {
        let mut rows: Vec<RebindViz> = Vec::new();
        for r in ctx.query::<HitRegion>() {
            let Some(key) = rebind_key_of(&r.action) else {
                continue;
            };
            let (Some(action), Some(value_id)) =
                (crate::gfx::keymap::Bindable::from_setting_key(key), r.label)
            else {
                continue;
            };
            rows.push(RebindViz { action, value_id });
        }
        // Sync each value label to the live bound key (persisted or default).
        for row in &rows {
            let name = self.keymap.get(row.action).display_name();
            set_label_content(ctx, row.value_id, name);
        }
        self.rebind_rows = rows;
    }

    // Capture each cycle row's setting key -> value-label id, so a runtime change
    // can relabel a row other than the one clicked (the master preset relabels
    // its dependents; a quality-toggle change relabels the master row). Runs at
    // init, before UiInputSystem drains the HitRegions (GraphicsSystem.init runs
    // first), since they cannot be re-queried once drained.
    pub(super) fn init_cycle_value_labels(&mut self, ctx: &mut PipelineContext) {
        let mut labels = std::collections::HashMap::new();
        for r in ctx.query::<HitRegion>() {
            if let (Some(key), Some(value_id)) = (cycle_next_key_of(&r.action), r.label) {
                labels.insert(key.to_string(), value_id);
            }
        }
        self.cycle_value_labels = labels;
    }

    // Capture each ScrollPanel's per-element clip band (reference space) so the
    // draw path scissors scroll-content elements to their panel and off-band
    // rows do not bleed over the chrome. Runs at init, before UiInputSystem
    // drains the ScrollPanels (GraphicsSystem.init runs first); the panels are
    // still queryable here. Every element listed in any row maps to its panel's
    // content band.
    pub(super) fn init_clip_rects(&mut self, ctx: &mut PipelineContext) {
        let mut clips: std::collections::HashMap<AssetId, [f32; 4]> =
            std::collections::HashMap::new();
        for panel in ctx.query::<crate::assets::ScrollPanel>() {
            let band = [panel.x, panel.y, panel.width, panel.height];
            for row in &panel.rows {
                for &id in &row.elements {
                    clips.insert(id, band);
                }
            }
        }
        self.clip_rects = clips;
    }

    // Gray out and disable every settings row whose feature the device cannot
    // provide (e.g. ray-traced reflections on a GPU without hardware ray
    // tracing). Runs once at init after the backend reports `self.caps`, while
    // the HitRegions / TextLabels / ScrollPanels are still present (before
    // UiInputSystem drains them). A disabled HitRegion is dropped by
    // UiInputSystem so it never hovers or fires; the row's labels are recolored
    // to a muted gray so it reads as unavailable.
    pub(super) fn apply_capability_gating(&mut self, ctx: &mut PipelineContext) {
        let caps = self.caps;
        // Mark each unavailable setting's region(s) disabled and collect their
        // value-label ids (both stepper regions of a row reference its value
        // label, so this is the row's anchor into the scroll element list).
        let mut gated_value_labels: std::collections::HashSet<AssetId> =
            std::collections::HashSet::new();
        for r in ctx.query_mut::<HitRegion>() {
            let Some(rest) = r.action.strip_prefix("setting:") else {
                continue;
            };
            let Some(key) = rest.split(':').next() else {
                continue;
            };
            if settings::setting_available(key, &caps) {
                continue;
            }
            r.disabled = true;
            if let Some(label) = r.label {
                gated_value_labels.insert(label);
            }
        }
        if gated_value_labels.is_empty() {
            return;
        }
        // Snapshot each scroll row's element id list (owned, so the ScrollPanel
        // borrow ends before the TextLabel write below), then expand the gated
        // value labels to every element of the rows that contain them.
        let rows: Vec<Vec<AssetId>> = ctx
            .query::<crate::assets::ScrollPanel>()
            .flat_map(|p| p.rows.iter().map(|r| r.elements.clone()))
            .collect();
        let dim = expand_dim_set(&gated_value_labels, &rows);
        for l in ctx.query_mut::<TextLabel>() {
            if dim.contains(&l.asset_id) {
                l.color = DISABLED_ROW_COLOR;
            }
        }
    }

    // The current user-facing value of a slider setting, derived from the live
    // post-process params. `None` for a key this system does not own.
    fn slider_current_value(&self, key: &str) -> Option<f32> {
        let stored = match key {
            "exposure" => self.post_process.exposure,
            "bloom_intensity" => self.post_process.bloom_intensity,
            "bloom_threshold" => self.post_process.bloom_threshold,
            "bloom_knee" => self.post_process.bloom_knee,
            "vignette" => self.post_process.vignette,
            "lut_strength" => self.post_process.lut_strength,
            "ambient_intensity" => self.ambient_intensity,
            // Per-feature sub-quality sliders read from the stored PostProcessConfig.
            "ssao_radius" => self.post_config.ssao_radius,
            "ssao_intensity" => self.post_config.ssao_intensity,
            "ssr_intensity" => self.post_config.ssr_intensity,
            "ssr_max_distance" => self.post_config.ssr_max_distance,
            "ssgi_intensity" => self.post_config.ssgi_intensity,
            "ssgi_max_distance" => self.post_config.ssgi_max_distance,
            "auto_exposure_min_ev" => self.post_config.auto_exposure_min_ev,
            "auto_exposure_max_ev" => self.post_config.auto_exposure_max_ev,
            "auto_exposure_speed" => self.post_config.auto_exposure_speed,
            // Mouse sensitivity lives in the controls store (radians/pixel), not
            // the render params; read the persisted value or the authored default.
            "mouse_sensitivity" => crate::config::Settings::load()
                .controls
                .mouse_sensitivity
                .unwrap_or(settings::DEFAULT_MOUSE_SENSITIVITY),
            _ => return None,
        };
        // Invert `slider_apply_value` to the user-facing value (exposure: 2^ev ->
        // EV; mouse sensitivity: radians/pixel -> 1..100).
        Some(settings::slider_recover_value(key, stored))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // A gated value label pulls in every element of the scroll row that holds
    // it (the row's background, name, value, and stepper glyphs), so the whole
    // row grays out; unrelated rows are untouched.
    #[test]
    fn dim_set_expands_a_gated_value_label_to_its_whole_row() {
        let value = AssetId(3);
        let gated: HashSet<AssetId> = [value].into_iter().collect();
        let rows = vec![
            // Row A: bg, name, prev_glyph, value, next_glyph (value is gated).
            vec![AssetId(1), AssetId(2), value, AssetId(4), AssetId(5)],
            // Row B: an unrelated row.
            vec![AssetId(10), AssetId(11)],
        ];
        let dim = expand_dim_set(&gated, &rows);
        for id in [1, 2, 3, 4, 5] {
            assert!(dim.contains(&AssetId(id)), "row A element {id} should dim");
        }
        assert!(!dim.contains(&AssetId(10)), "an unrelated row stays lit");
        assert!(!dim.contains(&AssetId(11)), "an unrelated row stays lit");
    }

    // With no scroll rows (a hand-authored menu outside a panel), only the gated
    // value label itself dims -- a graceful fallback, not a panic.
    #[test]
    fn dim_set_without_rows_falls_back_to_the_value_label() {
        let gated: HashSet<AssetId> = [AssetId(7)].into_iter().collect();
        assert_eq!(expand_dim_set(&gated, &[]), gated);
    }
}
