// src/gfx/animation.rs
//
// Skeletal animation playback. An internal system (not a declarable asset):
// `World::start` constructs one whenever the world contains any `Animation`
// component, then it samples the clip(s) targeting each `SkeletonPose` every
// frame to produce fresh skinning matrices.

use std::collections::HashMap;
use std::time::Instant;

use crate::assets::{Animation, SkeletonPose};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use crate::gfx::skinning::{self, AnimationClip};
use crate::jobs;

// A runtime clip plus the static metadata captured from its `Animation`
// asset. The live blend weight is stored separately on the owning
// [`TargetState`] so a runtime crossfade can re-weight clips without
// rewriting their data.
struct ClipEntry {
    clip: AnimationClip,
    // The declared steady-state weight from the asset. The bucket settles
    // on this once any initial fade-in or runtime crossfade completes.
    declared_weight: f32,
    // Seconds the clip ramps from zero to `declared_weight` at world start.
    // Zero plays the clip at full strength from the first frame.
    fade_in_secs: f32,
}

// One ramp between two weight vectors. The bucket holds at most one
// transition at a time; a new transition supersedes any in flight.
#[derive(Debug)]
struct Transition {
    source_weights: Vec<f32>,
    target_weights: Vec<f32>,
    // Wall-clock seconds since the system's first step.
    start_secs: f32,
    // Length of the ramp. Zero snaps to `target_weights` on the next
    // `step`.
    duration_secs: f32,
}

// Per-`SkinnedMesh` bucket of clips. `current_weights` is the live blend
// vector consumed by [`skinning::blend_many`].
struct TargetState {
    clips: Vec<ClipEntry>,
    current_weights: Vec<f32>,
    transition: Option<Transition>,
}

// One hot-reload entry for a file-backed `Animation`. Captured at init
// alongside the runtime clip; consulted by the per-step reload pass when the
// shared `PENDING_ANIMATIONS` flag fires (see
// [`crate::app::dev_flags::take_pending_animations`]). Inline-authored
// animations (no `source`) carry no entry; there's no file to watch and
// the build pipeline never expanded one.
//
// `pub` (with public fields) because the editor crate's hot-reload drive reads
// these to re-import the clip from source, then pushes the result back through
// `AnimationSystem::apply_reloaded_clip`. The GLB decode itself lives in the
// editor crate; the runtime crate only stores the catalogue.
#[derive(Debug, Clone)]
pub struct AnimationReloadEntry {
    // Target `SkinnedMesh` asset id, also the key into
    // [`AnimationSystem::targets`] where this clip lives.
    pub target: AssetId,
    // Position in the target bucket's `clips`. Set at init when the clip is
    // first pushed; stable for the process lifetime since the Vec is
    // neither rebuilt nor trimmed.
    pub clip_index: usize,
    // `.glb` source path verbatim from the asset declaration; used as-is by
    // the GLB parser at reload time.
    pub source: String,
    // Mirrors [`Animation::animation_index`].
    pub animation_index: u32,
    // Mirrors [`Animation::animation_name`] (precedence over index when
    // non-empty).
    pub animation_name: String,
    // Mirrors [`Animation::weight`]; the .glb has nothing equivalent, so
    // it's carried through the reload unchanged.
    pub weight: f32,
    // Mirrors [`Animation::looping`]; same rationale as `weight`.
    pub looping: bool,
}

// Skeletal animation playback behavior. Constructed internally by
// `World::start` when the world declares any `Animation`; never a
// world-declared asset, so it carries no config.
pub struct AnimationSystem {
    // Per-target clip buckets keyed by the `SkinnedMesh` asset id they
    // animate. Buckets with several clips blend each frame.
    targets: HashMap<AssetId, TargetState>,
    // Wall-clock origin, captured on the first step.
    start: Option<Instant>,
    // When a menu opened (and froze playback), if currently paused. On resume
    // the origin `start` is shifted forward by the paused span so clip time `t`
    // is continuous across the pause: the animation freezes on its current pose
    // and resumes from it, with no jump.
    pause_anchor: Option<Instant>,
    // One entry per file-backed Animation, captured at init under
    // `cn debug`. Empty when hot-reload is off or every clip is inline.
    reload_entries: Vec<AnimationReloadEntry>,
}

impl std::fmt::Debug for AnimationSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationSystem")
            .field("targets", &self.targets.len())
            .field("reload_entries", &self.reload_entries.len())
            .finish()
    }
}

impl Default for AnimationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl AnimationSystem {
    // Fresh playback state with no clips. Clips are drained from the world's
    // `Animation` components in [`System::init`].
    pub fn new() -> Self {
        Self {
            targets: HashMap::new(),
            start: None,
            pause_anchor: None,
            reload_entries: Vec::new(),
        }
    }

    // The file-backed clips captured at init under `cn debug`. The editor
    // crate's hot-reload drive reads these to re-import each clip from source.
    // Empty when hot-reload is off or every clip is inline.
    // `dead_code` allowed while the runtime crate still carries the legacy
    // binary; the only caller is the editor crate (external). Removed when the
    // binary moves out of the runtime crate.
    #[allow(dead_code)]
    pub fn reload_entries(&self) -> &[AnimationReloadEntry] {
        &self.reload_entries
    }

    // Swap a freshly re-imported `clip` into the bucket slot identified by
    // `target` + `clip_index`, restoring its declared `weight`. Returns false
    // if the target bucket disappeared or the slot index is out of range
    // (a half-applied reload is impossible: nothing is mutated on miss). The
    // editor crate calls this after decoding the source GLB; the runtime crate
    // does no decoding of its own.
    #[allow(dead_code)]
    pub fn apply_reloaded_clip(
        &mut self,
        target: AssetId,
        clip_index: usize,
        clip: AnimationClip,
        weight: f32,
    ) -> bool {
        let Some(bucket) = self.targets.get_mut(&target) else {
            return false;
        };
        let Some(slot) = bucket.clips.get_mut(clip_index) else {
            return false;
        };
        slot.clip = clip;
        slot.declared_weight = weight;
        true
    }

    // Drain pending runtime crossfade commands against the system's own clock.
    // Wraps [`Self::drain_runtime_commands`] with the same `start` / elapsed
    // bookkeeping `step` uses, so the binary-only `DebugHook::tick` drive can
    // apply `anim-crossfade` commands from outside the per-system step. The
    // library never calls this (the drive is in the `cn debug` binary), hence
    // the `dead_code` allowance. `step` runs after the hook on the same frame,
    // so the `start` anchor set here is shared.
    #[allow(dead_code)]
    pub fn apply_crossfade_commands(&mut self) {
        let now = Instant::now();
        let start = *self.start.get_or_insert(now);
        let t = (now - start).as_secs_f32();
        self.drain_runtime_commands(t);
    }

    // Drain pending runtime crossfade commands and apply them. Each
    // command targets one bucket and sets up a new ramp from the bucket's
    // current weights to the requested target over `duration_secs`.
    // Mismatched-length weight vectors or unknown targets fail without
    // touching the bucket. Commands run in queue order, so a later
    // command for the same target supersedes an earlier one.
    fn drain_runtime_commands(&mut self, now_secs: f32) {
        for cmd in crate::app::anim_runtime::drain() {
            match cmd {
                crate::app::anim_runtime::AnimCommand::Crossfade { req, reply } => {
                    let Some(state) = self.targets.get_mut(&req.target) else {
                        let _ = reply.send(Err(format!(
                            "anim-crossfade: no Animation registered for target {}",
                            req.target
                        )));
                        continue;
                    };
                    if req.weights.len() != state.clips.len() {
                        let _ = reply.send(Err(format!(
                            "anim-crossfade: weight count {} does not match clip count {} for target {}",
                            req.weights.len(),
                            state.clips.len(),
                            req.target,
                        )));
                        continue;
                    }
                    state.transition = Some(Transition {
                        source_weights: state.current_weights.clone(),
                        target_weights: req.weights,
                        start_secs: now_secs,
                        duration_secs: req.duration_secs.max(0.0),
                    });
                    let _ = reply.send(Ok(()));
                }
            }
        }
    }
}

// The animation origin to use on the frame a pause ends. Shifting the original
// origin forward by the paused span (now - anchor) holds clip time
// `t = now - origin` exactly where it was when the pause began, so playback
// resumes from the frozen pose with no jump. Split out so the continuity
// property is unit-testable without a live system.
fn resumed_origin(start: Instant, anchor: Instant, now: Instant) -> Instant {
    start + now.saturating_duration_since(anchor)
}

// Advance one bucket's `current_weights` along its active transition (if
// any). Returns the live weights to feed the blend.
fn advance_weights(state: &mut TargetState, now_secs: f32) -> &[f32] {
    if let Some(tr) = &state.transition {
        let finished = if tr.duration_secs <= 0.0 {
            true
        } else {
            now_secs >= tr.start_secs + tr.duration_secs
        };
        if finished {
            state.current_weights.clone_from(&tr.target_weights);
            state.transition = None;
        } else {
            let progress = ((now_secs - tr.start_secs) / tr.duration_secs).clamp(0.0, 1.0);
            for (slot, (src, dst)) in state
                .current_weights
                .iter_mut()
                .zip(tr.source_weights.iter().zip(tr.target_weights.iter()))
            {
                *slot = src + (dst - src) * progress;
            }
        }
    }
    &state.current_weights
}

impl System for AnimationSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        // Clips accumulate per target mesh; several clips on one target are
        // blended each frame rather than overwriting one another.
        let capture_sources = crate::app::dev_flags::enabled();
        let mut count = 0usize;
        for anim in ctx.drain::<Animation>() {
            let Some(target) = anim.target else {
                tracing::warn!("AnimationSystem: Animation has no target SkinnedMesh, ignored");
                continue;
            };
            let weight = anim.weight;
            let fade_in_secs = anim.fade_in_secs.max(0.0);
            let state = self.targets.entry(target).or_insert_with(|| TargetState {
                clips: Vec::new(),
                current_weights: Vec::new(),
                transition: None,
            });
            let clip_index = state.clips.len();
            state.clips.push(ClipEntry {
                clip: anim.to_clip(),
                declared_weight: weight,
                fade_in_secs,
            });
            // Each new clip starts at full declared weight unless it requests
            // a fade-in, in which case it begins at zero and ramps up.
            let initial = if fade_in_secs > 0.0 { 0.0 } else { weight };
            state.current_weights.push(initial);
            if capture_sources && !anim.source.is_empty() {
                self.reload_entries.push(AnimationReloadEntry {
                    target,
                    clip_index,
                    source: anim.source.clone(),
                    animation_index: anim.animation_index,
                    animation_name: anim.animation_name.clone(),
                    weight,
                    looping: anim.looping,
                });
            }
            count += 1;
        }
        // Build a startup transition for any bucket whose clips requested a
        // fade-in. The transition runs from zero to the declared weights over
        // the bucket's longest fade-in; clips with `fade_in_secs == 0` start
        // already at their declared weight via `current_weights`, so the lerp
        // leaves them alone.
        for state in self.targets.values_mut() {
            let max_fade = state
                .clips
                .iter()
                .fold(0.0f32, |m, c| m.max(c.fade_in_secs));
            if max_fade > 0.0 {
                let source = state.current_weights.clone();
                let target: Vec<f32> = state.clips.iter().map(|c| c.declared_weight).collect();
                state.transition = Some(Transition {
                    source_weights: source,
                    target_weights: target,
                    // Start the ramp on the first step (negative until then,
                    // overwritten in `step`).
                    start_secs: 0.0,
                    duration_secs: max_fade,
                });
            }
        }
        if capture_sources && !self.reload_entries.is_empty() {
            tracing::info!(
                "AnimationSystem: {} clip(s) loaded across {} target mesh(es); {} \
                 file-backed clip(s) captured for hot-reload",
                count,
                self.targets.len(),
                self.reload_entries.len()
            );
        } else {
            tracing::info!(
                "AnimationSystem: {} clip(s) loaded across {} target mesh(es)",
                count,
                self.targets.len()
            );
        }
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // Asset hot-reload of file-backed clips (`cn debug` only) is driven
        // from the binary's `DebugHook::tick` via `reload_clips_if_pending`,
        // not here. `cn run` has no debug hook, so this step is reload-free.

        let now = Instant::now();

        // Freeze while a menu is open: skip all sampling so animation stops
        // consuming CPU/GPU behind the menu, recording when the pause began.
        // The flag is published by GraphicsSystem, which runs first this tick.
        let paused = ctx
            .resource::<crate::ecs::MenuActive>()
            .is_some_and(|m| m.0);
        if paused {
            self.pause_anchor.get_or_insert(now);
            return StepResult::Continue;
        }
        // Resuming: advance the origin by the paused span so clip time `t` stays
        // continuous -- the animation resumes from the exact pose it froze on,
        // with no jump. (A pause before the first step has no origin yet, so it
        // just defers the capture below.)
        if let Some(anchor) = self.pause_anchor.take()
            && let Some(start) = self.start.as_mut()
        {
            *start = resumed_origin(*start, anchor, now);
        }

        let start = *self.start.get_or_insert(now);
        let t = (now - start).as_secs_f32();

        // First-frame fix-up: the startup transition built in `init` has
        // `start_secs == 0.0`. We don't know the wall-clock origin until the
        // first step, so re-anchor any in-flight transition that hasn't yet
        // started elapsing.
        for state in self.targets.values_mut() {
            if let Some(tr) = state.transition.as_mut()
                && tr.start_secs == 0.0
            {
                tr.start_secs = t;
            }
        }

        // Runtime crossfade commands (`cn debug` WS `anim-crossfade`) are
        // drained from the binary's `DebugHook::tick` via
        // `apply_crossfade_commands`, not here.

        // Advance each bucket's weights along its active transition (if any)
        // before sampling, so the per-pose blend uses the live values.
        for state in self.targets.values_mut() {
            let _ = advance_weights(state, t);
        }

        // Each `SkeletonPose` is sampled and skinned independently, so the
        // per-pose work fans across the job pool and joins before returning.
        let targets = &self.targets;
        let poses = ctx.query_slice_mut::<SkeletonPose>();
        jobs::pool().parallel_for(poses, |pose| {
            let Some(state) = targets.get(&pose.mesh_id) else {
                return;
            };
            let locals = match state.clips.as_slice() {
                [] => return,
                [single] => {
                    // One-clip buckets ignore weight and play at full
                    // strength; the blend would be a no-op anyway.
                    single.clip.sample(t, &pose.skeleton)
                }
                many => {
                    let sampled: Vec<_> = many
                        .iter()
                        .map(|c| c.clip.sample(t, &pose.skeleton))
                        .collect();
                    skinning::blend_many(&sampled, &state.current_weights)
                }
            };
            pose.joint_matrices = pose.skeleton.skinning_matrices(&locals);
        });

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::resumed_origin;
    use crate::assets::Animation;
    use crate::ecs::World;
    use std::time::{Duration, Instant};

    // Resuming after a pause must leave clip time `t = now - origin` exactly
    // where it was when the pause began, so playback continues from the frozen
    // pose with no jump, no matter how long the menu was open.
    #[test]
    fn resumed_origin_freezes_clip_time_across_pause() {
        let start = Instant::now();
        // Paused at t = 5s, menu held open for 30s of real time.
        let anchor = start + Duration::from_secs(5);
        let now = anchor + Duration::from_secs(30);

        let t_at_pause = (anchor - start).as_secs_f32();
        let new_origin = resumed_origin(start, anchor, now);
        let t_on_resume = (now - new_origin).as_secs_f32();

        assert!(
            (t_on_resume - t_at_pause).abs() < 1e-6,
            "clip time jumped across the pause: {t_at_pause} -> {t_on_resume}"
        );
    }

    // An `Animation` in the world implies the internal AnimationSystem: it is
    // constructed by `World::start`, not declared as an asset.
    #[test]
    fn animation_component_spawns_internal_system() {
        let mut world = World::new_empty();
        world.add_component(Animation::default());
        world.start().unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["AnimationSystem"]);
    }

    // No `Animation` means no AnimationSystem; the gate keys purely off world
    // content.
    #[test]
    fn no_animation_no_internal_system() {
        let mut world = World::new_empty();
        world.start().unwrap();
        assert!(world.systems().is_empty());
    }
}
