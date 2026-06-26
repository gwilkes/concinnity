// src/audio/system.rs
//
// 3D positional audio playback. An internal system (not a declarable asset):
// `World::start` constructs one whenever the world contains any `AudioEmitter`,
// so a world with no emitters never opens an audio device.

use std::collections::HashMap;

use crate::assets::{AudioClip, AudioCommand, AudioEmitter, Camera3D, Prop};
use crate::audio::{AudioEngine, EmitterId};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};

// 3D positional audio behavior. Constructed internally by `World::start` when
// the world declares any `AudioEmitter`; never a world-declared asset, so it
// carries no config.
pub struct AudioSystem {
    engine: AudioEngine,
    // One entry per `AudioEmitter` that became a live engine emitter.
    emitters: Vec<EmitterBinding>,
    // Cursor into the Events<AudioCommand> queue (live master-volume changes).
    audio_cmd_cursor: crate::ecs::EventCursor,
}

// Links one engine emitter to the world data that positions it.
struct EmitterBinding {
    id: EmitterId,
    // The Prop this emitter follows each frame, if any.
    follows: Option<AssetId>,
}

impl std::fmt::Debug for AudioSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioSystem")
            .field("engine", &self.engine)
            .field("emitters", &self.emitters.len())
            .finish()
    }
}

impl AudioSystem {
    // Fresh audio engine with no live emitters. Emitters are bound from the
    // world's `AudioEmitter` components in [`System::init`].
    pub fn new() -> Self {
        Self {
            engine: AudioEngine::new(),
            emitters: Vec::new(),
            audio_cmd_cursor: crate::ecs::EventCursor::default(),
        }
    }
}

impl System for AudioSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        // Snapshot the emitters, then resolve every clip's payload locator so
        // the borrow of `ctx` for the AudioClip query is released before the
        // `read_payload` calls below.
        let emitter_snaps: Vec<AudioEmitter> = ctx.query::<AudioEmitter>().cloned().collect();
        let clip_locators: HashMap<AssetId, crate::ecs::PayloadLocator> = ctx
            .query::<AudioClip>()
            .filter_map(|c| c.locator.clone().map(|l| (c.asset_id, l)))
            .collect();

        // The persisted master volume (settings menu) scales every emitter via
        // the main mix track, so it can be changed live (see `step`). `None`
        // leaves output at unity. Clips play at their authored gain; the master
        // is a separate output-stage multiplier.
        let master = crate::config::Settings::load()
            .audio
            .master_volume
            .unwrap_or(1.0);
        self.engine.set_master_volume(master);

        for emitter in emitter_snaps {
            let Some(id) = self.engine.add_emitter(emitter.position) else {
                continue;
            };
            match emitter.clip.and_then(|clip| clip_locators.get(&clip)) {
                Some(locator) => match ctx.read_payload(locator) {
                    Ok(bytes) => {
                        self.engine
                            .play_clip(id, bytes, emitter.looping, emitter.volume);
                    }
                    Err(e) => tracing::warn!("AudioSystem: clip payload read failed: {e}"),
                },
                None => tracing::warn!(
                    "AudioSystem: emitter has no clip with a compiled payload, silent"
                ),
            }
            self.emitters.push(EmitterBinding {
                id,
                follows: emitter.prop,
            });
        }

        tracing::info!(
            "AudioSystem: {} emitter(s), engine {}",
            self.emitters.len(),
            if self.engine.is_enabled() {
                "enabled"
            } else {
                "disabled"
            },
        );
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // Apply any live master-volume change sent this tick by GraphicsSystem,
        // which runs first. The last one this tick wins.
        if let Some(events) = ctx.events::<AudioCommand>() {
            for cmd in events.read(&mut self.audio_cmd_cursor) {
                self.engine.set_master_volume(cmd.master_volume);
            }
        }

        // The listener rides the camera.
        if let Some((pos, yaw, pitch)) = ctx
            .query::<Camera3D>()
            .next()
            .map(|c| (c.position, c.yaw, c.pitch))
        {
            self.engine.set_listener(pos, yaw, pitch);
        }

        // Prop-bound emitters track their Prop's current position.
        if self.emitters.iter().any(|b| b.follows.is_some()) {
            let prop_positions: HashMap<AssetId, [f32; 3]> = ctx
                .query::<Prop>()
                .map(|p| (p.asset_id, p.position))
                .collect();
            for binding in &self.emitters {
                if let Some(prop_id) = binding.follows
                    && let Some(&pos) = prop_positions.get(&prop_id)
                {
                    self.engine.set_emitter_position(binding.id, pos);
                }
            }
        }

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use crate::assets::AudioEmitter;
    use crate::ecs::World;

    // An `AudioEmitter` in the world spawns the internal AudioSystem; without
    // one, no audio device is opened.
    #[test]
    fn audio_emitter_spawns_internal_system() {
        let mut world = World::new_empty();
        world.add_component(AudioEmitter::default());
        world.start().unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["AudioSystem"]);
    }

    #[test]
    fn no_audio_emitter_means_no_system() {
        let mut world = World::new_empty();
        world.start().unwrap();
        assert!(world.systems().is_empty());
    }

    // The live master gain the world's AudioSystem currently holds. Mirrors how
    // the ControlsCommand test observes Camera3D.yaw: reach into the running
    // system to assert the command actually took effect (the gain is engine
    // state, not a queryable component).
    fn applied_master_volume(world: &World) -> f32 {
        world
            .systems()
            .iter()
            .find_map(|s| match s {
                crate::ecs::SystemAsset::AudioSystem(a) => Some(a.engine.last_master_volume),
                _ => None,
            })
            .expect("world has an AudioSystem")
    }

    // A master-volume AudioCommand sent mid-tick is read AND applied by the
    // audio system the same tick, so the new master takes effect without a
    // restart (the settings-menu master-volume row).
    #[test]
    fn audio_command_applies_master_volume_live() {
        use crate::assets::AudioCommand;

        let mut world = World::new_empty();
        world.add_component(AudioEmitter::default());
        world.start().unwrap();
        // Init applied the persisted master (unity by default in a test).
        assert!((applied_master_volume(&world) - 1.0).abs() < 1.0e-6);

        // GraphicsSystem would send this when the master-volume row is cycled;
        // the audio system reads it this same tick.
        world
            .events_mut::<AudioCommand>()
            .send(AudioCommand { master_volume: 0.5 });
        world.step();

        assert!(
            (applied_master_volume(&world) - 0.5).abs() < 1.0e-6,
            "master volume should be applied live this tick"
        );
    }

    // Several AudioCommands sent in one tick (e.g. a rapid double-cycle) are
    // all read in order; the last one sent is applied last and wins.
    #[test]
    fn audio_command_last_write_wins_per_tick() {
        use crate::assets::AudioCommand;

        let mut world = World::new_empty();
        world.add_component(AudioEmitter::default());
        world.start().unwrap();

        world
            .events_mut::<AudioCommand>()
            .send(AudioCommand { master_volume: 0.5 });
        world.events_mut::<AudioCommand>().send(AudioCommand {
            master_volume: 0.25,
        });
        world.step();

        assert!((applied_master_volume(&world) - 0.25).abs() < 1.0e-6);
    }
}
