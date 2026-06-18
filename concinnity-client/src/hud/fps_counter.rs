// src/hud/fps_counter.rs
//
// FPS-counter overlay behavior. An internal system (not a declarable asset):
// `World::start` constructs one from the world's `FpsCounter` component and it
// updates that component's `label` with the current rate once per second.

use crate::assets::{FpsCounter, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use std::time::Instant;

#[derive(Debug)]
pub struct FpsCounterSystem {
    last_time: Instant,
    frame_count: u32,
    label: Option<AssetId>,
}

impl FpsCounterSystem {
    // Build the counter from a world's `FpsCounter` request component.
    pub fn new(config: FpsCounter) -> Self {
        Self {
            last_time: Instant::now(),
            frame_count: 0,
            label: config.label,
        }
    }
}

impl System for FpsCounterSystem {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        self.frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_time).as_secs_f64();
        if elapsed >= 1.0 {
            let fps = self.frame_count as f64 / elapsed;
            if let Some(label_id) = self.label {
                for lbl in ctx.query_mut::<TextLabel>() {
                    if lbl.asset_id == label_id {
                        lbl.content = format!("FPS: {:.0}", fps);
                        break;
                    }
                }
            }
            self.frame_count = 0;
            self.last_time = now;
        }
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use crate::assets::FpsCounter;
    use crate::ecs::World;

    // An FpsCounter component spawns the internal counter system.
    #[test]
    fn fps_counter_component_spawns_internal_system() {
        let mut world = World::new_empty();
        world.add_component(FpsCounter::default());
        world.start().unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["FpsCounter"]);
    }

    #[test]
    fn no_fps_counter_no_system() {
        let mut world = World::new_empty();
        world.start().unwrap();
        assert!(world.systems().is_empty());
    }
}
