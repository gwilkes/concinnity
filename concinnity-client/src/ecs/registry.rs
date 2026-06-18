// src/ecs/registry.rs
//
// Client-side half of the asset registry: the runtime `SystemAsset` enum,
// generated from each system's `System` behavior impl. It holds a constructed
// system and dispatches init/step.
//
// Every system is internal: it has no declarable asset and carries no
// discriminant. `World::build_internal_systems` constructs each from the
// world's components. To add a system: implement `System` on it, list it here,
// and add a gated entry to that schedule.

use crate::ecs::{PipelineContext, StepResult, System};

crate::define_system_assets! {
    GraphicsSystem  => crate::gfx::graphics_system::GraphicsSystem,
    PhysicsSystem   => crate::physics::system::PhysicsSystem,
    Camera3DSystem  => crate::gfx::camera_controller::Camera3DSystem,
    AnimationSystem => crate::gfx::animation::AnimationSystem,
    AudioSystem     => crate::audio::system::AudioSystem,
    UiInputSystem   => crate::ui::UiInputSystem,
    FpsCounter      => crate::hud::fps_counter::FpsCounterSystem,
    StatHud         => crate::hud::stat_hud::StatHudSystem,
}
