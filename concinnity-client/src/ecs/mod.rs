// src/ecs/mod.rs
//
// Client-side ecs runtime. The renderer-free metadata, asset registry,
// registration macros, asset-construction API, and `PipelineContext` all live
// in concinnity-core; this module re-exports them under the historical
// `crate::ecs::*` paths and adds the runtime behavior half: the `System`
// behavior trait, `StepResult`, the `SystemAsset` value enum (generated from
// `System` in `registry`), the unified `Asset` handle, and the `World`.
//
// TO ADD A NEW COMPONENT: register it in concinnity-core's `ecs::registry`
// (`define_components!`). TO ADD A NEW SYSTEM: implement the `System` behavior
// trait on it, register it in this crate's `ecs::registry`
// (`define_system_assets!`), and add a gated entry to the
// `World::build_internal_systems` schedule below.

pub(crate) mod decompose;
mod registry;

// Renderer-free metadata, registry types, the asset-construction API, and the
// `PipelineContext`, re-exported from concinnity-core so the rest of the client
// keeps its historical `crate::ecs::*` import paths.
pub use concinnity_core::ecs::{
    BlobAssetDef, Component, ComponentAsset, ComponentSlot, ComponentStorage, ComponentType,
    Entity, EventCursor, Events, PayloadLocator, PipelineContext, Resources, asset_api, asset_id,
};

// The `SystemAsset` value enum is generated client-side from each system's
// `System` behavior impl (see `registry`).
pub use registry::SystemAsset;

use crate::blob::BlobData;
use crate::gfx::profile::FrameProfile;
use crate::result::CnResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    // Keep running.
    Continue,
    // This system is finished -- remove it from the active set.
    // The world exits naturally when no systems remain.
    Done,
    // Hard stop -- halt everything immediately.
    #[allow(dead_code)]
    Stop,
}

// Per-frame menu state, published as a resource by GraphicsSystem (which runs
// first in the schedule) and read by the simulation systems the same tick.
// `true` while any menu view is open: physics and animation then freeze so they
// stop consuming resources behind the menu. Each system keeps its own clock
// aligned across the freeze, so resuming costs one normal frame -- no catch-up
// burst, no pose jump.
#[derive(Debug, Clone, Copy, Default)]
pub struct MenuActive(pub bool);

// Per-frame stats-HUD visibility, published as a resource by GraphicsSystem
// (which runs first) and read by `StatHudSystem` the same tick. Each field is
// the effective on/off for that chip: the master "Display performance stats"
// toggle AND the per-readout toggle from the video settings. Absent (a HUD-only
// unit test with no GraphicsSystem) is treated as both shown.
#[derive(Debug, Clone, Copy)]
pub struct HudPrefs {
    pub show_fps: bool,
    pub show_vram: bool,
}

// Setting rows the engine has disabled at runtime (their keys, e.g. `show_fps`
// while "Display performance stats" is off). Published each frame by
// GraphicsSystem and read by `UiInputSystem`, which makes a matching row inert
// (no hover, no click) while its labels are grayed independently. Distinct from
// the init-time capability gating (which marks `HitRegion.disabled` before the
// regions are drained); this drives the same effect after they are drained.
#[derive(Debug, Clone, Default)]
pub struct DisabledSettingRows(pub std::collections::HashSet<String>);

// System -- has behavior, receives a PipelineContext each tick. Every system
// is internal engine code: `World::build_internal_systems` constructs it from
// world components (via the system's own `new(..)`), so a system is never
// loaded from or written to a blob. `init` runs once at `World::start`; `step`
// runs every tick.
pub trait System: Sized + std::fmt::Debug + 'static {
    fn init(&mut self, _ctx: &mut PipelineContext) {}
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult;
}

// System runtime registry. Generates the `SystemAsset` value enum that holds a
// constructed system and dispatches `init` / `step`.
//
// Every system is internal: it has no declarable asset, is never parsed from a
// world or written to a blob, and is constructed by
// `World::build_internal_systems` from world content. Each entry maps a variant
// name to the behavior type that implements `System`; the variant name doubles
// as the system's stable display name (`name()`) for profiling and logging.
#[macro_export]
macro_rules! define_system_assets {
    ( $( $variant:ident => $behavior:path ),* $(,)? ) => {
        // Variant sizes follow the behavior types; boxing them would only move
        // the per-system state behind a pointer for no real gain here.
        #[allow(clippy::large_enum_variant)]
        #[derive(Debug)]
        pub enum SystemAsset {
            $( $variant($behavior), )*
        }

        impl SystemAsset {
            // Stable display name used for profiling and logging. Every variant
            // name is the system's canonical name.
            pub fn name(&self) -> &'static str {
                match self {
                    $( SystemAsset::$variant(_) => stringify!($variant), )*
                }
            }

            pub fn init(&mut self, ctx: &mut PipelineContext) {
                match self {
                    $( SystemAsset::$variant(s) => s.init(ctx), )*
                }
            }

            pub fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
                match self {
                    $( SystemAsset::$variant(s) => s.step(ctx), )*
                }
            }
        }

        $( impl From<$behavior> for SystemAsset { fn from(s: $behavior) -> Self { SystemAsset::$variant(s) } } )*
    };
}

pub struct World {
    components: ComponentStorage,
    systems: Vec<SystemAsset>,
    blob: BlobData,
    profile: FrameProfile,
    // Type-keyed engine singletons (e.g. the per-frame FrameInput snapshot
    // GraphicsSystem publishes) and the event queues.
    resources: Resources,
    // Set once `build_internal_systems` has run, so a second `start()` on the
    // same world does not append the internal systems twice.
    internal_systems_built: bool,
}

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("World")
            .field("components", &self.components.len())
            .field("systems", &self.systems.len())
            .finish()
    }
}

impl World {
    pub fn new(blob: BlobData) -> Self {
        Self {
            components: ComponentStorage::default(),
            systems: Vec::new(),
            blob,
            profile: FrameProfile::default(),
            resources: Resources::new(),
            internal_systems_built: false,
        }
    }

    // Convenience constructor for contexts that have no blob data
    // (e.g. unit tests, or worlds built entirely from runtime-only assets).
    pub fn new_empty() -> Self {
        Self::new(BlobData::empty())
    }

    // Add a component loaded from a blob def. Systems are not added this way:
    // they are internal and constructed by `build_internal_systems`.
    pub fn add(&mut self, component: ComponentAsset) {
        self.components.push(component);
    }

    #[allow(dead_code)]
    pub fn add_component<C: Into<ComponentAsset>>(&mut self, c: C) {
        self.components.push(c.into());
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty() && self.systems.is_empty()
    }

    // Whether this world drives the renderer. True when it declares a
    // `GraphicsConfig` (pre-`start`) or has a constructed `GraphicsSystem`
    // (post-`start`, after the config component has been drained), so callers
    // can decide on the render loop / Metal activation regardless of timing.
    // Used only on the macOS NSApp-activation path in `app::run` (and in the
    // tests below), so it has no caller on other platforms in a non-test build.
    // Genuinely platform-conditional (unlike the dyn-dispatch dead-code blind
    // spots in the DX backend), so gate the allow on the same condition.
    #[cfg_attr(
        not(target_os = "macos"),
        allow(
            dead_code,
            reason = "used only on the macOS render-activation path in app::run, plus tests"
        )
    )]
    pub fn renders(&self) -> bool {
        self.query::<crate::assets::GraphicsConfig>()
            .next()
            .is_some()
            || self
                .systems
                .iter()
                .any(|s| matches!(s, SystemAsset::GraphicsSystem(_)))
    }

    #[allow(dead_code)]
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    #[allow(dead_code)]
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    // Iterate every stored component of a given type. Mirrors
    // `PipelineContext::query`; useful in tests that hold a `World` directly.
    #[allow(dead_code)]
    pub fn query<C: ComponentSlot>(&self) -> std::slice::Iter<'_, C> {
        C::slot(&self.components).iter()
    }

    // Mutable iteration over all components of type C. Mirror of
    // `PipelineContext::query_mut` for code holding a `World` directly rather
    // than a per-system `PipelineContext`, namely the `DebugHook::tick`
    // drive, which applies hot-reload skeleton-shape changes to the ECS-owned
    // `SkeletonPose` components from outside the system step.
    #[allow(dead_code)]
    pub fn query_mut<C: ComponentSlot>(&mut self) -> std::slice::IterMut<'_, C> {
        self.components.values_mut::<C>().iter_mut()
    }

    // Push a runtime-produced component into the matching typed slot. Mirror
    // of `PipelineContext::push`; used by the `DebugHook::tick` drive to insert
    // `Prop`s added by a world.jsonl hot-reload so subsequent systems see them.
    #[allow(dead_code)]
    pub fn push<C: ComponentSlot>(&mut self, c: C) {
        self.components.push_typed(c);
    }

    // Read-only join over two component types, for code holding a `World`
    // directly (the decomposition round-trip tests). Mirror of
    // `PipelineContext::join2`.
    #[allow(dead_code)]
    pub fn join2<A: ComponentSlot, B: ComponentSlot>(
        &self,
    ) -> impl Iterator<Item = (Entity, &A, &B)> {
        self.components.join2::<A, B>()
    }

    // Borrow the event queue for event type E, if any have been sent. Mirror of
    // `PipelineContext::events`, for code holding a `World` directly (tests).
    #[allow(dead_code)]
    pub fn events<E: 'static>(&self) -> Option<&Events<E>> {
        self.resources.get::<Events<E>>()
    }

    // Mutably borrow (creating if absent) the event queue for event type E.
    // Mirror of `PipelineContext::events_mut`, for code holding a `World`
    // directly: tests, and the editor's debug-driven command injection.
    #[allow(dead_code)]
    pub fn events_mut<E: 'static>(&mut self) -> &mut Events<E> {
        if !self.resources.contains::<Events<E>>() {
            self.resources.insert(Events::<E>::new());
        }
        self.resources
            .get_mut::<Events<E>>()
            .expect("Events<E> was just inserted")
    }

    #[allow(dead_code)]
    pub fn systems(&self) -> &[SystemAsset] {
        &self.systems
    }

    // Despawn an entity (all its components, recycling its id). Stands in for the
    // GraphicsSystem-mediated despawn in system tests that need an entity gone
    // before a later system step (e.g. physics-body reaping).
    #[cfg(test)]
    pub fn despawn(&mut self, entity: Entity) {
        self.components.despawn(entity);
    }

    // Seed a singleton resource that persists across steps. Stands in for the
    // GraphicsSystem-published resources (e.g. `MenuActive`) in system tests
    // that drive a later system directly without a GraphicsSystem in the world.
    #[cfg(test)]
    pub fn insert_resource<T: std::any::Any>(&mut self, value: T) {
        self.resources.insert(value);
    }

    // Mutable view of the active systems. Mirror of `systems()`; lets the
    // `DebugHook::tick` drive match out a `&mut GraphicsSystem` /
    // `&mut AnimationSystem` (the same enum-match `systems()` already serves
    // read-only) to drive hot-reload from outside the per-system step.
    #[allow(dead_code)]
    pub fn systems_mut(&mut self) -> &mut [SystemAsset] {
        &mut self.systems
    }

    // Re-serialize the world's components as blob defs. Systems are internal
    // and never serialized, so they do not appear here.
    #[allow(dead_code)]
    pub fn all_defs(&self) -> Vec<BlobAssetDef> {
        self.components.all_defs()
    }

    pub fn start(&mut self) -> Result<(), CnResult> {
        self.build_internal_systems();
        let mut ctx = PipelineContext {
            components: &mut self.components,
            blob: &mut self.blob,
            profile: &mut self.profile,
            resources: &mut self.resources,
        };
        // Give each loaded Prop's entity its per-instance components before
        // systems init. The Prop components remain; the decomposed ones ride
        // alongside until a consumer switches over.
        decompose::run(&mut ctx);
        for system in &mut self.systems {
            system.init(&mut ctx);
        }
        Ok(())
    }

    // Construct the internal systems implied by the world's content, in their
    // fixed run order, just before `init`. Internal systems are not declarable
    // assets: each is present only when its gating components are, and is built
    // from them. Runs at most once per world (guarded by
    // `internal_systems_built`) so a system whose gating components survive
    // `init` is not built twice.
    //
    // `SCHEDULE` is the single source of run order. The order encodes the
    // cross-system constraints:
    //   * GraphicsSystem first: it deposits `FrameInput`, publishes
    //     `SkeletonPose`, and makes payloads resident: everything below
    //     consumes these.
    //   * StatHud *queries* (does not drain) `FrameInput`, so it precedes the
    //     `FrameInput` drainers (Camera3DSystem / UiInputSystem).
    //   * PhysicsSystem before Camera3DSystem: physics consumes the camera's
    //     previous-frame `desired_move` (a one-frame-lagged resolution).
    //   * Camera3DSystem before AudioSystem: the audio listener reads the camera.
    fn build_internal_systems(&mut self) {
        if self.internal_systems_built {
            return;
        }
        self.internal_systems_built = true;

        // Each builder gates on world content and returns its system when
        // present. The array order is the run order.
        const SCHEDULE: &[fn(&World) -> Option<SystemAsset>] = &[
            World::build_graphics,
            World::build_stat_hud,
            World::build_debug_hud,
            World::build_physics,
            World::build_camera,
            World::build_fps_counter,
            World::build_animation,
            World::build_audio,
            World::build_ui_input,
        ];
        for build in SCHEDULE {
            if let Some(system) = build(&*self) {
                self.systems.push(system);
            }
        }
    }

    // GraphicsSystem: present whenever the world declares a `GraphicsConfig`
    // (the render marker).
    fn build_graphics(&self) -> Option<SystemAsset> {
        self.query::<crate::assets::GraphicsConfig>()
            .next()
            .map(|_| crate::gfx::graphics_system::GraphicsSystem::new().into())
    }

    // StatHud: present whenever the world declares a `StatHud`; built from that
    // component (the HUD's TextLabel refs).
    fn build_stat_hud(&self) -> Option<SystemAsset> {
        self.query::<crate::assets::StatHud>()
            .next()
            .cloned()
            .map(|cfg| crate::hud::stat_hud::StatHudSystem::new(cfg).into())
    }

    // PhysicsSystem: present whenever the world has physics content, namely a
    // `PhysicsConfig` (optional floor / terrain tuning), a `RigidBody` (character
    // capsule), or a `PropBody` (dynamic prop). Reads the `PhysicsConfig` if
    // present, otherwise a flat-floor default.
    fn build_physics(&self) -> Option<SystemAsset> {
        let needs = self
            .query::<crate::assets::PhysicsConfig>()
            .next()
            .is_some()
            || self.query::<crate::assets::RigidBody>().next().is_some()
            || self.query::<crate::assets::PropBody>().next().is_some();
        if !needs {
            return None;
        }
        let config = self
            .query::<crate::assets::PhysicsConfig>()
            .next()
            .cloned()
            .unwrap_or_default();
        Some(crate::physics::system::PhysicsSystem::new(config).into())
    }

    // Camera3DSystem: present whenever a `Camera3D` has a `controller` (the
    // default; `null` opts out for cutscene cameras). Built from the first
    // controlled camera's settings.
    fn build_camera(&self) -> Option<SystemAsset> {
        self.query::<crate::assets::Camera3D>()
            .find_map(|c| c.controller.clone())
            .map(|ctrl| crate::gfx::camera_controller::Camera3DSystem::new(ctrl).into())
    }

    // FpsCounter: present whenever the world declares an `FpsCounter`; built from
    // that component (its optional TextLabel ref).
    fn build_fps_counter(&self) -> Option<SystemAsset> {
        self.query::<crate::assets::FpsCounter>()
            .next()
            .cloned()
            .map(|cfg| crate::hud::fps_counter::FpsCounterSystem::new(cfg).into())
    }

    // DebugHud: present whenever the world declares a `DebugHud`; built from that
    // component (its developer-readout TextLabel refs).
    fn build_debug_hud(&self) -> Option<SystemAsset> {
        self.query::<crate::assets::DebugHud>()
            .next()
            .cloned()
            .map(|cfg| crate::hud::debug_hud::DebugHudSystem::new(cfg).into())
    }

    // AnimationSystem: present whenever the world declares any `Animation`. It
    // drains every `Animation` at init and writes `SkeletonPose` each frame.
    fn build_animation(&self) -> Option<SystemAsset> {
        self.query::<crate::assets::Animation>()
            .next()
            .map(|_| crate::gfx::animation::AnimationSystem::new().into())
    }

    // AudioSystem: present whenever the world declares any `AudioEmitter`.
    // Building it opens an audio device, so a world with no emitters stays silent
    // and device-free.
    fn build_audio(&self) -> Option<SystemAsset> {
        self.query::<crate::assets::AudioEmitter>()
            .next()
            .map(|_| crate::audio::system::AudioSystem::new().into())
    }

    // UiInputSystem: present whenever the world declares any `HitRegion`, `View`,
    // or `KeyBinding`. It drains all three at init.
    fn build_ui_input(&self) -> Option<SystemAsset> {
        let needs = self.query::<crate::assets::HitRegion>().next().is_some()
            || self.query::<crate::assets::View>().next().is_some()
            || self.query::<crate::assets::KeyBinding>().next().is_some();
        needs.then(|| crate::ui::UiInputSystem::new().into())
    }

    // Per-frame profiling data: system CPU timings and render-backend stats
    // from the most recently completed frame.
    #[allow(dead_code)]
    pub fn profile(&self) -> &FrameProfile {
        &self.profile
    }

    // Advance one event queue a frame so its two-frame retention holds for
    // readers that run after the writer.
    fn update_event_queue<E: 'static>(&mut self) {
        if let Some(events) = self.resources.get_mut::<Events<E>>() {
            events.update();
        }
    }

    // Advance every migrated event queue once per frame, before systems run.
    // Each migrated event type is listed here explicitly.
    fn update_events(&mut self) {
        self.update_event_queue::<crate::assets::SceneCommand>();
        self.update_event_queue::<crate::assets::ViewCommand>();
        self.update_event_queue::<crate::assets::SettingCommand>();
        self.update_event_queue::<crate::assets::ControlsCommand>();
        self.update_event_queue::<crate::assets::AudioCommand>();
        self.update_event_queue::<crate::assets::DespawnRequest>();
        self.update_event_queue::<crate::assets::ReparentRequest>();
        self.update_event_queue::<crate::assets::SpawnRequest>();
    }

    // Tick -- systems run in order, Done systems are removed.
    // Returns Done when no systems remain, Stop on hard halt.
    pub fn step(&mut self) -> StepResult {
        // Rotate the profiler's system-timing buffers so the frame that just
        // finished becomes the readable snapshot for this frame's readers.
        self.profile.begin_frame();
        self.update_events();
        let mut ctx = PipelineContext {
            components: &mut self.components,
            blob: &mut self.blob,
            profile: &mut self.profile,
            resources: &mut self.resources,
        };
        let mut i = 0;
        while i < self.systems.len() {
            let name = self.systems[i].name();
            let started = std::time::Instant::now();
            let result = self.systems[i].step(&mut ctx);
            let micros = started.elapsed().as_micros().min(u32::MAX as u128) as u32;
            ctx.profile.record_system(name, micros);
            match result {
                StepResult::Stop => return StepResult::Stop,
                StepResult::Done => {
                    let removed = self.systems.remove(i);
                    tracing::debug!("System '{}' finished", removed.name());
                }
                StepResult::Continue => {
                    i += 1;
                }
            }
        }
        if self.systems.is_empty() {
            StepResult::Done
        } else {
            StepResult::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A GraphicsConfig marks a rendering world. `renders()` reports it before
    // `start()` (while the component is present), the pre-start signal callers
    // use to choose the render loop. (The post-start GraphicsSystem path can't
    // be unit-tested here: its `init` builds the GPU backend.)
    #[test]
    fn graphics_config_makes_world_render() {
        let mut world = World::new_empty();
        assert!(!world.renders());
        world.add_component(crate::assets::GraphicsConfig::default());
        assert!(world.renders());
    }
}
