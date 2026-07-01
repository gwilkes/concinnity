// src/hud/mod.rs
//
// On-screen HUD overlays. Internal systems (not declarable assets): each is
// constructed by `World::start` when the world declares its matching request
// component (`FpsCounter` / `StatHud` / `DebugHud`), and writes live stats into
// the referenced `TextLabel`s.

pub(crate) mod debug_hud;
pub(crate) mod fps_counter;
pub(crate) mod stat_hud;
