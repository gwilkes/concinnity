// src/assets/mod.rs
//
// Client-side asset surface: a thin re-export of concinnity-core's asset types
// under the historical `crate::assets::*` paths. All assets are pure data now:
// every system became an internal client system living in its own domain module
// (`gfx::graphics_system`, `gfx::camera_controller`, `gfx::animation`,
// `physics::system`, `audio::system`, `ui`, `hud`), constructed by
// `World::start` from the components a world declares.
pub use concinnity_core::assets::*;

// Submodule paths referenced explicitly elsewhere in the client, e.g.
// `crate::assets::shader_stage::ShaderKind`,
// `crate::assets::audio_clip::audio_clip_blob_indices`,
// `crate::assets::sdf_volume::sdf_volume_blob_indices`. Re-export the modules
// themselves so those paths keep resolving against core.
pub use concinnity_core::assets::{audio_clip, procedural_mesh, sdf_volume, shader_stage};
// `instanced_prop`'s module path is only referenced from a `gfx::draw_list`
// unit test; gate the re-export to test builds so non-test builds don't flag it.
#[cfg(test)]
pub use concinnity_core::assets::instanced_prop;
