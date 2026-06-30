// src/vulkan/backend.rs
//
// RenderBackend impl for VkContext. Thin forwarders to the inherent
// methods scattered across vulkan/{context,resources}.rs.
//
// Most forwarders are a mechanical 1:1 call into the inherent method, so a
// `forward!` token-muncher generates them. The `&mut self` arms prepend
// `debug_assert_main_thread` so every mutation reached through the boxed trait
// object proves the main-thread invariant the `unsafe impl Send for VkContext`
// rests on. Forwarders that rename, drop args, or have a custom body stay
// hand-written below the macro. Mirrors src/directx/backend.rs.

use crate::gfx::backend::RenderBackend;
use crate::gfx::input::RenderInput;
use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
use crate::gfx::profile::RenderStats;
use crate::gfx::render_types::{MaterialUniforms, SkinnedDrawObject, TextDrawCall};

use super::context::{VkContext, debug_assert_main_thread};

// Generate a `RenderBackend` method that forwards to the inherent method of the
// same name. Inherent methods shadow trait methods in resolution, so
// `self.$name(...)` calls the inherent one (no recursion). The `&mut self` arms
// assert the main-thread invariant first; the `&self` arms are read-only and
// skip it. Recurses token-by-token over the method list.
macro_rules! forward {
    () => {};
    (fn $name:ident(&self $(, $arg:ident: $ty:ty)* $(,)?) -> $ret:ty; $($rest:tt)*) => {
        fn $name(&self $(, $arg: $ty)*) -> $ret { self.$name($($arg),*) }
        forward!($($rest)*);
    };
    (fn $name:ident(&self $(, $arg:ident: $ty:ty)* $(,)?); $($rest:tt)*) => {
        fn $name(&self $(, $arg: $ty)*) { self.$name($($arg),*) }
        forward!($($rest)*);
    };
    (fn $name:ident(&mut self $(, $arg:ident: $ty:ty)* $(,)?) -> $ret:ty; $($rest:tt)*) => {
        fn $name(&mut self $(, $arg: $ty)*) -> $ret {
            debug_assert_main_thread(stringify!($name));
            self.$name($($arg),*)
        }
        forward!($($rest)*);
    };
    (fn $name:ident(&mut self $(, $arg:ident: $ty:ty)* $(,)?); $($rest:tt)*) => {
        fn $name(&mut self $(, $arg: $ty)*) {
            debug_assert_main_thread(stringify!($name));
            self.$name($($arg),*)
        }
        forward!($($rest)*);
    };
}

impl RenderBackend for VkContext {
    forward! {
        fn window_closed(&mut self) -> bool;
        fn capture_cursor(&mut self);
        fn set_ui_cursor_hidden(&mut self, hidden: bool);
        fn set_menu_mode(&mut self, on: bool);
        fn set_camera_capture(&mut self, capture: bool);
        fn set_vsync(&mut self, on: bool);
        fn set_window_mode(&mut self, mode: crate::assets::WindowMode);
        fn set_window_size(&mut self, width: u32, height: u32);
        fn update_post_process(&mut self, params: crate::gfx::render_types::PostProcessParams);
        fn set_ambient_intensity(&mut self, value: f32);
        fn set_keymap(&mut self, keymap: &crate::gfx::keymap::KeyMap);
        fn set_reflection_probes(&mut self, probes: &[crate::gfx::reflection_probe::ProbePlacement]);
        fn apply_quality_settings(&mut self, settings: crate::gfx::backend::QualitySettings);
        fn set_shadow_update(&mut self, update: crate::assets::ShadowUpdate);
        fn set_shadow_distance(&mut self, distance: u32);
        fn set_shadow_cascades(&mut self, count: u32);
        fn update_quality_params(&mut self, settings: crate::gfx::backend::QualitySettings);
        fn take_input(&mut self) -> RenderInput;
        fn wait_idle(&self);
        fn draw_frame(&mut self, elapsed: f32, fov_y_radians: f32, near: f32, far: f32, cam_pos: [f32; 3], text_calls: &[TextDrawCall], world_hidden: bool) -> Result<(), String>;
        fn update_view(&mut self, matrix: [[f32; 4]; 4]);
        fn update_model(&mut self, index: usize, model: [[f32; 4]; 4]);
        fn retire_draw_object(&mut self, draw_idx: usize);
        fn update_skinned_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]);
        fn seed_skinned_instance_pool(&mut self, reservations: Vec<(usize, usize)>);
        fn spawn_skinned_instance(&mut self, template_skinned_index: usize, model: [[f32; 4]; 4]) -> Option<usize>;
        fn retire_skinned_draw_object(&mut self, skinned_index: usize);
        fn update_skinned_model(&mut self, skinned_index: usize, model: [[f32; 4]; 4]);
        fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String>;
        fn update_texture_slot(&mut self, slot: usize, w: u32, h: u32, px: &[u8]) -> Result<(), String>;
        fn evict_normal_map_slot(&mut self, slot: usize) -> Result<(), String>;
        fn update_normal_map_slot(&mut self, slot: usize, w: u32, h: u32, px: &[u8]) -> Result<(), String>;
        fn evict_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String>;
        fn upload_mesh(&mut self, draw_idx: usize, verts: &[Vertex], idxs: &[u16], frame: u64) -> Result<(), String>;
        fn seed_mesh_streaming(&mut self, vtx_offset: u64, vtx_bytes: u64, idx_offset: u64, idx_bytes: u64);
        fn setup_chunk_streaming(&mut self, chunk_vtx_bytes: usize, chunk_idx_bytes: usize, texture_slot: usize, normal_map_slot: usize) -> Result<(), String>;
        fn add_chunk_mesh(&mut self, verts: &[Vertex], idxs: &[u16], model: [[f32; 4]; 4], texture_slot: usize, normal_map_slot: usize, material: MaterialUniforms, frame: u64) -> Result<usize, String>;
        fn remove_chunk_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String>;
        fn set_chunk_model(&mut self, draw_idx: usize, model: [[f32; 4]; 4]) -> Result<(), String>;
        fn add_decal(&mut self, record: crate::gfx::decal::DecalRecord) -> Result<usize, String>;
        fn remove_decal(&mut self, decal_id: usize) -> Result<(), String>;
        fn render_stats(&self) -> RenderStats;
        fn capabilities(&self) -> crate::gfx::backend::DeviceCapabilities;
        fn gpu_profile(&self) -> crate::gfx::backend::GpuProfile;
        fn logical_size(&self) -> (f32, f32);
        fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String>;
        fn update_environment_map(&mut self, payload: &[u8]) -> Result<(), String>;
        fn update_mesh_geometry(&mut self, draw_idx: usize, verts: &[crate::gfx::mesh_payload::Vertex], idxs: &[u16], lod_alternates: &[(f32, Vec<u16>)]) -> Result<(), String>;
        fn update_world_shader_pipelines(&mut self, vert_bytes: Option<&[u8]>, frag_bytes: Option<&[u8]>, shadow_bytes: Option<&[u8]>, vert_instanced_bytes: Option<&[u8]>) -> Result<(), String>;
        fn update_skinned_mesh_geometry(&mut self, skinned_index: usize, vertex_base: u16, verts: &[crate::gfx::mesh_payload::SkinnedVertex], idxs: &[u16]) -> Result<(), String>;
        fn update_skinned_skeleton(&mut self, skinned_index: usize, new_joint_count: usize) -> Result<(), String>;
        fn rebuild_static_geometry(&mut self, changes: Vec<crate::gfx::backend::DrawGeometryUpdate>) -> Result<(), String>;
        fn rebuild_skinned_geometry(&mut self, changes: Vec<crate::gfx::backend::SkinnedDrawGeometryUpdate>) -> Result<Vec<crate::gfx::backend::SkinnedSlotLayout>, String>;
        fn clone_static_draw_object(&mut self, src_draw_idx: usize, model: [[f32; 4]; 4]) -> Result<usize, String>;
    }

    // Non-1:1 forwarders kept explicit.

    fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u16],
        draw_objects: Vec<SkinnedDrawObject>,
        _vert_bytes: &[u8],
        frag_bytes: &[u8],
        _shadow_bytes: &[u8],
    ) -> Result<(), String> {
        debug_assert_main_thread("upload_skinned");
        // Vulkan compiles the vertex / shadow paths from inline GLSL; only the
        // fragment shader is supplied as a precompiled SPIR-V payload.
        self.upload_skinned(vertices, indices, draw_objects, frag_bytes)
    }

    // Inherent particle methods carry the `_particle_` infix; the trait names
    // do not, so these stay out of the macro to avoid a name mismatch.
    fn add_emitter(
        &mut self,
        record: crate::gfx::particles::ParticleEmitterRecord,
    ) -> Result<usize, String> {
        debug_assert_main_thread("add_emitter");
        self.add_particle_emitter(record)
    }
    fn remove_emitter(&mut self, emitter_id: usize) -> Result<(), String> {
        debug_assert_main_thread("remove_emitter");
        self.remove_particle_emitter(emitter_id)
    }

    fn update_fog_settings(&mut self, settings: Option<crate::gfx::volumetric_fog::FogSettings>) {
        debug_assert_main_thread("update_fog_settings");
        self.apply_fog_settings(settings)
    }

    // Inherent method is named `capture_screenshot` to keep the forwarder
    // unambiguous (an inherent `screenshot` would shadow the trait method and
    // recurse); kept explicit out of the `forward!` macro for that rename.
    fn screenshot(&mut self, path: &str) -> Result<String, String> {
        debug_assert_main_thread("screenshot");
        self.capture_screenshot(path)
    }

    fn shader_reload_flag(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        self.shader_reload_pending()
    }

    fn draw_geometry_size(&self, draw_idx: usize) -> Option<(usize, usize)> {
        self.draw_objects
            .get(draw_idx)
            .map(|o| (o.vertex_count, o.index_count))
    }
    fn draw_lod_index_counts(&self, draw_idx: usize) -> Option<Vec<usize>> {
        self.draw_objects
            .get(draw_idx)
            .map(|o| o.lod_alternates.iter().map(|s| s.index_count).collect())
    }
}
