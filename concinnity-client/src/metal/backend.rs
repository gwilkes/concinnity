// src/metal/backend.rs
//
// RenderBackend impl for MtlContext. Thin forwarders to the inherent
// methods scattered across metal/{context,resources,streaming,draw}.rs.
// Method resolution picks the inherent over the trait method when both
// have the same name, so `self.draw_frame(...)` calls the inherent here.

use crate::gfx::backend::{QualitySettings, RenderBackend};
use crate::gfx::input::RenderInput;
use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
use crate::gfx::profile::RenderStats;
use crate::gfx::render_types::{
    MaterialUniforms, PostProcessParams, SkinnedDrawObject, TextDrawCall,
};

use super::context::{MtlContext, debug_assert_main_thread};

// Generate `RenderBackend` methods that forward 1:1 to the inherent
// `MtlContext` method of the same name: each entry is the trait signature, and
// the generated body is `self.<name>(<args…>)`. Inherent-over-trait method
// resolution makes that call bind the inherent method (not this trait one), so
// there is no recursion. Methods that diverge from a straight forward (a
// receiver mismatch, dropped/renamed args, or a custom body) are written out
// by hand below the invocation.
//
// The `&mut self` arms prepend `debug_assert_main_thread` so every mutation
// entry point reached through the boxed `RenderBackend` proves the
// main-thread invariant the `unsafe impl Send for MtlContext` rests on:
// loud in debug, free in release. The `&self` arms stay unguarded: read-only
// access is the in-order parallel fan-out's whole point and is allowed off
// the main thread. `draw_frame` is hand-written below (it needs the
// `MainThreadMarker` as a value and self-checks), so it is not listed here.
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

impl RenderBackend for MtlContext {
    forward! {
        fn capture_cursor(&mut self);
        fn set_ui_cursor_hidden(&mut self, hidden: bool);
        fn set_menu_mode(&mut self, on: bool);
        fn set_camera_capture(&mut self, capture: bool);
        fn set_reflection_probes(&mut self, probes: &[crate::gfx::reflection_probe::ProbePlacement]);
        fn set_vsync(&mut self, on: bool);
        fn set_window_mode(&mut self, mode: crate::assets::WindowMode);
        fn set_window_size(&mut self, width: u32, height: u32);
        fn update_post_process(&mut self, params: PostProcessParams);
        fn set_ambient_intensity(&mut self, value: f32);
        fn set_keymap(&mut self, keymap: &crate::gfx::keymap::KeyMap);
        fn apply_quality_settings(&mut self, settings: QualitySettings);
        fn set_shadow_update(&mut self, update: crate::assets::ShadowUpdate);
        fn update_quality_params(&mut self, settings: QualitySettings);
        fn take_input(&mut self) -> RenderInput;
        fn wait_idle(&self);
        fn update_view(&mut self, matrix: [[f32; 4]; 4]);
        fn update_model(&mut self, index: usize, model: [[f32; 4]; 4]);
        fn retire_draw_object(&mut self, draw_idx: usize);
        fn upload_skinned(&mut self, vertices: &[SkinnedVertex], indices: &[u16], draw_objects: Vec<SkinnedDrawObject>, vert_bytes: &[u8], frag_bytes: &[u8], shadow_bytes: &[u8]) -> Result<(), String>;
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
        fn add_chunk_mesh(&mut self, verts: &[Vertex], idxs: &[u16], model: [[f32; 4]; 4], texture_slot: usize, normal_map_slot: usize, material: MaterialUniforms, frame: u64) -> Result<usize, String>;
        fn remove_chunk_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String>;
        fn set_chunk_model(&mut self, draw_idx: usize, model: [[f32; 4]; 4]) -> Result<(), String>;
        fn capabilities(&self) -> crate::gfx::backend::DeviceCapabilities;
        fn gpu_profile(&self) -> crate::gfx::backend::GpuProfile;
        fn logical_size(&self) -> (f32, f32);
        fn render_stats(&self) -> RenderStats;
        fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String>;
        fn update_environment_map(&mut self, payload: &[u8]) -> Result<(), String>;
        fn update_fog_settings(&mut self, settings: Option<crate::gfx::volumetric_fog::FogSettings>);
        fn update_mesh_geometry(&mut self, draw_idx: usize, verts: &[crate::gfx::mesh_payload::Vertex], idxs: &[u16], lod_alternates: &[(f32, Vec<u16>)]) -> Result<(), String>;
        fn rebuild_static_geometry(&mut self, changes: Vec<crate::gfx::backend::DrawGeometryUpdate>) -> Result<(), String>;
        fn update_skinned_mesh_geometry(&mut self, skinned_index: usize, vertex_base: u16, verts: &[crate::gfx::mesh_payload::SkinnedVertex], idxs: &[u16]) -> Result<(), String>;
        fn rebuild_skinned_geometry(&mut self, changes: Vec<crate::gfx::backend::SkinnedDrawGeometryUpdate>) -> Result<Vec<crate::gfx::backend::SkinnedSlotLayout>, String>;
        fn update_skinned_skeleton(&mut self, skinned_index: usize, new_joint_count: usize) -> Result<(), String>;
        fn clone_static_draw_object(&mut self, src_draw_idx: usize, model: [[f32; 4]; 4]) -> Result<usize, String>;
        fn set_draw_material(&mut self, draw_idx: usize, material: MaterialUniforms, texture_slot: usize, normal_map_slot: usize);
        fn set_draw_cull_distance(&mut self, draw_idx: usize, cull_distance: f32);
        fn add_decal(&mut self, record: crate::gfx::decal::DecalRecord) -> Result<usize, String>;
        fn remove_decal(&mut self, decal_id: usize) -> Result<(), String>;
        fn add_emitter(&mut self, record: crate::gfx::particles::ParticleEmitterRecord) -> Result<usize, String>;
        fn remove_emitter(&mut self, emitter_id: usize) -> Result<(), String>;
        fn update_world_shader_pipelines(&mut self, vert_bytes: Option<&[u8]>, frag_bytes: Option<&[u8]>, shadow_bytes: Option<&[u8]>, vert_instanced_bytes: Option<&[u8]>) -> Result<(), String>;
    }

    // Methods that are NOT a 1:1 forward; written out by hand.

    fn draw_frame(
        &mut self,
        elapsed: f32,
        fov_y_radians: f32,
        near: f32,
        far: f32,
        cam_pos: [f32; 3],
        text_calls: &[TextDrawCall],
    ) -> Result<(), String> {
        // Not in the guarded `forward!` block: draw_frame needs the
        // MainThreadMarker as a *value* (it threads it into NSEvent pumping and
        // window ops), so it proves the invariant itself and returns Err off
        // the main thread rather than asserting: no point double-checking.
        self.draw_frame(elapsed, fov_y_radians, near, far, cam_pos, text_calls)
    }

    fn window_closed(&mut self) -> bool {
        // Metal's inherent method is &self; the trait takes &mut self for
        // parity with DX/VK.
        MtlContext::window_closed(self)
    }

    // Inherent method is named `capture_screenshot` to keep the forwarder
    // unambiguous (an inherent `screenshot` would shadow the trait method and
    // recurse). Mirrors the DX/VK backends.
    fn screenshot(&mut self, path: &str) -> Result<String, String> {
        debug_assert_main_thread("screenshot");
        self.capture_screenshot(path)
    }

    fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
        _texture_slot: usize,
        _normal_map_slot: usize,
    ) -> Result<(), String> {
        debug_assert_main_thread("setup_chunk_streaming");
        // Metal binds chunk textures per draw, so the slot args are unused.
        self.setup_chunk_streaming(chunk_vtx_bytes, chunk_idx_bytes)
    }

    fn shader_reload_flag(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        self.shader_reload_pending
            .as_ref()
            .map(std::sync::Arc::clone)
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
