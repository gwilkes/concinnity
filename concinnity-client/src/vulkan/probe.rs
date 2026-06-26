// src/vulkan/probe.rs
//
// Scene-captured reflection probes on Vulkan. Each declared `ReflectionProbe`
// (or an auto-seeded grid when a world declares none) describes a cube to bake
// DISTINCT from `env_map`: the specular reflection term box-projects against the
// probe's influence box and samples its cube, so glossy surfaces reflect the
// actual surrounding geometry instead of the imported HDR sky, while the skybox +
// diffuse irradiance keep sampling `env_map` so the visible sky is never replaced.
//
// The cube math + the staggered-bake state machine are backend-agnostic
// (`crate::gfx::reflection_probe`); this module drives the placement intake + the
// GPU capture, mirroring `crate::directx::probe` / `crate::metal::probe`.
//
// `set_reflection_probes` converts the graphics-system placements (auto-seeding a
// grid from the scene bounds when a world declares none) into the stored placement
// list + an EMPTY `ProbeSet`, then enqueues them. `bake_pending_probes` (driven each
// frame from `draw_frame`) advances the shared `next_bake_action` transition table:
// it renders one cube face per frame into a bake-owned target on a per-face fence,
// reads the six faces back, runs the GGX prefilter convolution on a worker thread,
// and installs the prefiltered cube into the forward / SSR / RT cube array -- all
// without blocking the render loop (the sky reflection covers a probe until its
// cube installs). The forward / SSR / RT sampling lives in the main / resolve
// shaders (see the reflection_probes.md DX/VK port checklist).

use ash::vk;

use super::context::{HDR_FORMAT, VkContext};
use super::descriptor_layout::PROBE_CUBE_ARRAY_BINDING;
use super::draw::ViewUniforms;
use super::hiz::CullHizParams;
use super::probe_uniforms::{MAX_PROBES, ProbeSet, ProbeUniforms};
use super::resources::alloc_descriptor_sets;
use super::texture::{
    GpuImage, create_buffer, create_image, create_image_view, upload_probe_prefilter_cube,
};
use crate::gfx::frustum::Frustum;
use crate::gfx::reflection_probe::{self, BakeAction, BakePhase, ProbePlacement};

// Captured cube-face resolution (mip 0 of the prefilter chain). Matches the
// `EnvironmentMap` asset default + the DirectX / Metal `PROBE_FACE_SIZE`.
const PROBE_FACE_SIZE: u32 = 512;
// Irradiance cube resolution (diffuse is low frequency, so this stays small).
const PROBE_IRRADIANCE_FACE: u32 = 16;
// GGX prefilter samples per output texel (a runtime bake uses far fewer than the
// importer's 1024; the convolution is rayon-parallel).
const PROBE_PREFILTER_SAMPLES: u32 = 128;
// Firefly clamp during the prefilter convolution (matches the asset default).
const PROBE_PREFILTER_CLAMP: f32 = 12.0;
// Cube faces per probe.
const PROBE_FACE_COUNT: usize = 6;
// Depth format of the probe-face target (matches the main pass's DSV).
const PROBE_DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
// Near / far for the 90-degree probe-face projection. A fixed wide range keeps
// the capture independent of the live camera; the cube is sampled by direction,
// so the exact far plane only affects depth precision during the bake.
const PROBE_NEAR: f32 = 0.05;
const PROBE_FAR: f32 = 2000.0;

impl VkContext {
    // Set the reflection-probe placements (declared `ReflectionProbe` assets,
    // converted to `ProbePlacement`s by the graphics system). An empty list
    // auto-seeds a grid from the scene bounds, so existing scenes still get local
    // reflections without authoring. Capped at `MAX_PROBES`. Pushed once after
    // construction; the cube capture that fills the probe set runs across later
    // frames (next slice).
    pub(super) fn set_reflection_probes(&mut self, declared: &[ProbePlacement]) {
        let mut placements: Vec<ProbePlacement> = if declared.is_empty() {
            match self.scene_world_bounds() {
                Some((mn, mx)) => {
                    // Object AABBs as occupancy so a probe is not auto-captured from
                    // inside a wall; skip degenerate (non-finite) boxes.
                    let occupancy: Vec<([f32; 3], [f32; 3])> = self
                        .draw_objects
                        .iter()
                        .map(|o| (o.bb_min, o.bb_max))
                        .filter(|(mn, mx)| mn.iter().chain(mx).all(|c| c.is_finite()))
                        .collect();
                    reflection_probe::auto_seed_probes(mn, mx, &occupancy)
                }
                None => Vec::new(),
            }
        } else {
            declared.to_vec()
        };
        if placements.len() > MAX_PROBES {
            tracing::warn!(
                "reflection probes: {} placements, capping at MAX_PROBES={}",
                placements.len(),
                MAX_PROBES
            );
            placements.truncate(MAX_PROBES);
        }
        // A re-placement (rare -- this is normally a one-time init call) abandons any
        // in-flight staggered bake and frees the previously baked cubes. Idle first
        // when a capture is in flight (its targets may still be on the GPU) or cubes
        // exist (the forward shader may sample them), reset every cube-array slot back
        // to the sky so none dangles, then drop the in-flight bake + the cubes. The
        // common first call has an empty queue + `probe_maps`, so it skips all of this.
        if self.probe_rendering.is_some() || !self.probe_maps.is_empty() {
            self.wait_idle();
        }
        let device = self.device.clone();
        if let Some(rendering) = self.probe_rendering.take() {
            rendering.destroy(&device, self.commands.command_pool);
        }
        self.probe_converting = None;
        if !self.probe_maps.is_empty() {
            self.reset_probe_cube_slots_to_sky();
            for cube in self.probe_maps.drain(..) {
                cube.destroy(&device);
            }
        }
        self.probe_placements = placements;
        self.probe_set = ProbeSet::EMPTY;
        // Enqueue the placements; `bake_pending_probes` (driven each frame from
        // `draw_frame`) renders + installs them staggered across later frames, so the
        // construction call no longer blocks on the capture.
        self.probe_bake_queue = reflection_probe::ProbeBakeQueue::new(self.probe_placements.len());
    }

    // World-space bounds over every static draw object, skipping degenerate
    // (non-finite) AABBs. `None` for an empty scene. Mirrors
    // `directx/probe.rs::scene_world_bounds`.
    pub(super) fn scene_world_bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        reflection_probe::fold_world_bounds(self.draw_objects.iter().map(|o| (o.bb_min, o.bb_max)))
    }

    // Point every probe-cube-array slot (binding 8) of every frame's global set
    // back at the sky prefilter cube. The init path leaves them this way; this
    // restores it before a re-placement drops the old baked cubes, so no slot
    // dangles a freed view (Vulkan requires every descriptor in a bound set be
    // valid, even slots the shader's `i < count` loop never samples).
    fn reset_probe_cube_slots_to_sky(&self) {
        let sky: Vec<vk::DescriptorImageInfo> = (0..MAX_PROBES)
            .map(|_| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(self.env_map.prefilter.view)
                    .sampler(self.cube_sampler)
            })
            .collect();
        for &set in &self.descriptors.global_sets {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(PROBE_CUBE_ARRAY_BINDING)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&sky);
            unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        }
    }

    // Advance the staggered asynchronous reflection-probe bake by one step. Called
    // every frame from `draw_frame` after this frame's slot fence wait; cheap once the
    // queue drains. Drives the shared `next_bake_action` transition table over two
    // pipelined slots (one Rendering, one Converting), so the capture spreads across
    // frames instead of blocking construction. Non-fatal: a failure abandons the
    // remaining bakes, keeping what already installed. Mirrors `directx::probe`.
    //
    // V1 simplifications (documented; shared with DirectX / Metal):
    //   * Static + streamed-chunk geometry only -- instanced + skinned draws are left
    //     disabled in the bake cull buffers (the kernel skips them). They still
    //     RECEIVE probe reflections.
    //   * Cold lighting -- shadows may be unpopulated on the first frames, exactly
    //     like the DX / Metal first-frame bake.
    pub(super) fn bake_pending_probes(&mut self) -> Result<(), String> {
        // Nothing queued and nothing in flight: cheap early-out once the bake drains.
        if !self.probe_bake_queue.pending()
            && self.probe_rendering.is_none()
            && self.probe_converting.is_none()
        {
            return Ok(());
        }
        // Permanent ineligibility: a probe only improves on a real captured
        // environment, and the capture renders through the bindless GPU cull. These
        // never become true after init, so abandon the queue rather than re-checking
        // forever (the forward specular keeps sampling the sky).
        if self.env_map.prefilter_mip_count <= 1
            || self.cull.cull_pipeline.is_none()
            || self.cull.bindless_pipeline.is_none()
        {
            if self.probe_rendering.is_some() {
                self.wait_idle();
            }
            let device = self.device.clone();
            if let Some(rendering) = self.probe_rendering.take() {
                rendering.destroy(&device, self.commands.command_pool);
            }
            self.probe_converting = None;
            self.probe_bake_queue.abort();
            return Ok(());
        }

        // Converting slot first: install the convolved cube once the worker finishes,
        // freeing the slot so the rendering slot can read its finished capture back
        // this same frame (keeps installs in queue order -> `probe_maps` aligned with
        // the placement list).
        let converting_occupied = self.probe_converting.is_some();
        let payload_ready = self
            .probe_converting
            .as_ref()
            .is_some_and(|c| c.payload.get().is_some());
        let install = reflection_probe::next_bake_action(
            if converting_occupied {
                BakePhase::Converting
            } else {
                BakePhase::Idle
            },
            false,
            payload_ready,
            false,
            false,
            false,
        ) == BakeAction::Install;
        if install && let Err(e) = self.probe_install() {
            self.fail_bake(e);
            return Ok(());
        }
        let converting_free = !converting_occupied || install;

        // Rendering slot: submit one face per frame; once all six retired on the GPU
        // (the last face's fence signalled) AND the converting slot is free, read the
        // faces back and hand them to the worker, or start the next placement.
        let rendering_occupied = self.probe_rendering.is_some();
        let more_faces = self
            .probe_rendering
            .as_ref()
            .is_some_and(|r| r.cursor < PROBE_FACE_COUNT);
        let done = self.probe_rendering.as_ref().is_some_and(|r| {
            r.cursor >= PROBE_FACE_COUNT
                && unsafe { self.device.get_fence_status(r.face_fences[r.last_fence()]) }
                    .unwrap_or(false)
        });
        // Transient ineligibility: geometry may still be streaming. A zero cull keeps
        // the queue cursor so a later frame retries rather than baking an empty cube.
        let eligible = self.cull_count() > 0;
        match reflection_probe::next_bake_action(
            if rendering_occupied {
                BakePhase::Rendering
            } else {
                BakePhase::Idle
            },
            done && converting_free,
            false,
            self.probe_bake_queue.pending(),
            eligible,
            more_faces,
        ) {
            BakeAction::RenderFace => {
                if let Err(e) = self.probe_render_next_face() {
                    self.fail_bake(e);
                }
            }
            BakeAction::Readback => {
                if let Err(e) = self.probe_readback_and_convolve() {
                    self.fail_bake(e);
                }
            }
            BakeAction::StartNext => {
                if let Err(e) = self.probe_start_next() {
                    self.fail_bake(e);
                }
            }
            BakeAction::Install | BakeAction::Idle => {}
        }
        Ok(())
    }

    // Abandon the rest of the bake after an unrecoverable error, keeping the cubes
    // already installed. The queue cursor advanced when the current probe started, so
    // aborting (cursor -> end) keeps `probe_maps` aligned with the placement list.
    fn fail_bake(&mut self, e: String) {
        tracing::warn!(
            "reflection probe bake failed, keeping {} baked: {e}",
            self.probe_maps.len()
        );
        // Idle before dropping the in-flight capture's GPU resources: its command
        // buffers may still be executing. A bake failure is rare (allocation / device
        // error), so the one-time stall is acceptable.
        if self.probe_rendering.is_some() {
            self.wait_idle();
        }
        let device = self.device.clone();
        if let Some(rendering) = self.probe_rendering.take() {
            rendering.destroy(&device, self.commands.command_pool);
        }
        self.probe_converting = None;
        self.probe_bake_queue.abort();
    }

    // Begin baking the next pending placement: build the bake-owned capture resources
    // (target + cull ring + per-face view UBOs + readback buffers) and fill the cull
    // buffers + the six per-face view uniforms ONCE (frustum-independent; each face
    // re-runs only the cull with its own frustum). No face is submitted here; the six
    // follow one per frame via `probe_render_next_face`.
    fn probe_start_next(&mut self) -> Result<(), String> {
        let Some(index) = self.probe_bake_queue.take_next() else {
            return Ok(());
        };
        let placement = self.probe_placements[index];
        let eye = placement.position;
        let bake = BakeResources::new(self)?;

        // Bake-owned cull buffers, zeroed first so the untouched instance tail reads
        // as disabled (a probe omits instanced geometry in V1), then filled with this
        // probe's static + chunk + skinned records (LOD by probe eye).
        let object_size =
            self.cull_count() * std::mem::size_of::<crate::gfx::render_types::GpuObjectData>();
        let args_size =
            self.cull_count() * std::mem::size_of::<crate::gfx::render_types::GpuDrawArgs>();
        unsafe {
            std::ptr::write_bytes(bake.object_ptr, 0, object_size);
            std::ptr::write_bytes(bake.draw_args_ptr, 0, args_size);
        }
        self.build_object_records_into(bake.object_ptr);
        self.build_draw_args_records_into(bake.draw_args_ptr, eye);

        // Per-face view uniforms (the only per-face binding), all six filled once.
        // reflections_enabled stays 0: no resolve runs over a probe face, so the bake
        // captures the full forward probe specular -- here the sky, since the bake
        // binds an EMPTY ProbeSet.
        let prefilter_mip_count = self.env_map.prefilter_mip_count as f32;
        for face in 0..PROBE_FACE_COUNT {
            let vp = reflection_probe::face_view_projection(eye, face, PROBE_NEAR, PROBE_FAR);
            let view_mat = reflection_probe::face_view_matrix(eye, face);
            let view = ViewUniforms {
                vp,
                view_mat,
                elapsed: 0.0,
                reflections_enabled: 0.0,
                cam_x: eye[0],
                cam_y: eye[1],
                cam_z: eye[2],
                prefilter_mip_count,
                _ep0: 0.0,
                _ep1: 0.0,
            };
            // SAFETY: each view UBO was sized for one ViewUniforms.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &view as *const ViewUniforms as *const u8,
                    bake.view_ptrs[face],
                    std::mem::size_of::<ViewUniforms>(),
                );
            }
        }

        self.probe_rendering = Some(RenderingBake {
            index,
            placement,
            eye,
            cursor: 0,
            bake,
            face_cmds: Vec::with_capacity(PROBE_FACE_COUNT),
            face_fences: Vec::with_capacity(PROBE_FACE_COUNT),
        });
        Ok(())
    }

    // Submit one cube face of the in-flight probe: a fresh command buffer that culls
    // for this face's frustum, draws the bindless main into the bake target, and
    // copies the resolved face into its readback buffer, on a per-face fence (polled,
    // never waited). The command buffer + fence are held in the `RenderingBake` until
    // readback, so the last face's fence retiring means the whole capture is done. One
    // face per frame spreads the capture so no frame pays the whole cost.
    fn probe_render_next_face(&mut self) -> Result<(), String> {
        let device = self.device.clone();
        let extent = vk::Extent2D {
            width: PROBE_FACE_SIZE,
            height: PROBE_FACE_SIZE,
        };
        // Copy the bake handles out (all Copy) so no borrow of `self.probe_rendering`
        // is held across the `&self` encode calls below.
        let (
            face,
            eye,
            cull_set,
            hiz_set,
            framebuffer,
            global_set,
            bindless_set,
            indirect,
            copy_src,
            readback,
        ) = {
            let r = self
                .probe_rendering
                .as_ref()
                .ok_or("probe: render face with no bake in flight")?;
            let b = &r.bake;
            (
                r.cursor,
                r.eye,
                b.cull_set,
                b.hiz_set,
                b.framebuffer,
                b.global_sets[r.cursor],
                b.bindless_set,
                b.indirect_buf,
                b.copy_source(),
                b.readback_bufs[r.cursor],
            )
        };

        // A fresh command buffer + fence for this face, from the one-shot pool.
        // Register both in the `RenderingBake` the instant they exist so a later
        // record / submit error still reclaims them via `fail_bake` ->
        // `RenderingBake::destroy` (which `wait_idle`s first); on the success path the
        // last-pushed fence is `face_fences[last_fence()]` after `cursor` advances.
        let cmd = {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.commands.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            unsafe { device.allocate_command_buffers(&info) }
                .map_err(|e| format!("probe face cmd alloc: {e}"))?[0]
        };
        let fence = match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(f) => f,
            Err(e) => {
                // The command buffer is allocated but not yet tracked; free it before
                // bailing so it does not leak.
                unsafe {
                    device.free_command_buffers(
                        self.commands.command_pool,
                        std::slice::from_ref(&cmd),
                    );
                }
                return Err(format!("probe face fence: {e}"));
            }
        };
        {
            let r = self
                .probe_rendering
                .as_mut()
                .ok_or("probe: render face slot vanished")?;
            r.face_cmds.push(cmd);
            r.face_fences.push(fence);
        }

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(cmd, &begin) }
            .map_err(|e| format!("probe face begin: {e}"))?;
        // Order the previous face's readback copy + indirect-draw read (a prior
        // frame's submit) before this face's cull (rewrites the shared indirect
        // buffer) and resolve (rewrites the shared colour). Intra-queue, so the
        // queue's submission order preserves it across the separate submits.
        if face > 0 {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::INDIRECT_COMMAND_READ,
                )
                .dst_access_mask(
                    vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                );
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::DRAW_INDIRECT,
                    vk::PipelineStageFlags::COMPUTE_SHADER
                        | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&barrier),
                    &[],
                    &[],
                );
            }
        }
        let vp = reflection_probe::face_view_projection(eye, face, PROBE_NEAR, PROBE_FAR);
        let frustum = Frustum::from_view_projection(vp);
        self.encode_probe_cull(cmd, cull_set, hiz_set, &frustum, eye);
        self.encode_main_into_face(cmd, framebuffer, extent, global_set, bindless_set, indirect);
        // The face colour rests in SHADER_READ_ONLY_OPTIMAL after the render pass;
        // flip it to TRANSFER_SRC for the readback copy. This exact transition is the
        // one the shared layout-transition table omits.
        let to_src = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(copy_src)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: PROBE_FACE_SIZE,
                height: PROBE_FACE_SIZE,
                depth: 1,
            });
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_src),
            );
            device.cmd_copy_image_to_buffer(
                cmd,
                copy_src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback,
                std::slice::from_ref(&region),
            );
            device
                .end_command_buffer(cmd)
                .map_err(|e| format!("probe face end: {e}"))?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            device
                .queue_submit(self.graphics_queue, std::slice::from_ref(&submit), fence)
                .map_err(|e| format!("probe face submit: {e}"))?;
        }

        // The command buffer + fence are already tracked (registered at allocation);
        // advance the cursor now that this face submitted, so `last_fence()` points at
        // it and `done` polls the right fence.
        let r = self
            .probe_rendering
            .as_mut()
            .ok_or("probe: render face slot vanished")?;
        r.cursor += 1;
        Ok(())
    }

    // The capture finished on the GPU (the last face's fence signalled): map the six
    // readback buffers, decode RGBA16F -> f32, free the capture's GPU resources, and
    // hand the faces to a worker thread that runs the GGX prefilter convolution off
    // the render thread. Moves the bake to the Converting slot.
    fn probe_readback_and_convolve(&mut self) -> Result<(), String> {
        let rendering = self
            .probe_rendering
            .take()
            .ok_or("probe: readback with no bake in flight")?;
        let device = self.device.clone();

        // Decode the six readbacks (tightly packed RGBA16F) to f32.
        let mut faces: [Vec<f32>; PROBE_FACE_COUNT] = std::array::from_fn(|_| Vec::new());
        let face_bytes = (PROBE_FACE_SIZE as u64) * (PROBE_FACE_SIZE as u64) * 8;
        for (slot, &mem) in faces.iter_mut().zip(rendering.bake.readback_mems.iter()) {
            let ptr = unsafe { device.map_memory(mem, 0, face_bytes, vk::MemoryMapFlags::empty()) }
                .map_err(|e| format!("map probe readback: {e}"))?
                as *const u8;
            // SAFETY: the buffer is HOST_COHERENT and `face_bytes` long; the last
            // face's fence is signalled, so on the single graphics queue all six copies
            // completed.
            let raw = unsafe { std::slice::from_raw_parts(ptr, face_bytes as usize) };
            *slot = decode_probe_face_rgba16f(raw, PROBE_FACE_SIZE);
            unsafe { device.unmap_memory(mem) };
        }
        let index = rendering.index;
        let placement = rendering.placement;
        // The capture's GPU resources (target + cull + per-face command buffers +
        // fences + readbacks) free here; the last face's fence signalled, so the GPU
        // is done with all of them.
        rendering.destroy(&device, self.commands.command_pool);

        // Convolve off the render thread: only the decoded CPU floats + the payload
        // slot cross the boundary (no vk handle), so it is Send-safe. A worker panic
        // yields an empty payload, which `probe_install` rejects -> `fail_bake`.
        let payload = std::sync::Arc::new(std::sync::OnceLock::new());
        let slot = std::sync::Arc::clone(&payload);
        std::thread::spawn(move || {
            let bytes = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                reflection_probe::build_probe_payload(
                    &faces,
                    PROBE_FACE_SIZE,
                    PROBE_IRRADIANCE_FACE,
                    PROBE_PREFILTER_SAMPLES,
                    PROBE_PREFILTER_CLAMP,
                )
            }))
            .unwrap_or_else(|_| {
                tracing::error!("reflection probe convolution panicked; abandoning this probe");
                Vec::new()
            });
            let _ = slot.set(bytes);
        });

        self.probe_converting = Some(ConvertingBake {
            index,
            placement,
            payload,
        });
        Ok(())
    }

    // The off-thread convolution finished: deserialise the worker's payload, upload
    // the prefiltered radiance cube, and install it as probe `index` -- point this
    // probe's slot in every frame's cube array at the baked cube and record its
    // parallax box, bumping `probe_set.count` so the forward specular samples it.
    // Leaves `env_map` / the sky untouched. Mirrors `directx/probe.rs::probe_install`.
    fn probe_install(&mut self) -> Result<(), String> {
        let ConvertingBake {
            index,
            placement: p,
            payload,
        } = self
            .probe_converting
            .take()
            .ok_or("probe: install with no bake in flight")?;
        let bytes = payload.get().ok_or("probe: install before payload ready")?;
        let view = crate::build::environment_map::deserialise(bytes)
            .map_err(|e| format!("deserialise probe payload: {e}"))?;
        if view.prefilter_mip_bytes.is_empty() {
            return Err("probe payload has no prefilter mips".into());
        }
        let cube = upload_probe_prefilter_cube(
            &self.instance,
            &self.device,
            self.physical_device,
            self.commands.command_pool,
            self.graphics_queue,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )?;

        // Point this probe's slot in every frame's global set at the baked cube (it
        // held the sky prefilter until now). Safe to rewrite mid-frame-loop: the cube
        // upload's `one_shot_submit` just idled the graphics queue (no in-flight frame
        // is reading the global sets), and the shader's `i < count` loop never reaches
        // slot `index` until the count bump below, so no frame samples it mid-rewrite.
        let img_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(cube.view)
            .sampler(self.cube_sampler);
        for &set in &self.descriptors.global_sets {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(PROBE_CUBE_ARRAY_BINDING)
                .dst_array_element(index as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&img_info));
            unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        }

        // Installs run in queue order, so the cube array stays aligned with the
        // placement list.
        debug_assert_eq!(index, self.probe_maps.len());
        self.probe_maps.push(cube);
        self.probe_set.probes[index] = ProbeUniforms {
            box_min: [p.box_min[0], p.box_min[1], p.box_min[2], 1.0],
            box_max: [p.box_max[0], p.box_max[1], p.box_max[2], 0.0],
            probe_pos: [p.position[0], p.position[1], p.position[2], 0.0],
        };
        self.probe_set.count = self.probe_maps.len() as u32;
        if !self.probe_bake_queue.pending() && self.probe_rendering.is_none() {
            tracing::info!(
                "reflection probes: baked {}/{}",
                self.probe_maps.len(),
                self.probe_placements.len()
            );
        }
        Ok(())
    }

    // Dispatch the compute cull for one probe face (or one planar mirror plane)
    // into the caller's indirect buffer. A thin sibling of `encode_cull`: it binds
    // the given cull set (set 0) and -- when the world runs Hi-Z -- a Hi-Z set
    // (set 1, written with `hiz_enabled = 0` so the frustum-only cull never samples
    // the pyramid; the cull layout statically references set 1, so it must be
    // bound), pushes the face/plane frustum + eye, dispatches one invocation per
    // record, and orders the writes before the indirect draw's read. Shared by the
    // probe bake + the planar reflection's reflected-frustum cull.
    pub(in crate::vulkan) fn encode_probe_cull(
        &self,
        cmd: vk::CommandBuffer,
        cull_set: vk::DescriptorSet,
        hiz_set: Option<vk::DescriptorSet>,
        frustum: &Frustum,
        cam_pos: [f32; 3],
    ) {
        let (Some(pipeline), Some(layout)) =
            (self.cull.cull_pipeline, self.cull.cull_pipeline_layout)
        else {
            return;
        };
        let device = &self.device;
        // The cull push-constant block (six planes + cam_pos + object_count, 112 B
        // std430). Built inline to avoid exposing cull.rs's private CullParams.
        let mut push = [0u8; 112];
        for (i, p) in frustum.planes.iter().enumerate().take(6) {
            let plane = [p.normal[0], p.normal[1], p.normal[2], p.d];
            push[i * 16..i * 16 + 16].copy_from_slice(unsafe {
                std::slice::from_raw_parts(plane.as_ptr() as *const u8, 16)
            });
        }
        push[96..108].copy_from_slice(unsafe {
            std::slice::from_raw_parts(cam_pos.as_ptr() as *const u8, 12)
        });
        push[108..112].copy_from_slice(&(self.cull_count() as u32).to_le_bytes());
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                std::slice::from_ref(&cull_set),
                &[],
            );
            if let Some(hs) = hiz_set {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    1,
                    std::slice::from_ref(&hs),
                    &[],
                );
            }
            device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
            device.cmd_dispatch(cmd, (self.cull_count() as u32).div_ceil(64), 1, 1);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    // Render the bindless static + instance + chunk prefix into a probe face (or a
    // planar mirror plane). A thin sibling of `encode_main_pass`'s bindless branch:
    // begins the render pass (reusing `main_render_pass`, render-pass-compatible
    // with the bindless pipeline), binds the caller's face/plane global set (set 0)
    // + bindless set (set 1), and issues one indirect draw of
    // `[0, skinned_record_base())` from the given indirect buffer. The skinned tail
    // is omitted (V1). Shared by the probe bake + the planar reflection render.
    pub(in crate::vulkan) fn encode_main_into_face(
        &self,
        cmd: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        extent: vk::Extent2D,
        global_set: vk::DescriptorSet,
        bindless_set: vk::DescriptorSet,
        indirect: vk::Buffer,
    ) {
        let (Some(pipeline), Some(layout)) = (
            self.cull.bindless_pipeline,
            self.cull.bindless_pipeline_layout,
        ) else {
            return;
        };
        let device = &self.device;
        let [r, g, b, a] = self.clear_color;
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [r, g, b, a],
            },
        };
        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let clears: &[vk::ClearValue] = if self.msaa_samples != vk::SampleCountFlags::TYPE_1 {
            &[clear_color, clear_depth, vk::ClearValue::default()]
        } else {
            &[clear_color, clear_depth]
        };
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.main_render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D::default().extent(extent))
            .clear_values(clears);
        // Negative-height viewport (Y flip), matching the main pass so the captured
        // faces share the cube convention `face_view_projection` was built against.
        let vp = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_vertex_buffers(
                cmd,
                0,
                std::slice::from_ref(&self.geometry.vertex_buffer),
                &[0],
            );
            device.cmd_bind_index_buffer(cmd, self.geometry.index_buffer, 0, vk::IndexType::UINT32);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[global_set, bindless_set],
                &[],
            );
            device.cmd_draw_indexed_indirect(
                cmd,
                indirect,
                0,
                self.skinned_record_base() as u32,
                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
            device.cmd_end_render_pass(cmd);
        }
    }
}

// One in-flight probe's GPU capture state, held on `VkContext::probe_rendering`
// while its six faces submit one per frame. Reuses one `BakeResources` (built in
// `probe_start_next`, freed in `probe_readback_and_convolve`) across the faces; the
// per-face command buffers + fences accumulate until readback, when the last face's
// fence retiring guarantees the GPU is done with all of them. Mirrors
// `directx::probe::RenderingBake`.
pub(super) struct RenderingBake {
    index: usize,
    placement: ProbePlacement,
    eye: [f32; 3],
    // Next of `PROBE_FACE_COUNT` faces to submit; `more_faces = cursor < FACE_COUNT`.
    cursor: usize,
    bake: BakeResources,
    face_cmds: Vec<vk::CommandBuffer>,
    face_fences: Vec<vk::Fence>,
}

impl RenderingBake {
    // Index of the face whose fence completion means the whole capture retired (the
    // last submitted face; the single graphics queue retires the rest in order).
    fn last_fence(&self) -> usize {
        self.cursor.saturating_sub(1)
    }

    // Free every owned GPU resource: the per-face command buffers (back to the
    // one-shot pool), the per-face fences, and the bake target / cull / sets. The
    // caller has ensured the GPU retired them (the last face's fence is signalled, or
    // the device is idle).
    pub(super) fn destroy(self, device: &ash::Device, command_pool: vk::CommandPool) {
        unsafe {
            if !self.face_cmds.is_empty() {
                device.free_command_buffers(command_pool, &self.face_cmds);
            }
            for &fence in &self.face_fences {
                device.destroy_fence(fence, None);
            }
        }
        self.bake.destroy(device);
    }
}

// The prior probe whose read-back faces are convolving on a worker thread. Holds
// only the worker's payload slot (plain bytes), so it drops freely (no vk handle).
// Mirrors `directx::probe::ConvertingBake`.
pub(super) struct ConvertingBake {
    index: usize,
    placement: ProbePlacement,
    payload: std::sync::Arc<std::sync::OnceLock<Vec<u8>>>,
}

// The GPU resources for ONE reflection-probe bake: the 512x512 colour/depth
// (/resolve) target + framebuffer, a bake-owned cull ring + its descriptor sets,
// six per-face global sets carrying the face view + snapshot lighting, and six
// readback buffers. One per in-flight probe (held in `RenderingBake`); `destroy`
// frees it after the faces read back.
struct BakeResources {
    color: GpuImage,
    depth: GpuImage,
    resolve: Option<GpuImage>,
    framebuffer: vk::Framebuffer,
    object_buf: vk::Buffer,
    object_mem: vk::DeviceMemory,
    object_ptr: *mut u8,
    draw_args_buf: vk::Buffer,
    draw_args_mem: vk::DeviceMemory,
    draw_args_ptr: *mut u8,
    indirect_buf: vk::Buffer,
    indirect_mem: vk::DeviceMemory,
    status_buf: vk::Buffer,
    status_mem: vk::DeviceMemory,
    pool: vk::DescriptorPool,
    cull_set: vk::DescriptorSet,
    bindless_set: vk::DescriptorSet,
    hiz_set: Option<vk::DescriptorSet>,
    hiz_ubo: Option<(vk::Buffer, vk::DeviceMemory, *mut u8)>,
    global_sets: Vec<vk::DescriptorSet>,
    view_bufs: Vec<vk::Buffer>,
    view_mems: Vec<vk::DeviceMemory>,
    view_ptrs: Vec<*mut u8>,
    light: (vk::Buffer, vk::DeviceMemory, *mut u8),
    shadow: (vk::Buffer, vk::DeviceMemory, *mut u8),
    probeset: (vk::Buffer, vk::DeviceMemory, *mut u8),
    readback_bufs: Vec<vk::Buffer>,
    readback_mems: Vec<vk::DeviceMemory>,
}

impl BakeResources {
    // The image the readback copy reads: the single-sample resolve when MSAA is on,
    // else the (single-sample) colour attachment. Both rest in SHADER_READ_ONLY
    // after the render pass.
    fn copy_source(&self) -> vk::Image {
        match &self.resolve {
            Some(r) => r.image,
            None => self.color.image,
        }
    }

    fn new(ctx: &VkContext) -> Result<BakeResources, String> {
        use crate::gfx::render_types::{GpuDrawArgs, GpuObjectData, LightUniforms, ShadowUniforms};
        let device = &ctx.device;
        let instance = &ctx.instance;
        let pd = ctx.physical_device;
        let msaa = ctx.msaa_samples != vk::SampleCountFlags::TYPE_1;
        let size = PROBE_FACE_SIZE;

        // Colour + depth (+ single-sample resolve when MSAA), then a framebuffer
        // compatible with `main_render_pass`.
        let (color_img, color_mem) = create_image(
            instance,
            device,
            pd,
            size,
            size,
            HDR_FORMAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ctx.msaa_samples,
        )?;
        let color_view =
            create_image_view(device, color_img, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
        let color = GpuImage {
            image: color_img,
            memory: color_mem,
            view: color_view,
            aux_views: Vec::new(),
        };
        let (depth_img, depth_mem) = create_image(
            instance,
            device,
            pd,
            size,
            size,
            PROBE_DEPTH_FORMAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ctx.msaa_samples,
        )?;
        let depth_view = create_image_view(
            device,
            depth_img,
            PROBE_DEPTH_FORMAT,
            vk::ImageAspectFlags::DEPTH,
        )?;
        let depth = GpuImage {
            image: depth_img,
            memory: depth_mem,
            view: depth_view,
            aux_views: Vec::new(),
        };
        let resolve = if msaa {
            let (img, mem) = create_image(
                instance,
                device,
                pd,
                size,
                size,
                HDR_FORMAT,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::SampleCountFlags::TYPE_1,
            )?;
            let view = create_image_view(device, img, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
            Some(GpuImage {
                image: img,
                memory: mem,
                view,
                aux_views: Vec::new(),
            })
        } else {
            None
        };
        let fb_attachments: Vec<vk::ImageView> = if msaa {
            vec![color.view, depth.view, resolve.as_ref().unwrap().view]
        } else {
            vec![color.view, depth.view]
        };
        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(ctx.main_render_pass)
            .attachments(&fb_attachments)
            .width(size)
            .height(size)
            .layers(1);
        let framebuffer = unsafe { device.create_framebuffer(&fb_info, None) }
            .map_err(|e| format!("probe framebuffer: {e}"))?;

        // Bake-owned cull ring, sized like the per-frame rings.
        let n = ctx.cull_count();
        let object_size = (n * std::mem::size_of::<GpuObjectData>()) as u64;
        let args_size = (n * std::mem::size_of::<GpuDrawArgs>()) as u64;
        let indirect_size = (n * std::mem::size_of::<vk::DrawIndexedIndirectCommand>()) as u64;
        let status_size = (n * std::mem::size_of::<u32>()) as u64;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let (object_buf, object_mem) = create_buffer(
            instance,
            device,
            pd,
            object_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            host,
        )?;
        let object_ptr = map(device, object_mem, object_size)?;
        let (draw_args_buf, draw_args_mem) = create_buffer(
            instance,
            device,
            pd,
            args_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            host,
        )?;
        let draw_args_ptr = map(device, draw_args_mem, args_size)?;
        let (indirect_buf, indirect_mem) = create_buffer(
            instance,
            device,
            pd,
            indirect_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let (status_buf, status_mem) = create_buffer(
            instance,
            device,
            pd,
            status_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Snapshot lighting (so all faces share one set), an EMPTY ProbeSet (count 0
        // so a probe face reflects only the sky), and six per-face view UBOs.
        let light = make_ubo_bytes(
            instance,
            device,
            pd,
            light_bytes(&ctx.uniforms.light_uniforms),
        )?;
        let shadow = make_ubo_bytes(instance, device, pd, shadow_bytes(&ctx.shadow.uniforms))?;
        let probeset = make_ubo_bytes(instance, device, pd, probeset_bytes(&ProbeSet::EMPTY))?;
        let view_size = std::mem::size_of::<ViewUniforms>() as u64;
        let mut view_bufs = Vec::with_capacity(PROBE_FACE_COUNT);
        let mut view_mems = Vec::with_capacity(PROBE_FACE_COUNT);
        let mut view_ptrs = Vec::with_capacity(PROBE_FACE_COUNT);
        for _ in 0..PROBE_FACE_COUNT {
            let (buf, mem) = create_buffer(
                instance,
                device,
                pd,
                view_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                host,
            )?;
            let ptr = map(device, mem, view_size)?;
            view_bufs.push(buf);
            view_mems.push(mem);
            view_ptrs.push(ptr);
        }

        // A bake Hi-Z set (cull set 1) only when the world runs Hi-Z; written with
        // hiz_enabled = 0 so the pyramid is never sampled. The UBO is kept so it can
        // be freed in `destroy`.
        let mut hiz_ubo: Option<(vk::Buffer, vk::DeviceMemory, *mut u8)> = None;

        // One dedicated descriptor pool for the bake's cull + bindless + global +
        // Hi-Z sets.
        let tex_pool = (ctx.textures.len() + ctx.normal_map_textures.len()) as u32;
        let has_hiz = ctx.cull.hiz.is_some();
        let uniform_count = PROBE_FACE_COUNT as u32 * 4 + u32::from(has_hiz);
        let storage_count = 4 + 1;
        let sampler_count =
            tex_pool + PROBE_FACE_COUNT as u32 * (4 + MAX_PROBES as u32) + u32::from(has_hiz);
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(uniform_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(storage_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(sampler_count.max(1)),
        ];
        let max_sets = 2 + PROBE_FACE_COUNT as u32 + u32::from(has_hiz);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets);
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| format!("probe descriptor pool: {e}"))?;

        // Cull set (set 0): object / draw-args / indirect / status SSBOs.
        let cull_set = alloc_descriptor_sets(
            device,
            pool,
            std::slice::from_ref(&ctx.cull.cull_set_layout.unwrap()),
        )?[0];
        write_storage(device, cull_set, 0, object_buf, object_size);
        write_storage(device, cull_set, 1, draw_args_buf, args_size);
        write_storage(device, cull_set, 2, indirect_buf, indirect_size);
        write_storage(device, cull_set, 3, status_buf, status_size);

        // Bindless set (set 1): object SSBO + the shared texture pool array.
        let bindless_set = alloc_descriptor_sets(
            device,
            pool,
            std::slice::from_ref(&ctx.cull.bindless_set_layout.unwrap()),
        )?[0];
        let pool_infos: Vec<vk::DescriptorImageInfo> = ctx
            .textures
            .iter()
            .chain(ctx.normal_map_textures.iter())
            .map(|img| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(img.view)
                    .sampler(ctx.linear_sampler)
            })
            .collect();
        {
            let obj_info = vk::DescriptorBufferInfo::default()
                .buffer(object_buf)
                .offset(0)
                .range(object_size);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(bindless_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&obj_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(bindless_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&pool_infos),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        // Bake Hi-Z set (cull set 1), hiz_enabled = 0.
        let hiz_set = if let Some(hiz) = ctx.cull.hiz.as_ref() {
            let params = CullHizParams {
                prev_view_proj: [[0.0; 4]; 4],
                hiz_size: [1.0, 1.0],
                hiz_mip_count: 1,
                hiz_enabled: 0,
            };
            let ubo = make_ubo_bytes(instance, device, pd, hiz_params_bytes(&params))?;
            let (view, sampler) = hiz.read_set_sources();
            let layout = hiz.read_set_layout;
            let set = alloc_descriptor_sets(device, pool, std::slice::from_ref(&layout))?[0];
            let img = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view)
                .sampler(sampler);
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(ubo.0)
                .offset(0)
                .range(std::mem::size_of::<CullHizParams>() as u64);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&img)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&ubo_info)),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            hiz_ubo = Some(ubo);
            Some(set)
        } else {
            None
        };

        // Six per-face global sets (set 0 of the bindless main pass): the face view
        // + shared snapshot lighting + env cubes + the SSAO white fallback + an
        // EMPTY ProbeSet + the sky-filled probe cube array. Mirrors init.rs.
        let layouts: Vec<_> = (0..PROBE_FACE_COUNT)
            .map(|_| ctx.descriptors.global_set_layout)
            .collect();
        let global_sets = alloc_descriptor_sets(device, pool, &layouts)?;
        let probe_cube_sky: Vec<vk::DescriptorImageInfo> = (0..MAX_PROBES)
            .map(|_| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(ctx.env_map.prefilter.view)
                    .sampler(ctx.cube_sampler)
            })
            .collect();
        for (face, &set) in global_sets.iter().enumerate() {
            let view_info = buf_info(view_bufs[face], view_size);
            let light_info = buf_info(light.0, std::mem::size_of::<LightUniforms>() as u64);
            let shadow_info = buf_info(shadow.0, std::mem::size_of::<ShadowUniforms>() as u64);
            let probeset_info = buf_info(probeset.0, std::mem::size_of::<ProbeSet>() as u64);
            let shadow_img = img_info(ctx.shadow.map.view, ctx.shadow.sampler);
            let irr_img = img_info(ctx.env_map.irradiance.view, ctx.cube_sampler);
            let pre_img = img_info(ctx.env_map.prefilter.view, ctx.cube_sampler);
            let ssao_img = img_info(ctx.ssao_white.view, ctx.linear_sampler);
            let writes = [
                ubo_write(set, 0, &view_info),
                ubo_write(set, 1, &light_info),
                ubo_write(set, 2, &shadow_info),
                sampler_write(set, 3, &shadow_img),
                sampler_write(set, 4, &irr_img),
                sampler_write(set, 5, &pre_img),
                sampler_write(set, 6, &ssao_img),
                ubo_write(set, 7, &probeset_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(PROBE_CUBE_ARRAY_BINDING)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&probe_cube_sky),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        // Six readback buffers (one RGBA16F face each, tightly packed).
        let readback_size = (size as u64) * (size as u64) * 8;
        let mut readback_bufs = Vec::with_capacity(PROBE_FACE_COUNT);
        let mut readback_mems = Vec::with_capacity(PROBE_FACE_COUNT);
        for _ in 0..PROBE_FACE_COUNT {
            let (buf, mem) = create_buffer(
                instance,
                device,
                pd,
                readback_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                host,
            )?;
            readback_bufs.push(buf);
            readback_mems.push(mem);
        }

        Ok(BakeResources {
            color,
            depth,
            resolve,
            framebuffer,
            object_buf,
            object_mem,
            object_ptr,
            draw_args_buf,
            draw_args_mem,
            draw_args_ptr,
            indirect_buf,
            indirect_mem,
            status_buf,
            status_mem,
            pool,
            cull_set,
            bindless_set,
            hiz_set,
            hiz_ubo,
            global_sets,
            view_bufs,
            view_mems,
            view_ptrs,
            light,
            shadow,
            probeset,
            readback_bufs,
            readback_mems,
        })
    }

    fn destroy(self, device: &ash::Device) {
        let mut bufs: Vec<vk::Buffer> = vec![
            self.object_buf,
            self.draw_args_buf,
            self.indirect_buf,
            self.status_buf,
            self.light.0,
            self.shadow.0,
            self.probeset.0,
        ];
        bufs.extend(self.readback_bufs.iter().copied());
        bufs.extend(self.view_bufs.iter().copied());
        let mut mems: Vec<vk::DeviceMemory> = vec![
            self.object_mem,
            self.draw_args_mem,
            self.indirect_mem,
            self.status_mem,
            self.light.1,
            self.shadow.1,
            self.probeset.1,
        ];
        mems.extend(self.readback_mems.iter().copied());
        mems.extend(self.view_mems.iter().copied());
        if let Some((buf, mem, _)) = self.hiz_ubo {
            bufs.push(buf);
            mems.push(mem);
        }
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
            self.color.destroy(device);
            self.depth.destroy(device);
            if let Some(r) = &self.resolve {
                r.destroy(device);
            }
            for buf in bufs {
                device.destroy_buffer(buf, None);
            }
            for mem in mems {
                device.free_memory(mem, None);
            }
            // The pool frees every set allocated from it.
            device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

// Map a freshly created HOST_VISIBLE buffer's whole range.
fn map(device: &ash::Device, mem: vk::DeviceMemory, size: u64) -> Result<*mut u8, String> {
    unsafe { device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()) }
        .map(|p| p as *mut u8)
        .map_err(|e| format!("map bake buffer: {e}"))
}

// Create a HOST_VISIBLE uniform buffer holding `bytes`, persistently mapped.
fn make_ubo_bytes(
    instance: &ash::Instance,
    device: &ash::Device,
    pd: vk::PhysicalDevice,
    bytes: &[u8],
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let (buf, mem) = create_buffer(
        instance,
        device,
        pd,
        bytes.len() as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        host,
    )?;
    let ptr = map(device, mem, bytes.len() as u64)?;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    Ok((buf, mem, ptr))
}

fn light_bytes(u: &crate::gfx::render_types::LightUniforms) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            u as *const _ as *const u8,
            std::mem::size_of::<crate::gfx::render_types::LightUniforms>(),
        )
    }
}

fn shadow_bytes(u: &crate::gfx::render_types::ShadowUniforms) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            u as *const _ as *const u8,
            std::mem::size_of::<crate::gfx::render_types::ShadowUniforms>(),
        )
    }
}

fn probeset_bytes(p: &ProbeSet) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<ProbeSet>())
    }
}

fn hiz_params_bytes(p: &CullHizParams) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            p as *const _ as *const u8,
            std::mem::size_of::<CullHizParams>(),
        )
    }
}

fn buf_info(buffer: vk::Buffer, range: u64) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(0)
        .range(range)
}

fn img_info(view: vk::ImageView, sampler: vk::Sampler) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(view)
        .sampler(sampler)
}

fn ubo_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(info))
}

fn sampler_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorImageInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(info))
}

fn write_storage(
    device: &ash::Device,
    set: vk::DescriptorSet,
    binding: u32,
    buffer: vk::Buffer,
    range: u64,
) {
    let info = buf_info(buffer, range);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

// Decode a tightly-packed `R16G16B16A16_SFLOAT` probe-cube face (the format the
// bake renders + copies back into a host buffer) to linear f32 RGBA, row major.
// The bake's `cmd_copy_image_to_buffer` uses `buffer_row_length(0)`, so the
// readback is tightly packed (8 bytes per texel, no row padding) -- unlike
// DirectX, whose `CopyTextureRegion` footprint is 256-byte-row-aligned and needs
// an explicit unpad. The six decoded faces feed
// `reflection_probe::build_probe_payload`, which wants each as
// `face_size * face_size` RGBA f32 in row-major order. Mirrors the decode half of
// `directx/probe.rs::read_face_rgba_f32`.
#[allow(dead_code)] // consumed by the probe capture-pass readback (next slice).
fn decode_probe_face_rgba16f(raw: &[u8], face_size: u32) -> Vec<f32> {
    let texels = (face_size as usize) * (face_size as usize);
    let mut out = vec![0.0f32; texels * 4];
    for (texel, px) in raw.chunks_exact(8).take(texels).enumerate() {
        let half = |o: usize| super::screenshot::f16_to_f32(u16::from_le_bytes([px[o], px[o + 1]]));
        let base = texel * 4;
        out[base] = half(0);
        out[base + 1] = half(2);
        out[base + 2] = half(4);
        out[base + 3] = half(6);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // The readback decode unpacks tightly-packed RGBA16F (4 halfs per texel, no
    // row padding) into row-major f32, the layout `build_probe_payload` consumes.
    #[test]
    fn decode_probe_face_unpacks_tightly_packed_rgba16f() {
        // A 2x2 face = 4 texels. Each texel is four little-endian halfs.
        let texels: [[u16; 4]; 4] = [
            [0x3c00, 0x3800, 0x0000, 0x3c00], // (1.0, 0.5, 0.0, 1.0)
            [0x4000, 0xbc00, 0x0000, 0x3c00], // (2.0, -1.0, 0.0, 1.0)
            [0x0000, 0x0000, 0x0000, 0x0000], // (0.0, 0.0, 0.0, 0.0)
            [0x3800, 0x3800, 0x3800, 0x3c00], // (0.5, 0.5, 0.5, 1.0)
        ];
        let mut raw = Vec::new();
        for t in texels {
            for h in t {
                raw.extend_from_slice(&h.to_le_bytes());
            }
        }
        let out = decode_probe_face_rgba16f(&raw, 2);
        assert_eq!(out.len(), 16, "2x2 face decodes to 4 RGBA texels");
        assert_eq!(&out[0..4], &[1.0, 0.5, 0.0, 1.0]);
        assert_eq!(&out[4..8], &[2.0, -1.0, 0.0, 1.0]);
        assert_eq!(&out[8..12], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&out[12..16], &[0.5, 0.5, 0.5, 1.0]);
    }
}
