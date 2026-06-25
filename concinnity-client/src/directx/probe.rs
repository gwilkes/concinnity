// src/directx/probe.rs
//
// Scene-captured reflection probes on DirectX. Each declared `ReflectionProbe`
// (or an auto-seeded grid when a world declares none) is baked into its own cube,
// DISTINCT from `env_map`: the specular reflection term box-projects against the
// probe's influence box and samples its cube, so glossy surfaces reflect the
// actual surrounding geometry instead of the imported HDR sky, while the skybox +
// diffuse irradiance keep sampling `env_map` so the visible sky is never replaced.
//
// The cube math + the staggered-bake state machine are backend-agnostic
// (`crate::gfx::reflection_probe`); this module drives the GPU capture, mirroring
// `crate::metal::probe`. The bake is STAGGERED + ASYNCHRONOUS across frames so the
// render thread never blocks: one probe is in flight at a time, its six cube faces
// submitted one per frame, then read back and convolved on a worker thread.
//
// DirectX simplification vs Metal: a per-face fence VALUE gives ordered GPU
// completion for free (the queue is FIFO), so there is no completion handler / atomic
// -- a face is done when `frame_sync.fence` reaches the value signalled after it. The
// bake never calls `wait_idle` (that would reintroduce a multi-hundred-ms freeze);
// readback is deferred until the fence reaches the last face's value.
//
// Each probe passes through three phases (`gfx::reflection_probe::BakePhase`, driven by
// the pure `next_bake_action` transition table called once per pipeline slot per frame):
//   * Rendering   -- six cube faces submitted to the GPU (one per frame) into a RESERVED
//                    ring slot (`bake_ring_slot`) the frame never overwrites.
//   * Converting  -- the six faces are read back (`CopyTextureRegion` into READBACK
//                    buffers, fence-gated) and the GGX prefilter convolution runs on a
//                    WORKER THREAD.
//   * (install)   -- the convolved cube is uploaded into `probe_maps` + `probe_set`.
//
// Known V1 simplifications (documented intentionally; mirror Metal where noted):
//   * Static + instanced geometry only -- skinned meshes are not captured into the
//     probe (no per-bake deformed buffer yet). They still receive probe reflections.
//   * Single bounce + cold-first-frame lighting (the shadow map may be unpopulated when
//     a probe bakes on an early frame), exactly like Metal.
#![allow(clippy::incompatible_msrv)]

use std::sync::Arc;
use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::context::{DxContext, FRAMES};
use super::screenshot::f16_to_f32;
use super::texture::{
    HDR_FORMAT, create_buffer, create_hdr_color_target, create_hdr_resolve_target,
    transition_barrier, upload_probe_prefilter_cube,
};
use crate::gfx::reflection_probe::{self, BakeAction, BakePhase};

// Captured cube-face resolution (mip 0 of the prefilter chain). Matches the
// `EnvironmentMap` asset default + Metal's `PROBE_FACE_SIZE`.
const PROBE_FACE_SIZE: u32 = 512;
// Irradiance cube resolution (diffuse is low frequency, so this stays small).
const PROBE_IRRADIANCE_FACE: u32 = 16;
// GGX prefilter samples per output texel (a runtime bake uses far fewer than the
// importer's 1024; the convolution is rayon-parallel).
const PROBE_PREFILTER_SAMPLES: u32 = 128;
// Firefly clamp during the prefilter convolution (matches the asset default).
const PROBE_PREFILTER_CLAMP: f32 = 12.0;
// Cube faces per probe, rendered one per frame.
const PROBE_FACE_COUNT: usize = 6;

// A baked prefilter cube, one per installed probe. Distinct from `env_map`;
// sampled only by the specular reflection term. The SRV into the probe cube array
// is written when the array is bound to the shaders.
pub(in crate::directx) struct ProbeCube {
    #[allow(dead_code)] // bound to the forward shader (next slice)
    pub(in crate::directx) prefilter: ID3D12Resource,
    #[allow(dead_code)] // bound to the forward shader (next slice)
    pub(in crate::directx) mip_count: u32,
}

// The GPU resources + state of one in-flight capture. The six faces share one
// (MSAA) colour + depth target reused across frames; each face has its own view
// CBV + readback buffer + command allocator/list (held until the faces are read
// back, so the fence guarantees their GPU work has retired before they drop).
pub(in crate::directx) struct RenderingBake {
    index: usize,
    placement: reflection_probe::ProbePlacement,
    // Next of `PROBE_FACE_COUNT` faces to submit (one per frame).
    cursor: usize,
    eye: [f32; 3],
    near: f32,
    far: f32,
    sample_count: u32,
    // Reused across the six faces.
    color: ID3D12Resource,
    _depth: ID3D12Resource,
    resolve: Option<ID3D12Resource>,
    _rtv_heap: ID3D12DescriptorHeap,
    _dsv_heap: ID3D12DescriptorHeap,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    // Per-face: a 160-byte ViewUniforms CBV (kept mapped) + its GVA, and a READBACK
    // buffer the resolved face is copied into.
    _view_cbvs: Vec<ID3D12Resource>,
    view_gvas: Vec<u64>,
    // Per-capture light + shadow snapshots (so the six faces share one consistent
    // lighting set, decoupled from the frame's per-frame CBV writes).
    light_gva: u64,
    shadow_gva: u64,
    _light_cbv: ID3D12Resource,
    _shadow_cbv: ID3D12Resource,
    readbacks: Vec<ID3D12Resource>,
    readback_layout: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    // One fresh allocator + list per submitted face, held until readback.
    cmd_allocs: Vec<ID3D12CommandAllocator>,
    cmd_lists: Vec<ID3D12GraphicsCommandList>,
    // Fence value signalled after the LAST face; readback waits for the shared
    // `frame_sync.fence` to reach it.
    last_fence_value: u64,
}

// The prior probe whose read-back faces are convolving on a worker thread. Holds
// only the worker's payload slot (plain data), so it drops freely.
pub(in crate::directx) struct ConvertingBake {
    index: usize,
    placement: reflection_probe::ProbePlacement,
    payload: Arc<OnceLock<Vec<u8>>>,
}

impl DxContext {
    // Set the reflection-probe placements (declared `ReflectionProbe` assets,
    // converted to `ProbePlacement`s by the graphics system). An empty list
    // auto-seeds a grid from the scene bounds, so existing scenes still get local
    // reflections without authoring. Resets the staggered bake; capped at
    // `MAX_PROBES`.
    pub(super) fn set_reflection_probes(&mut self, declared: &[reflection_probe::ProbePlacement]) {
        use super::probe_uniforms::{MAX_PROBES, ProbeSet};
        let mut placements: Vec<reflection_probe::ProbePlacement> = if declared.is_empty() {
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
        // A re-placement mid-flight (rare -- this is normally an init-time call) would
        // free capture resources the GPU may still be reading. Idle the GPU first so
        // the dropped command lists + reserved-slot buffers are safe to release. The
        // first call has nothing in flight, so it never idles.
        if self.probe_rendering.is_some() {
            self.wait_idle();
        }
        self.probe_placements = placements;
        self.probe_maps.clear();
        self.probe_set = ProbeSet::EMPTY;
        self.probe_bake_queue = reflection_probe::ProbeBakeQueue::new(self.probe_placements.len());
        self.probe_rendering = None;
        self.probe_converting = None;
    }

    // The reserved transient-ring slot the asynchronous bake builds its bindless
    // buffers into: one past the frame's range `[0, FRAMES)`. The frame never writes
    // this slot, so the bake's CPU-written buffers stay valid across its capture.
    // The cull rings are sized `FRAMES + 1` in `init/pipelines.rs` to make room.
    fn bake_ring_slot(&self) -> usize {
        FRAMES
    }

    // GPU descriptor handle of the reflection-probe cube array table base (root param
    // [10] of the bindless main pass). The MAX_PROBES contiguous cube SRVs start here.
    pub(in crate::directx) fn probe_cube_table_gpu(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let base = unsafe {
            self.descriptors
                .srv_heap
                .GetGPUDescriptorHandleForHeapStart()
        };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr
                + (self.descriptors.probe_cube_base_slot * self.descriptors.srv_descriptor_size)
                    as u64,
        }
    }

    // CPU descriptor handle of probe cube array slot `i` (for writing a baked cube's
    // SRV into the array at install time).
    fn probe_cube_slot_cpu(&self, i: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let base = unsafe {
            self.descriptors
                .srv_heap
                .GetCPUDescriptorHandleForHeapStart()
        };
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr
                + (self.descriptors.probe_cube_base_slot + i)
                    * self.descriptors.srv_descriptor_size,
        }
    }

    // Whether the capture path can run: the bindless GPU-driven cull must be active
    // (the capture renders through the indirect command buffer) and the reserved ring
    // slot must exist.
    fn probe_capture_supported(&self) -> bool {
        self.cull.main_bindless_pso.is_some()
            && self.cull.cull_pso.is_some()
            && self.cull.object_buffer_resources.len() > FRAMES
            && self.cull.draw_args_buffer_resources.len() > FRAMES
            && self.cull.indirect_cmd_buffers.len() > FRAMES
    }

    // Advance the asynchronous reflection-probe bake by one step. Called every frame
    // from `draw_frame` after the frame-slot fence wait; cheap once the queue drains.
    // Drives the pure `next_bake_action` transition table over two pipelined slots.
    // Non-fatal: a failure abandons the remaining bakes, keeping what baked.
    pub(super) fn bake_pending_probes(
        &mut self,
        elapsed: f32,
        near: f32,
        far: f32,
    ) -> Result<(), String> {
        let _ = elapsed;
        if !self.probe_bake_queue.pending()
            && self.probe_rendering.is_none()
            && self.probe_converting.is_none()
        {
            return Ok(());
        }
        // Permanent ineligibility: a probe only improves on a real environment, and
        // the capture renders through the bindless cull. Abandon the queue rather than
        // re-checking forever.
        if self.env_map.prefilter_mip_count <= 1 || !self.probe_capture_supported() {
            self.probe_rendering = None;
            self.probe_converting = None;
            self.probe_bake_queue.abort();
            return Ok(());
        }

        // Converting slot first: install the convolved cube once the worker finishes,
        // freeing the slot so the rendering slot can read its finished capture back
        // this same frame.
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

        // Rendering slot: submit one face per frame; once all six are done on the GPU
        // (the fence reached the last face's value) AND the converting slot is free,
        // read the faces back and hand them to the worker; or start the next placement.
        let rendering_occupied = self.probe_rendering.is_some();
        let more_faces = self
            .probe_rendering
            .as_ref()
            .is_some_and(|r| r.cursor < PROBE_FACE_COUNT);
        let done = self.probe_rendering.as_ref().is_some_and(|r| {
            r.cursor >= PROBE_FACE_COUNT
                && unsafe { self.frame_sync.fence.GetCompletedValue() } >= r.last_fence_value
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
                if let Err(e) = self.probe_start_next(near, far) {
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
        // lists may still be executing. A bake failure is rare (allocation / device
        // error), so the one-time stall is acceptable.
        if self.probe_rendering.is_some() {
            self.wait_idle();
        }
        self.probe_rendering = None;
        self.probe_converting = None;
        self.probe_bake_queue.abort();
    }

    // Begin baking the next pending placement: build the reserved-slot bindless
    // buffers (object + draw-args, frustum-independent) ONCE, and allocate the capture
    // targets + per-face view CBVs + readback buffers. No face is submitted here; the
    // faces follow one per frame via `probe_render_next_face`.
    fn probe_start_next(&mut self, near: f32, far: f32) -> Result<(), String> {
        let Some(index) = self.probe_bake_queue.take_next() else {
            return Ok(());
        };
        let placement = self.probe_placements[index];
        let eye = placement.position;
        let slot = self.bake_ring_slot();

        // Build the reserved-slot bindless buffers once: the per-object record buffer
        // and the draw-args buffer (LOD by distance from the probe eye). Both are
        // frustum-independent, reused by every face's cull.
        self.build_object_buffer(slot);
        self.build_draw_args_buffer(slot, eye);

        let device = &self.device;
        let sample_count = self.hdr.msaa_samples.max(1);
        let size = PROBE_FACE_SIZE;

        // One MSAA (or single-sample) colour + depth pair, reused across the six faces.
        let rtv_heap = create_rtv_heap(device)?;
        let dsv_heap = create_dsv_heap(device)?;
        let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        let dsv = unsafe { dsv_heap.GetCPUDescriptorHandleForHeapStart() };
        let color =
            create_hdr_color_target(device, size, size, sample_count, rtv, self.clear_color)?;
        let depth = create_bake_depth(device, size, sample_count, dsv)?;
        // A single-sample resolve target only when MSAA is on.
        let resolve = if sample_count > 1 {
            Some(create_hdr_resolve_target(device, size, size)?)
        } else {
            None
        };

        // Snapshot the frame's light + shadow uniforms into bake-owned CBVs so all six
        // faces share one temporally-consistent lighting set, and so the capture does
        // not read `light_ubo` / `shadow_ubo[frame]` while `record_frame` (which runs
        // after this) overwrites them on the same frame -- a CPU/GPU race on a mapped
        // buffer. The capture's lighting is the env live when it started.
        let light_bytes = unsafe {
            std::slice::from_raw_parts(
                &self.uniforms.light_uniforms as *const crate::gfx::render_types::LightUniforms
                    as *const u8,
                std::mem::size_of::<crate::gfx::render_types::LightUniforms>(),
            )
        };
        let (light_cbv, light_gva) = make_snapshot_cbv(device, light_bytes)?;
        let shadow_bytes = unsafe {
            std::slice::from_raw_parts(
                &self.shadow.uniforms as *const crate::gfx::render_types::ShadowUniforms
                    as *const u8,
                std::mem::size_of::<crate::gfx::render_types::ShadowUniforms>(),
            )
        };
        let (shadow_cbv, shadow_gva) = make_snapshot_cbv(device, shadow_bytes)?;

        // Per-face ViewUniforms CBVs (the only per-face binding) + readback buffers.
        // The capture renders with the real env IBL (so the scene carries ambient
        // lighting), exactly like the main pass minus the SSR/RT resolve.
        let prefilter_mip_count = self.env_map.prefilter_mip_count as f32;
        let mut view_cbvs = Vec::with_capacity(PROBE_FACE_COUNT);
        let mut view_gvas = Vec::with_capacity(PROBE_FACE_COUNT);
        for face in 0..PROBE_FACE_COUNT {
            let vp = reflection_probe::face_view_projection(eye, face, near, far);
            let view_mat = reflection_probe::face_view_matrix(eye, face);
            let view = super::draw::ViewUniforms {
                vp,
                view_mat,
                elapsed: 0.0,
                // No reflection resolve runs over the probe cube, so the forward
                // probe specular is the only reflection source here; keep it.
                reflections_enabled: 0.0,
                cam_x: eye[0],
                cam_y: eye[1],
                cam_z: eye[2],
                prefilter_mip_count,
                _ep0: 0.0,
                _ep1: 0.0,
            };
            let cbv = create_buffer(
                device,
                256,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            unsafe { cbv.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("probe: map view cbv: {e}"))?;
            // SAFETY: the buffer is 256 bytes; ViewUniforms is 160.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &view as *const super::draw::ViewUniforms as *const u8,
                    ptr as *mut u8,
                    std::mem::size_of::<super::draw::ViewUniforms>(),
                );
            }
            view_gvas.push(unsafe { cbv.GetGPUVirtualAddress() });
            view_cbvs.push(cbv);
        }

        // Readback footprint for one RGBA16Float face (same for all six).
        let face_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: size as u64,
            Height: size,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: HDR_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let mut readback_layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut readback_total: u64 = 0;
        unsafe {
            device.GetCopyableFootprints(
                &face_desc,
                0,
                1,
                0,
                Some(&mut readback_layout),
                None,
                None,
                Some(&mut readback_total),
            );
        }
        let mut readbacks = Vec::with_capacity(PROBE_FACE_COUNT);
        for _ in 0..PROBE_FACE_COUNT {
            readbacks.push(create_buffer(
                device,
                readback_total,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?);
        }

        self.probe_rendering = Some(RenderingBake {
            index,
            placement,
            cursor: 0,
            eye,
            near,
            far,
            sample_count,
            color,
            _depth: depth,
            resolve,
            _rtv_heap: rtv_heap,
            _dsv_heap: dsv_heap,
            rtv,
            dsv,
            _view_cbvs: view_cbvs,
            view_gvas,
            light_gva,
            shadow_gva,
            _light_cbv: light_cbv,
            _shadow_cbv: shadow_cbv,
            readbacks,
            readback_layout,
            cmd_allocs: Vec::with_capacity(PROBE_FACE_COUNT),
            cmd_lists: Vec::with_capacity(PROBE_FACE_COUNT),
            last_fence_value: 0,
        });
        Ok(())
    }

    // Submit the in-flight capture's next cube face (one per frame): a fresh command
    // list that culls the face frustum into the reserved slot, renders the bindless
    // static + instance geometry into the face target, (resolves +) copies it into the
    // face's readback buffer, then signals a fence value. The last face's value is what
    // readback waits for.
    fn probe_render_next_face(&mut self) -> Result<(), String> {
        let slot = self.bake_ring_slot();
        let (face, eye, near, far, sample_count, view_gva, light_gva, shadow_gva) = {
            let bake = self
                .probe_rendering
                .as_ref()
                .ok_or("probe: render face with no capture in flight")?;
            (
                bake.cursor,
                bake.eye,
                bake.near,
                bake.far,
                bake.sample_count,
                bake.view_gvas[bake.cursor],
                bake.light_gva,
                bake.shadow_gva,
            )
        };

        let vp = reflection_probe::face_view_projection(eye, face, near, far);
        let frustum = crate::gfx::frustum::Frustum::from_view_projection(vp);

        // A fresh allocator + list per face (held until readback, so no reset of an
        // in-flight allocator).
        let alloc: ID3D12CommandAllocator = unsafe {
            self.device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .map_err(|e| format!("probe: face allocator: {e}"))?;
        let cmd: ID3D12GraphicsCommandList = unsafe {
            self.device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
        }
        .map_err(|e| format!("probe: face cmd list: {e}"))?;

        // Cull this face's frustum into the reserved indirect buffer, then render.
        self.encode_probe_cull(&cmd, slot, &frustum, eye);
        let (rtv, dsv) = {
            let bake = self.probe_rendering.as_ref().unwrap();
            (bake.rtv, bake.dsv)
        };
        let indirect = &self.cull.indirect_cmd_buffers[slot];
        let object_gva = unsafe { self.cull.object_buffer_resources[slot].GetGPUVirtualAddress() };
        self.encode_main_into_face(
            &cmd,
            rtv,
            dsv,
            view_gva,
            light_gva,
            shadow_gva,
            indirect,
            0,
            object_gva,
            PROBE_FACE_SIZE,
            PROBE_FACE_SIZE,
        );

        // Resolve (MSAA) + copy the face into its readback buffer.
        self.copy_face_to_readback(&cmd, face, sample_count)?;

        unsafe { cmd.Close() }.map_err(|e| format!("probe: face close: {e}"))?;
        let list: ID3D12CommandList =
            windows::core::Interface::cast(&cmd).map_err(|e| format!("probe: face cast: {e}"))?;
        unsafe { self.command_queue.ExecuteCommandLists(&[Some(list)]) };

        // Signal a unique fence value on the shared fence; the readback waits for it.
        let fence_val = self.frame_sync.next_fence_value.get();
        self.frame_sync.next_fence_value.set(fence_val + 1);
        unsafe { self.command_queue.Signal(&self.frame_sync.fence, fence_val) }
            .map_err(|e| format!("probe: face signal: {e}"))?;

        if let Some(bake) = self.probe_rendering.as_mut() {
            bake.cmd_allocs.push(alloc);
            bake.cmd_lists.push(cmd);
            bake.last_fence_value = fence_val;
            bake.cursor += 1;
        }
        Ok(())
    }

    // Resolve (when MSAA) + copy the just-rendered face colour into readback buffer
    // `face`. The colour rests in RENDER_TARGET and is restored to it for the next
    // face; the resolve target rests in PIXEL_SHADER_RESOURCE.
    fn copy_face_to_readback(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        face: usize,
        sample_count: u32,
    ) -> Result<(), String> {
        let bake = self.probe_rendering.as_ref().unwrap();
        let layout = bake.readback_layout;
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(&bake.readbacks[face]) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layout,
            },
        };
        if sample_count > 1 {
            let resolve = bake.resolve.as_ref().unwrap();
            unsafe {
                cmd.ResourceBarrier(&[
                    transition_barrier(
                        &bake.color,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                    ),
                    transition_barrier(
                        resolve,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_RESOLVE_DEST,
                    ),
                ]);
                cmd.ResolveSubresource(resolve, 0, &bake.color, 0, HDR_FORMAT);
                cmd.ResourceBarrier(&[
                    transition_barrier(
                        resolve,
                        D3D12_RESOURCE_STATE_RESOLVE_DEST,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ),
                    transition_barrier(
                        &bake.color,
                        D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                    ),
                ]);
                let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(resolve),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
                cmd.ResourceBarrier(&[transition_barrier(
                    resolve,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                )]);
            }
        } else {
            unsafe {
                cmd.ResourceBarrier(&[transition_barrier(
                    &bake.color,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&bake.color),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
                cmd.ResourceBarrier(&[transition_barrier(
                    &bake.color,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                )]);
            }
        }
        Ok(())
    }

    // The GPU has finished the capture (the fence reached the last face's value): map
    // the six readback buffers, decode RGBA16Float -> f32, free the capture's GPU
    // resources, and hand the faces to a worker thread that runs the GGX prefilter
    // convolution off the render thread. Moves the bake to Converting.
    fn probe_readback_and_convolve(&mut self) -> Result<(), String> {
        let bake = self
            .probe_rendering
            .take()
            .ok_or("probe: readback with no bake in flight")?;
        let row_pitch = bake.readback_layout.Footprint.RowPitch as usize;
        let tight_row = PROBE_FACE_SIZE as usize * 8; // four halfs per texel
        let mut faces: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::new());
        for (face, readback) in bake.readbacks.iter().enumerate() {
            faces[face] = read_face_rgba_f32(readback, row_pitch, tight_row)?;
        }
        // The capture's GPU resources (targets + command lists + readbacks) drop here;
        // the fence reached `last_fence_value`, so the GPU is done with all of them.
        let index = bake.index;
        let placement = bake.placement;
        drop(bake);

        let payload = Arc::new(OnceLock::new());
        let slot = Arc::clone(&payload);
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
                tracing::error!("reflection probe convolution panicked; abandoning bake");
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

    // The off-thread convolution finished: deserialise its payload, upload the
    // prefiltered radiance cube, and install it into `probe_maps` + `probe_set` (the
    // specular reflection source), leaving `env_map` / the sky untouched.
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
            .map_err(|e| format!("probe: deserialise payload: {e}"))?;
        if view.prefilter_mip_bytes.is_empty() {
            return Err("probe: payload has no prefilter mips".into());
        }
        let mip_count = view.prefilter_mip_bytes.len() as u32;
        let prefilter = upload_probe_prefilter_cube(
            &self.device,
            &self.command_queue,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )?;

        // Point this probe's slot in the cube array at the baked cube (it held the
        // sky prefilter until now). The forward shader samples it once `probe_set.count`
        // covers this index.
        super::texture::write_cube_srv_mips(
            &self.device,
            &prefilter,
            mip_count,
            self.probe_cube_slot_cpu(index),
        );

        debug_assert_eq!(index, self.probe_maps.len());
        self.probe_maps.push(ProbeCube {
            prefilter,
            mip_count,
        });
        self.probe_set.probes[index] = super::probe_uniforms::ProbeUniforms {
            box_min: [p.box_min[0], p.box_min[1], p.box_min[2], 1.0],
            box_max: [p.box_max[0], p.box_max[1], p.box_max[2], 0.0],
            probe_pos: [p.position[0], p.position[1], p.position[2], 0.0],
        };
        self.probe_set.count = self.probe_maps.len() as u32;
        tracing::info!(
            "reflection probes: baked {}/{}",
            index + 1,
            self.probe_placements.len()
        );
        Ok(())
    }

    // Render the bindless static + instance geometry into an off-screen target. A
    // thin sibling of `encode_main_pass`'s bindless branch: it clears + targets the
    // RTV/DSV, binds a per-view ViewUniforms CBV, and issues the static + instance
    // prefix `ExecuteIndirect` from `slot`'s indirect buffer. Skinned geometry is not
    // drawn (V1). No SSAO pre-pass, no HDR resolve -- the caller copies / resolves the
    // target out. Shared by the probe-face capture (square face, reserved bake
    // slot's indirect at offset 0) and the planar reflection mirror render
    // (render-resolution target, the planar indirect buffer at the plane's region
    // byte offset, drawn against the frame's object buffer). `indirect_offset` is a
    // byte offset into `indirect` to the region's first command.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn encode_main_into_face(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
        view_gva: u64,
        light_gva: u64,
        shadow_ubo_gva: u64,
        indirect: &ID3D12Resource,
        indirect_offset: u32,
        object_gva: u64,
        width: u32,
        height: u32,
    ) {
        let bindless_pso = self.cull.main_bindless_pso.as_ref().unwrap();
        let bindless_root = self.cull.main_bindless_root_sig.as_ref().unwrap();
        let cull_sig = self.cull.cull_command_signature.as_ref().unwrap();

        unsafe {
            cmd.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            cmd.ClearRenderTargetView(rtv, &self.clear_color, None);
            cmd.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);

            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.IASetVertexBuffers(0, Some(&[self.geometry.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);

            cmd.SetPipelineState(bindless_pso);
            cmd.SetGraphicsRootSignature(bindless_root);
            cmd.SetGraphicsRootConstantBufferView(1, view_gva);
            cmd.SetGraphicsRootConstantBufferView(2, light_gva);
            cmd.SetGraphicsRootConstantBufferView(3, shadow_ubo_gva);
            cmd.SetGraphicsRootDescriptorTable(4, self.shadow.srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(5, self.cull.bindless_pool_gpu);
            cmd.SetGraphicsRootDescriptorTable(6, self.descriptors.shadow_sampler_gpu);
            cmd.SetGraphicsRootDescriptorTable(7, self.descriptors.linear_sampler_gpu);
            cmd.SetGraphicsRootShaderResourceView(8, object_gva);
            cmd.SetGraphicsRootDescriptorTable(9, self.ssao_ao_srv_gpu());
            // [10] probe cube array (valid -- filled with the sky) + [11] the EMPTY
            // ProbeSet (count 0), so a probe face samples only the sky, not other
            // probes, and never reads the live ProbeSet ring while it is rewritten.
            cmd.SetGraphicsRootDescriptorTable(10, self.probe_cube_table_gpu());
            cmd.SetGraphicsRootConstantBufferView(
                11,
                self.probe_set_empty_cbv.GetGPUVirtualAddress(),
            );
            // Static + instance prefix `[0, skinned_record_base())`. Skinned tail
            // omitted (not captured into the probe in V1).
            cmd.ExecuteIndirect(
                cull_sig,
                self.skinned_record_base() as u32,
                indirect,
                indirect_offset as u64,
                None::<&ID3D12Resource>,
                0,
            );
        }
        self.inc_draw_calls(1);
    }

    // World-space bounds over every static draw object, skipping degenerate
    // (non-finite) AABBs. `None` for an empty scene. Mirrors
    // `metal/probe.rs::scene_world_bounds`.
    pub(super) fn scene_world_bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        reflection_probe::fold_world_bounds(self.draw_objects.iter().map(|o| (o.bb_min, o.bb_max)))
    }
}

// Read one READBACK buffer back as tightly-packed linear f32 RGBA, decoding the
// RGBA16Float half values and stripping the 256-byte row padding. Row 0 is the top
// of the framebuffer = the `v = -1` cube edge, exactly the layout the build-time
// convolutions consume (matching the shared `face_view_projection` orientation).
fn read_face_rgba_f32(
    readback: &ID3D12Resource,
    row_pitch: usize,
    tight_row: usize,
) -> Result<Vec<f32>, String> {
    let h = PROBE_FACE_SIZE as usize;
    let w = PROBE_FACE_SIZE as usize;
    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    unsafe { readback.Map(0, None, Some(&mut map_ptr)) }
        .map_err(|e| format!("probe: map readback: {e}"))?;
    let mut out = vec![0.0f32; w * h * 4];
    for row in 0..h {
        // SAFETY: the buffer holds `row_pitch * h` (padded) bytes; each row's tight
        // span of `tight_row` bytes is valid within it. The fence reached the face's
        // value, so the copy completed.
        let src = unsafe { (map_ptr as *const u8).add(row * row_pitch) };
        for col in 0..w {
            let px = unsafe { src.add(col * 8) };
            let half =
                |o: usize| unsafe { f16_to_f32(u16::from_le_bytes([*px.add(o), *px.add(o + 1)])) };
            let base = (row * w + col) * 4;
            out[base] = half(0);
            out[base + 1] = half(2);
            out[base + 2] = half(4);
            out[base + 3] = half(6);
        }
    }
    unsafe { readback.Unmap(0, None) };
    let _ = tight_row;
    Ok(out)
}

// Create a persistently-mapped UPLOAD constant buffer holding `bytes` (256-aligned)
// and return it with its GPU virtual address. Used for the bake's per-capture light
// + shadow snapshots, so the six faces share one lighting set decoupled from the
// frame's per-frame CBV writes.
fn make_snapshot_cbv(device: &ID3D12Device, bytes: &[u8]) -> Result<(ID3D12Resource, u64), String> {
    let size = (((bytes.len() as u64) + 255) & !255).max(256);
    let cbv = create_buffer(
        device,
        size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    unsafe { cbv.Map(0, None, Some(&mut ptr)) }
        .map_err(|e| format!("probe: map snapshot cbv: {e}"))?;
    // SAFETY: the buffer is at least `bytes.len()` bytes (256-aligned).
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
    }
    let gva = unsafe { cbv.GetGPUVirtualAddress() };
    Ok((cbv, gva))
}

// A one-entry non-shader-visible RTV heap for a probe face colour target.
fn create_rtv_heap(device: &ID3D12Device) -> Result<ID3D12DescriptorHeap, String> {
    let desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    unsafe { device.CreateDescriptorHeap(&desc) }.map_err(|e| format!("probe: rtv heap: {e}"))
}

// A one-entry non-shader-visible DSV heap for a probe face depth target.
fn create_dsv_heap(device: &ID3D12Device) -> Result<ID3D12DescriptorHeap, String> {
    let desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    unsafe { device.CreateDescriptorHeap(&desc) }.map_err(|e| format!("probe: dsv heap: {e}"))
}

// Create a probe face depth target (D32_FLOAT, matching the main pass's DSV format
// + the face colour's sample count) and write its DSV. Created in DEPTH_WRITE and
// left there (only the bake uses it; it is cleared every face).
fn create_bake_depth(
    device: &ID3D12Device,
    size: u32,
    sample_count: u32,
    dsv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = D3D12_CLEAR_VALUE {
        Format: DXGI_FORMAT_D32_FLOAT,
        Anonymous: D3D12_CLEAR_VALUE_0 {
            DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                Depth: 1.0,
                Stencil: 0,
            },
        },
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: size as u64,
        Height: size,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            Some(&clear_value),
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("probe: create face depth: {e}"))?;
    let texture = tex_opt.ok_or_else(|| "probe: create face depth returned None".to_string())?;
    let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
        Format: DXGI_FORMAT_D32_FLOAT,
        ViewDimension: if sample_count > 1 {
            D3D12_DSV_DIMENSION_TEXTURE2DMS
        } else {
            D3D12_DSV_DIMENSION_TEXTURE2D
        },
        Flags: D3D12_DSV_FLAG_NONE,
        ..Default::default()
    };
    unsafe { device.CreateDepthStencilView(&texture, Some(&dsv_desc), dsv_cpu) };
    Ok(texture)
}
