// src/metal/probe.rs
//
// Scene-captured reflection probes. Each declared `ReflectionProbe` (or an
// auto-seeded grid when a world declares none) is baked into its own cube,
// DISTINCT from `env_map`: the specular reflection term box-projects against the
// probe's influence box and samples its cube, so glossy surfaces and windows
// reflect the actual surrounding geometry instead of the imported (often foreign)
// HDR sky, while the skybox + diffuse irradiance keep sampling `env_map` so the
// visible sky is never replaced by a capture.
//
// Each cube mirrors the main pass exactly -- it reuses the GPU-driven bindless
// cull + the three main-pass geometry sub-paths (`encode_main_into_face`) so the
// folded static + instanced + skinned geometry, and the skybox (a non-cullable
// draw object), all render into each face. The six faces are rendered through
// the cube view-projections in `gfx::reflection_probe` (orientation unit-tested
// there), read back, and handed to the shared build-time convolutions
// (`build_probe_payload`) so a scene-captured probe and an imported HDR produce
// byte-compatible payloads.
//
// The bake is STAGGERED, ASYNCHRONOUS, and PIPELINED across frames so the render
// thread NEVER blocks on a capture, walking a `ProbeBakeQueue` cursor so a not-yet-
// baked probe falls back to the sky until its turn. Each probe passes through three
// phases (`gfx::reflection_probe::BakePhase`, driven by the pure `next_bake_action`
// transition table, called once per pipeline slot per frame):
//   * Rendering   -- six cube faces submitted to the GPU WITHOUT
//                    `waitUntilCompleted`; a completion handler flags GPU
//                    completion. The faces draw from a RESERVED ring slot
//                    (`bake_ring_slot`) the frame never overwrites, so the bake's
//                    CPU-written bindless buffers stay valid across the async work.
//   * Converting  -- on completion the six faces are read back (a fast `getBytes`,
//                    the GPU being done) and the heavy GGX prefilter convolution
//                    runs on a WORKER THREAD, off the render thread.
//   * (install)   -- when the worker finishes, the convolved cube is uploaded and
//                    installed into `probe_maps` + `probe_set` on the main thread.
// The Rendering and Converting phases run in PARALLEL across two slots
// (`probe_rendering` / `probe_converting`): once a probe's faces are read back its
// GPU resources (the reserved ring slot included) are freed, so the NEXT probe starts
// rendering while the prior probe's faces convolve on the worker -- shortening the
// warm-up vs serialising render-then-convolve per probe. Only ONE probe renders at a
// time (so a single reserved ring slot suffices, GPU lifetime unchanged) and only ONE
// converts at a time (so installs stay in queue order, keeping `probe_maps` aligned
// with the placement list). A re-placement (`set_reflection_probes`) parks the
// rendering slot's GPU resources in a frame-tagged retire pool so they outlive any
// still-running capture; the converting slot holds only plain data and drops freely.
//
// Known simplifications (documented intentionally):
//   * The scene is captured lit by whatever environment is live at bake time
//     (single bounce): surfaces carry the old env's ambient. The dominant,
//     visible change is that reflections now show real geometry.
//   * Captured before that frame's shadow map is populated, so the probe bakes
//     direct + ambient lighting without contact shadows.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer as _, MTLCommandBuffer as _, MTLCommandBufferStatus, MTLCommandQueue as _,
    MTLDevice as _, MTLOrigin, MTLPixelFormat, MTLRegion, MTLResourceOptions, MTLSize,
    MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use super::context::{HDR_SAMPLE_COUNT, MtlContext};
use super::screenshot::f16_to_f32;
use crate::gfx::reflection_probe::{self, BakeAction, BakePhase};

// Captured cube-face resolution (mip 0 of the prefilter chain). The fragment
// shaders sample env-map mip 0 directly for the on-screen skybox, so this also
// sets the visible sky resolution; 512 matches the `EnvironmentMap` asset
// default so swapping in a captured probe keeps the sky as sharp as the imported
// HDR (256 visibly softens a detailed sky). Mips 1..N feed the glossy reflection
// lookup. The convolution + readback cost scales with this.
const PROBE_FACE_SIZE: u32 = 512;
// Irradiance cube resolution. Diffuse irradiance is low frequency, so this stays
// small and cheap. Power of two in the build-time valid range.
const PROBE_IRRADIANCE_FACE: u32 = 16;
// GGX prefilter samples per output texel. The importer uses 1024 at build time;
// a runtime bake uses far fewer (the convolution is rayon-parallel, so this is
// sub-second on Apple Silicon) at a small quality cost acceptable for reflections.
const PROBE_PREFILTER_SAMPLES: u32 = 128;
// Firefly clamp applied during the prefilter convolution, matching the
// EnvironmentMap asset default so a captured probe and an imported HDR suppress
// the same bright-square aliasing.
const PROBE_PREFILTER_CLAMP: f32 = 12.0;
// Cube faces per probe. Rendered one per frame (spread) so a single bake never adds
// the cost of six full-scene captures to one frame.
const PROBE_FACE_COUNT: usize = 6;

// The two pipelined bake slots. One probe renders its six cube faces on the GPU
// (`RenderingBake`, owning the reserved-ring-slot buffers + capture targets) while a
// PRIOR probe's faces convolve off-thread (`ConvertingBake`, owning only the worker's
// payload slot -- plain data). Overlapping the off-thread convolution with the next
// probe's render shortens the bake warm-up. Only ONE probe holds the reserved ring slot
// at a time (the rendering one), so the GPU-resource lifetime is identical to a single
// in-flight bake; at most one probe converts at a time, which keeps installs in queue
// order (`probe_maps` is appended one cube per probe).
pub(in crate::metal) struct RenderingBake {
    // Placement index being captured; its cube lands at `probe_maps[index]` on install.
    index: usize,
    placement: reflection_probe::ProbePlacement,
    // Set by the LAST face's completion handler once every face has been submitted.
    done: Arc<AtomicBool>,
    // The next of `PROBE_FACE_COUNT` faces to submit (one per frame, so no single frame
    // pays the whole capture).
    cursor: usize,
    // Capture vantage, snapshotted at start so the six faces are temporally consistent.
    eye: [f32; 3],
    near: f32,
    far: f32,
    elapsed: f32,
    // Loop-invariant buffers + targets shared across the six faces (reserved ring slot).
    gpu: BakeGpu,
}

pub(in crate::metal) struct ConvertingBake {
    // Placement index; its convolved cube lands at `probe_maps[index]` on install.
    index: usize,
    placement: reflection_probe::ProbePlacement,
    // The off-thread prefilter convolution writes its `ENVM` payload here exactly once;
    // the install polls it. Plain data (no Metal handles), so this slot drops freely.
    payload: Arc<OnceLock<Vec<u8>>>,
}

// The GPU resources of one capture, built once at the start and reused across all
// six faces: the shared MSAA pair, the six Shared resolve faces, and the
// reserved-slot bindless buffers (+ a skinned deformed buffer). Held resident for
// the whole asynchronous capture (every field is bound while rendering a face;
// `resolves` is additionally read at the end). Also the retire-pool payload when a
// re-placement interrupts a still-running bake: parked until the frames-in-flight
// fence guarantees the capture command buffers have retired.
pub(in crate::metal) struct BakeGpu {
    msaa_color: Retained<ProtocolObject<dyn MTLTexture>>,
    msaa_depth: Retained<ProtocolObject<dyn MTLTexture>>,
    resolves: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    object_buffer: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    draw_args: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    tex_args: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    joint_bufs: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    deformed: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
}

impl MtlContext {
    // Set the reflection-probe placements (declared `ReflectionProbe` assets,
    // converted to `ProbePlacement`s by the graphics system). An empty list
    // auto-seeds a grid from the scene bounds, so existing scenes still get local
    // reflections without authoring. Resets the staggered bake so the next
    // eligible frames re-bake from scratch; capped at `MAX_PROBES`.
    pub(in crate::metal) fn set_reflection_probes(
        &mut self,
        declared: &[reflection_probe::ProbePlacement],
    ) {
        use super::uniforms::MAX_PROBES;
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
        self.probe_placements = placements;
        self.probe_maps.clear();
        self.probe_set = super::uniforms::ProbeSet::EMPTY;
        self.probe_bake_queue = reflection_probe::ProbeBakeQueue::new(self.probe_placements.len());
        // Park the rendering capture's GPU resources instead of dropping them: its
        // command buffers may still be reading the reserved-slot buffers + resolve
        // targets, so defer their free until the frames-in-flight fence guarantees those
        // command buffers have retired. The converting bake holds only the worker's
        // payload slot (plain data) and drops freely.
        if let Some(bake) = self.probe_rendering.take() {
            self.probe_retire_pool.push(self.frame_ring_index, bake.gpu);
        }
        self.probe_converting = None;
    }

    // The reserved transient-ring slot the asynchronous bake builds its bindless
    // buffers into: one past the frame's range `[0, frames_in_flight)`. The frame
    // never writes this slot, so the bake's CPU-written buffers stay valid across
    // its `waitUntilCompleted`-free capture. The bake-relevant rings are sized
    // `frames_in_flight + 1` in `init` to make room for it.
    fn bake_ring_slot(&self) -> usize {
        self.frames_in_flight
    }

    // Advance the asynchronous reflection-probe bake by one step. Called every
    // frame from `draw_frame_inner` after the frames-in-flight fence; cheap once the
    // queue drains and nothing is in flight. Drives the pure `next_bake_action`
    // transition table: start the next probe's capture, read a finished capture back
    // and kick its off-thread convolution, or install a convolved cube. Never blocks
    // the render thread. Non-fatal: a failure keeps the current state.
    pub(in crate::metal) fn bake_pending_probes(
        &mut self,
        elapsed: f32,
        near: f32,
        far: f32,
    ) -> Result<(), String> {
        // Free any parked (interrupted) bake resources the fence now guarantees have
        // retired on the GPU.
        self.probe_retire_pool
            .collect(self.frame_ring_index, self.frames_in_flight as u64);

        // Permanent ineligibility: the capture renders through the bindless ICB
        // (needs the GPU-driven static path); a world with no real geometry keeps
        // the sky; and a probe only adds value over a real environment (a world on
        // the 1x1 grey fallback has no prefilter chain). The environment is built at
        // init, so its readiness never changes for a world. None of these can become
        // eligible later, so abandon the queue rather than re-checking it forever.
        // Under normal play no bake is in flight here (the gate is stable from the
        // first frame), but a debug shader hot-reload can flip `self.bindless` false
        // after a bake started, so park any in-flight capture behind the fence rather
        // than leaking it (its command buffers may still be reading those resources).
        if !self.bindless || self.geometry_less || self.env_map.prefilter_mip_count <= 1 {
            if let Some(bake) = self.probe_rendering.take() {
                self.probe_retire_pool.push(self.frame_ring_index, bake.gpu);
            }
            self.probe_converting = None;
            self.probe_bake_queue.abort();
            return Ok(());
        }

        // Two pipelined slots advance independently each frame, the pure
        // `next_bake_action` transition table called once per slot. Every transition
        // that can fail routes through `fail_bake` (abandon the rest, keep what baked):
        // the queue cursor advanced when a probe started, so leaving it pending after a
        // mid-bake failure would desync `probe_maps` from the placement list.

        // Converting slot: install the convolved cube once the worker finishes. Doing
        // this FIRST frees the slot so the rendering slot can read its finished capture
        // back this same frame.
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
        // The converting slot is free this frame if it was empty or we just installed it.
        let converting_free = !converting_occupied || install;

        // Rendering slot: submit one cube face per frame; once the GPU signals all six
        // done AND the converting slot is free, read the faces back and hand them to the
        // worker (moving to the converting slot); or, when no probe is rendering, start
        // the next pending placement. Gating `Readback` on the converting slot being free
        // keeps at most one probe converting, so installs stay in queue order -- and the
        // next probe's render overlaps the prior probe's off-thread convolution, so the
        // warm-up no longer serialises render-then-convolve per probe.
        let rendering_occupied = self.probe_rendering.is_some();
        // `done` only matters once every face is submitted; the completion handler is
        // attached on the last face, so it cannot be set while faces remain.
        let more_faces = self
            .probe_rendering
            .as_ref()
            .is_some_and(|r| r.cursor < PROBE_FACE_COUNT);
        let done = self
            .probe_rendering
            .as_ref()
            .is_some_and(|r| r.done.load(Ordering::Acquire));
        // Transient ineligibility: geometry may still be streaming in on the first
        // frames. A zero cull keeps the queue cursor so a later frame retries rather than
        // starting an empty capture.
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
                if let Err(e) = self.probe_start_next(near, far, elapsed) {
                    self.fail_bake(e);
                }
            }
            BakeAction::Install | BakeAction::Idle => {}
        }
        Ok(())
    }

    // Abandon the rest of the bake after an unrecoverable error, keeping the cubes
    // already installed. The queue cursor advanced when the current probe started,
    // so aborting (cursor -> end) is what keeps `probe_maps` aligned with the
    // placement list; the sky covers the remaining placements.
    fn fail_bake(&mut self, e: String) {
        tracing::warn!(
            "reflection probe bake failed, keeping {} baked: {e}",
            self.probe_maps.len()
        );
        // Abandon BOTH slots: a converting-slot (install) failure leaves `probe_maps`
        // short by one, so a later rendering probe would install at a gapped index and
        // desync the box alignment. Dropping the rendering gpu here is safe -- Metal
        // retains resources referenced by in-flight command buffers until they retire --
        // and the queue abort means the reserved ring slot is never reused.
        self.probe_rendering = None;
        self.probe_converting = None;
        self.probe_bake_queue.abort();
    }

    // Begin baking the next pending placement: build the reserved-slot bindless
    // buffers + capture targets ONCE (they are loop-invariant across the six faces),
    // and enter `Rendering` with the face cursor at 0. No face is submitted here; the
    // faces follow one per frame via `probe_render_next_face`, so a single frame never
    // pays the cost of all six full-scene captures.
    fn probe_start_next(&mut self, near: f32, far: f32, elapsed: f32) -> Result<(), String> {
        let Some(index) = self.probe_bake_queue.take_next() else {
            return Ok(());
        };
        // Note: unlike the install-time check, `index == probe_maps.len()` does NOT hold
        // here -- with the pipeline this probe can START rendering while the PRIOR probe
        // is still converting (not yet installed), so `probe_maps` may be one entry
        // behind. The box-alignment invariant is enforced at install instead, where the
        // single-converting rule guarantees installs land in queue order.
        let placement = self.probe_placements[index];
        let eye = placement.position;
        let slot = self.bake_ring_slot();

        // Build into the reserved ring slot (the frame never touches it), so these
        // CPU-written buffers stay valid for the whole asynchronous capture. They are
        // frustum-independent (only the per-face view/projection differs), so they are
        // built once and reused by every face.
        let object_buffer = self
            .build_object_buffer(slot)?
            .ok_or("probe: no static geometry to bake")?;
        let draw_args = self
            .build_draw_args_buffer(eye, slot)?
            .ok_or("probe: no draw args to bake")?;
        self.ensure_icb_capacity(self.cull_count())?;
        let tex_args = self
            .build_bindless_texture_args(slot)?
            .ok_or("probe: no bindless texture args")?;
        let joint_bufs = self.build_joint_buffers(slot)?;
        // The folded skinned tail draws compute-deformed vertices. The frame's
        // deformed ring is overwritten every frame, so an async capture needs its
        // OWN deformed buffer (Shared storage -- a Private one page-faults in this
        // cross-command-buffer producer/consumer pattern, like the frame's). `None`
        // for static worlds.
        let deformed: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>> =
            if self.n_skinned > 0 {
                match self.skinned.deformed.first().map(|b| b.length()) {
                    Some(len) if len > 0 => Some(
                        self.device
                            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                            .ok_or("probe: failed to allocate deformed buffer")?,
                    ),
                    _ => None,
                }
            } else {
                None
            };

        // One reused MSAA colour + depth pair (faces render serially across frames),
        // and six Shared single-sample resolve targets read back on completion.
        let msaa_color = make_msaa_color(&self.device, PROBE_FACE_SIZE)?;
        let msaa_depth = make_msaa_depth(&self.device, PROBE_FACE_SIZE)?;
        let resolves: Vec<Retained<ProtocolObject<dyn MTLTexture>>> = (0..PROBE_FACE_COUNT)
            .map(|_| make_resolve_shared(&self.device, PROBE_FACE_SIZE))
            .collect::<Result<_, _>>()?;

        self.probe_rendering = Some(RenderingBake {
            index,
            placement,
            done: Arc::new(AtomicBool::new(false)),
            cursor: 0,
            eye,
            near,
            far,
            elapsed,
            gpu: BakeGpu {
                msaa_color,
                msaa_depth,
                resolves,
                object_buffer,
                draw_args,
                tex_args,
                joint_bufs,
                deformed,
            },
        });
        Ok(())
    }

    // Submit the in-flight capture's next cube face (one per frame). On the LAST face
    // a completion handler is attached (before that face's commit, as Metal requires)
    // to flag GPU completion: single-queue FIFO completion means every face is done
    // when this one is. The shared `cull.icb` is GPU-written, so Metal hazard-tracks
    // it: each face's cull (and the frame's own cull) waits for the prior read,
    // ordering the reuse correctly with no explicit barrier or `waitUntilCompleted`.
    fn probe_render_next_face(&mut self) -> Result<(), String> {
        let Some(RenderingBake {
            done,
            cursor,
            eye,
            near,
            far,
            elapsed,
            gpu,
            ..
        }) = &self.probe_rendering
        else {
            return Err("probe: render face with no capture in flight".into());
        };
        let face = *cursor;
        let (eye, near, far, elapsed) = (*eye, *near, *far, *elapsed);
        let attach_done = face + 1 == PROBE_FACE_COUNT;

        let vp = reflection_probe::face_view_projection(eye, face, near, far);
        let view = reflection_probe::face_view_matrix(eye, face);
        let frustum = crate::gfx::frustum::Frustum::from_view_projection(vp);

        // Cull command buffer: fills the shared ICB for this face's frustum.
        let cull_cb = self
            .command_queue
            .commandBuffer()
            .ok_or("probe: failed to get cull command buffer")?;
        // Skin once, on the first face: the deformed vertices are a pure function of
        // the bind pose + joint palettes (both loop-invariant), so the pose is
        // identical for every face. FIFO + hazard tracking on the Shared deformed
        // buffer order that single write before every face render reads it.
        if face == 0
            && let Some(def) = gpu.deformed.as_ref()
        {
            self.encode_main_skin(&cull_cb, def, &gpu.joint_bufs)?;
        }
        self.encode_cull(&cull_cb, &gpu.object_buffer, &gpu.draw_args, &frustum, eye)?;
        cull_cb.commit();

        // Render command buffer: reads the ICB into this face. Instances fold into
        // the bindless ICB, so the legacy prepared set draws nothing here (empty).
        let render_cb = self
            .command_queue
            .commandBuffer()
            .ok_or("probe: failed to get render command buffer")?;
        let prepared = super::instanced::PreparedInstances {
            clusters: Vec::new(),
        };
        self.encode_main_into_face(
            &render_cb,
            &gpu.msaa_color,
            &gpu.msaa_depth,
            &gpu.resolves[face],
            view,
            vp,
            eye,
            elapsed,
            &[],
            &prepared,
            &gpu.joint_bufs,
            Some(&gpu.object_buffer),
            Some(&gpu.tex_args),
            gpu.deformed.as_ref(),
            // Probe cube bake reuses the main cull ICB (no per-face mirror cull).
            None,
        )?;
        if attach_done {
            let flag = Arc::clone(done);
            let handler = block2::RcBlock::new(
                move |cb: NonNull<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>| {
                    // SAFETY: the completion handler is invoked by Metal with a valid
                    // command-buffer pointer.
                    let cb = unsafe { cb.as_ref() };
                    if cb.status() == MTLCommandBufferStatus::Error {
                        tracing::error!(
                            "reflection probe face bake faulted (async): {:?}",
                            cb.error()
                        );
                    }
                    flag.store(true, Ordering::Release);
                },
            );
            // SAFETY: addCompletedHandler copies the block, so the RcBlock may drop
            // here; it must be added before the commit below.
            unsafe {
                render_cb.addCompletedHandler(block2::RcBlock::as_ptr(&handler));
            }
        }
        render_cb.commit();

        // Advance the cursor (a separate mutable borrow now the render borrows ended).
        if let Some(RenderingBake { cursor, .. }) = &mut self.probe_rendering {
            *cursor += 1;
        }
        Ok(())
    }

    // The GPU has finished the in-flight capture: read the six faces back as linear
    // f32 RGBA (a fast `getBytes`, the resolves being Shared and the GPU done), free
    // the capture's GPU resources, and hand the faces to a worker thread that runs
    // the heavy GGX prefilter convolution off the render thread. The bake moves to
    // Converting; the worker writes its `ENVM` payload into the shared slot once.
    fn probe_readback_and_convolve(&mut self) -> Result<(), String> {
        let RenderingBake {
            index,
            placement,
            gpu,
            ..
        } = self
            .probe_rendering
            .take()
            .ok_or("probe: readback with no bake in flight")?;
        let mut faces: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::new());
        for (face, resolve) in gpu.resolves.iter().enumerate() {
            faces[face] = read_face_rgba_f32(resolve, PROBE_FACE_SIZE)?;
        }
        // `gpu` (MSAA pair + resolves + reserved-slot buffers) drops here -- safe, the
        // GPU is done with all of it.
        drop(gpu);

        let payload = Arc::new(OnceLock::new());
        let slot = Arc::clone(&payload);
        // The convolution touches only plain `Vec<f32>` data (no Metal handles), so
        // it runs on a worker thread with nothing to synchronise; it sets the payload
        // exactly once, which the install polls for. The work is caught so a panic
        // can never leave the slot empty: that would wedge Converting forever (the
        // poll never sees a payload, and one bake is in flight at a time, so no
        // further probe would ever start). On a panic an empty payload is stored;
        // `build_probe_textures` then rejects it, and `fail_bake` abandons the rest.
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

    // The off-thread convolution finished: upload its payload as a probe cube and
    // install it into `probe_maps` + `probe_set` (the specular reflection source),
    // leaving `env_map` / the sky untouched. Runs on the main thread.
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
        let textures = self.build_probe_textures(bytes)?;
        debug_assert_eq!(index, self.probe_maps.len());
        self.probe_maps.push(textures);
        self.probe_set.probes[index] = super::uniforms::ProbeUniforms {
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

    // World-space bounds over every static draw object, skipping degenerate
    // (non-finite) AABBs. `None` for an empty scene. Folded instances + skinned
    // objects sit inside the static extent for the scenes this bakes, so the
    // static objects' union is a good probe-centring volume.
    pub(in crate::metal) fn scene_world_bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        reflection_probe::fold_world_bounds(self.draw_objects.iter().map(|o| (o.bb_min, o.bb_max)))
    }
}

// MSAA HDR colour face: RGBA16Float, 4x, render-target only -- matches the main
// pipeline's attachment format + sample count so `self.pipeline_state` binds.
fn make_msaa_color(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    size: u32,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let desc = MTLTextureDescriptor::new();
    unsafe {
        desc.setTextureType(MTLTextureType::Type2DMultisample);
        desc.setPixelFormat(MTLPixelFormat::RGBA16Float);
        desc.setWidth(size as usize);
        desc.setHeight(size as usize);
        desc.setSampleCount(HDR_SAMPLE_COUNT as usize);
        desc.setUsage(MTLTextureUsage::RenderTarget);
        desc.setStorageMode(MTLStorageMode::Private);
    }
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| "probe: failed to create MSAA colour face".into())
}

// MSAA depth face: Depth32Float, 4x, render-target only. Cleared per face and
// discarded -- the probe consumes only the resolved colour.
fn make_msaa_depth(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    size: u32,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let desc = MTLTextureDescriptor::new();
    unsafe {
        desc.setTextureType(MTLTextureType::Type2DMultisample);
        desc.setPixelFormat(MTLPixelFormat::Depth32Float);
        desc.setWidth(size as usize);
        desc.setHeight(size as usize);
        desc.setSampleCount(HDR_SAMPLE_COUNT as usize);
        desc.setUsage(MTLTextureUsage::RenderTarget);
        desc.setStorageMode(MTLStorageMode::Private);
    }
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| "probe: failed to create MSAA depth face".into())
}

// Single-sample resolve face: RGBA16Float, Shared storage so `getBytes` can read
// it on the CPU after the render pass resolves into it.
fn make_resolve_shared(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    size: u32,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let desc = MTLTextureDescriptor::new();
    unsafe {
        desc.setTextureType(MTLTextureType::Type2D);
        desc.setPixelFormat(MTLPixelFormat::RGBA16Float);
        desc.setWidth(size as usize);
        desc.setHeight(size as usize);
        desc.setUsage(MTLTextureUsage(
            MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0,
        ));
        desc.setStorageMode(MTLStorageMode::Shared);
    }
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| "probe: failed to create resolve face".into())
}

// Read one Shared RGBA16Float face back as tightly-packed linear f32 RGBA, row
// major (row 0 = the v = -1 edge), exactly the layout the build-time
// convolutions consume.
fn read_face_rgba_f32(
    tex: &ProtocolObject<dyn MTLTexture>,
    face_size: u32,
) -> Result<Vec<f32>, String> {
    let w = face_size as usize;
    let h = w;
    let bytes_per_row = w * 8; // four halfs per texel
    let mut raw = vec![0u8; bytes_per_row * h];
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: w,
            height: h,
            depth: 1,
        },
    };
    // SAFETY: `raw` is exactly `bytes_per_row * h` bytes (the tight footprint
    // requested), the texture is StorageModeShared, and its resolve completed
    // (`waitUntilCompleted` before this call), so the copy is valid.
    unsafe {
        tex.getBytes_bytesPerRow_fromRegion_mipmapLevel(
            std::ptr::NonNull::new(raw.as_mut_ptr() as *mut std::ffi::c_void)
                .ok_or("probe: null readback pointer")?,
            bytes_per_row,
            region,
            0,
        );
    }
    let mut out = vec![0.0f32; w * h * 4];
    for (i, px) in raw.chunks_exact(8).enumerate() {
        out[i * 4] = f16_to_f32(u16::from_le_bytes([px[0], px[1]]));
        out[i * 4 + 1] = f16_to_f32(u16::from_le_bytes([px[2], px[3]]));
        out[i * 4 + 2] = f16_to_f32(u16::from_le_bytes([px[4], px[5]]));
        out[i * 4 + 3] = f16_to_f32(u16::from_le_bytes([px[6], px[7]]));
    }
    Ok(out)
}
