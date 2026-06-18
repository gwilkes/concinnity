// src/metal/particle.rs
//
// GPU-compute particle system on Metal. Each `ParticleEmitter` declared in
// the world produces one persistent `ParticleEmitterGpuState` carrying a pool
// of `Particle` slots and an atomic spawn-counter buffer. Each frame the
// renderer:
//
//   1. Computes the per-emitter spawn budget CPU-side (a fractional
//      accumulator drives integer particle spawns per dispatch).
//   2. Writes that budget into the atomic counter buffer.
//   3. Dispatches the `particle_simulate` compute kernel to age + integrate +
//      respawn the pool.
//   4. Dispatches the `particle_vertex`/`particle_fragment` render pipeline
//      with `instance_count = max_particles`, drawing one camera-facing
//      billboard quad per live particle.
//
// The render pass alpha-blends into `hdr_resolve` after the volumetric fog
// pass and before SSR, so particles appear in screen-space reflections and
// are temporally stabilised by TAA.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlendFactor, MTLBuffer, MTLCommandBuffer as _, MTLComputeCommandEncoder as _,
    MTLComputePassDescriptor, MTLComputePipelineState, MTLDevice as _, MTLLibrary as _,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
    MTLResourceOptions, MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
    MTLSamplerState, MTLSize, MTLStoreAction,
};

use crate::gfx::particles::{ParticleEmitterRecord, ParticleSpawnState};
use crate::gfx::render_types::ParticleParams;

use super::context::MtlContext;
use super::pipeline::{ns_str, shader_source};
use super::scoped_encoder::ScopedEncoder;
use super::uniforms::ParticleView;

// One particle slot on the GPU. Layout must match the `Particle` MSL struct
// in `shaders/particle.metal` (32 bytes per slot: `packed_float3 + float`
// twice).
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct GpuParticle {
    position: [f32; 3],
    age: f32,
    velocity: [f32; 3],
    lifetime: f32,
}

// Per-emitter persistent GPU state. The pool buffer lives in shared storage
// so the CPU can zero-init it once; the atomic counter buffer is rewritten
// per frame with the integer spawn budget.
pub(super) struct ParticleEmitterGpuState {
    // Particle pool: `record.max_particles` slots of `GpuParticle`.
    pub pool: Retained<ProtocolObject<dyn MTLBuffer>>,
    // One `u32` atomic counter; the compute kernel decrements it as threads
    // claim spawn slots. Reset to `spawn_budget` at the start of each frame.
    pub spawn_counter: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Carry-over spawn fraction. Combined with `dt` and the emitter's
    // `spawn_rate` to produce the integer spawn budget for each dispatch.
    pub spawn_state: ParticleSpawnState,
}

// Pair of pipelines driving the particle system: the compute kernel that
// ages + integrates + respawns the pool, and the render pipeline that draws
// each live particle as a camera-facing billboard quad. Built only when the
// world declared at least one `ParticleEmitter`.
pub(super) struct ParticlePipelines {
    pub simulate: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub render: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
}

// All particle-system state grouped into one feature unit: the per-emitter
// records (with their tombstone free-list), the parallel per-emitter GPU
// pools, the shared compute + render pipelines, and the per-frame timing
// bookkeeping. `records` and `emitter_state` are parallel: the dispatch
// loop walks both in lockstep, skipping `None` pairs. `pipelines` is built
// lazily at init (≥1 declared emitter) or on the first runtime
// [`MtlContext::add_emitter`].
pub(crate) struct ParticleState {
    // One slot per emitter; `None` slots are tombstones from
    // [`MtlContext::remove_emitter`], reused by the next add via `free_slots`.
    pub records: Vec<Option<ParticleEmitterRecord>>,
    // Per-emitter persistent GPU state, parallel to `records`; `None` matches
    // a tombstoned record.
    pub emitter_state: Vec<Option<ParticleEmitterGpuState>>,
    pub free_slots: Vec<usize>,
    pub pipelines: Option<ParticlePipelines>,
    // Last frame's `elapsed`; the diff drives spawn budgets + integration.
    pub last_elapsed: f32,
    // Frame counter mixed into the compute kernel's per-thread RNG seed.
    pub frame_index: u32,
}

impl MtlContext {
    // Encode the per-emitter compute + render passes. A no-op when no
    // emitters are declared; `record.visible` and `max_particles == 0`
    // filtering is done at `build_particle_records` time, so any emitter
    // that reached this point is drawn.
    //
    // `elapsed` is the same value the rest of the frame already computed:
    // the previous-frame snapshot lives in `particle_last_elapsed`, and the
    // diff is the frame `dt` driving spawn rates + integration.
    // pub(in crate::metal) so the render-graph executor in
    // metal/graph_exec.rs can dispatch this pass from a CompiledGraph.
    // Bundles ParticlesSim (compute) + ParticlesDraw (render); the
    // graph only adds a node for `PassId::ParticlesDraw`, but the
    // bundled sim sub-pass keeps its own per-pass timing slot via the
    // inline `pass_timing.attach_compute` call below.
    // Mutate the per-frame particle state (dt against
    // `particle_last_elapsed`, monotonic `particle_frame_index`,
    // per-emitter spawn budgets) and write each emitter's spawn-counter
    // buffer in-place. Returns the per-frame `(dt, frame_index,
    // per_emitter_budgets)` tuple the read-only `encode_particles` then
    // consumes. Split out so `encode_particles` can take `&self` and run
    // on a parallel-recording worker; the mutating prelude stays on the
    // frame's main `&mut self` path inside `execute_graph`.
    pub(in crate::metal) fn prepare_particle_pass(
        &mut self,
        elapsed: f32,
    ) -> Option<(f32, u32, Vec<u32>)> {
        self.particle.pipelines.as_ref()?;
        if self.particle.records.is_empty() || self.particle.emitter_state.is_empty() {
            return None;
        }
        let dt = (elapsed - self.particle.last_elapsed).max(0.0);
        self.particle.last_elapsed = elapsed;
        self.particle.frame_index = self.particle.frame_index.wrapping_add(1);
        let frame_index = self.particle.frame_index;
        let mut budgets = Vec::with_capacity(self.particle.records.len());
        for (rec_slot, gpu_slot) in self
            .particle
            .records
            .iter()
            .zip(self.particle.emitter_state.iter_mut())
        {
            let budget = match (rec_slot.as_ref(), gpu_slot.as_mut()) {
                (Some(rec), Some(gpu)) => {
                    let spawn = gpu
                        .spawn_state
                        .take_budget(dt, rec.spawn_rate, rec.max_particles);
                    // Reset the atomic counter to this frame's budget.
                    // Shared storage means the kernel sees the write
                    // immediately.
                    unsafe {
                        let dst = gpu.spawn_counter.contents().as_ptr() as *mut u32;
                        dst.write(spawn);
                    }
                    spawn
                }
                _ => 0,
            };
            budgets.push(budget);
        }
        Some((dt, frame_index, budgets))
    }

    pub(in crate::metal) fn encode_particles(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        dt: f32,
        frame_index: u32,
        spawn_budgets: &[u32],
        vp: [[f32; 4]; 4],
        frustum: &crate::gfx::frustum::Frustum,
    ) -> Result<u32, String> {
        let Some(pipelines) = self.particle.pipelines.as_ref() else {
            return Ok(0);
        };
        if self.particle.records.is_empty() || self.particle.emitter_state.is_empty() {
            return Ok(0);
        }
        let last_tex = self.textures.len().saturating_sub(1);

        // Visibility-cull per emitter for the *render* pass only. The compute
        // simulation still ticks every pool so particles spawn / age / die
        // while the camera looks away: that way the emitter is in a
        // realistic mid-life state the moment the camera turns back. The
        // compute cost is per-slot work in a single threadgroup, so leaving
        // it un-culled is cheap. Tombstoned (None) slots are always invisible.
        let visible: Vec<bool> = self
            .particle
            .records
            .iter()
            .map(|slot| match slot {
                Some(r) => {
                    let (mn, mx) = r.aabb();
                    frustum.intersects_aabb(mn, mx)
                }
                None => false,
            })
            .collect();

        // Camera basis for camera-facing billboards: rows 0 and 1 of the view
        // matrix's 3×3 are the world-space right and up vectors (the view
        // matrix is column-major, so we read those rows out element-wise).
        let v = self.view_matrix;
        let cam_right = [v[0][0], v[1][0], v[2][0]];
        let cam_up = [v[0][1], v[1][1], v[2][1]];
        let view = ParticleView {
            vp,
            cam_right,
            _pad0: 0.0,
            cam_up,
            _pad1: 0.0,
        };

        // Compute: age + integrate + respawn each pool in turn. One
        // dispatch per emitter; cheap enough to not bother packing them.
        {
            let sim_desc = MTLComputePassDescriptor::new();
            if let Some(t) = &self.pass_timing {
                t.attach_compute(&sim_desc, super::pass_timing::PassId::ParticlesSim);
            }
            // Guard drops at the end of this block, ending the compute pass
            // before the render encoder below opens.
            let enc = ScopedEncoder::new(
                cmd_buf
                    .computeCommandEncoderWithDescriptor(&sim_desc)
                    .ok_or("failed to get particle compute encoder")?,
                "particles: simulate",
            );
            enc.setComputePipelineState(&pipelines.simulate);
            for (i, (rec_slot, gpu_slot)) in self
                .particle
                .records
                .iter()
                .zip(self.particle.emitter_state.iter())
                .enumerate()
            {
                let (rec, gpu) = match (rec_slot.as_ref(), gpu_slot.as_ref()) {
                    (Some(r), Some(g)) => (r, g),
                    _ => continue,
                };
                let spawn_budget = spawn_budgets.get(i).copied().unwrap_or(0);
                let params = rec.params(dt, spawn_budget, frame_index);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(gpu.pool.as_ref()), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(gpu.spawn_counter.as_ref()), 0, 1);
                    enc.setBytes_length_atIndex(
                        std::ptr::NonNull::from(&params).cast(),
                        std::mem::size_of::<ParticleParams>(),
                        2,
                    );
                }
                let grid = MTLSize {
                    width: rec.max_particles as usize,
                    height: 1,
                    depth: 1,
                };
                // 64-thread groups: a multiple of the SIMD width on every Apple
                // GPU since A11 and small enough that a thin pool still
                // dispatches efficiently.
                let tg = MTLSize {
                    width: 64,
                    height: 1,
                    depth: 1,
                };
                enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            }
        }

        // Render: one alpha-blended quad per live particle, drawn into
        // `hdr_resolve`. Caller has already ended the previous render
        // pass (fog), so we open a fresh Load/Store pass here. When every
        // emitter culls out we skip the render encoder entirely.
        if !visible.iter().any(|v| *v) {
            return Ok(0);
        }
        let pass_desc = MTLRenderPassDescriptor::new();
        unsafe {
            let ca = pass_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.hdr_targets.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::Store);
        }

        if let Some(t) = &self.pass_timing {
            t.attach_render(&pass_desc, super::pass_timing::PassId::ParticlesDraw);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or("failed to get particle render encoder")?,
            "particles: draw",
        );
        enc.setRenderPipelineState(&pipelines.render);
        unsafe {
            enc.setVertexBytes_length_atIndex(
                std::ptr::NonNull::from(&view).cast(),
                std::mem::size_of::<ParticleView>(),
                1,
            );
            enc.setFragmentSamplerState_atIndex(Some(&pipelines.sampler), 0);
        }

        let mut draw_calls: u32 = 0;
        for (i, (rec_slot, gpu_slot)) in self
            .particle
            .records
            .iter()
            .zip(self.particle.emitter_state.iter())
            .enumerate()
        {
            if !visible[i] {
                continue;
            }
            let (rec, gpu) = match (rec_slot.as_ref(), gpu_slot.as_ref()) {
                (Some(r), Some(g)) => (r, g),
                _ => continue,
            };
            // Spawn budget and frame seed only matter to the compute kernel,
            // but we share the uniform layout so the render path passes its
            // own zero-budget copy. `dt` is irrelevant to the vertex shader
            // (it reads `age` / `lifetime` straight from the pool).
            let params = rec.params(0.0, 0, frame_index);
            let slot = rec.texture_slot.min(last_tex);
            unsafe {
                enc.setVertexBuffer_offset_atIndex(Some(gpu.pool.as_ref()), 0, 0);
                enc.setVertexBytes_length_atIndex(
                    std::ptr::NonNull::from(&params).cast(),
                    std::mem::size_of::<ParticleParams>(),
                    2,
                );
                enc.setFragmentTexture_atIndex(Some(self.textures[slot].as_ref()), 0);
                enc.drawPrimitives_vertexStart_vertexCount_instanceCount(
                    MTLPrimitiveType::TriangleStrip,
                    0,
                    4,
                    rec.max_particles as usize,
                );
            }
            draw_calls += 1;
        }

        Ok(draw_calls)
    }
}

// Build the particle compute + render pipelines plus the shared sampler.
// Returned only when the world declares at least one `ParticleEmitter`.
pub(super) fn build_particle_pipelines(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<ParticlePipelines, String> {
    let msl = shader_source(hot_reload, "particle.metal");
    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(msl.as_ref()), Some(&options))
        .map_err(|e| format!("particle shader compile error: {:?}", e))?;

    // Compute kernel.
    let sim_fn = library
        .newFunctionWithName(&ns_str("particle_simulate"))
        .ok_or("particle_simulate not found")?;
    let simulate = device
        .newComputePipelineStateWithFunction_error(&sim_fn)
        .map_err(|e| format!("failed to create particle_simulate pipeline: {:?}", e))?;

    // Render pipeline. No vertex descriptor: the vertex shader reads from the
    // particle pool storage buffer directly via `[[vertex_id]]` + `[[instance_id]]`.
    let vert_fn = library
        .newFunctionWithName(&ns_str("particle_vertex"))
        .ok_or("particle_vertex not found")?;
    let frag_fn = library
        .newFunctionWithName(&ns_str("particle_fragment"))
        .ok_or("particle_fragment not found")?;
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(1);
    unsafe {
        let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca.setPixelFormat(MTLPixelFormat::RGBA16Float);
        ca.setBlendingEnabled(true);
        ca.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca.setSourceAlphaBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    }
    let render = device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| format!("failed to create particle render pipeline: {:?}", e))?;

    // Sampler: linear-clamp, same envelope the decal pass uses.
    let sampler = {
        let sdesc = MTLSamplerDescriptor::new();
        sdesc.setMinFilter(MTLSamplerMinMagFilter::Linear);
        sdesc.setMagFilter(MTLSamplerMinMagFilter::Linear);
        sdesc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        sdesc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        device
            .newSamplerStateWithDescriptor(&sdesc)
            .ok_or("failed to create particle sampler state")?
    };

    Ok(ParticlePipelines {
        simulate,
        render,
        sampler,
    })
}

// Allocate the per-emitter GPU state for one record: a zero-initialised
// particle pool plus a one-`u32` atomic counter buffer. Both buffers use
// shared storage so the CPU can reset the spawn counter each frame without
// a staging copy.
pub(super) fn build_emitter_gpu_state(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    record: &ParticleEmitterRecord,
) -> Result<ParticleEmitterGpuState, String> {
    let slots = record.max_particles as usize;
    let pool_bytes = slots * std::mem::size_of::<GpuParticle>();
    let pool = device
        .newBufferWithLength_options(pool_bytes, MTLResourceOptions::StorageModeShared)
        .ok_or("failed to allocate particle pool buffer")?;
    // Zero-init: every slot starts dead (`lifetime = 0`).
    unsafe {
        let dst = pool.contents().as_ptr() as *mut u8;
        std::ptr::write_bytes(dst, 0, pool_bytes);
    }

    let spawn_counter = device
        .newBufferWithLength_options(
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or("failed to allocate particle spawn counter")?;
    unsafe {
        let dst = spawn_counter.contents().as_ptr() as *mut u32;
        dst.write(0);
    }

    Ok(ParticleEmitterGpuState {
        pool,
        spawn_counter,
        spawn_state: ParticleSpawnState::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_particle_layout_matches_msl() {
        // Mirrors the `Particle` struct in `shaders/particle.metal`:
        // packed_float3 + float, twice = 32 bytes, layout 0/12/16/28.
        assert_eq!(std::mem::size_of::<GpuParticle>(), 32);
        assert_eq!(std::mem::offset_of!(GpuParticle, position), 0);
        assert_eq!(std::mem::offset_of!(GpuParticle, age), 12);
        assert_eq!(std::mem::offset_of!(GpuParticle, velocity), 16);
        assert_eq!(std::mem::offset_of!(GpuParticle, lifetime), 28);
    }
}
