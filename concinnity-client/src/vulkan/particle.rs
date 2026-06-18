// src/vulkan/particle.rs
//
// GPU-compute particle system for the Vulkan backend. Each `ParticleEmitter`
// declared in the world produces one persistent `ParticleEmitterGpuState`
// carrying a device-local pool SSBO (read-write in the compute pass, read-only
// in the vertex pass) and a device-local 4-byte atomic spawn-counter SSBO.
// Each frame the renderer:
//
//   1. Computes the per-emitter spawn budget CPU-side (a fractional
//      accumulator drives integer particle spawns per dispatch).
//   2. Writes that budget into the per-emitter counter buffer via a
//      `vkCmdUpdateBuffer` (the value fits in the inline-update 64 KiB cap).
//   3. Dispatches the `particle_simulate` compute kernel to age + integrate +
//      respawn each pool.
//   4. Transitions visible pools to SHADER_READ for the vertex stage and
//      rasterises one alpha-blended billboard quad per live particle into
//      `hdr_resolve_images[frame_idx]`.
//
// Runs after the volumetric-fog pass and before SSR / TAA so particles
// appear in screen-space reflections and are temporally stabilised by the
// TAA history. Mirrors src/directx/particle.rs and src/metal/particle.rs.

use std::cell::Cell;
use std::ffi::CString;

use ash::{Device, vk};

use crate::gfx::particles::{ParticleEmitterRecord, ParticleSpawnState};
use crate::gfx::render_types::ParticleParams;

use super::context::{HDR_FORMAT, VkContext};
use super::pipeline::{compile_glsl, shader_source, spv_module};
use super::texture::create_buffer;

// GLSL sources, shared with the host so the hot-reload pass can pick them
// up the same way the existing built-in shaders do.
pub(in crate::vulkan) const PARTICLE_SIMULATE_GLSL: &str =
    include_str!("shaders/particle_simulate.comp");
pub(in crate::vulkan) const PARTICLE_VERT_GLSL: &str = include_str!("shaders/particle.vert");
pub(in crate::vulkan) const PARTICLE_FRAG_GLSL: &str = include_str!("shaders/particle.frag");

// Cap on the number of simultaneously-live particle emitters. The
// per-emitter descriptor pool reserves a fixed block of `2 * MAX_EMITTERS`
// sets at init (one compute set + one render set per emitter), so runtime
// `add_emitter` past this many returns an error. Matches the Metal /
// DirectX cap.
pub(in crate::vulkan) const MAX_EMITTERS: usize = 256;

// One particle slot on the GPU. Layout must match the `Particle` GLSL
// struct in `shaders/particle_simulate.comp` and `shaders/particle.vert`
// (32 bytes per slot; std430 packs vec3+float into 16 bytes).
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct GpuParticle {
    position: [f32; 3],
    age: f32,
    velocity: [f32; 3],
    lifetime: f32,
}

// Per-frame view inputs to the particle render pass. Mirrors the
// `ParticleView` uniform block in `particle.vert` (96 bytes: mat4 + two
// (vec3, float) slots; std140 packs vec3+float into a single 16-byte block).
#[derive(Copy, Clone)]
#[repr(C)]
struct ParticleView {
    vp: [[f32; 4]; 4],
    cam_right: [f32; 3],
    _pad0: f32,
    cam_up: [f32; 3],
    _pad1: f32,
}

// Compile the particle compute + vertex + fragment shaders to SPIR-V. Used
// by [`ParticleResources::new`] at init and by shader hot-reload to rebuild
// the two pipelines against the existing layouts.
#[allow(clippy::type_complexity)]
pub(in crate::vulkan) fn compile_particle_shaders(
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), String> {
    let cs_src = shader_source(hot_reload, "particle_simulate.comp", PARTICLE_SIMULATE_GLSL);
    let vs_src = shader_source(hot_reload, "particle.vert", PARTICLE_VERT_GLSL);
    let fs_src = shader_source(hot_reload, "particle.frag", PARTICLE_FRAG_GLSL);
    let cs = compile_glsl(
        &cs_src,
        shaderc::ShaderKind::Compute,
        "particle_simulate.comp",
    )?;
    let vs = compile_glsl(&vs_src, shaderc::ShaderKind::Vertex, "particle.vert")?;
    let fs = compile_glsl(&fs_src, shaderc::ShaderKind::Fragment, "particle.frag")?;
    Ok((cs, vs, fs))
}

// Per-emitter persistent GPU state: the particle pool, the atomic spawn
// counter, the CPU-side fractional spawn accumulator, and the descriptor
// sets that bind them. Pool + counter sit in DEVICE_LOCAL memory; both
// rest in the same access state across frames (the encoder flips the
// pool's barrier between the compute write and the vertex read).
pub(in crate::vulkan) struct ParticleEmitterGpuState {
    // Particle pool: `record.max_particles` slots of `GpuParticle`. Used
    // as a storage buffer by both the compute pass and the vertex pass.
    pub pool_buffer: vk::Buffer,
    pub pool_memory: vk::DeviceMemory,
    // Pool size in bytes. Kept around so a future hot-reload that
    // resizes a live emitter's pool can reuse the descriptor write
    // helper (`write_compute_set`) with the new range.
    #[allow(dead_code)]
    pub pool_bytes: u64,
    // One u32 atomic counter (4 bytes). Reset to the integer spawn budget
    // each frame via `vkCmdUpdateBuffer`; decremented by the compute
    // kernel as threads claim spawn slots.
    pub counter_buffer: vk::Buffer,
    pub counter_memory: vk::DeviceMemory,
    // Carry-over fractional spawn count. Combined with `dt` and the
    // emitter's `spawn_rate` to produce the integer spawn budget for each
    // dispatch. Interior-mutable so `encode_particles` (which is reached
    // through `&self` from the graph executor) can advance it without
    // taking `&mut self`.
    pub spawn_state: Cell<ParticleSpawnState>,
    // Compute descriptor set (set 0): binding 0 the pool SSBO, binding 1
    // the counter SSBO. Allocated from the particle descriptor pool at
    // emitter creation and re-pointed on a future pool/counter swap (none
    // today; emitters keep their pool for the emitter's whole lifetime).
    pub compute_set: vk::DescriptorSet,
    // Render emitter descriptor set (set 1): binding 0 the pool SSBO
    // (read-only here), binding 1 the emitter's albedo combined image
    // sampler. The albedo binding is rewritten by [`VkContext::add_emitter`]
    // from the live texture pool.
    pub render_set: vk::DescriptorSet,
    // Texture-pool slot last written into `render_set`'s albedo binding.
    // Read by `rewrite_particle_albedo_slot` so a streamed or hot-reloaded
    // albedo swap that recreates this slot's view re-points the binding.
    pub texture_slot: usize,
}

impl ParticleEmitterGpuState {
    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_buffer(self.pool_buffer, None);
            device.free_memory(self.pool_memory, None);
            device.destroy_buffer(self.counter_buffer, None);
            device.free_memory(self.counter_memory, None);
        }
    }
}

// Pipelines + per-frame view uniform ring + per-emitter descriptor pool
// shared across every emitter. Owned by `VkContext` at most once; built
// either at init (when the world declares ≥1 emitter) or on the first
// runtime `add_emitter`.
pub(in crate::vulkan) struct ParticleResources {
    // Compute pass: particle_simulate.comp.
    pub(in crate::vulkan) compute_pipeline: vk::Pipeline,
    pub(in crate::vulkan) compute_pipeline_layout: vk::PipelineLayout,
    // set 0: (pool SSBO, counter SSBO) per emitter.
    pub(in crate::vulkan) compute_set_layout: vk::DescriptorSetLayout,

    // Render pass: particle.vert / particle.frag.
    pub(in crate::vulkan) render_pass: vk::RenderPass,
    pub(in crate::vulkan) render_pipeline: vk::Pipeline,
    pub(in crate::vulkan) render_pipeline_layout: vk::PipelineLayout,
    // set 0: per-frame ParticleView UBO. Single binding (binding 0).
    pub(in crate::vulkan) view_set_layout: vk::DescriptorSetLayout,
    // set 1: per-emitter (pool SSBO, albedo). Allocated for each
    // `ParticleEmitterGpuState` from `descriptor_pool` and written by
    // `add_emitter`.
    pub(in crate::vulkan) emitter_set_layout: vk::DescriptorSetLayout,

    // Per-emitter descriptor pool. Holds `MAX_EMITTERS` compute sets +
    // `MAX_EMITTERS` render emitter sets + `frames` view sets. Sized at
    // init; runtime `add_emitter` past the cap returns an error.
    pub(in crate::vulkan) descriptor_pool: vk::DescriptorPool,

    // Per-frame view UBO (single 96-byte block), persistently mapped.
    pub(in crate::vulkan) view_ubos: Vec<vk::Buffer>,
    pub(in crate::vulkan) view_ubo_memories: Vec<vk::DeviceMemory>,
    pub(in crate::vulkan) view_ubo_ptrs: Vec<*mut u8>,
    // Per-frame view set (binding 0 = view UBO). One per frame slot.
    pub(in crate::vulkan) view_sets: Vec<vk::DescriptorSet>,

    // One framebuffer per frame-in-flight slot, each binding its frame
    // slot's `hdr_resolve_images[i].view` as the sole colour attachment.
    pub(in crate::vulkan) framebuffers: Vec<vk::Framebuffer>,

    // Linear-clamp sampler shared by every emitter's albedo binding.
    pub(in crate::vulkan) sampler: vk::Sampler,
}

impl ParticleResources {
    // Build the particle compute + render pipelines, the per-frame view
    // UBO ring, the shared sampler, the descriptor pool, and the per-frame
    // framebuffers. Called from `VkContext::new` only when the world
    // declared at least one `ParticleEmitter`. The encoder is a no-op
    // when this is `None`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vulkan) fn new(
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        frames: usize,
        hdr_resolve_views: &[vk::ImageView],
        extent: vk::Extent2D,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let render_pass = create_render_pass(device, HDR_FORMAT)?;
        let compute_set_layout = create_compute_set_layout(device)?;
        let (view_set_layout, emitter_set_layout) = create_render_set_layouts(device)?;
        let compute_pipeline_layout = create_compute_pipeline_layout(device, compute_set_layout)?;
        let render_pipeline_layout =
            create_render_pipeline_layout(device, view_set_layout, emitter_set_layout)?;

        let (cs_spv, vs_spv, fs_spv) = compile_particle_shaders(hot_reload)?;
        let compute_pipeline = create_compute_pipeline(device, compute_pipeline_layout, &cs_spv)?;
        let render_pipeline = create_render_pipeline(
            device,
            render_pass,
            render_pipeline_layout,
            &vs_spv,
            &fs_spv,
        )?;

        // Per-frame ParticleView UBOs (HOST_VISIBLE | HOST_COHERENT,
        // persistently mapped).
        let view_size = std::mem::size_of::<ParticleView>() as u64;
        let mut view_ubos = Vec::with_capacity(frames);
        let mut view_ubo_memories = Vec::with_capacity(frames);
        let mut view_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let (buf, mem) = create_buffer(
                instance,
                device,
                physical_device,
                view_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let ptr = unsafe { device.map_memory(mem, 0, view_size, vk::MemoryMapFlags::empty()) }
                .map_err(|e| format!("map particle view ubo: {e}"))?
                as *mut u8;
            view_ubos.push(buf);
            view_ubo_memories.push(mem);
            view_ubo_ptrs.push(ptr);
        }

        let sampler = create_sampler(device)?;
        let descriptor_pool = create_descriptor_pool(device, frames)?;

        // Per-frame view sets (one per frame slot).
        let view_layouts: Vec<_> = (0..frames).map(|_| view_set_layout).collect();
        let view_sets = alloc_descriptor_sets(device, descriptor_pool, &view_layouts)?;
        for (i, &set) in view_sets.iter().enumerate() {
            write_view_set(device, set, view_ubos[i]);
        }

        // Per-frame framebuffers (one per frame slot binding that slot's
        // hdr_resolve view as the colour attachment).
        let mut framebuffers = Vec::with_capacity(frames);
        for &view in hdr_resolve_views.iter().take(frames) {
            let attachments = [view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width.max(1))
                .height(extent.height.max(1))
                .layers(1);
            let fb = unsafe { device.create_framebuffer(&fb_info, None) }
                .map_err(|e| format!("particle framebuffer: {e}"))?;
            framebuffers.push(fb);
        }

        Ok(Self {
            compute_pipeline,
            compute_pipeline_layout,
            compute_set_layout,
            render_pass,
            render_pipeline,
            render_pipeline_layout,
            view_set_layout,
            emitter_set_layout,
            descriptor_pool,
            view_ubos,
            view_ubo_memories,
            view_ubo_ptrs,
            view_sets,
            framebuffers,
            sampler,
        })
    }

    // Rebuild the framebuffers after a swapchain resize. Called from
    // `VkContext::rebuild_swapchain`; same pattern as `FogResources` /
    // `DecalResources`. The pipelines, layouts, buffers, sampler, and
    // per-emitter descriptor sets all survive.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        device: &Device,
        hdr_resolve_views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Result<(), String> {
        for &fb in &self.framebuffers {
            unsafe { device.destroy_framebuffer(fb, None) };
        }
        self.framebuffers.clear();
        for &view in hdr_resolve_views.iter().take(self.view_ubos.len()) {
            let attachments = [view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(self.render_pass)
                .attachments(&attachments)
                .width(extent.width.max(1))
                .height(extent.height.max(1))
                .layers(1);
            let fb = unsafe { device.create_framebuffer(&fb_info, None) }
                .map_err(|e| format!("particle framebuffer (rebuild): {e}"))?;
            self.framebuffers.push(fb);
        }
        Ok(())
    }

    // Construct the compute + render pipelines against the existing
    // layouts. Used by the shader hot-reload pass.
    pub(in crate::vulkan) fn rebuild_pipelines(
        &self,
        device: &Device,
        hot_reload: bool,
    ) -> Result<(vk::Pipeline, vk::Pipeline), String> {
        let (cs_spv, vs_spv, fs_spv) = compile_particle_shaders(hot_reload)?;
        let cp = create_compute_pipeline(device, self.compute_pipeline_layout, &cs_spv)?;
        let rp = create_render_pipeline(
            device,
            self.render_pass,
            self.render_pipeline_layout,
            &vs_spv,
            &fs_spv,
        )?;
        Ok((cp, rp))
    }

    // Swap the freshly-built pipelines in. The caller has already
    // `device_wait_idle`'d so the old pipelines are not in flight.
    pub(in crate::vulkan) fn swap_pipelines(
        &mut self,
        device: &Device,
        compute: vk::Pipeline,
        render: vk::Pipeline,
    ) {
        unsafe {
            device.destroy_pipeline(self.compute_pipeline, None);
            device.destroy_pipeline(self.render_pipeline, None);
        }
        self.compute_pipeline = compute;
        self.render_pipeline = render;
    }

    // Free every owned handle. Called from `Drop for VkContext` after
    // `device_wait_idle`. Per-emitter pools + counters live in
    // `VkContext::particle_emitter_state`; their destruction is the
    // caller's responsibility.
    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        unsafe {
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            for (&buf, &mem) in self.view_ubos.iter().zip(self.view_ubo_memories.iter()) {
                device.unmap_memory(mem);
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
            device.destroy_sampler(self.sampler, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.compute_pipeline, None);
            device.destroy_pipeline(self.render_pipeline, None);
            device.destroy_pipeline_layout(self.compute_pipeline_layout, None);
            device.destroy_pipeline_layout(self.render_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.compute_set_layout, None);
            device.destroy_descriptor_set_layout(self.view_set_layout, None);
            device.destroy_descriptor_set_layout(self.emitter_set_layout, None);
            device.destroy_render_pass(self.render_pass, None);
        }
        self.framebuffers.clear();
        self.view_ubos.clear();
        self.view_ubo_memories.clear();
        self.view_ubo_ptrs.clear();
    }
}

// Allocate the per-emitter GPU state: a zero-initialised pool SSBO and a
// 4-byte atomic spawn counter SSBO, both DEVICE_LOCAL. Also allocates the
// emitter's compute + render descriptor sets and writes the pool/counter
// bindings. The albedo binding stays unwritten; `add_emitter` writes it
// from the live texture pool.
#[allow(clippy::too_many_arguments)]
pub(in crate::vulkan) fn build_emitter_gpu_state(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    resources: &ParticleResources,
    record: &ParticleEmitterRecord,
) -> Result<ParticleEmitterGpuState, String> {
    let slots = record.max_particles as u64;
    let pool_bytes = slots * std::mem::size_of::<GpuParticle>() as u64;

    // Pool buffer: DEVICE_LOCAL, used as STORAGE by both passes. The
    // compute kernel writes through it; the vertex stage reads it. A WAR
    // barrier in the encoder transitions accesses between dispatches.
    let (pool_buffer, pool_memory) = create_buffer(
        instance,
        device,
        physical_device,
        pool_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    zero_device_buffer(
        instance,
        device,
        physical_device,
        command_pool,
        queue,
        pool_buffer,
        pool_bytes,
    )?;

    // Counter buffer: DEVICE_LOCAL, 4 bytes, used as STORAGE by the
    // compute kernel and TRANSFER_DST for the per-frame
    // `vkCmdUpdateBuffer` that resets it to the integer budget.
    let counter_bytes = std::mem::size_of::<u32>() as u64;
    let (counter_buffer, counter_memory) = create_buffer(
        instance,
        device,
        physical_device,
        counter_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    zero_device_buffer(
        instance,
        device,
        physical_device,
        command_pool,
        queue,
        counter_buffer,
        counter_bytes,
    )?;

    // Allocate the (compute, render) descriptor set pair.
    let set_layouts = [resources.compute_set_layout, resources.emitter_set_layout];
    let sets = alloc_descriptor_sets(device, resources.descriptor_pool, &set_layouts)?;
    let compute_set = sets[0];
    let render_set = sets[1];

    // Write the pool + counter bindings on the compute set (set 0).
    write_compute_set(device, compute_set, pool_buffer, pool_bytes, counter_buffer);
    // Write the pool binding on the render set (set 1, binding 0). The
    // albedo binding (set 1, binding 1) is written by `add_emitter` from
    // the live texture pool.
    write_render_pool_binding(device, render_set, pool_buffer, pool_bytes);

    Ok(ParticleEmitterGpuState {
        pool_buffer,
        pool_memory,
        pool_bytes,
        counter_buffer,
        counter_memory,
        spawn_state: Cell::new(ParticleSpawnState::default()),
        compute_set,
        render_set,
        texture_slot: usize::MAX,
    })
}

// Render pass / descriptor / pipeline construction

fn create_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass, String> {
    // One colour attachment: the resolved HDR scene. The fog pass left
    // it in SHADER_READ_ONLY_OPTIMAL; we want it in COLOR_ATTACHMENT
    // during the subpass and SHADER_READ_ONLY_OPTIMAL again on exit so
    // SSR / TAA / bloom / composite can sample it. Mirrors the decal /
    // fog render passes.
    let attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));
    let dep_in = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_READ,
        );
    let dep_out = vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    let deps = [dep_in, dep_out];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&deps);
    unsafe { device.create_render_pass(&info, None) }
        .map_err(|e| format!("particle render pass: {e}"))
}

fn create_compute_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("particle compute set layout: {e}"))
}

fn create_render_set_layouts(
    device: &Device,
) -> Result<(vk::DescriptorSetLayout, vk::DescriptorSetLayout), String> {
    // set 0: per-frame ParticleView UBO. Vertex stage only.
    let view_bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX)];
    let view_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&view_bindings);
    let view_set_layout = unsafe { device.create_descriptor_set_layout(&view_info, None) }
        .map_err(|e| format!("particle view set layout: {e}"))?;

    // set 1: per-emitter (pool SSBO, albedo).
    let emitter_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let emitter_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&emitter_bindings);
    let emitter_set_layout = unsafe { device.create_descriptor_set_layout(&emitter_info, None) }
        .map_err(|e| format!("particle emitter set layout: {e}"))?;
    Ok((view_set_layout, emitter_set_layout))
}

// Push-constant range covering the full 112-byte `ParticleParams` block.
// Visible to vertex (size_start/end, color_start/end) + fragment (none:
// vertex emits the colour; fragment reads it via varyings) + compute
// (every field). The vertex stage actually only reads the gradient + size
// fields, but binding the full struct keeps the host upload single-shot.
const PARTICLE_PUSH_BYTES: u32 = 112;

fn create_compute_pipeline_layout(
    device: &Device,
    compute_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, String> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(PARTICLE_PUSH_BYTES);
    let set_layouts = [compute_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push_range));
    unsafe { device.create_pipeline_layout(&info, None) }
        .map_err(|e| format!("particle compute pipeline layout: {e}"))
}

fn create_render_pipeline_layout(
    device: &Device,
    view_set_layout: vk::DescriptorSetLayout,
    emitter_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, String> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(PARTICLE_PUSH_BYTES);
    let set_layouts = [view_set_layout, emitter_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push_range));
    unsafe { device.create_pipeline_layout(&info, None) }
        .map_err(|e| format!("particle render pipeline layout: {e}"))
}

fn create_descriptor_pool(device: &Device, frames: usize) -> Result<vk::DescriptorPool, String> {
    let frames = frames as u32;
    let max_emitters = MAX_EMITTERS as u32;
    // Pool sizing:
    //   - UNIFORM_BUFFER: `frames` (one ParticleView UBO per frame slot)
    //   - STORAGE_BUFFER: `2 * MAX_EMITTERS` for compute (pool + counter)
    //                     + `MAX_EMITTERS` for render (pool, read-only)
    //   - COMBINED_IMAGE_SAMPLER: `MAX_EMITTERS` (one albedo per emitter)
    let sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: frames,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3 * max_emitters,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: max_emitters,
        },
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(frames + 2 * max_emitters)
        .pool_sizes(&sizes);
    unsafe { device.create_descriptor_pool(&info, None) }
        .map_err(|e| format!("particle descriptor pool: {e}"))
}

fn alloc_descriptor_sets(
    device: &Device,
    pool: vk::DescriptorPool,
    layouts: &[vk::DescriptorSetLayout],
) -> Result<Vec<vk::DescriptorSet>, String> {
    let info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(layouts);
    unsafe { device.allocate_descriptor_sets(&info) }
        .map_err(|e| format!("particle descriptor sets: {e}"))
}

fn write_view_set(device: &Device, set: vk::DescriptorSet, view_ubo: vk::Buffer) {
    let info = vk::DescriptorBufferInfo::default()
        .buffer(view_ubo)
        .offset(0)
        .range(std::mem::size_of::<ParticleView>() as u64);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

fn write_compute_set(
    device: &Device,
    set: vk::DescriptorSet,
    pool_buffer: vk::Buffer,
    pool_bytes: u64,
    counter_buffer: vk::Buffer,
) {
    let pool_info = vk::DescriptorBufferInfo::default()
        .buffer(pool_buffer)
        .offset(0)
        .range(pool_bytes);
    let counter_info = vk::DescriptorBufferInfo::default()
        .buffer(counter_buffer)
        .offset(0)
        .range(std::mem::size_of::<u32>() as u64);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&pool_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&counter_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn write_render_pool_binding(
    device: &Device,
    set: vk::DescriptorSet,
    pool_buffer: vk::Buffer,
    pool_bytes: u64,
) {
    let info = vk::DescriptorBufferInfo::default()
        .buffer(pool_buffer)
        .offset(0)
        .range(pool_bytes);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

fn create_sampler(device: &Device) -> Result<vk::Sampler, String> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
        .max_lod(vk::LOD_CLAMP_NONE);
    unsafe { device.create_sampler(&info, None) }.map_err(|e| format!("particle sampler: {e}"))
}

fn create_compute_pipeline(
    device: &Device,
    layout: vk::PipelineLayout,
    spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let module = spv_module(device, spv)?;
    let entry = CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create particle compute pipeline: {e}"))?[0];
    unsafe { device.destroy_shader_module(module, None) };
    Ok(pipeline)
}

fn create_render_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert = spv_module(device, vert_spv)?;
    let frag = spv_module(device, frag_spv)?;
    let entry = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag)
            .name(&entry),
    ];
    // No vertex buffers: the vertex shader emits the quad from
    // gl_VertexIndex and reads the particle from the pool by
    // gl_InstanceIndex.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false);
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend_attachments = [blend_attachment];
    let blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&blend_state)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass);
    let pipeline = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create particle render pipeline: {e}"))?[0];
    unsafe {
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
    }
    Ok(pipeline)
}

// Zero-initialise a DEVICE_LOCAL buffer by recording a `vkCmdFillBuffer`
// inside a one-shot command buffer. Cheaper than the staging-buffer
// alternative and trivially correct since `vkCmdFillBuffer` writes a
// 32-bit pattern; `bytes` is guaranteed to be a multiple of 4 for both
// the pool (32 bytes per slot) and the counter (4 bytes).
#[allow(clippy::too_many_arguments)]
fn zero_device_buffer(
    _instance: &ash::Instance,
    device: &Device,
    _physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    target: vk::Buffer,
    bytes: u64,
) -> Result<(), String> {
    super::texture::one_shot_submit(device, command_pool, queue, |cmd| {
        unsafe { device.cmd_fill_buffer(cmd, target, 0, bytes, 0) };
    })
}

// Encoder

impl VkContext {
    // Mutating prelude for the particle pass, run on `&mut self` before the
    // render-graph fan-out: advance the frame `dt` (against
    // `particle_last_elapsed`), the monotonic `particle_frame_index`, and each
    // emitter's fractional spawn accumulator, returning the per-frame
    // `(dt, frame_index, per_emitter_spawn_budgets)` the read-only
    // `encode_particles` then consumes. Split out so `encode_particles` can
    // take `&self` and run on a parallel-recording worker without touching the
    // `Cell` state. Returns `None` when the pass is inert (no pipeline / no
    // live emitter). Mirrors `metal::MtlContext::prepare_particle_pass`.
    pub(in crate::vulkan) fn prepare_particle_pass(
        &mut self,
        elapsed: f32,
    ) -> Option<(f32, u32, Vec<u32>)> {
        self.particle_resources.as_ref()?;
        if self.particles.is_empty() || self.particle_emitter_state.is_empty() {
            return None;
        }
        let dt = (elapsed - self.particle_last_elapsed.get()).max(0.0);
        self.particle_last_elapsed.set(elapsed);
        let frame_index = self.particle_frame_index.get().wrapping_add(1);
        self.particle_frame_index.set(frame_index);

        let mut budgets = Vec::with_capacity(self.particles.len());
        for (rec_slot, gpu_slot) in self
            .particles
            .iter()
            .zip(self.particle_emitter_state.iter())
        {
            let budget = match (rec_slot.as_ref(), gpu_slot.as_ref()) {
                (Some(rec), Some(gpu)) => {
                    let mut spawn_state = gpu.spawn_state.get();
                    let b = spawn_state.take_budget(dt, rec.spawn_rate, rec.max_particles);
                    gpu.spawn_state.set(spawn_state);
                    b
                }
                _ => 0,
            };
            budgets.push(budget);
        }
        Some((dt, frame_index, budgets))
    }

    // Encode the per-emitter compute + render passes. A no-op when no
    // pipeline has been built (no emitter has ever existed in this
    // session) or when every slot is tombstoned. `frame` is the
    // `(dt, frame_index, per_emitter_spawn_budgets)` tuple
    // `prepare_particle_pass` computed on `&mut self`; this method takes
    // `&self` (no `Cell` mutation) so it can run on a parallel-recording
    // worker.
    pub(in crate::vulkan) fn encode_particles(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        frame: &(f32, u32, Vec<u32>),
        vp: [[f32; 4]; 4],
        frustum: &crate::gfx::frustum::Frustum,
    ) {
        let Some(resources) = self.particle_resources.as_ref() else {
            return;
        };
        if self.particles.is_empty() || self.particle_emitter_state.is_empty() {
            return;
        }
        let (dt, frame_index, spawn_budgets) = (frame.0, frame.1, frame.2.as_slice());

        let device = &self.device;
        let extent = self.render_extent;

        // Visibility-cull per emitter for the *render* pass only. The
        // compute simulation still ticks every live pool so off-screen
        // emitters stay in a realistic mid-life state when the camera
        // turns back. Tombstoned (None) slots are always invisible.
        let visible: Vec<bool> = self
            .particles
            .iter()
            .map(|slot| match slot {
                Some(r) => {
                    let (mn, mx) = r.aabb();
                    frustum.intersects_aabb(mn, mx)
                }
                None => false,
            })
            .collect();

        // Camera basis for camera-facing billboards: rows 0 and 1 of the
        // view matrix's 3×3 are the world-space right and up vectors (the
        // view matrix is column-major, so we read those rows out
        // element-wise). Mirrors metal/directx particle encoders.
        let v = self.view_matrix;
        let cam_right = [v[0][0], v[1][0], v[2][0]];
        let cam_up = [v[0][1], v[1][1], v[2][1]];
        let view_uni = ParticleView {
            vp,
            cam_right,
            _pad0: 0.0,
            cam_up,
            _pad1: 0.0,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &view_uni as *const ParticleView as *const u8,
                resources.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<ParticleView>(),
            );
        }

        // Per-emitter spawn budget + ParticleParams pre-compute. Each
        // emitter advances its own fractional accumulator and we cache
        // the resulting params so the compute + render loops below can
        // upload the same value (the compute kernel needs the spawn
        // budget; the vertex stage zeroes its copy since it only reads
        // gradient + size fields).
        let mut params_per_emitter: Vec<Option<(ParticleParams, u32)>> =
            Vec::with_capacity(self.particles.len());
        for (i, (rec_slot, gpu_slot)) in self
            .particles
            .iter()
            .zip(self.particle_emitter_state.iter())
            .enumerate()
        {
            let (rec, _gpu) = match (rec_slot.as_ref(), gpu_slot.as_ref()) {
                (Some(r), Some(g)) => (r, g),
                _ => {
                    params_per_emitter.push(None);
                    continue;
                }
            };
            // Spawn budget was advanced on `&mut self` in
            // `prepare_particle_pass`; consume the precomputed value here.
            let spawn_budget = spawn_budgets.get(i).copied().unwrap_or(0);
            let params = rec.params(dt, spawn_budget, frame_index);
            params_per_emitter.push(Some((params, spawn_budget)));
        }

        // Pass 1: counter resets. Each emitter's counter buffer is
        // updated to its integer spawn budget via `vkCmdUpdateBuffer`
        // (a transfer write). A single TRANSFER_WRITE → SHADER_READ
        // barrier between the resets and the dispatch makes the writes
        // visible to the compute kernel.
        for (data, gpu_slot) in params_per_emitter
            .iter()
            .zip(self.particle_emitter_state.iter())
        {
            let (Some((_, spawn_budget)), Some(gpu)) = (data.as_ref(), gpu_slot.as_ref()) else {
                continue;
            };
            // `vkCmdUpdateBuffer` inlines `data` into the command stream
            // (4-byte aligned, ≤ 65536 bytes), perfect for a 4-byte
            // counter reset.
            let bytes = spawn_budget.to_ne_bytes();
            unsafe {
                device.cmd_update_buffer(cmd, gpu.counter_buffer, 0, &bytes);
            }
        }
        // Barrier: TRANSFER_WRITE → SHADER_READ on every emitter's
        // counter so the upcoming compute dispatch sees the fresh value.
        // Use a single global memory barrier (cheaper than per-buffer).
        unsafe {
            let mem_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&mem_barrier),
                &[],
                &[],
            );
        }

        // Pass 2: compute dispatches. One per live emitter; resources
        // are disjoint between emitters so no inter-dispatch barrier is
        // needed.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                resources.compute_pipeline,
            );
        }
        for (i, data) in params_per_emitter.iter().enumerate() {
            let Some((params, _)) = data.as_ref() else {
                continue;
            };
            let Some(gpu) = self.particle_emitter_state[i].as_ref() else {
                continue;
            };
            let Some(rec) = self.particles[i].as_ref() else {
                continue;
            };
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    resources.compute_pipeline_layout,
                    0,
                    std::slice::from_ref(&gpu.compute_set),
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    resources.compute_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    std::slice::from_raw_parts(
                        params as *const ParticleParams as *const u8,
                        PARTICLE_PUSH_BYTES as usize,
                    ),
                );
                let groups = rec.max_particles.div_ceil(64);
                device.cmd_dispatch(cmd, groups, 1, 1);
            }
        }

        // Pass 3: render pass. SHADER_WRITE (compute) → SHADER_READ
        // (vertex) on every visible emitter's pool: pool stays in the
        // same memory but the access kind changes between the dispatch
        // and the draw.
        let any_visible = visible.iter().any(|v| *v);
        if !any_visible {
            return;
        }
        unsafe {
            let mem_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::VERTEX_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&mem_barrier),
                &[],
                &[],
            );
        }

        // Begin the render pass into this frame's framebuffer (which
        // binds the resolved HDR target as colour attachment 0). The
        // render pass declares the round-trip
        // SHADER_READ_ONLY_OPTIMAL → COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
        // via its subpass dependencies, so no explicit image barrier is
        // needed here.
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(resources.render_pass)
            .framebuffer(resources.framebuffers[frame_idx])
            .render_area(vk::Rect2D::default().extent(extent));
        // Negative-height viewport flips clip-space Y to match the main +
        // shadow + decal passes (the engine's `perspective()` produces +Y-up
        // clip coords, OpenGL-style; the Vulkan framebuffer has +Y down, so
        // the flip happens in the viewport). The fog pass dodges this with
        // a positive-height viewport because it emits NDC-space verts
        // directly; we MVP-transform world geometry, so we need the same
        // convention as the main pass.
        let viewport = vk::Viewport {
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
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                resources.render_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                resources.render_pipeline_layout,
                0,
                std::slice::from_ref(&resources.view_sets[frame_idx]),
                &[],
            );
        }

        for (i, data) in params_per_emitter.iter().enumerate() {
            if !visible[i] {
                continue;
            }
            let Some((params, _)) = data.as_ref() else {
                continue;
            };
            let Some(gpu) = self.particle_emitter_state[i].as_ref() else {
                continue;
            };
            let Some(rec) = self.particles[i].as_ref() else {
                continue;
            };
            // Vertex stage reads only gradient + size fields; sending the
            // full struct keeps the push-constant range the same shape
            // across compute + render.
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    resources.render_pipeline_layout,
                    1,
                    std::slice::from_ref(&gpu.render_set),
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    resources.render_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    std::slice::from_raw_parts(
                        params as *const ParticleParams as *const u8,
                        PARTICLE_PUSH_BYTES as usize,
                    ),
                );
                device.cmd_draw(cmd, 4, rec.max_particles, 0, 0);
            }
            self.inc_draw_calls(1);
        }
        unsafe {
            device.cmd_end_render_pass(cmd);
        }
    }
}

// Runtime mutation (RenderBackend::add_emitter / remove_emitter)

impl VkContext {
    // Append a runtime emitter. Builds the particle pipelines + per-frame
    // uniform ring on first use (matching the init-time path) so a world
    // that never declared an emitter pays zero pipeline cost until the
    // first add. Reuses tombstoned slots from a prior `remove_emitter`
    // before growing the vec.
    pub(in crate::vulkan) fn add_particle_emitter(
        &mut self,
        record: ParticleEmitterRecord,
    ) -> Result<usize, String> {
        if self.particle_resources.is_none() {
            let hdr_resolve_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            let resources = ParticleResources::new(
                &self.instance,
                &self.device,
                self.physical_device,
                self.frames_in_flight,
                &hdr_resolve_views,
                self.render_extent,
                self.hot_reload,
            )?;
            self.particle_resources = Some(resources);
        }

        // Reuse a tombstoned slot if available; otherwise grow the vec.
        // The cap check is independent of slot availability.
        let live_count = self.particles.iter().filter(|s| s.is_some()).count();
        if live_count >= MAX_EMITTERS {
            return Err(format!(
                "add_emitter: MAX_EMITTERS ({MAX_EMITTERS}) exceeded"
            ));
        }

        let gpu_state = build_emitter_gpu_state(
            &self.instance,
            &self.device,
            self.physical_device,
            self.commands.command_pool,
            self.graphics_queue,
            self.particle_resources.as_ref().unwrap(),
            &record,
        )?;

        // Write the albedo binding from the live texture pool.
        let last_tex = self.textures.len().saturating_sub(1);
        let tex_idx = record.texture_slot.min(last_tex);
        let sampler = self.particle_resources.as_ref().unwrap().sampler;
        write_render_albedo_binding(
            &self.device,
            gpu_state.render_set,
            self.textures[tex_idx].view,
            sampler,
        );

        let id = if let Some(slot) = self.particle_free_slots.pop() {
            // Slot recycle: destroy any leftover state (none today,
            // since `remove_emitter` already destroyed it) and overwrite.
            self.particles[slot] = Some(record);
            let new_state = ParticleEmitterGpuState {
                texture_slot: tex_idx,
                ..gpu_state
            };
            self.particle_emitter_state[slot] = Some(new_state);
            slot
        } else {
            let new_state = ParticleEmitterGpuState {
                texture_slot: tex_idx,
                ..gpu_state
            };
            self.particles.push(Some(record));
            self.particle_emitter_state.push(Some(new_state));
            self.particles.len() - 1
        };
        Ok(id)
    }

    // Tombstone a runtime emitter slot. The id becomes invalid; the next
    // `add_emitter` may reuse it. The pool + counter buffers are dropped
    // after a `device_wait_idle`: Vulkan has no driver-side keep-alive
    // for in-flight buffer references, so we must drain the queue before
    // freeing the backing memory. Reached only through the bin's `cn debug`
    // runtime-mutation path (dead in the FFI lib, live in the bin).
    #[allow(dead_code)]
    pub(in crate::vulkan) fn remove_particle_emitter(
        &mut self,
        emitter_id: usize,
    ) -> Result<(), String> {
        let rec_slot = self
            .particles
            .get_mut(emitter_id)
            .ok_or_else(|| format!("remove_emitter: id {emitter_id} out of range"))?;
        if rec_slot.is_none() {
            return Err(format!("remove_emitter: id {emitter_id} already removed"));
        }
        *rec_slot = None;
        if let Some(gpu_slot) = self.particle_emitter_state.get_mut(emitter_id)
            && let Some(state) = gpu_slot.take()
        {
            // Drain the queue before freeing the pool/counter so an
            // in-flight command buffer can't dereference the freed
            // memory. `cn debug` is the only consumer; this is not
            // on a hot path.
            self.wait_idle();
            state.destroy(&self.device);
            // Free the (compute, render) descriptor sets back to the
            // particle descriptor pool so the next `add_emitter` can
            // re-allocate them. Requires
            // `FREE_DESCRIPTOR_SET_BIT` on the pool; see the
            // descriptor pool creation. (We don't set it today; a
            // tombstoned slot's sets are reused at the next add via
            // the freelist path on Metal/DirectX. Here, since the
            // pool was sized for `2 * MAX_EMITTERS` sets, leaking
            // the slot's sets until the context dies is safe; the
            // freelist guarantees we never exceed the cap.)
        }
        self.particle_free_slots.push(emitter_id);
        Ok(())
    }

    // Wire every world-authored particle emitter through `add_particle_emitter`
    // so the same descriptor / SRV / GPU-state path serves both init and
    // runtime adds. Called from `VkContext::new` after the texture pool
    // is uploaded.
    pub(in crate::vulkan) fn upload_initial_particles(
        &mut self,
        records: Vec<ParticleEmitterRecord>,
    ) -> Result<(), String> {
        if records.is_empty() {
            return Ok(());
        }
        if records.len() > MAX_EMITTERS {
            return Err(format!(
                "particles: {} authored emitters exceed MAX_EMITTERS ({})",
                records.len(),
                MAX_EMITTERS
            ));
        }
        for record in records {
            self.add_particle_emitter(record)?;
        }
        Ok(())
    }

    // Re-point every emitter's albedo binding (set 1, binding 1) that samples
    // texture-pool `slot` at the just-swapped `self.textures[slot]` view. The
    // emitter albedo lives in the shared texture pool, so a streamed or
    // hot-reloaded albedo swap recreates the view and leaves a dangling
    // descriptor unless every emitter sampling that slot is re-pointed. Called
    // from `rewrite_albedo_slot`, the sibling of the per-object / clone rewires.
    pub(in crate::vulkan) fn rewrite_particle_albedo_slot(&self, slot: usize) {
        let Some(resources) = self.particle_resources.as_ref() else {
            return;
        };
        let last = self.textures.len().saturating_sub(1);
        let view = self.textures[slot].view;
        for state in self.particle_emitter_state.iter().flatten() {
            if state.texture_slot.min(last) == slot {
                write_render_albedo_binding(
                    &self.device,
                    state.render_set,
                    view,
                    resources.sampler,
                );
            }
        }
    }

    // Free every per-emitter pool/counter buffer. Called from
    // `Drop for VkContext` after `device_wait_idle`. Sibling of
    // `ParticleResources::destroy`, which handles the shared pipelines.
    pub(in crate::vulkan) fn destroy_particle_emitter_states(&mut self, device: &Device) {
        for state in self.particle_emitter_state.drain(..).flatten() {
            state.destroy(device);
        }
    }
}

fn write_render_albedo_binding(
    device: &Device,
    set: vk::DescriptorSet,
    view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(view)
        .sampler(sampler);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_particle_layout_matches_glsl() {
        // Mirrors the `Particle` struct in `shaders/particle_simulate.comp`:
        // std430 packs (vec3, float) into a 16-byte block, so the struct is
        // 32 bytes total laid out at 0/12/16/28.
        assert_eq!(std::mem::size_of::<GpuParticle>(), 32);
        assert_eq!(std::mem::offset_of!(GpuParticle, position), 0);
        assert_eq!(std::mem::offset_of!(GpuParticle, age), 12);
        assert_eq!(std::mem::offset_of!(GpuParticle, velocity), 16);
        assert_eq!(std::mem::offset_of!(GpuParticle, lifetime), 28);
    }

    #[test]
    fn particle_view_layout_matches_glsl() {
        // Mirrors the `ParticleView` uniform block in `particle.vert`:
        // mat4 (64) + (vec3 + pad) + (vec3 + pad) = 96.
        assert_eq!(std::mem::size_of::<ParticleView>(), 96);
        assert_eq!(std::mem::offset_of!(ParticleView, vp), 0);
        assert_eq!(std::mem::offset_of!(ParticleView, cam_right), 64);
        assert_eq!(std::mem::offset_of!(ParticleView, cam_up), 80);
    }

    #[test]
    fn particle_params_push_size_matches_glsl() {
        // The push-constant range size declared in the pipeline layout
        // must match the 112-byte ParticleParams struct exactly; neither
        // the compute shader nor the vertex shader reaches past it.
        assert_eq!(
            std::mem::size_of::<ParticleParams>() as u32,
            PARTICLE_PUSH_BYTES
        );
    }
}
