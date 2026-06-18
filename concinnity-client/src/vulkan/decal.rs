// src/vulkan/decal.rs
//
// Projected (deferred) decals for the Vulkan backend. Each decal is drawn
// as a unit cube (positions in `[-0.5, 0.5]^3`) transformed by its world
// model matrix and the camera VP; the fragment shader samples the main
// pass's depth attachment to reconstruct the world-space sample point at
// each pixel and tests it against the decal's local bounding box,
// stamping the texture onto whatever fills the box.
//
// Runs after the main HDR resolve and before SSR resolve / TAA, so
// decals are reflected and tracked by the temporal history just like
// the rest of the scene. Mirrors `src/directx/decal.rs` and
// `src/metal/decal.rs`.

use std::cell::Cell;
use std::ffi::CString;

use ash::{Device, vk};

use crate::gfx::decal::DecalRecord;

use super::context::VkContext;
use super::pipeline::{compile_glsl, inject_define, spv_module};
use super::texture::{GpuImage, create_buffer};

// GLSL sources, shared with the host so a future hot-reload pass can
// pick them up the same way the existing built-in shaders do.
pub(in crate::vulkan) const DECAL_VERT_GLSL: &str = include_str!("shaders/decal.vert");
pub(in crate::vulkan) const DECAL_FRAG_GLSL: &str = include_str!("shaders/decal.frag");

// Cap on the number of active decals: the descriptor pool reserves a
// fixed block of `MAX_DECALS` per-decal albedo sets at init, so runtime
// adds past this many return an error.
pub(in crate::vulkan) const MAX_DECALS: usize = 256;

// Eight unit-cube corners in [-0.5, 0.5]^3. Matches the DirectX / Metal
// vertex lists.
const CUBE_VERTS: [f32; 24] = [
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5,
    0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
];

// 36 indices forming 12 triangles wound CCW outward. Matches the
// DirectX / Metal index list so the rasterised cube exactly mirrors the
// reference.
const CUBE_INDICES: [u16; 36] = [
    // -Z face                +Z face
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, // -Y                     +Y
    0, 1, 5, 0, 5, 4, 3, 6, 2, 3, 7, 6, // -X                     +X
    0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
];

// Stride for the per-decal params uniform buffer ring. Vulkan's
// `minUniformBufferOffsetAlignment` is at most 256 bytes on every
// desktop GPU we target (spec-guaranteed upper bound), so a constant
// 256 byte stride keeps each dynamic-offset slot naturally aligned
// without querying the device.
const PARAMS_STRIDE: u64 = 256;

// Per-frame view inputs to the decal pass. Mirrors the `DecalViewBlock`
// uniform in `decal.vert` / `decal.frag`. 144 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
struct DecalView {
    vp: [[f32; 4]; 4],
    inv_vp: [[f32; 4]; 4],
    viewport: [f32; 2],
    _pad: [f32; 2],
}

// Per-decal uniforms uploaded into the per-frame params ring before
// each draw. Mirrors `DecalParamsBlock` in the shaders. 160 bytes
// (within the 256-byte stride slot).
#[derive(Copy, Clone)]
#[repr(C)]
struct DecalParams {
    model: [[f32; 4]; 4],
    inv_model: [[f32; 4]; 4],
    tint: [f32; 4],
    fade: [f32; 4], // .x = fade_pow, .yzw padding
}

// Owned by `VkContext` exactly once: the decal pipeline + its dependent
// resources. `decals` plus the freelist live on `VkContext` itself
// (mirroring the DirectX / Metal layout).
//
// Decal-pass descriptor sets follow a two-set layout:
//   * **set 0** (per-frame, FRAMES sets):
//       - binding 0: UNIFORM_BUFFER, `DecalView` (per-frame)
//       - binding 1: UNIFORM_BUFFER_DYNAMIC, `DecalParams` ring (per-frame,
//         MAX_DECALS slots; dynamic offset picks the per-decal slot)
//       - binding 2: COMBINED_IMAGE_SAMPLER, main depth view (per-frame
//         so a future-frame depth swap doesn't break a binding allocated
//         from a sibling frame slot)
//   * **set 1** (per-decal, MAX_DECALS sets):
//       - binding 0: COMBINED_IMAGE_SAMPLER, decal albedo
pub(in crate::vulkan) struct DecalResources {
    pub(in crate::vulkan) render_pass: vk::RenderPass,
    pub(in crate::vulkan) pipeline: vk::Pipeline,
    pub(in crate::vulkan) pipeline_layout: vk::PipelineLayout,
    pub(in crate::vulkan) view_set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) albedo_set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) descriptor_pool: vk::DescriptorPool,

    // Unit-cube vertex + index buffers (shared across frames).
    pub(in crate::vulkan) vertex_buffer: vk::Buffer,
    pub(in crate::vulkan) vertex_memory: vk::DeviceMemory,
    pub(in crate::vulkan) index_buffer: vk::Buffer,
    pub(in crate::vulkan) index_memory: vk::DeviceMemory,

    // Per-frame view UBO (DecalView, 144 bytes). Persistently mapped.
    pub(in crate::vulkan) view_ubos: Vec<vk::Buffer>,
    pub(in crate::vulkan) view_ubo_memories: Vec<vk::DeviceMemory>,
    pub(in crate::vulkan) view_ubo_ptrs: Vec<*mut u8>,

    // Per-frame per-decal params ring (PARAMS_STRIDE * MAX_DECALS bytes).
    // Persistently mapped; bound as UNIFORM_BUFFER_DYNAMIC with a
    // per-draw offset of `decal_id * PARAMS_STRIDE`.
    pub(in crate::vulkan) params_ubos: Vec<vk::Buffer>,
    pub(in crate::vulkan) params_ubo_memories: Vec<vk::DeviceMemory>,
    pub(in crate::vulkan) params_ubo_ptrs: Vec<*mut u8>,

    // Per-frame view sets (binding 0 view UBO, 1 params dynamic, 2 depth).
    pub(in crate::vulkan) view_sets: Vec<vk::DescriptorSet>,
    // Per-decal albedo sets (binding 0 albedo sampler). Indexed by
    // `decal_id`; tombstoned slots keep their last write, but the
    // executor only binds the set for visible records.
    pub(in crate::vulkan) albedo_sets: Vec<vk::DescriptorSet>,

    // One framebuffer per frame-in-flight slot, each binding its frame
    // slot's `hdr_resolve_images[i].view` as the sole colour attachment.
    pub(in crate::vulkan) framebuffers: Vec<vk::Framebuffer>,

    pub(in crate::vulkan) sampler: vk::Sampler,

    // Last-uploaded texture-pool slot per decal id. Used by
    // `rewrite_albedo_slot` to detect which decal albedo sets need
    // re-writing when a streamed texture pool entry swaps.
    pub(in crate::vulkan) decal_texture_slots: Cell<[usize; MAX_DECALS]>,
}

impl DecalResources {
    // Build the decal pipeline + its dependent resources. Called
    // unconditionally from `VkContext::new` so runtime `add_decal`
    // works from a world that started empty; the cost is one pipeline
    // + small buffers.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vulkan) fn new(
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        frames: usize,
        msaa: bool,
        hdr_format: vk::Format,
        hdr_resolve_views: &[vk::ImageView],
        depth_views: &[vk::ImageView],
        sampler: vk::Sampler,
        extent: vk::Extent2D,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let render_pass = create_decal_render_pass(device, hdr_format)?;
        let (view_set_layout, albedo_set_layout) = create_decal_set_layouts(device)?;
        let pipeline_layout =
            create_decal_pipeline_layout(device, view_set_layout, albedo_set_layout)?;

        let (vert_spv, frag_spv) = compile_decal_shaders(hot_reload, msaa)?;
        let pipeline =
            create_decal_pipeline(device, render_pass, pipeline_layout, &vert_spv, &frag_spv)?;

        // Unit-cube vertex + index buffers (single device-local upload).
        let (vertex_buffer, vertex_memory) = upload_static_buffer(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            bytemuck_cast(&CUBE_VERTS),
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let (index_buffer, index_memory) = upload_static_buffer(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            bytemuck_cast(&CUBE_INDICES),
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;

        // Per-frame view UBOs (HOST_VISIBLE | HOST_COHERENT, persistently
        // mapped).
        let mut view_ubos = Vec::with_capacity(frames);
        let mut view_ubo_memories = Vec::with_capacity(frames);
        let mut view_ubo_ptrs = Vec::with_capacity(frames);
        for _ in 0..frames {
            let (buf, mem) = create_buffer(
                instance,
                device,
                physical_device,
                std::mem::size_of::<DecalView>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let ptr = unsafe {
                device.map_memory(
                    mem,
                    0,
                    std::mem::size_of::<DecalView>() as u64,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|e| format!("map decal view ubo: {e}"))? as *mut u8;
            view_ubos.push(buf);
            view_ubo_memories.push(mem);
            view_ubo_ptrs.push(ptr);
        }

        // Per-frame per-decal params ring.
        let params_total = PARAMS_STRIDE * MAX_DECALS as u64;
        let mut params_ubos = Vec::with_capacity(frames);
        let mut params_ubo_memories = Vec::with_capacity(frames);
        let mut params_ubo_ptrs = Vec::with_capacity(frames);
        for _ in 0..frames {
            let (buf, mem) = create_buffer(
                instance,
                device,
                physical_device,
                params_total,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let ptr =
                unsafe { device.map_memory(mem, 0, params_total, vk::MemoryMapFlags::empty()) }
                    .map_err(|e| format!("map decal params ubo: {e}"))? as *mut u8;
            params_ubos.push(buf);
            params_ubo_memories.push(mem);
            params_ubo_ptrs.push(ptr);
        }

        let descriptor_pool = create_decal_descriptor_pool(device, frames)?;

        // Per-frame view sets (one per frame slot).
        let view_layouts: Vec<_> = (0..frames).map(|_| view_set_layout).collect();
        let view_sets = alloc_descriptor_sets(device, descriptor_pool, &view_layouts)?;
        for (i, &set) in view_sets.iter().enumerate() {
            write_view_set(
                device,
                set,
                view_ubos[i],
                params_ubos[i],
                depth_views[i.min(depth_views.len().saturating_sub(1))],
                sampler,
            );
        }

        // Per-decal albedo sets (MAX_DECALS sets, pre-allocated).
        let albedo_layouts: Vec<_> = (0..MAX_DECALS).map(|_| albedo_set_layout).collect();
        let albedo_sets = alloc_descriptor_sets(device, descriptor_pool, &albedo_layouts)?;

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
                .map_err(|e| format!("decal framebuffer: {e}"))?;
            framebuffers.push(fb);
        }

        Ok(Self {
            render_pass,
            pipeline,
            pipeline_layout,
            view_set_layout,
            albedo_set_layout,
            descriptor_pool,
            vertex_buffer,
            vertex_memory,
            index_buffer,
            index_memory,
            view_ubos,
            view_ubo_memories,
            view_ubo_ptrs,
            params_ubos,
            params_ubo_memories,
            params_ubo_ptrs,
            view_sets,
            albedo_sets,
            framebuffers,
            sampler,
            decal_texture_slots: Cell::new([usize::MAX; MAX_DECALS]),
        })
    }

    // Rebuild the framebuffers + re-point the per-frame view set's depth
    // binding after a swapchain resize. Called from
    // `VkContext::rebuild_swapchain`; same pattern as `SsrResources` /
    // `SsaoResources`. The pipeline, layouts, buffers, sampler, and
    // per-decal albedo sets all survive.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        device: &Device,
        hdr_resolve_views: &[vk::ImageView],
        depth_views: &[vk::ImageView],
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
                .map_err(|e| format!("decal framebuffer (rebuild): {e}"))?;
            self.framebuffers.push(fb);
        }
        // Re-point each per-frame view set's depth binding (binding 2)
        // at the rebuilt depth view.
        for (i, &set) in self.view_sets.iter().enumerate() {
            let depth_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(depth_views[i.min(depth_views.len().saturating_sub(1))])
                .sampler(self.sampler);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&depth_info));
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }
        Ok(())
    }

    // Destroy every GPU resource. Called from `VkContext::destroy` after
    // `wait_idle`. Buffer memory is unmapped first.
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
            for (&buf, &mem) in self.params_ubos.iter().zip(self.params_ubo_memories.iter()) {
                device.unmap_memory(mem);
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_memory, None);
            device.destroy_buffer(self.index_buffer, None);
            device.free_memory(self.index_memory, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.view_set_layout, None);
            device.destroy_descriptor_set_layout(self.albedo_set_layout, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);
        }
        self.framebuffers.clear();
        self.view_ubos.clear();
        self.view_ubo_memories.clear();
        self.view_ubo_ptrs.clear();
        self.params_ubos.clear();
        self.params_ubo_memories.clear();
        self.params_ubo_ptrs.clear();
    }
}

// Render pass / pipeline construction

fn create_decal_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass, String> {
    // One colour attachment: the resolved HDR scene. The main pass left
    // it in SHADER_READ_ONLY_OPTIMAL; we want it in COLOR_ATTACHMENT
    // during the subpass, then SHADER_READ_ONLY_OPTIMAL again on exit so
    // SSR / TAA / bloom / composite can sample it. The subpass
    // dependencies + (initial=SHADER_READ_ONLY, final=SHADER_READ_ONLY)
    // do the round-trip transition without an explicit barrier from the
    // caller.
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
    // Entry: anyone sampling the resolved HDR (e.g. the main pass's resolve
    // attachment writer, which finished with this image in
    // SHADER_READ_ONLY) must complete before the decal subpass starts
    // writing it.
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
    // Exit: any subsequent SSR / TAA / bloom / composite pass reading the
    // resolved HDR must wait for our writes to complete and become
    // available + visible to the fragment shader.
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
    unsafe { device.create_render_pass(&info, None) }.map_err(|e| format!("decal render pass: {e}"))
}

fn create_decal_set_layouts(
    device: &Device,
) -> Result<(vk::DescriptorSetLayout, vk::DescriptorSetLayout), String> {
    // set 0: per-frame view UBO + per-decal params dynamic UBO + depth.
    let view_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let view_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&view_bindings);
    let view_set_layout = unsafe { device.create_descriptor_set_layout(&view_info, None) }
        .map_err(|e| format!("decal view set layout: {e}"))?;

    // set 1: per-decal albedo sampler.
    let albedo_bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    let albedo_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&albedo_bindings);
    let albedo_set_layout = unsafe { device.create_descriptor_set_layout(&albedo_info, None) }
        .map_err(|e| format!("decal albedo set layout: {e}"))?;

    Ok((view_set_layout, albedo_set_layout))
}

fn create_decal_pipeline_layout(
    device: &Device,
    view_set_layout: vk::DescriptorSetLayout,
    albedo_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, String> {
    let set_layouts = [view_set_layout, albedo_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    unsafe { device.create_pipeline_layout(&info, None) }
        .map_err(|e| format!("decal pipeline layout: {e}"))
}

fn create_decal_descriptor_pool(
    device: &Device,
    frames: usize,
) -> Result<vk::DescriptorPool, String> {
    let frames = frames as u32;
    let max_decals = MAX_DECALS as u32;
    // Pool sizing: FRAMES sets for view + (MAX_DECALS) sets for albedo.
    //   - UNIFORM_BUFFER: FRAMES (one DecalView per frame slot)
    //   - UNIFORM_BUFFER_DYNAMIC: FRAMES (one params ring per frame slot)
    //   - COMBINED_IMAGE_SAMPLER: FRAMES (depth) + MAX_DECALS (albedo)
    let sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: frames,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: frames,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: frames + max_decals,
        },
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(frames + max_decals)
        .pool_sizes(&sizes);
    unsafe { device.create_descriptor_pool(&info, None) }
        .map_err(|e| format!("decal descriptor pool: {e}"))
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
        .map_err(|e| format!("decal descriptor sets: {e}"))
}

fn write_view_set(
    device: &Device,
    set: vk::DescriptorSet,
    view_ubo: vk::Buffer,
    params_ubo: vk::Buffer,
    depth_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let view_info = vk::DescriptorBufferInfo::default()
        .buffer(view_ubo)
        .offset(0)
        .range(std::mem::size_of::<DecalView>() as u64);
    let params_info = vk::DescriptorBufferInfo::default()
        .buffer(params_ubo)
        .offset(0)
        // Dynamic-offset descriptor binds a window of `range` bytes
        // starting at the offset supplied at cmd_bind_descriptor_sets
        // time. Setting range to PARAMS_STRIDE means each per-decal bind
        // touches exactly one slot.
        .range(PARAMS_STRIDE);
    let depth_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(depth_view)
        .sampler(sampler);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&view_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .buffer_info(std::slice::from_ref(&params_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&depth_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn compile_decal_shaders(hot_reload: bool, msaa: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let define = if msaa {
        "#define USE_MSAA 1\n"
    } else {
        "#define USE_MSAA 0\n"
    };
    // shaderc accepts a single source string. The vert source doesn't
    // branch on USE_MSAA but it costs nothing to define it there too.
    use super::pipeline::shader_source;
    let vert_src = inject_define(
        &shader_source(hot_reload, "decal.vert", DECAL_VERT_GLSL),
        define,
    );
    let frag_src = inject_define(
        &shader_source(hot_reload, "decal.frag", DECAL_FRAG_GLSL),
        define,
    );
    let vert = compile_glsl(&vert_src, shaderc::ShaderKind::Vertex, "decal.vert")?;
    let frag = compile_glsl(&frag_src, shaderc::ShaderKind::Fragment, "decal.frag")?;
    Ok((vert, frag))
}

// Rebuild the decal graphics pipeline against the existing render pass +
// layout. Used by the Vulkan shader hot-reload path. The caller is
// responsible for destroying the previous pipeline only after this call
// succeeds.
pub(in crate::vulkan) fn rebuild_decal_pipeline(
    device: &Device,
    decals: &DecalResources,
    msaa: bool,
    hot_reload: bool,
) -> Result<vk::Pipeline, String> {
    let (vert_spv, frag_spv) = compile_decal_shaders(hot_reload, msaa)?;
    create_decal_pipeline(
        device,
        decals.render_pass,
        decals.pipeline_layout,
        &vert_spv,
        &frag_spv,
    )
}

fn create_decal_pipeline(
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
    let bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(12) // vec3 position
        .input_rate(vk::VertexInputRate::VERTEX)];
    let attrs = [vk::VertexInputAttributeDescription::default()
        .location(0)
        .binding(0)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset(0)];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        // Cull front faces: the camera may be inside the decal volume.
        // With back-face culling on (the default) entering the volume
        // would make the unit cube disappear; culling the front face
        // keeps the back faces rasterised in both cases. Mirrors
        // DirectX / Metal.
        .cull_mode(vk::CullModeFlags::FRONT)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        // The decal pass writes the SINGLE-SAMPLE resolved HDR, not the
        // MSAA colour. Sample count here matches the attachment, not the
        // main pass's MSAA count.
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
    .map_err(|(_, e)| format!("create decal pipeline: {e}"))?[0];
    unsafe {
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
    }
    Ok(pipeline)
}

// Helpers for upload + casting

fn upload_static_buffer(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    data: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
    let size = data.len() as vk::DeviceSize;
    let (staging, staging_mem) = create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        let ptr = device
            .map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("map decal staging: {e}"))? as *mut u8;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        device.unmap_memory(staging_mem);
    }
    let (buf, mem) = create_buffer(
        instance,
        device,
        physical_device,
        size,
        usage | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    super::texture::one_shot_submit(device, command_pool, queue, |cmd| {
        let region = vk::BufferCopy::default().size(size);
        unsafe { device.cmd_copy_buffer(cmd, staging, buf, std::slice::from_ref(&region)) };
    })?;
    unsafe {
        device.destroy_buffer(staging, None);
        device.free_memory(staging_mem, None);
    }
    Ok((buf, mem))
}

fn bytemuck_cast<T: Copy>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

// Encoder

impl VkContext {
    // Encode the projected-decal pass. Called between the main HDR
    // resolve and the SSR resolve so a decal is reflected by SSR and
    // tracked by TAA's history buffer like the rest of the scene.
    //
    // `vp` is the same jittered view-projection the main pass
    // rasterised with; the inverse drives the world-space reconstruction
    // in the fragment shader.
    pub(in crate::vulkan) fn encode_decals(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        vp: [[f32; 4]; 4],
        frustum: &crate::gfx::frustum::Frustum,
    ) {
        let decals = match &self.decals_state {
            Some(s) => s,
            None => return,
        };
        if self.decals.iter().all(|slot| slot.is_none()) {
            return;
        }
        // Frustum-cull first so a frame where every live decal lands
        // off-screen skips the pass, including the depth-transition
        // barriers. Tombstoned (None) slots are always invisible.
        let visible_count = self
            .decals
            .iter()
            .filter(|slot| {
                slot.as_ref()
                    .map(|d| {
                        let (mn, mx) = d.aabb();
                        frustum.intersects_aabb(mn, mx)
                    })
                    .unwrap_or(false)
            })
            .count();
        if visible_count == 0 {
            return;
        }

        let device = &self.device;
        let extent = self.render_extent;

        // Upload this frame's view UBO.
        let inv_vp = super::math::mat4_inverse(vp);
        let viewport_pix = [extent.width as f32, extent.height as f32];
        let view_uni = DecalView {
            vp,
            inv_vp,
            viewport: viewport_pix,
            _pad: [0.0; 2],
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &view_uni as *const DecalView as *const u8,
                decals.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<DecalView>(),
            );
        }

        // Upload per-decal params slots for every live record (visible or
        // not; easier to skip the visibility check here and pay one
        // 160-byte write per slot).
        for (i, slot) in self.decals.iter().enumerate() {
            let d = match slot {
                Some(d) => d,
                None => continue,
            };
            let params = DecalParams {
                model: d.model,
                inv_model: d.inv_model,
                tint: d.tint,
                fade: [2.0, 0.0, 0.0, 0.0],
            };
            unsafe {
                let dst = decals.params_ubo_ptrs[frame_idx].add(i * PARAMS_STRIDE as usize);
                std::ptr::copy_nonoverlapping(
                    &params as *const DecalParams as *const u8,
                    dst,
                    std::mem::size_of::<DecalParams>(),
                );
            }
        }

        // Transition main depth → SHADER_READ_ONLY so the fragment can
        // sample it; restore to DEPTH_STENCIL_ATTACHMENT after the pass
        // so the next frame's main pass can clear/write it again.
        let depth_image = self.depth_images[frame_idx].image;
        unsafe {
            let to_read = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(depth_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_read),
            );
        }

        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(decals.render_pass)
            .framebuffer(decals.framebuffers[frame_idx])
            .render_area(vk::Rect2D::default().extent(extent));

        // Negative-height viewport matches the main pass so the
        // rasterised pixel grid lines up with the depth attachment we're
        // sampling. Without this, the cube would be drawn Y-flipped
        // relative to the scene depth.
        let vp_state = vk::Viewport {
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
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp_state));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, decals.pipeline);
            device.cmd_bind_vertex_buffers(
                cmd,
                0,
                std::slice::from_ref(&decals.vertex_buffer),
                &[0],
            );
            device.cmd_bind_index_buffer(cmd, decals.index_buffer, 0, vk::IndexType::UINT16);
        }

        for (i, slot) in self.decals.iter().enumerate() {
            let d = match slot {
                Some(d) => d,
                None => continue,
            };
            let (mn, mx) = d.aabb();
            if !frustum.intersects_aabb(mn, mx) {
                continue;
            }
            let dynamic_offset = (i as u64 * PARAMS_STRIDE) as u32;
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    decals.pipeline_layout,
                    0,
                    std::slice::from_ref(&decals.view_sets[frame_idx]),
                    std::slice::from_ref(&dynamic_offset),
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    decals.pipeline_layout,
                    1,
                    std::slice::from_ref(&decals.albedo_sets[i]),
                    &[],
                );
                device.cmd_draw_indexed(cmd, 36, 1, 0, 0, 0);
            }
            self.inc_draw_calls(1);
        }

        unsafe {
            device.cmd_end_render_pass(cmd);

            // Restore main depth → DEPTH_STENCIL_ATTACHMENT for the next
            // frame's main pass.
            let to_depth = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .image(depth_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_depth),
            );
        }
    }
}

// Runtime mutation (RenderBackend::add_decal / remove_decal)

impl VkContext {
    // Append a runtime decal. Writes the per-decal albedo descriptor
    // into the reserved slot for `id`; the encoder reads it next frame.
    // Reuses tombstoned slots from a prior `remove_decal` before growing
    // the vec.
    pub fn add_decal(&mut self, record: DecalRecord) -> Result<usize, String> {
        let last_tex = self.textures.len().saturating_sub(1);
        let tex_idx = record.texture_slot.min(last_tex);

        let id = if let Some(slot) = self.decal_free_slots.pop() {
            self.decals[slot] = Some(record);
            slot
        } else {
            if self.decals.len() >= MAX_DECALS {
                return Err(format!("add_decal: MAX_DECALS ({MAX_DECALS}) exceeded"));
            }
            self.decals.push(Some(record));
            self.decals.len() - 1
        };

        // Write the albedo descriptor for this slot. The texture pool
        // entry is referenced live; a future eviction routes through
        // `rewrite_albedo_slot` to re-point.
        let decals = self
            .decals_state
            .as_ref()
            .ok_or_else(|| "add_decal: decal pipeline unavailable".to_string())?;
        write_albedo_set(
            &self.device,
            decals.albedo_sets[id],
            self.textures[tex_idx].view,
            decals.sampler,
        );
        let mut slots = decals.decal_texture_slots.get();
        slots[id] = tex_idx;
        decals.decal_texture_slots.set(slots);
        Ok(id)
    }

    // Tombstone a runtime decal slot. The id becomes invalid; the next
    // `add_decal` may reuse it. Reached only through the bin's `cn debug`
    // runtime-mutation path (dead in the FFI lib, live in the bin).
    #[allow(dead_code)]
    pub fn remove_decal(&mut self, decal_id: usize) -> Result<(), String> {
        let slot = self
            .decals
            .get_mut(decal_id)
            .ok_or_else(|| format!("remove_decal: id {decal_id} out of range"))?;
        if slot.is_none() {
            return Err(format!("remove_decal: id {decal_id} already removed"));
        }
        *slot = None;
        self.decal_free_slots.push(decal_id);
        if let Some(decals) = &self.decals_state {
            let mut slots = decals.decal_texture_slots.get();
            slots[decal_id] = usize::MAX;
            decals.decal_texture_slots.set(slots);
        }
        Ok(())
    }

    // Re-point every decal albedo set that pointed at texture-pool slot
    // `slot` to the new `GpuImage` at that slot. Called from the
    // streaming-texture path when an evicted slot is replaced. Walks
    // `decals_state.decal_texture_slots` so a world with no decals pays
    // nothing.
    pub(in crate::vulkan) fn rewrite_decal_albedo_slot(&self, slot: usize) {
        let decals = match &self.decals_state {
            Some(s) => s,
            None => return,
        };
        let slots = decals.decal_texture_slots.get();
        let last_tex = self.textures.len().saturating_sub(1);
        for (id, &tex_slot) in slots.iter().enumerate() {
            if tex_slot == slot {
                let view = self.textures[tex_slot.min(last_tex)].view;
                write_albedo_set(&self.device, decals.albedo_sets[id], view, decals.sampler);
            }
        }
    }
}

fn write_albedo_set(
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
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

// Wire authored decals into the runtime state at init.

impl VkContext {
    // Push every world-authored `DecalRecord` through `add_decal` so its
    // albedo descriptor lands in the reserved slot. Called once from
    // `VkContext::new` after `decals_state` is built.
    pub(in crate::vulkan) fn upload_initial_decals(
        &mut self,
        records: Vec<DecalRecord>,
    ) -> Result<(), String> {
        if records.len() > MAX_DECALS {
            return Err(format!(
                "decals: {} authored decals exceed MAX_DECALS ({})",
                records.len(),
                MAX_DECALS
            ));
        }
        for rec in records {
            self.add_decal(rec)?;
        }
        Ok(())
    }
}

// Re-export GpuImage so the textures module compiles cleanly when the
// decal module is the only consumer of a few of its helpers.
#[allow(dead_code)]
type _Marker = GpuImage;
