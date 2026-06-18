// src/vulkan/fog.rs
//
// Volumetric fog for the Vulkan backend. Frostbite-style froxel volume:
//
//   * The `fog_froxel.comp` compute pass (`encode_fog_froxel`) populates a
//     screen-aligned `(80 x 45 x 64)` 3D `RGBA16F` volume of
//     `(scattered_rgb, 1 - T)` across the view frustum. One thread per
//     (x, y) tile; each thread walks Z front-to-back, accumulating the
//     per-slab scatter + transmittance with a CSM shadow tap per slice.
//
//   * The fullscreen `Fog` render pass (`encode_fog`) samples the volume by
//     `(screen_uv, view_z)` instead of marching per pixel and composites
//     `(scattered, 1 - T)` over the resolved HDR target with the standard
//     `over` blend (`final = scene * T + scattered`).
//
// Runs between the projected-decal pass and the SSR resolve so the fog wraps
// the decal-stamped scene and SSR reflects through it; TAA history then
// reprojects the integrated fog colour and transmittance.
//
// Mirrors src/directx/fog.rs and src/metal/fog.rs.

use std::ffi::CString;

use ash::{Device, vk};

use crate::gfx::render_graph::{FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z};
use crate::gfx::render_types::{FogFroxelParams, FogParams, ShadowUniforms};

use super::context::VkContext;
use super::pipeline::{compile_glsl, inject_define, spv_module};
use super::texture::{
    create_buffer, find_memory_type, one_shot_submit, transition_image_layout_range,
};

// GLSL sources, shared with the host so the hot-reload pass can pick them up
// the same way the existing built-in shaders do.
pub(in crate::vulkan) const FOG_VERT_GLSL: &str = include_str!("shaders/fog.vert");
pub(in crate::vulkan) const FOG_FRAG_GLSL: &str = include_str!("shaders/fog.frag");
pub(in crate::vulkan) const FOG_FROXEL_GLSL: &str = include_str!("shaders/fog_froxel.comp");

// Threadgroup tile for the froxel kernel (8x8, one thread per (x, y) froxel),
// matching the DirectX `[numthreads(8, 8, 1)]` and the Metal dispatch.
const FROXEL_TILE: u32 = 8;

// 3D froxel volume pixel format. RGBA16F holds `(scattered_rgb, 1 - T)` per
// slice; mirrors the DirectX / Metal `RGBA16Float` volume.
const VOLUME_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

// Owned by `VkContext` exactly when the world declared a `VolumetricFog`:
// the fog render pipeline + the froxel compute pipeline + the per-frame
// uniform rings + the shared 3D froxel volume the kernel writes and the
// fragment shader samples.
pub(in crate::vulkan) struct FogResources {
    pub(in crate::vulkan) render_pass: vk::RenderPass,
    pub(in crate::vulkan) pipeline: vk::Pipeline,
    pub(in crate::vulkan) pipeline_layout: vk::PipelineLayout,
    pub(in crate::vulkan) view_set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) descriptor_pool: vk::DescriptorPool,

    // Per-frame FogParams view UBO (176 bytes). Persistently mapped.
    pub(in crate::vulkan) params_ubos: Vec<vk::Buffer>,
    pub(in crate::vulkan) params_ubo_memories: Vec<vk::DeviceMemory>,
    pub(in crate::vulkan) params_ubo_ptrs: Vec<*mut u8>,

    // Per-frame FogFroxelParams UBO (96 bytes). Persistently mapped. Bound at
    // the froxel kernel's set binding 1 and the fog fragment's set binding 2.
    pub(in crate::vulkan) froxel_ubos: Vec<vk::Buffer>,
    pub(in crate::vulkan) froxel_ubo_memories: Vec<vk::DeviceMemory>,
    pub(in crate::vulkan) froxel_ubo_ptrs: Vec<*mut u8>,

    // Per-frame fog-render view sets (binding 0 FogParams, 1 depth, 2
    // FogFroxelParams, 3 volume sampler3D).
    pub(in crate::vulkan) view_sets: Vec<vk::DescriptorSet>,

    // Froxel compute pipeline + its per-frame sets (binding 0 FogParams, 1
    // FogFroxelParams, 2 ShadowUniforms, 3 shadow_map, 4 volume image3D).
    pub(in crate::vulkan) froxel_pipeline: vk::Pipeline,
    pub(in crate::vulkan) froxel_pipeline_layout: vk::PipelineLayout,
    pub(in crate::vulkan) froxel_set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) froxel_sets: Vec<vk::DescriptorSet>,

    // Shared 3D RGBA16F volume: written by the compute kernel (GENERAL),
    // sampled by the fog fragment (SHADER_READ_ONLY). The open
    // (SHADER_READ_ONLY -> GENERAL) and close (GENERAL -> SHADER_READ_ONLY)
    // transitions are graph-driven (fog_froxel_volume's FogFroxel producer + Fog
    // consumer barriers, emitted by the executor); the cross-frame hazard chain
    // spans submission order on the one queue (same reasoning as the Hi-Z
    // pyramid).
    pub(in crate::vulkan) volume_image: vk::Image,
    pub(in crate::vulkan) volume_memory: vk::DeviceMemory,
    pub(in crate::vulkan) volume_storage_view: vk::ImageView,
    pub(in crate::vulkan) volume_sampled_view: vk::ImageView,

    // One framebuffer per frame-in-flight slot, each binding its frame slot's
    // `hdr_resolve_images[i].view` as the sole colour attachment.
    pub(in crate::vulkan) framebuffers: Vec<vk::Framebuffer>,

    // Depth sampler (the shared linear sampler; depth is read via texelFetch so
    // the filter mode is irrelevant).
    pub(in crate::vulkan) sampler: vk::Sampler,
    // Linear-clamp sampler for the trilinear volume read.
    pub(in crate::vulkan) volume_sampler: vk::Sampler,
}

impl FogResources {
    // Build the fog render pipeline + the froxel compute pipeline + their
    // dependent resources. Called from `VkContext::new` only when the world
    // declared a `VolumetricFog` and `FogSettings::resolve` returned a value.
    // `shadow_ubo` / `shadow_map_view` / `shadow_sampler` are the shared CSM
    // resources the froxel kernel taps per slab; `command_pool` / `queue`
    // are used once to move the volume into `SHADER_READ_ONLY_OPTIMAL`.
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
        shadow_ubo: vk::Buffer,
        shadow_map_view: vk::ImageView,
        shadow_sampler: vk::Sampler,
        extent: vk::Extent2D,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let render_pass = create_fog_render_pass(device, hdr_format)?;
        let view_set_layout = create_fog_set_layout(device)?;
        let pipeline_layout = create_fog_pipeline_layout(device, view_set_layout)?;

        let (vert_spv, frag_spv) = compile_fog_shaders(hot_reload, msaa)?;
        let pipeline =
            create_fog_pipeline(device, render_pass, pipeline_layout, &vert_spv, &frag_spv)?;

        // Froxel compute pipeline.
        let froxel_set_layout = create_froxel_set_layout(device)?;
        let froxel_pipeline_layout = create_froxel_pipeline_layout(device, froxel_set_layout)?;
        let froxel_spv = compile_fog_froxel_shader(hot_reload)?;
        let froxel_pipeline = create_compute_pipeline(device, froxel_pipeline_layout, &froxel_spv)?;

        // The shared 3D volume + its storage (compute write) + sampled
        // (fragment read) views. Rest it in SHADER_READ_ONLY so the first
        // froxel build's opening barrier (SHADER_READ_ONLY -> GENERAL) matches.
        let (volume_image, volume_memory) = create_volume_image(instance, device, physical_device)?;
        one_shot_submit(device, command_pool, queue, |cmd| {
            transition_image_layout_range(
                device,
                cmd,
                volume_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                0,
                1,
                0,
                1,
            );
        })?;
        let volume_storage_view = create_volume_view(device, volume_image)?;
        let volume_sampled_view = create_volume_view(device, volume_image)?;
        let volume_sampler = create_volume_sampler(device)?;

        // Per-frame FogParams UBOs (HOST_VISIBLE | HOST_COHERENT, mapped).
        let (params_ubos, params_ubo_memories, params_ubo_ptrs) = alloc_ubo_ring(
            instance,
            device,
            physical_device,
            frames,
            std::mem::size_of::<FogParams>() as u64,
        )?;
        // Per-frame FogFroxelParams UBOs.
        let (froxel_ubos, froxel_ubo_memories, froxel_ubo_ptrs) = alloc_ubo_ring(
            instance,
            device,
            physical_device,
            frames,
            std::mem::size_of::<FogFroxelParams>() as u64,
        )?;

        let descriptor_pool = create_fog_descriptor_pool(device, frames)?;
        let view_layouts: Vec<_> = (0..frames).map(|_| view_set_layout).collect();
        let view_sets = alloc_descriptor_sets(device, descriptor_pool, &view_layouts)?;
        let froxel_layouts: Vec<_> = (0..frames).map(|_| froxel_set_layout).collect();
        let froxel_sets = alloc_descriptor_sets(device, descriptor_pool, &froxel_layouts)?;

        let last_depth = depth_views.len().saturating_sub(1);
        for (i, &set) in view_sets.iter().enumerate() {
            write_view_set(
                device,
                set,
                params_ubos[i],
                depth_views[i.min(last_depth)],
                sampler,
                froxel_ubos[i],
                volume_sampled_view,
                volume_sampler,
            );
        }
        for (i, &set) in froxel_sets.iter().enumerate() {
            write_froxel_set(
                device,
                set,
                params_ubos[i],
                froxel_ubos[i],
                shadow_ubo,
                shadow_map_view,
                shadow_sampler,
                volume_storage_view,
            );
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
                .map_err(|e| format!("fog framebuffer: {e}"))?;
            framebuffers.push(fb);
        }

        Ok(Self {
            render_pass,
            pipeline,
            pipeline_layout,
            view_set_layout,
            descriptor_pool,
            params_ubos,
            params_ubo_memories,
            params_ubo_ptrs,
            froxel_ubos,
            froxel_ubo_memories,
            froxel_ubo_ptrs,
            view_sets,
            froxel_pipeline,
            froxel_pipeline_layout,
            froxel_set_layout,
            froxel_sets,
            volume_image,
            volume_memory,
            volume_storage_view,
            volume_sampled_view,
            framebuffers,
            sampler,
            volume_sampler,
        })
    }

    // Rebuild the framebuffers + re-point the per-frame view set's depth
    // binding after a swapchain resize. Called from
    // `VkContext::rebuild_swapchain`; same pattern as `DecalResources`. The
    // pipelines, layouts, buffers, the froxel sets, the 3D volume, and the
    // samplers all survive (the volume is screen-aligned via the per-froxel
    // reconstruction, not tied to render resolution).
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
        for &view in hdr_resolve_views.iter().take(self.params_ubos.len()) {
            let attachments = [view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(self.render_pass)
                .attachments(&attachments)
                .width(extent.width.max(1))
                .height(extent.height.max(1))
                .layers(1);
            let fb = unsafe { device.create_framebuffer(&fb_info, None) }
                .map_err(|e| format!("fog framebuffer (rebuild): {e}"))?;
            self.framebuffers.push(fb);
        }
        let last_depth = depth_views.len().saturating_sub(1);
        for (i, &set) in self.view_sets.iter().enumerate() {
            let depth_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(depth_views[i.min(last_depth)])
                .sampler(self.sampler);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&depth_info));
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }
        Ok(())
    }

    // Destroy every GPU resource. Called from `VkContext::drop` after
    // `wait_idle`. Buffer memory is unmapped first.
    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        unsafe {
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            for (&buf, &mem) in self.params_ubos.iter().zip(self.params_ubo_memories.iter()) {
                device.unmap_memory(mem);
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
            for (&buf, &mem) in self.froxel_ubos.iter().zip(self.froxel_ubo_memories.iter()) {
                device.unmap_memory(mem);
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
            device.destroy_sampler(self.volume_sampler, None);
            device.destroy_image_view(self.volume_storage_view, None);
            device.destroy_image_view(self.volume_sampled_view, None);
            device.destroy_image(self.volume_image, None);
            device.free_memory(self.volume_memory, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.view_set_layout, None);
            device.destroy_descriptor_set_layout(self.froxel_set_layout, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline(self.froxel_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_pipeline_layout(self.froxel_pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);
        }
        self.framebuffers.clear();
        self.params_ubos.clear();
        self.params_ubo_memories.clear();
        self.params_ubo_ptrs.clear();
        self.froxel_ubos.clear();
        self.froxel_ubo_memories.clear();
        self.froxel_ubo_ptrs.clear();
    }
}

// Allocate `count` host-visible/coherent uniform buffers of `size` bytes,
// each persistently mapped. Returns the buffers, their memory, and the mapped
// host pointers.
type UboRing = (Vec<vk::Buffer>, Vec<vk::DeviceMemory>, Vec<*mut u8>);
fn alloc_ubo_ring(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    count: usize,
    size: u64,
) -> Result<UboRing, String> {
    let mut bufs = Vec::with_capacity(count);
    let mut mems = Vec::with_capacity(count);
    let mut ptrs: Vec<*mut u8> = Vec::with_capacity(count);
    for _ in 0..count {
        let (buf, mem) = create_buffer(
            instance,
            device,
            physical_device,
            size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let ptr = unsafe { device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()) }
            .map_err(|e| format!("map fog ubo: {e}"))? as *mut u8;
        bufs.push(buf);
        mems.push(mem);
        ptrs.push(ptr);
    }
    Ok((bufs, mems, ptrs))
}

// Render pass / pipeline construction

fn create_fog_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass, String> {
    // One colour attachment: the resolved HDR scene. The main pass (and
    // any preceding decal pass) left it in SHADER_READ_ONLY_OPTIMAL; we
    // want it in COLOR_ATTACHMENT during the subpass and
    // SHADER_READ_ONLY_OPTIMAL again on exit so SSR / TAA / bloom /
    // composite can sample it. Mirrors the decal render pass.
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
    unsafe { device.create_render_pass(&info, None) }.map_err(|e| format!("fog render pass: {e}"))
}

fn create_fog_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
    let bindings = [
        // 0: FogParams UBO.
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // 1: scene depth sampler.
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // 2: FogFroxelParams UBO.
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // 3: froxel volume sampler3D.
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("fog set layout: {e}"))
}

fn create_froxel_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
    let bindings = [
        // 0: FogParams UBO.
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 1: FogFroxelParams UBO.
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 2: ShadowUniforms UBO.
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 3: shadow map array (sampler2DArrayShadow).
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 4: froxel volume image3D (storage).
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("fog froxel set layout: {e}"))
}

fn create_fog_pipeline_layout(
    device: &Device,
    view_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, String> {
    let set_layouts = [view_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    unsafe { device.create_pipeline_layout(&info, None) }
        .map_err(|e| format!("fog pipeline layout: {e}"))
}

fn create_froxel_pipeline_layout(
    device: &Device,
    froxel_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, String> {
    let set_layouts = [froxel_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    unsafe { device.create_pipeline_layout(&info, None) }
        .map_err(|e| format!("fog froxel pipeline layout: {e}"))
}

fn create_fog_descriptor_pool(
    device: &Device,
    frames: usize,
) -> Result<vk::DescriptorPool, String> {
    let f = frames as u32;
    let sizes = [
        // view: FogParams + FogFroxelParams (2). froxel: FogParams +
        // FogFroxelParams + ShadowUniforms (3).
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 5 * f,
        },
        // view: depth + volume sampled (2). froxel: shadow map (1).
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 3 * f,
        },
        // froxel: volume storage (1).
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: f,
        },
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(2 * f)
        .pool_sizes(&sizes);
    unsafe { device.create_descriptor_pool(&info, None) }
        .map_err(|e| format!("fog descriptor pool: {e}"))
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
        .map_err(|e| format!("fog descriptor sets: {e}"))
}

#[allow(clippy::too_many_arguments)]
fn write_view_set(
    device: &Device,
    set: vk::DescriptorSet,
    params_ubo: vk::Buffer,
    depth_view: vk::ImageView,
    depth_sampler: vk::Sampler,
    froxel_ubo: vk::Buffer,
    volume_view: vk::ImageView,
    volume_sampler: vk::Sampler,
) {
    let params_info = vk::DescriptorBufferInfo::default()
        .buffer(params_ubo)
        .offset(0)
        .range(std::mem::size_of::<FogParams>() as u64);
    let depth_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(depth_view)
        .sampler(depth_sampler);
    let froxel_info = vk::DescriptorBufferInfo::default()
        .buffer(froxel_ubo)
        .offset(0)
        .range(std::mem::size_of::<FogFroxelParams>() as u64);
    let volume_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(volume_view)
        .sampler(volume_sampler);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&params_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&depth_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&froxel_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&volume_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

#[allow(clippy::too_many_arguments)]
fn write_froxel_set(
    device: &Device,
    set: vk::DescriptorSet,
    params_ubo: vk::Buffer,
    froxel_ubo: vk::Buffer,
    shadow_ubo: vk::Buffer,
    shadow_map_view: vk::ImageView,
    shadow_sampler: vk::Sampler,
    volume_storage_view: vk::ImageView,
) {
    let params_info = vk::DescriptorBufferInfo::default()
        .buffer(params_ubo)
        .offset(0)
        .range(std::mem::size_of::<FogParams>() as u64);
    let froxel_info = vk::DescriptorBufferInfo::default()
        .buffer(froxel_ubo)
        .offset(0)
        .range(std::mem::size_of::<FogFroxelParams>() as u64);
    let shadow_info = vk::DescriptorBufferInfo::default()
        .buffer(shadow_ubo)
        .offset(0)
        .range(std::mem::size_of::<ShadowUniforms>() as u64);
    let shadow_map_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(shadow_map_view)
        .sampler(shadow_sampler);
    let volume_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(volume_storage_view);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&params_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&froxel_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&shadow_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&shadow_map_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&volume_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

// Create the shared 3D RGBA16F froxel volume (STORAGE | SAMPLED, GPU-local).
fn create_volume_image(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
) -> Result<(vk::Image, vk::DeviceMemory), String> {
    let img_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_3D)
        .extent(vk::Extent3D {
            width: FOG_FROXEL_X,
            height: FOG_FROXEL_Y,
            depth: FOG_FROXEL_Z,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(VOLUME_FORMAT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let image = unsafe { device.create_image(&img_info, None) }
        .map_err(|e| format!("fog volume image: {e}"))?;
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(find_memory_type(
            instance,
            physical_device,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?);
    let memory = unsafe { device.allocate_memory(&alloc, None) }
        .map_err(|e| format!("fog volume memory: {e}"))?;
    unsafe { device.bind_image_memory(image, memory, 0) }
        .map_err(|e| format!("fog volume bind memory: {e}"))?;
    Ok((image, memory))
}

// A whole-image 3D view of the froxel volume (used for both the compute
// storage bind and the fragment sampled bind).
fn create_volume_view(device: &Device, image: vk::Image) -> Result<vk::ImageView, String> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_3D)
        .format(VOLUME_FORMAT)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe { device.create_image_view(&info, None) }.map_err(|e| format!("fog volume view: {e}"))
}

// Linear clamp-to-edge sampler for the trilinear volume read.
fn create_volume_sampler(device: &Device) -> Result<vk::Sampler, String> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    unsafe { device.create_sampler(&info, None) }.map_err(|e| format!("fog volume sampler: {e}"))
}

fn compile_fog_shaders(hot_reload: bool, msaa: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    use super::pipeline::shader_source;
    let define = if msaa {
        "#define USE_MSAA 1\n"
    } else {
        "#define USE_MSAA 0\n"
    };
    let vert_src = inject_define(
        &shader_source(hot_reload, "fog.vert", FOG_VERT_GLSL),
        define,
    );
    let frag_src = inject_define(
        &shader_source(hot_reload, "fog.frag", FOG_FRAG_GLSL),
        define,
    );
    let vert = compile_glsl(&vert_src, shaderc::ShaderKind::Vertex, "fog.vert")?;
    let frag = compile_glsl(&frag_src, shaderc::ShaderKind::Fragment, "fog.frag")?;
    Ok((vert, frag))
}

// Compile the froxel-volume compute kernel. MSAA-independent (the kernel does
// not read the scene depth attachment).
fn compile_fog_froxel_shader(hot_reload: bool) -> Result<Vec<u8>, String> {
    use super::pipeline::shader_source;
    let src = shader_source(hot_reload, "fog_froxel.comp", FOG_FROXEL_GLSL);
    compile_glsl(&src, shaderc::ShaderKind::Compute, "fog_froxel.comp")
}

// Rebuild the fog graphics pipeline against the existing render pass +
// layout. Used by the Vulkan shader hot-reload path. The caller is
// responsible for destroying the previous pipeline only after this call
// succeeds.
pub(in crate::vulkan) fn rebuild_fog_pipeline(
    device: &Device,
    fog: &FogResources,
    msaa: bool,
    hot_reload: bool,
) -> Result<vk::Pipeline, String> {
    let (vert_spv, frag_spv) = compile_fog_shaders(hot_reload, msaa)?;
    create_fog_pipeline(
        device,
        fog.render_pass,
        fog.pipeline_layout,
        &vert_spv,
        &frag_spv,
    )
}

// Rebuild the froxel compute pipeline against the existing layout. Hot-reload.
pub(in crate::vulkan) fn rebuild_fog_froxel_pipeline(
    device: &Device,
    fog: &FogResources,
    hot_reload: bool,
) -> Result<vk::Pipeline, String> {
    let spv = compile_fog_froxel_shader(hot_reload)?;
    create_compute_pipeline(device, fog.froxel_pipeline_layout, &spv)
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
    .map_err(|(_, e)| format!("create fog froxel pipeline: {e}"))?[0];
    unsafe { device.destroy_shader_module(module, None) };
    Ok(pipeline)
}

fn create_fog_pipeline(
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
    // Fullscreen triangle is emitted by gl_VertexIndex; no vertex buffer.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        // The fog pass writes the SINGLE-SAMPLE resolved HDR target, not
        // the MSAA colour, regardless of whether the main pass uses MSAA.
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false);
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        // (scattered, 1 - T) over scene: dst = src + (1 - src.a) * dst,
        // resolving to `final = scattered + transmittance * scene`. Matches
        // the DirectX / Metal blend.
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
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
    .map_err(|(_, e)| format!("create fog pipeline: {e}"))?[0];
    unsafe {
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
    }
    Ok(pipeline)
}

// Encoder

impl VkContext {
    // Hot-reload entry point for the volumetric-fog tunables (driven by
    // `world.jsonl` hot-reload under `cn debug`). Writes the new
    // `Option<FogSettings>` into `self.fog_settings`; the next frame's graph
    // seed re-reads it (so `None` drops the FogFroxel + Fog passes) and
    // `encode_fog_froxel` rebuilds `FogParams` / `FogFroxelParams` from it.
    // Mirrors `MtlContext::update_fog_settings`.
    //
    // If the world started with no `VolumetricFog` (so `fog_resources` is
    // `None`), a `Some` update logs once and is dropped: re-enabling fog
    // mid-run requires a relaunch (the froxel pipeline + volume were never
    // built).
    //
    // Named distinctly from the `RenderBackend::update_fog_settings` trait
    // method so the backend forwarder's `self.apply_fog_settings(...)` is
    // unambiguous. `#[allow(dead_code)]`: reached only through the
    // `RenderBackend` vtable (the bin's `cn debug` world.jsonl hot-reload), so
    // it is dead in the FFI lib but live in the bin.
    #[allow(dead_code)]
    pub(in crate::vulkan) fn apply_fog_settings(
        &mut self,
        settings: Option<crate::gfx::volumetric_fog::FogSettings>,
    ) {
        if settings.is_some() && self.fog_resources.is_none() {
            tracing::warn!(
                "VolumetricFog hot-reload: world started without fog, so the fog \
                 pipeline + froxel volume were never built: re-enabling fog mid-run \
                 is not supported (relaunch required). Ignoring update."
            );
            return;
        }
        self.fog_settings = settings;
    }

    // Encode the volumetric-fog froxel-volume compute pass. Populates the 3D
    // `(scattered, 1 - T)` volume the fog fragment shader samples. The shared
    // graph seeds `PassId::FogFroxel` before `Fog` so the RAW edge orders the
    // dispatch ahead of the render-pass read. Uploads both per-frame UBOs
    // (`FogParams` + `FogFroxelParams`) so `encode_fog` only reads them.
    pub(in crate::vulkan) fn encode_fog_froxel(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        near: f32,
        vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
    ) {
        let fog_settings = match &self.fog_settings {
            Some(s) => *s,
            None => return,
        };
        let fog = match &self.fog_resources {
            Some(f) => f,
            None => return,
        };

        let device = &self.device;

        // Per-frame FogParams (drives the volume integration + the fragment's
        // viewport / reconstruction). Uploaded here so `encode_fog` only reads.
        let inv_vp = super::math::mat4_inverse(vp);
        let viewport_pix = [
            self.render_extent.width as f32,
            self.render_extent.height as f32,
        ];
        let params = fog_settings.params(
            inv_vp,
            cam_pos,
            self.fog_sun_dir,
            self.fog_sun_color,
            viewport_pix,
        );
        // Per-frame FogFroxelParams: world->view matrix + the volume's discrete
        // dimensions + the linear-Z `[near, max_distance]` mapping. `near` is
        // clamped to >= 1e-3 so the linear-Z reconstruction stays finite.
        let froxel_params = FogFroxelParams {
            view: self.view_matrix,
            froxel_dims: [FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z],
            _pad_align: 0,
            z_near: near.max(1e-3),
            z_far: fog_settings.max_distance,
            _pad: [0.0; 2],
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &params as *const FogParams as *const u8,
                fog.params_ubo_ptrs[frame_idx],
                std::mem::size_of::<FogParams>(),
            );
            std::ptr::copy_nonoverlapping(
                &froxel_params as *const FogFroxelParams as *const u8,
                fog.froxel_ubo_ptrs[frame_idx],
                std::mem::size_of::<FogFroxelParams>(),
            );
        }

        // The froxel volume's SHADER_READ_ONLY -> GENERAL open transition is
        // graph-driven: fog_froxel_volume is the graph's resource and the
        // executor emits its FogFroxel producer (Undefined -> Write) barrier
        // before this pass, whose (FRAGMENT_SHADER, SHADER_READ) source scope
        // also orders the previous frame's fog fragment read before this write.
        // Only the shadow-map sync stays inline: shadow_map isn't a graph read of
        // this pass (the kernel's CSM tap is hand-rolled), so we order the Shadow
        // pass's depth write before this compute tap here. Same-layout barrier:
        // the map stays SHADER_READ_ONLY (the main pass already sampled it).
        let shadow_to_compute = shadow_sync_barrier(
            self.shadow.map.image,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&shadow_to_compute),
            );
        }

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, fog.froxel_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                fog.froxel_pipeline_layout,
                0,
                std::slice::from_ref(&fog.froxel_sets[frame_idx]),
                &[],
            );
            device.cmd_dispatch(
                cmd,
                FOG_FROXEL_X.div_ceil(FROXEL_TILE),
                FOG_FROXEL_Y.div_ceil(FROXEL_TILE),
                1,
            );
        }

        // The froxel volume's GENERAL -> SHADER_READ_ONLY close transition is
        // graph-driven: the executor emits the Fog consumer (Write -> Read)
        // barrier before the Fog pass. Only the shadow-map sync stays inline:
        // order this compute shadow read before next frame's shadow depth write
        // (the end-of-frame SHADER_READ -> DEPTH_STENCIL reset covers only the
        // fragment reads).
        let shadow_to_depth = shadow_sync_barrier(
            self.shadow.map.image,
            vk::AccessFlags::empty(),
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&shadow_to_depth),
            );
        }
    }

    // Encode the volumetric-fog pass. Samples the 3D froxel volume the
    // `FogFroxel` compute pass populated this frame. Caller has already ended
    // the main HDR resolve and the projected-decal pass (if any), so
    // `depth_images[frame_idx]` holds the scene depth and
    // `hdr_resolve_images[frame_idx]` holds the resolved scene + decal colour
    // in SHADER_READ_ONLY_OPTIMAL. Alpha-blends `(scattered, 1 - T)` over the
    // resolved HDR target. `FogParams` / `FogFroxelParams` were uploaded by
    // `encode_fog_froxel` for this frame's slot, so this pass only binds.
    pub(in crate::vulkan) fn encode_fog(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        _vp: [[f32; 4]; 4],
        _cam_pos: [f32; 3],
    ) {
        if self.fog_settings.is_none() {
            return;
        }
        let fog = match &self.fog_resources {
            Some(f) => f,
            None => return,
        };

        let device = &self.device;
        let extent = self.render_extent;

        // Transition main depth -> SHADER_READ_ONLY so the fragment can sample
        // it; restore to DEPTH_STENCIL_ATTACHMENT after the pass so the next
        // frame's main pass can clear/write it again. Mirrors the decal
        // encoder.
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
            .render_pass(fog.render_pass)
            .framebuffer(fog.framebuffers[frame_idx])
            .render_area(vk::Rect2D::default().extent(extent));

        // Standard positive-height viewport: the fullscreen triangle is
        // emitted in NDC and the fragment shader's reconstruction handles
        // the Y flip against the main pass's depth (which was written with
        // a negative-height viewport).
        let vp_state = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);

        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp_state));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, fog.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                fog.pipeline_layout,
                0,
                std::slice::from_ref(&fog.view_sets[frame_idx]),
                &[],
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);

            // Restore main depth -> DEPTH_STENCIL_ATTACHMENT for the next
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

// A same-layout (SHADER_READ_ONLY) image memory barrier on the shadow map
// array, used to thread the per-slab compute tap into the CSM write/reset
// chain that otherwise only orders fragment reads. Stage masks are supplied to
// `cmd_pipeline_barrier`; the layout never changes.
fn shadow_sync_barrier(
    image: vk::Image,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        })
}

#[cfg(test)]
mod tests {
    use crate::gfx::render_types::{FogFroxelParams, FogParams};
    use std::mem::size_of;

    #[test]
    fn fog_params_ubo_size_matches_glsl() {
        // The fog.vert/frag + fog_froxel.comp FogBlock std140 layout is 176 B.
        assert_eq!(size_of::<FogParams>(), 176);
    }

    #[test]
    fn fog_froxel_params_ubo_size_matches_glsl() {
        // The FogFroxelBlock std140 layout (mat4 + uvec3 + uint + 2 float + vec2)
        // is 96 B; the offsets are pinned by the core render_types tests.
        assert_eq!(size_of::<FogFroxelParams>(), 96);
    }

    #[test]
    fn fog_shaders_compile() {
        // Compile the rewritten froxel-sampling fragment shader (both MSAA
        // modes) + the froxel compute kernel so a GLSL regression fails the
        // test suite without needing a GPU. Mirrors the cull-shader compile
        // guard the two-pass occlusion landing added.
        super::compile_fog_shaders(false, false).expect("fog shaders (no MSAA) compile");
        super::compile_fog_shaders(false, true).expect("fog shaders (MSAA) compile");
        super::compile_fog_froxel_shader(false).expect("fog froxel kernel compiles");
    }
}
