// Vulkan image/texture creation helpers.
// All uploads go through a host-visible staging buffer that is blit-copied to
// a device-local image via a one-shot command buffer.

use ash::{Device, vk};

// Opaque handle to a GPU image and its backing memory.
pub(super) struct GpuImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    // Auxiliary image views for the same image (e.g. per-cascade DSVs for an
    // array shadow map). Destroyed alongside `view`.
    pub aux_views: Vec<vk::ImageView>,
}

impl GpuImage {
    pub(super) fn destroy(&self, device: &Device) {
        unsafe {
            for &v in &self.aux_views {
                device.destroy_image_view(v, None);
            }
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

// Find a memory type index that satisfies both the type filter and required properties.
pub(super) fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32, String> {
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for i in 0..mem_props.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(properties)
        {
            return Ok(i);
        }
    }
    Err("no suitable memory type found".to_string())
}

// Allocate a VkBuffer with its own DeviceMemory.
pub(super) fn create_buffer(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    mem_props: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
    let buf_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buf_info, None) }
        .map_err(|e| format!("create_buffer: {e}"))?;
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    // A buffer created with SHADER_DEVICE_ADDRESS usage (the ray-tracing
    // acceleration-structure buffers + their build inputs) needs its backing
    // memory allocated with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, or
    // `get_buffer_device_address` is invalid. The flag is inert (and the chain
    // omitted) for every other buffer, so this is a no-op when RT is off.
    let mut flags_info =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let needs_device_address = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
    let mut alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(find_memory_type(
            instance,
            physical_device,
            reqs.memory_type_bits,
            mem_props,
        )?);
    if needs_device_address {
        alloc_info = alloc_info.push_next(&mut flags_info);
    }
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| format!("allocate_memory (buffer): {e}"))?;
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }
        .map_err(|e| format!("bind_buffer_memory: {e}"))?;
    Ok((buffer, memory))
}

// Allocate a VkImage with its own DeviceMemory.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_image(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    mem_props: vk::MemoryPropertyFlags,
    samples: vk::SampleCountFlags,
) -> Result<(vk::Image, vk::DeviceMemory), String> {
    let img_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(samples);
    let image = unsafe { device.create_image(&img_info, None) }
        .map_err(|e| format!("create_image: {e}"))?;
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(find_memory_type(
            instance,
            physical_device,
            reqs.memory_type_bits,
            mem_props,
        )?);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| format!("allocate_memory (image): {e}"))?;
    unsafe { device.bind_image_memory(image, memory, 0) }
        .map_err(|e| format!("bind_image_memory: {e}"))?;
    Ok((image, memory))
}

// Create a VkImageView for a 2-D image.
pub(super) fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView, String> {
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    unsafe { device.create_image_view(&view_info, None) }
        .map_err(|e| format!("create_image_view: {e}"))
}

// Execute a short-lived command buffer and wait for it to complete.
pub(super) fn one_shot_submit<F>(
    device: &Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    f: F,
) -> Result<(), String>
where
    F: FnOnce(vk::CommandBuffer),
{
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }
        .map_err(|e| format!("one_shot allocate: {e}"))?[0];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd, &begin_info) }
        .map_err(|e| format!("one_shot begin: {e}"))?;

    f(cmd);

    unsafe { device.end_command_buffer(cmd) }.map_err(|e| format!("one_shot end: {e}"))?;

    let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
    unsafe { device.queue_submit(queue, std::slice::from_ref(&submit_info), vk::Fence::null()) }
        .map_err(|e| format!("one_shot submit: {e}"))?;
    unsafe { device.queue_wait_idle(queue) }.map_err(|e| format!("one_shot wait: {e}"))?;

    unsafe { device.free_command_buffers(command_pool, std::slice::from_ref(&cmd)) };
    Ok(())
}

// Transition `layer_count` layers of an image from one layout to another via
// a pipeline barrier. Used by the array shadow map and cube uploads.
#[allow(clippy::too_many_arguments)]
pub(super) fn transition_image_layout_range(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
    base_layer: u32,
    layer_count: u32,
    base_mip: u32,
    mip_count: u32,
) {
    let (src_access, dst_access, src_stage, dst_stage) =
        layout_transition_access(old_layout, new_layout);

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(base_mip)
                .level_count(mip_count)
                .base_array_layer(base_layer)
                .layer_count(layer_count),
        )
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);

    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&barrier),
        );
    }
}

fn layout_transition_access(
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> (
    vk::AccessFlags,
    vk::AccessFlags,
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
) {
    match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        ),
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ),
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        (
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ) => (
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        (
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ) => (
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        ),
        _ => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
        ),
    }
}

// Transition an image from one layout to another via a pipeline barrier.
pub(super) fn transition_image_layout(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
) {
    transition_image_layout_range(
        device, cmd, image, old_layout, new_layout, aspect, 0, 1, 0, 1,
    );
}

// Variant of `transition_image_layout` that covers every layer of an array
// image (e.g. the 4-layer shadow array).
pub(super) fn transition_image_layout_array(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
    layer_count: u32,
) {
    transition_image_layout_range(
        device,
        cmd,
        image,
        old_layout,
        new_layout,
        aspect,
        0,
        layer_count,
        0,
        1,
    );
}

// Upload RGBA pixel data to a device-local RGBA8_UNORM image with a full mip
// chain. The chain is box-filtered on the CPU (`crate::gfx::mipmap`) and every
// level is uploaded so the texture minifies through hardware trilinear / aniso
// selection instead of aliasing from a single mip-0 sample at a distance.
#[allow(clippy::too_many_arguments)]
pub(super) fn upload_texture(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<GpuImage, String> {
    let base = (width as usize) * (height as usize) * 4;
    if pixels.len() < base {
        return Err(format!(
            "pixel data too short for {}x{} RGBA texture ({} bytes, need {})",
            width,
            height,
            pixels.len(),
            base
        ));
    }

    let chain = crate::gfx::mipmap::generate_mip_chain(width, height, pixels);
    let mip_count = chain.len() as u32;

    // One packed staging buffer holding mip 0..N concatenated.
    let total: usize = chain.iter().map(|m| m.pixels.len()).sum();
    let (staging_buf, staging_mem) = create_buffer(
        instance,
        device,
        physical_device,
        total as vk::DeviceSize,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        let ptr = device
            .map_memory(
                staging_mem,
                0,
                total as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("map staging: {e}"))? as *mut u8;
        let mut off = 0usize;
        for m in &chain {
            std::ptr::copy_nonoverlapping(m.pixels.as_ptr(), ptr.add(off), m.pixels.len());
            off += m.pixels.len();
        }
        device.unmap_memory(staging_mem);
    }

    // Device-local image with the full mip chain.
    let img_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(mip_count)
        .array_layers(1)
        .format(vk::Format::R8G8B8A8_UNORM)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let image = unsafe { device.create_image(&img_info, None) }
        .map_err(|e| format!("create_image: {e}"))?;
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(find_memory_type(
            instance,
            physical_device,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| format!("allocate_memory (image): {e}"))?;
    unsafe { device.bind_image_memory(image, memory, 0) }
        .map_err(|e| format!("bind_image_memory: {e}"))?;

    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout_range(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            0,
            1,
            0,
            mip_count,
        );
        let mut regions: Vec<vk::BufferImageCopy> = Vec::with_capacity(mip_count as usize);
        let mut off = 0u64;
        for (m, level) in chain.iter().enumerate() {
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(off)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(m as u32)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D::default())
                    .image_extent(vk::Extent3D {
                        width: level.width,
                        height: level.height,
                        depth: 1,
                    }),
            );
            off += level.pixels.len() as u64;
        }
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                staging_buf,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
        transition_image_layout_range(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            0,
            1,
            0,
            mip_count,
        );
    })?;

    unsafe {
        device.destroy_buffer(staging_buf, None);
        device.free_memory(staging_mem, None);
    }

    // View spanning every mip.
    let view = {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_count)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        unsafe { device.create_image_view(&info, None) }
            .map_err(|e| format!("create_image_view: {e}"))?
    };

    Ok(GpuImage {
        image,
        memory,
        view,
        aux_views: Vec::new(),
    })
}

// Create a 1x1 opaque white RGBA texture (fallback when no albedo asset is present).
pub(super) fn create_fallback_white(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Result<GpuImage, String> {
    upload_texture(
        instance,
        device,
        physical_device,
        command_pool,
        queue,
        1,
        1,
        &[255u8, 255, 255, 255],
    )
}

// Create a 1x1 flat-normal RGBA texture (tangent-space (0,0,1) = no perturbation).
pub(super) fn create_fallback_flat_normal(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Result<GpuImage, String> {
    upload_texture(
        instance,
        device,
        physical_device,
        command_pool,
        queue,
        1,
        1,
        &[128u8, 128, 255, 255],
    )
}

// Upload a 3D colour-grading LUT from a `ColorLut` payload. `data` is the raw
// RGBA8 emitted by `build/color_lut.rs`: `size`³ texels ordered red-fastest,
// then green, then blue, which is the natural row/slice order of a `TYPE_3D`
// image, so the byte slice copies in verbatim. The returned `GpuImage` has a
// `VK_IMAGE_VIEW_TYPE_3D` view left in `SHADER_READ_ONLY_OPTIMAL`, ready for
// the composite pass to sample as a `sampler3D`.
pub(super) fn upload_color_lut(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    size: u32,
    data: &[u8],
) -> Result<GpuImage, String> {
    let needed = (size as usize).pow(3) * 4;
    if data.len() < needed {
        return Err(format!(
            "color LUT data too short for size {}: {} bytes, need {}",
            size,
            data.len(),
            needed
        ));
    }

    // Staging buffer (host visible).
    let byte_size = needed as vk::DeviceSize;
    let (staging_buf, staging_mem) = create_buffer(
        instance,
        device,
        physical_device,
        byte_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        let ptr = device
            .map_memory(staging_mem, 0, byte_size, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("map LUT staging: {e}"))? as *mut u8;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, needed);
        device.unmap_memory(staging_mem);
    }

    // Device-local 3D image. `create_image` is TYPE_2D only, so the LUT image
    // is built inline.
    let img_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_3D)
        .extent(vk::Extent3D {
            width: size,
            height: size,
            depth: size,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(vk::Format::R8G8B8A8_UNORM)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let image = unsafe { device.create_image(&img_info, None) }
        .map_err(|e| format!("create_image (LUT): {e}"))?;
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(find_memory_type(
            instance,
            physical_device,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| format!("allocate_memory (LUT): {e}"))?;
    unsafe { device.bind_image_memory(image, memory, 0) }
        .map_err(|e| format!("bind_image_memory (LUT): {e}"))?;

    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        );
        let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width: size,
                height: size,
                depth: size,
            });
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                staging_buf,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy_region),
            );
        }
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        );
    })?;

    unsafe {
        device.destroy_buffer(staging_buf, None);
        device.free_memory(staging_mem, None);
    }

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_3D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    let view = unsafe { device.create_image_view(&view_info, None) }
        .map_err(|e| format!("create_image_view (LUT): {e}"))?;

    Ok(GpuImage {
        image,
        memory,
        view,
        aux_views: Vec::new(),
    })
}

// Build a 2x2x2 identity colour LUT: the eight corners of the unit RGB cube.
// Mirrors `metal/texture.rs::create_fallback_color_lut`. With the identity LUT
// the composite grade is a no-op at any `lut_strength`, so the `sampler3D`
// binding stays valid even when the world declares no `ColorLut`.
pub(super) fn create_fallback_color_lut(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Result<GpuImage, String> {
    // Red-fastest, then green, then blue, matching the payload texel order.
    let mut data = Vec::with_capacity(2 * 2 * 2 * 4);
    for b in 0..2u8 {
        for g in 0..2u8 {
            for r in 0..2u8 {
                data.extend_from_slice(&[r * 255, g * 255, b * 255, 255]);
            }
        }
    }
    upload_color_lut(
        instance,
        device,
        physical_device,
        command_pool,
        queue,
        2,
        &data,
    )
}

// Create a `layers`-slice D32_SFLOAT array shadow map. The returned `view` is
// a single sampled 2D-array view covering every layer (bound at descriptor
// set=0 binding=3 in the main pass); `aux_views` holds one single-layer 2D
// view per cascade for use as a per-slice depth attachment in the shadow
// pass.
//
// When `size > 0`, creates a full shadow map; otherwise a 1×1 single-layer
// fallback (depth=1.0 = fully lit). The fallback intentionally uses a single
// array layer because the shader's cascade selection falls back to cascade 0
// when `cascade_splits == +inf`, so layer 0 is the only one ever sampled.
pub(super) fn create_shadow_map_array(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    size: u32,
    layers: u32,
) -> Result<GpuImage, String> {
    let (w, h, layer_count) = if size > 0 {
        (size, size, layers.max(1))
    } else {
        (1, 1, 1)
    };
    let img_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(layer_count)
        .format(vk::Format::D32_SFLOAT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let image = unsafe { device.create_image(&img_info, None) }
        .map_err(|e| format!("create_image (shadow array): {e}"))?;
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(find_memory_type(
            instance,
            physical_device,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| format!("allocate_memory (shadow array): {e}"))?;
    unsafe { device.bind_image_memory(image, memory, 0) }
        .map_err(|e| format!("bind_image_memory (shadow array): {e}"))?;

    // Rest the cascades sampled. The graph's Shadow producer barrier transitions
    // them to DEPTH_STENCIL_ATTACHMENT before each shadow loop and Main's consumer
    // returns them here, so the cross-frame reset is the graph's producer barrier,
    // not an inline end-of-frame transition. Initialising sampled makes frame 0's
    // producer barrier (SHADER_READ_ONLY -> DEPTH_STENCIL_ATTACHMENT) start from
    // the image's real layout.
    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout_array(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::DEPTH,
            layer_count,
        );
    })?;

    // Sampled view: 2D array over every layer.
    let view = {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(layer_count),
            );
        unsafe { device.create_image_view(&info, None) }
            .map_err(|e| format!("shadow array view: {e}"))?
    };

    // Per-slice attachment views (one per cascade).
    let mut aux_views = Vec::with_capacity(layer_count as usize);
    for i in 0..layer_count {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(i)
                    .layer_count(1),
            );
        let v = unsafe { device.create_image_view(&info, None) }
            .map_err(|e| format!("shadow slice view {i}: {e}"))?;
        aux_views.push(v);
    }

    Ok(GpuImage {
        image,
        memory,
        view,
        aux_views,
    })
}

// Create a device-local depth image for the main render pass.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_depth_image(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    width: u32,
    height: u32,
    samples: vk::SampleCountFlags,
) -> Result<GpuImage, String> {
    let (image, memory) = create_image(
        instance,
        device,
        physical_device,
        width,
        height,
        vk::Format::D32_SFLOAT,
        vk::ImageTiling::OPTIMAL,
        // SAMPLED so the projected-decal pass (and any future depth-
        // sampling effect) can read it from a fragment shader. Without
        // this, the validation layer rejects the SHADER_READ_ONLY layout
        // transition and the SAMPLED-bit-required `vkUpdateDescriptorSets`
        // write that bind the depth view to the decal's set 0 binding 2.
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        samples,
    )?;
    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageAspectFlags::DEPTH,
        );
    })?;
    let view = create_image_view(
        device,
        image,
        vk::Format::D32_SFLOAT,
        vk::ImageAspectFlags::DEPTH,
    )?;
    Ok(GpuImage {
        image,
        memory,
        view,
        aux_views: Vec::new(),
    })
}

// Create a multisampled color image for the MSAA resolve target.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_msaa_color_image(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    width: u32,
    height: u32,
    format: vk::Format,
    samples: vk::SampleCountFlags,
) -> Result<GpuImage, String> {
    let (image, memory) = create_image(
        instance,
        device,
        physical_device,
        width,
        height,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        samples,
    )?;
    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        );
    })?;
    let view = create_image_view(device, image, format, vk::ImageAspectFlags::COLOR)?;
    Ok(GpuImage {
        image,
        memory,
        view,
        aux_views: Vec::new(),
    })
}

// Create a single-sample colour image usable as both a render target and a
// sampled texture. This is the HDR resolve target: the main pass resolves
// (or, with MSAA off, draws directly) into it, and the composite pass samples
// it to tonemap. No pre-transition is needed: the main render pass declares
// an `UNDEFINED` initial layout for it.
pub(super) fn create_hdr_resolve_image(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<GpuImage, String> {
    let (image, memory) = create_image(
        instance,
        device,
        physical_device,
        width,
        height,
        format,
        vk::ImageTiling::OPTIMAL,
        // TRANSFER_SRC so the raymarch pass can snapshot the resolved scene into
        // its `scene_color` refraction tap before compositing the SDF volumes.
        vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::SampleCountFlags::TYPE_1,
    )?;
    let view = create_image_view(device, image, format, vk::ImageAspectFlags::COLOR)?;
    Ok(GpuImage {
        image,
        memory,
        view,
        aux_views: Vec::new(),
    })
}

// Linear repeat sampler for albedo and normal map sampling. Now that scene
// textures carry a full mip chain, `max_lod` is unclamped so minified surfaces
// trilinear-select down the chain. `max_anisotropy > 1.0` enables anisotropic
// filtering (the caller passes the device-supported degree, or <= 1.0 when the
// `samplerAnisotropy` feature is unavailable).
pub(super) fn create_sampler_linear_repeat(
    device: &Device,
    max_anisotropy: f32,
) -> Result<vk::Sampler, String> {
    let aniso = max_anisotropy > 1.0;
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(aniso)
        .max_anisotropy(if aniso { max_anisotropy } else { 1.0 })
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .min_lod(0.0)
        .max_lod(vk::LOD_CLAMP_NONE);
    unsafe { device.create_sampler(&info, None) }.map_err(|e| format!("linear repeat sampler: {e}"))
}

// Compare sampler for PCF shadow sampling (LessEqual compare op).
pub(super) fn create_sampler_shadow(device: &Device) -> Result<vk::Sampler, String> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
        .unnormalized_coordinates(false)
        .compare_enable(true)
        .compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR);
    unsafe { device.create_sampler(&info, None) }.map_err(|e| format!("shadow sampler: {e}"))
}

// Linear clamp sampler for text atlas lookups.
pub(super) fn create_sampler_linear_clamp(device: &Device) -> Result<vk::Sampler, String> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR);
    unsafe { device.create_sampler(&info, None) }.map_err(|e| format!("linear clamp sampler: {e}"))
}

// IBL textures produced by a single `EnvironmentMap` asset. Mirrors the Metal
// `EnvironmentMapTextures` shape so the fragment-shader code stays portable.
// `prefilter_mip_count == 0` is the runtime signal for "IBL disabled": the
// fragment shader keys off it and falls back to the legacy ambient path.
pub(super) struct EnvironmentMapTextures {
    pub irradiance: GpuImage,
    pub prefilter: GpuImage,
    pub prefilter_mip_count: u32,
}

// Create a RGBA32_SFLOAT cubemap image with `mip_count` mips, then upload the
// supplied byte slices via a staging buffer. `mip_bytes[m]` must hold
// `6 * (face_size >> m)² * 16` bytes in face-major order
// (+X, -X, +Y, -Y, +Z, -Z). Returns a `GpuImage` whose `view` is a
// `VK_IMAGE_VIEW_TYPE_CUBE` view spanning every mip.
fn create_cube_image(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    face_size: u32,
    mip_bytes: &[&[u8]],
) -> Result<GpuImage, String> {
    let mip_count = mip_bytes.len() as u32;
    if mip_count == 0 {
        return Err("cubemap upload: mip_bytes must not be empty".into());
    }

    // Validate each mip and compute the staging buffer footprint.
    let mut mip_sizes: Vec<usize> = Vec::with_capacity(mip_count as usize);
    let mut total: usize = 0;
    for (m, bytes) in mip_bytes.iter().enumerate() {
        let s = (face_size >> m) as usize;
        if s == 0 {
            return Err(format!(
                "cubemap mip {} would have zero face size (face_size {} too small)",
                m, face_size
            ));
        }
        let face_bytes = s * s * 16;
        let needed = 6 * face_bytes;
        if bytes.len() < needed {
            return Err(format!(
                "cubemap mip {} too short: {} bytes, need {}",
                m,
                bytes.len(),
                needed
            ));
        }
        mip_sizes.push(needed);
        total += needed;
    }

    // Create the image: array_layers=6, with the CUBE_COMPATIBLE flag.
    let img_info = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: face_size,
            height: face_size,
            depth: 1,
        })
        .mip_levels(mip_count)
        .array_layers(6)
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let image = unsafe { device.create_image(&img_info, None) }
        .map_err(|e| format!("create_image (cube): {e}"))?;
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(find_memory_type(
            instance,
            physical_device,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| format!("allocate_memory (cube): {e}"))?;
    unsafe { device.bind_image_memory(image, memory, 0) }
        .map_err(|e| format!("bind_image_memory (cube): {e}"))?;

    // Build one packed staging buffer with mip 0..N concatenated.
    let (staging_buf, staging_mem) = create_buffer(
        instance,
        device,
        physical_device,
        total as vk::DeviceSize,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        let ptr = device
            .map_memory(
                staging_mem,
                0,
                total as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("map cube staging: {e}"))? as *mut u8;
        let mut off = 0usize;
        for (m, bytes) in mip_bytes.iter().enumerate() {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(off), mip_sizes[m]);
            off += mip_sizes[m];
        }
        device.unmap_memory(staging_mem);
    }

    // Transition all 6 layers / N mips to TRANSFER_DST, copy each face per mip,
    // then transition to SHADER_READ_ONLY_OPTIMAL.
    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout_range(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            0,
            6,
            0,
            mip_count,
        );

        // One BufferImageCopy per (mip, face).
        let mut regions: Vec<vk::BufferImageCopy> = Vec::with_capacity((mip_count * 6) as usize);
        let mut off = 0u64;
        for m in 0..mip_count as usize {
            let s = face_size >> m;
            let face_bytes = (s as u64) * (s as u64) * 16;
            for face in 0..6u32 {
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(off)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(m as u32)
                                .base_array_layer(face)
                                .layer_count(1),
                        )
                        .image_offset(vk::Offset3D::default())
                        .image_extent(vk::Extent3D {
                            width: s,
                            height: s,
                            depth: 1,
                        }),
                );
                off += face_bytes;
            }
        }
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                staging_buf,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }

        transition_image_layout_range(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            0,
            6,
            0,
            mip_count,
        );
    })?;

    unsafe {
        device.destroy_buffer(staging_buf, None);
        device.free_memory(staging_mem, None);
    }

    // Single cube view covering all mips.
    let view = {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::CUBE)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_count)
                    .base_array_layer(0)
                    .layer_count(6),
            );
        unsafe { device.create_image_view(&info, None) }.map_err(|e| format!("cube view: {e}"))?
    };

    Ok(GpuImage {
        image,
        memory,
        view,
        aux_views: Vec::new(),
    })
}

// Upload a six-face HDR cubemap from a `CubemapTexture` payload. RGBA32F,
// 6 * face_size * face_size * 16 bytes in face-major order
// (+X, -X, +Y, -Y, +Z, -Z). Single-mip.
#[allow(dead_code)]
pub(super) fn upload_cubemap(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    face_size: u32,
    bytes: &[u8],
) -> Result<GpuImage, String> {
    create_cube_image(
        instance,
        device,
        physical_device,
        command_pool,
        queue,
        face_size,
        &[bytes],
    )
}

// Create a 1×1 RGBA32F cube of `value` for every face. Used as the IBL
// fallback when no `EnvironmentMap` is bound: the fragment shader keys off
// `prefilter_mip_count == 0` and skips IBL math, but the cube binding must
// still resolve to a valid texture.
pub(super) fn create_fallback_cubemap(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    value: [f32; 4],
) -> Result<GpuImage, String> {
    let mut face_bytes = Vec::with_capacity(6 * 16);
    for _ in 0..6 {
        for v in &value {
            face_bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    create_cube_image(
        instance,
        device,
        physical_device,
        command_pool,
        queue,
        1,
        &[&face_bytes],
    )
}

// Upload an `EnvironmentMap` payload into two cube textures: a single-mip
// irradiance cube and a multi-mip prefiltered radiance cube. Mirrors the
// Metal and DirectX upload paths.
#[allow(clippy::too_many_arguments)]
pub(super) fn upload_environment_map(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    irradiance_face: u32,
    irradiance_bytes: &[u8],
    prefilter_face: u32,
    mip_bytes: &[&[u8]],
) -> Result<EnvironmentMapTextures, String> {
    if mip_bytes.is_empty() {
        return Err("envmap upload: prefilter mip_bytes must not be empty".into());
    }
    let irradiance = create_cube_image(
        instance,
        device,
        physical_device,
        command_pool,
        queue,
        irradiance_face,
        &[irradiance_bytes],
    )
    .map_err(|e| format!("envmap irradiance: {e}"))?;
    let prefilter = create_cube_image(
        instance,
        device,
        physical_device,
        command_pool,
        queue,
        prefilter_face,
        mip_bytes,
    )
    .map_err(|e| format!("envmap prefilter: {e}"))?;
    Ok(EnvironmentMapTextures {
        irradiance,
        prefilter,
        prefilter_mip_count: mip_bytes.len() as u32,
    })
}

// Linear-clamp sampler with full mipmap support, used by the IBL prefilter
// cube (roughness → mip selection) and the irradiance cube.
pub(super) fn create_sampler_cube_linear(device: &Device) -> Result<vk::Sampler, String> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .min_lod(0.0)
        .max_lod(vk::LOD_CLAMP_NONE);
    unsafe { device.create_sampler(&info, None) }.map_err(|e| format!("cube sampler: {e}"))
}
