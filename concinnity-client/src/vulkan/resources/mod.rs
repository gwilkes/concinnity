// src/vulkan/resources/mod.rs
//
// Runtime GPU resource management for VkContext, split per-category to mirror
// the Metal reference shape (`metal/resources/`):
//
//   textures.rs   Texture-pool slot updates + descriptor rewires (`update_*`,
//                 `evict_*`, `write_object_image`, `write_pool_image`)
//   geometry.rs   Streamed-mesh upload + eviction (`upload_mesh`,
//                 `evict_mesh`, the shared `write_geometry_region` helper)
//   streaming.rs  VoxelWorld chunk streaming (`setup_chunk_streaming`,
//                 `add_chunk_mesh`, `remove_chunk_mesh`, `set_chunk_model`)
//   skinning.rs   Skinned-mesh upload + per-frame joint upload
//                 (`upload_skinned`, `update_skinned_pose`,
//                 `upload_joint_matrices`, `skinned_geometry`)
//   geometry_rebuild.rs  Size-changing static + skinned VB/IB rebuilds
//                 driven by asset hot-reload (`rebuild_static_geometry`,
//                 `rebuild_skinned_geometry`)
//
// The shared low-level helpers (`create_descriptor_set_layout`,
// `alloc_descriptor_sets`, `upload_geometry_buffer{,_raw}`) live in this file
// because every submodule + `init.rs` needs them.

use ash::{Device, vk};

use super::texture::{self, create_buffer};

mod geometry;
mod geometry_rebuild;
mod skinning;
mod streaming;
mod textures;

pub(in crate::vulkan) fn create_descriptor_set_layout(
    device: &Device,
    bindings: &[(u32, vk::DescriptorType, vk::ShaderStageFlags)],
) -> Result<vk::DescriptorSetLayout, String> {
    let vk_bindings: Vec<_> = bindings
        .iter()
        .map(|&(b, ty, stage)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(stage)
        })
        .collect();
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("descriptor set layout: {e}"))
}

pub(in crate::vulkan) fn alloc_descriptor_sets(
    device: &Device,
    pool: vk::DescriptorPool,
    layouts: &[vk::DescriptorSetLayout],
) -> Result<Vec<vk::DescriptorSet>, String> {
    if layouts.is_empty() {
        return Ok(vec![]);
    }
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(layouts);
    unsafe { device.allocate_descriptor_sets(&alloc) }
        .map_err(|e| format!("allocate descriptor sets: {e}"))
}

pub(in crate::vulkan) fn upload_geometry_buffer<T>(
    instance: &ash::Instance,
    device: &Device,
    pd: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    data: &[T],
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
    let size = std::mem::size_of_val(data) as u64;
    upload_geometry_buffer_raw(
        instance,
        device,
        pd,
        command_pool,
        queue,
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, size as usize) },
        usage,
    )
}

pub(in crate::vulkan) fn upload_geometry_buffer_raw(
    instance: &ash::Instance,
    device: &Device,
    pd: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    data: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
    // TRANSFER_SRC lets `setup_chunk_streaming` copy the build-time geometry
    // out of these buffers when it grows them for chunk-streaming headroom;
    // TRANSFER_DST lets the staging copy below and `write_geometry_region`
    // write into them.
    let usage = usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    let size = data.len() as u64;
    if size == 0 {
        // Return a minimal 4-byte buffer to keep Vulkan happy.
        return create_buffer(
            instance,
            device,
            pd,
            4,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
    }
    let (staging, staging_mem) = create_buffer(
        instance,
        device,
        pd,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        let ptr = device
            .map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("map staging geo: {e}"))? as *mut u8;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, size as usize);
        device.unmap_memory(staging_mem);
    }
    let (buf, mem) = create_buffer(
        instance,
        device,
        pd,
        size,
        usage,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    texture::one_shot_submit(device, command_pool, queue, |cmd| {
        let copy = vk::BufferCopy::default().size(size);
        unsafe { device.cmd_copy_buffer(cmd, staging, buf, std::slice::from_ref(&copy)) };
    })?;
    unsafe {
        device.destroy_buffer(staging, None);
        device.free_memory(staging_mem, None);
    }
    Ok((buf, mem))
}
