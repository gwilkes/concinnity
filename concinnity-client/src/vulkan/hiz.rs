// src/vulkan/hiz.rs
//
// Hi-Z (depth-mip pyramid) build pass used by the GPU-cull compute kernel for
// occlusion culling. Each frame, after the main depth buffer has been written
// by the graph, we reduce it into an `R32_SFLOAT` mip chain (MAX reduction:
// standard depth, so larger = farther). The *next* frame's `Cull` kernel
// projects each `DrawObject` AABB through the previous frame's un-jittered
// view-projection, picks the Hi-Z mip whose texels are ~the size of the
// projected rect, 4-tap-samples the max occluder depth, and culls the AABB when
// its nearest projected NDC depth is strictly behind. Mirrors the DirectX
// implementation in `directx/hiz.rs` + `directx/shaders/hiz_build.hlsl` and the
// Metal one in `metal/hiz.rs` + `metal/shaders/hiz_build.metal`.
//
// Two compute kernels build the pyramid:
//
//   * `hiz_init.comp`     : reduce the (MSAA) main depth into mip 0, taking the
//                           MAX over every sample so the result is conservative.
//   * `hiz_downsample.comp`: MAX-reduce 2x2 source texels into the next mip.
//
// The pyramid is *not* a graph node: it runs inline on the frame's command
// buffer at the end of `record_frame`, after `execute_graph` returns (which
// already recorded the Main pass that writes depth, plus any decal / fog passes
// that restore depth to `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`). Treating it as an
// end-of-frame action keeps it off the render-graph dispatch and off the main
// depth attachment's in-graph layout chain.
//
// Each mip is written through its own single-level R32F storage-image view; the
// whole Hi-Z image stays in GENERAL during the build, with a compute
// write -> read memory barrier between each step. Between frames the image
// rests in `SHADER_READ_ONLY_OPTIMAL` so the cull kernel samples it via a
// `sampler2D` (set 1). A single shared image read one frame / written the next
// is hazard-free on a single queue: the build's closing GENERAL ->
// SHADER_READ_ONLY barrier (dstStage COMPUTE, dstAccess SHADER_READ) orders the
// write before the next frame's cull read, and the build's opening
// SHADER_READ_ONLY -> GENERAL barrier orders that read before this frame's
// write.

use ash::{Device, vk};

use super::pipeline::{compile_glsl, inject_define, shader_source, spv_module};
use super::resources::alloc_descriptor_sets;
use super::texture::{
    create_buffer, find_memory_type, one_shot_submit, transition_image_layout_range,
};

const HIZ_INIT_GLSL: &str = include_str!("shaders/hiz_init.comp");
const HIZ_DOWNSAMPLE_GLSL: &str = include_str!("shaders/hiz_downsample.comp");

// Upper bound on the Hi-Z mip count, used to size the dedicated descriptor pool
// for the per-downsample-step sets. `hiz_mip_count` caps at 32 - leading_zeros,
// so 16 covers any render target up to 32768 px on its longer edge.
const MAX_HIZ_MIPS: usize = 16;

// Compute threadgroup tile size for the Hi-Z build kernels (8x8, matching the
// DirectX `[numthreads(8, 8, 1)]` and the Metal `HIZ_TILE`).
const HIZ_TILE: u32 = 8;

// One per-frame ring of cull-read uniform buffers: the buffers, their backing
// memory, and their persistently-mapped host pointers.
type CullUboRing = (Vec<vk::Buffer>, Vec<vk::DeviceMemory>, Vec<*mut u8>);

// Per-dispatch params pushed inline at the build kernels' push-constant block.
// Must match the `HizParams` block in `shaders/hiz_init.comp` /
// `shaders/hiz_downsample.comp`.
#[derive(Copy, Clone)]
#[repr(C)]
struct HizParams {
    dst_width: u32,
    dst_height: u32,
    src_mip: u32,
    sample_count: u32,
}

// Cull-side Hi-Z uniforms (set 1, binding 1). Bound by `encode_cull` each
// frame. Layout must match the `CullHizParams` std140 UBO in
// `shaders/cull.comp` (80 bytes) and the Metal / DirectX CullUniforms tail.
#[derive(Copy, Clone)]
#[repr(C)]
pub(in crate::vulkan) struct CullHizParams {
    // Previous frame's un-jittered view-projection. Projects each AABB into the
    // depth space the Hi-Z pyramid was reduced from (`M * v`).
    pub prev_view_proj: [[f32; 4]; 4],
    // Hi-Z mip-0 dimensions (in texels).
    pub hiz_size: [f32; 2],
    // How many mip levels live in the bound texture.
    pub hiz_mip_count: u32,
    // 0 skips the Hi-Z test entirely (first frame / after a resize, before a
    // valid pyramid exists).
    pub hiz_enabled: u32,
}

// Mip count for a Hi-Z of size (w, h): `floor(log2(max(w, h))) + 1`. Power-of-
// two sources end exactly at 1x1; non-power-of-two sources stop one mip short
// of true 1x1 in the smaller dimension, which is fine: the cull kernel clamps
// to the actual mip dims. Mirrors `directx::hiz::hiz_mip_count` /
// `metal::hiz::hiz_mip_count`.
pub(super) fn hiz_mip_count(width: u32, height: u32) -> u32 {
    let m = width.max(height).max(1);
    32 - m.leading_zeros()
}

// Compute pipelines + image + per-mip views + descriptor sets for the Hi-Z
// build, plus the cull-read set (set 1 of the cull pipeline) and its per-frame
// uniform buffers. `Some` on the context exactly when the GPU-cull pipeline is
// active (same gating as `cull_pipeline`).
pub(super) struct HiZResources {
    // Build pipelines + their layouts (the init and downsample kernels bind
    // different set layouts, so each needs its own pipeline layout).
    init_pipeline: vk::Pipeline,
    downsample_pipeline: vk::Pipeline,
    init_pipeline_layout: vk::PipelineLayout,
    downsample_pipeline_layout: vk::PipelineLayout,
    init_set_layout: vk::DescriptorSetLayout,
    downsample_set_layout: vk::DescriptorSetLayout,

    // Cull-read set layout (set 1 of the cull pipeline): sampler2D Hi-Z +
    // CullHizParams UBO. Held here because `init.rs` threads it into the cull
    // pipeline layout, and the layout survives a resize.
    pub(super) read_set_layout: vk::DescriptorSetLayout,

    // Dedicated descriptor pool for every Hi-Z set (build + cull-read).
    descriptor_pool: vk::DescriptorPool,

    // R32F mip-chain image. Written mip-by-mip during the build (GENERAL),
    // sampled by the cull kernel between frames (SHADER_READ_ONLY).
    image: vk::Image,
    memory: vk::DeviceMemory,
    // All-mips sampled view bound in the cull-read set.
    sampled_view: vk::ImageView,
    // One single-level storage view per mip, bound as the init dst (mip 0) and
    // the downsample src/dst. Length = `mip_count`.
    mip_views: Vec<vk::ImageView>,
    // Nearest sampler the cull kernel reads the Hi-Z with (texelFetch ignores
    // filtering, but a sampler is still required for the combined-image-sampler
    // binding).
    sampler: vk::Sampler,

    // Build sets: one init set per frame (depth differs per frame slot), one
    // downsample set per mip step (frame-independent, Hi-Z mips only).
    init_sets: Vec<vk::DescriptorSet>,
    downsample_sets: Vec<vk::DescriptorSet>,
    // Cull-read sets, one per frame (the UBO differs per frame slot).
    pub(super) read_sets: Vec<vk::DescriptorSet>,
    // Per-frame CullHizParams uniform buffers (host-mapped), bound in
    // `read_sets[i]` binding 1 and written by `encode_cull`.
    cull_ubo_memories: Vec<vk::DeviceMemory>,
    pub(super) cull_ubo_ptrs: Vec<*mut u8>,
    cull_ubos: Vec<vk::Buffer>,

    // Two-pass occlusion phase-2 cull-read sets + their own per-frame UBOs.
    // Empty unless two-pass occlusion is active. The phase-2 `Cull2` dispatch
    // needs a separate UBO from phase 1 because both consume their UBO at
    // different GPU times within one frame (phase 1's `prev_view_proj`, phase
    // 2's current-frame VP); sharing one host-mapped buffer would clobber
    // phase 1's value before the GPU reads it. The sampler binding points at
    // the same pyramid `sampled_view`, re-pointed alongside `read_sets` on a
    // resize. Uses the shared `read_set_layout`.
    pub(super) read_sets2: Vec<vk::DescriptorSet>,
    cull_ubo2_memories: Vec<vk::DeviceMemory>,
    pub(super) cull_ubo2_ptrs: Vec<*mut u8>,
    cull_ubos2: Vec<vk::Buffer>,

    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) mip_count: u32,
    // MSAA sample count of the main depth the init kernel reduces (1 when the
    // world is single-sampled).
    sample_count: u32,
}

// Create the R32F mip-chain image + memory (STORAGE + SAMPLED), GPU-local.
fn create_hiz_image(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    mip_count: u32,
) -> Result<(vk::Image, vk::DeviceMemory), String> {
    let img_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: width.max(1),
            height: height.max(1),
            depth: 1,
        })
        .mip_levels(mip_count.max(1))
        .array_layers(1)
        .format(vk::Format::R32_SFLOAT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let image =
        unsafe { device.create_image(&img_info, None) }.map_err(|e| format!("hiz image: {e}"))?;
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
        .map_err(|e| format!("hiz image memory: {e}"))?;
    unsafe { device.bind_image_memory(image, memory, 0) }
        .map_err(|e| format!("hiz bind image memory: {e}"))?;
    Ok((image, memory))
}

// A single-level (`mip`) or all-mips (`base_mip = 0`, `count = mip_count`) 2D
// R32F view of the Hi-Z image.
fn create_hiz_view(
    device: &Device,
    image: vk::Image,
    base_mip: u32,
    level_count: u32,
) -> Result<vk::ImageView, String> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R32_SFLOAT)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: base_mip,
            level_count,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe { device.create_image_view(&info, None) }.map_err(|e| format!("hiz view: {e}"))
}

// Build the init/downsample pipelines for the given MSAA mode. Returns the two
// pipelines; the layouts are created by the caller and outlive a hot-reload.
fn build_hiz_pipelines(
    device: &Device,
    init_layout: vk::PipelineLayout,
    downsample_layout: vk::PipelineLayout,
    sample_count: u32,
    hot_reload: bool,
) -> Result<(vk::Pipeline, vk::Pipeline), String> {
    // The init kernel branches on a `USE_MSAA` define injected after `#version`
    // (the depth resource is a `sampler2DMS` when multisampled, a `sampler2D`
    // otherwise), mirroring the decal shader's MSAA split.
    let init_src_raw = shader_source(hot_reload, "hiz_init.comp", HIZ_INIT_GLSL);
    let define = if sample_count > 1 {
        "#define USE_MSAA 1\n"
    } else {
        "#define USE_MSAA 0\n"
    };
    let init_src = inject_define(&init_src_raw, define);
    let init_spv = compile_glsl(&init_src, shaderc::ShaderKind::Compute, "hiz_init.glsl")?;
    let downsample_src = shader_source(hot_reload, "hiz_downsample.comp", HIZ_DOWNSAMPLE_GLSL);
    let downsample_spv = compile_glsl(
        &downsample_src,
        shaderc::ShaderKind::Compute,
        "hiz_downsample.glsl",
    )?;
    let init = create_compute_pipeline(device, init_layout, &init_spv)?;
    let downsample = create_compute_pipeline(device, downsample_layout, &downsample_spv)?;
    Ok((init, downsample))
}

fn create_compute_pipeline(
    device: &Device,
    layout: vk::PipelineLayout,
    spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let module = spv_module(device, spv)?;
    let entry = std::ffi::CString::new("main").unwrap();
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
    .map_err(|(_, e)| format!("create hiz pipeline: {e}"))?[0];
    unsafe { device.destroy_shader_module(module, None) };
    Ok(pipeline)
}

impl HiZResources {
    // Build every Hi-Z resource sized to the render (depth) resolution. Called
    // from the init path when the GPU-cull pipeline is active. `depth_views`
    // are the per-frame main-depth views the init kernel reduces.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
        sample_count: u32,
        frames: usize,
        depth_views: &[vk::ImageView],
        // When set, allocate the phase-2 cull-read sets + UBOs for two-pass
        // occlusion (`Cull2`). Gated on the world's `occlusion_two_pass`.
        two_pass: bool,
        hot_reload: bool,
    ) -> Result<Self, String> {
        // Set layouts.
        // Init: binding 0 depth sampler, binding 1 dst-mip storage image.
        let init_set_layout = create_set_layout(
            device,
            &[
                (0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                (1, vk::DescriptorType::STORAGE_IMAGE),
            ],
        )?;
        // Downsample: binding 0 src-mip storage image, binding 1 dst-mip.
        let downsample_set_layout = create_set_layout(
            device,
            &[
                (0, vk::DescriptorType::STORAGE_IMAGE),
                (1, vk::DescriptorType::STORAGE_IMAGE),
            ],
        )?;
        // Cull-read (set 1 of the cull pipeline): sampler2D Hi-Z + UBO.
        let read_set_layout = create_set_layout(
            device,
            &[
                (0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                (1, vk::DescriptorType::UNIFORM_BUFFER),
            ],
        )?;

        // Pipeline layouts (shared 16-byte push range for both build kernels).
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<HizParams>() as u32);
        let init_pipeline_layout = create_pipeline_layout(device, init_set_layout, push_range)?;
        let downsample_pipeline_layout =
            create_pipeline_layout(device, downsample_set_layout, push_range)?;

        let (init_pipeline, downsample_pipeline) = build_hiz_pipelines(
            device,
            init_pipeline_layout,
            downsample_pipeline_layout,
            sample_count,
            hot_reload,
        )?;

        // Dedicated descriptor pool, sized for the worst-case mip count.
        let descriptor_pool = create_pool(device, frames, two_pass)?;

        // Per-frame cull-read uniform buffers (host-mapped). The phase-2 set
        // gets its own ring (`cull_ubos2`) when two-pass occlusion is active.
        let ubo_size = std::mem::size_of::<CullHizParams>() as u64;
        let alloc_ubo_ring = |count: usize| -> Result<CullUboRing, String> {
            let mut bufs = Vec::with_capacity(count);
            let mut mems = Vec::with_capacity(count);
            let mut ptrs: Vec<*mut u8> = Vec::with_capacity(count);
            for _ in 0..count {
                let (buf, mem) = create_buffer(
                    instance,
                    device,
                    physical_device,
                    ubo_size,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?;
                let ptr = unsafe {
                    device
                        .map_memory(mem, 0, ubo_size, vk::MemoryMapFlags::empty())
                        .map_err(|e| format!("map hiz cull ubo: {e}"))?
                        as *mut u8
                };
                bufs.push(buf);
                mems.push(mem);
                ptrs.push(ptr);
            }
            Ok((bufs, mems, ptrs))
        };
        let (cull_ubos, cull_ubo_memories, cull_ubo_ptrs) = alloc_ubo_ring(frames)?;
        let (cull_ubos2, cull_ubo2_memories, cull_ubo2_ptrs) =
            alloc_ubo_ring(if two_pass { frames } else { 0 })?;

        let mut res = Self {
            init_pipeline,
            downsample_pipeline,
            init_pipeline_layout,
            downsample_pipeline_layout,
            init_set_layout,
            downsample_set_layout,
            read_set_layout,
            descriptor_pool,
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            sampled_view: vk::ImageView::null(),
            mip_views: Vec::new(),
            sampler: create_sampler(device)?,
            init_sets: Vec::new(),
            downsample_sets: Vec::new(),
            read_sets: Vec::new(),
            cull_ubo_memories,
            cull_ubo_ptrs,
            cull_ubos,
            read_sets2: Vec::new(),
            cull_ubo2_memories,
            cull_ubo2_ptrs,
            cull_ubos2,
            width,
            height,
            mip_count: 0,
            sample_count,
        };
        res.create_image_and_sets(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            width,
            height,
            depth_views,
        )?;
        Ok(res)
    }

    // (Re)create the mip-chain image + views and (re)allocate every set bound
    // to it. Resets the descriptor pool, so all Hi-Z sets are freshly
    // allocated; the cull-read UBO buffers themselves survive (only their
    // descriptors are rewritten). The caller must have idled the GPU.
    #[allow(clippy::too_many_arguments)]
    fn create_image_and_sets(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
        depth_views: &[vk::ImageView],
    ) -> Result<(), String> {
        let mip_count = hiz_mip_count(width, height).min(MAX_HIZ_MIPS as u32).max(1);
        let (image, memory) =
            create_hiz_image(instance, device, physical_device, width, height, mip_count)?;
        // Rest in SHADER_READ_ONLY so the cull-read descriptor's layout is
        // satisfied on the first frame (the cull kernel won't sample it -
        // `hiz_enabled` is 0 - but the descriptor layout must still match).
        one_shot_submit(device, command_pool, queue, |cmd| {
            transition_image_layout_range(
                device,
                cmd,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                0,
                1,
                0,
                mip_count,
            );
        })?;

        let sampled_view = create_hiz_view(device, image, 0, mip_count)?;
        let mut mip_views = Vec::with_capacity(mip_count as usize);
        for mip in 0..mip_count {
            mip_views.push(create_hiz_view(device, image, mip, 1)?);
        }

        // Reset the pool and reallocate every set (init / downsample / read).
        unsafe {
            device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| format!("reset hiz pool: {e}"))?;
        }
        let frames = self.cull_ubos.len();
        let init_layouts: Vec<_> = (0..frames).map(|_| self.init_set_layout).collect();
        let init_sets = alloc_descriptor_sets(device, self.descriptor_pool, &init_layouts)?;
        let downsample_layouts: Vec<_> =
            (1..mip_count).map(|_| self.downsample_set_layout).collect();
        let downsample_sets =
            alloc_descriptor_sets(device, self.descriptor_pool, &downsample_layouts)?;
        let read_layouts: Vec<_> = (0..frames).map(|_| self.read_set_layout).collect();
        let read_sets = alloc_descriptor_sets(device, self.descriptor_pool, &read_layouts)?;
        // Phase-2 cull-read sets (two-pass occlusion), one per frame, only when
        // the phase-2 UBO ring was allocated.
        let two_pass = !self.cull_ubos2.is_empty();
        let read_layouts2: Vec<_> = (0..if two_pass { frames } else { 0 })
            .map(|_| self.read_set_layout)
            .collect();
        let read_sets2 = alloc_descriptor_sets(device, self.descriptor_pool, &read_layouts2)?;

        // Init sets: binding 0 = that frame's main depth, binding 1 = mip 0.
        for (i, &set) in init_sets.iter().enumerate() {
            let depth = depth_views[i.min(depth_views.len().saturating_sub(1))];
            write_sampler(device, set, 0, depth, self.sampler);
            write_storage_image(device, set, 1, mip_views[0]);
        }
        // Downsample sets: step m reads mip m-1, writes mip m.
        for (step, &set) in downsample_sets.iter().enumerate() {
            let m = step + 1;
            write_storage_image(device, set, 0, mip_views[m - 1]);
            write_storage_image(device, set, 1, mip_views[m]);
        }
        // Read sets: binding 0 = all-mips Hi-Z sampler, binding 1 = cull UBO.
        for (i, &set) in read_sets.iter().enumerate() {
            write_sampler(device, set, 0, sampled_view, self.sampler);
            write_uniform_buffer(
                device,
                set,
                1,
                self.cull_ubos[i],
                std::mem::size_of::<CullHizParams>() as u64,
            );
        }
        // Phase-2 read sets: same pyramid sampler, the phase-2 per-frame UBO.
        for (i, &set) in read_sets2.iter().enumerate() {
            write_sampler(device, set, 0, sampled_view, self.sampler);
            write_uniform_buffer(
                device,
                set,
                1,
                self.cull_ubos2[i],
                std::mem::size_of::<CullHizParams>() as u64,
            );
        }

        self.image = image;
        self.memory = memory;
        self.sampled_view = sampled_view;
        self.mip_views = mip_views;
        self.init_sets = init_sets;
        self.downsample_sets = downsample_sets;
        self.read_sets = read_sets;
        self.read_sets2 = read_sets2;
        self.width = width;
        self.height = height;
        self.mip_count = mip_count;
        Ok(())
    }

    // Recreate the image + views + sets at new render-target dimensions. The
    // pipelines, layouts, sampler, and cull-read UBO buffers survive. The
    // caller flips `hiz_valid` to false so the next cull dispatch ignores the
    // now-stale pyramid, and must have idled the GPU first.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resize_to(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
        depth_views: &[vk::ImageView],
    ) -> Result<(), String> {
        self.destroy_image_and_views(device);
        self.create_image_and_sets(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            width,
            height,
            depth_views,
        )
    }

    // Swap freshly-rebuilt pipelines into the live resource. Used by the shader
    // hot-reload pass; the image, views, sets, and layouts are kept.
    pub(super) fn swap_pipelines(
        &mut self,
        device: &Device,
        init: vk::Pipeline,
        downsample: vk::Pipeline,
    ) {
        unsafe {
            device.destroy_pipeline(self.init_pipeline, None);
            device.destroy_pipeline(self.downsample_pipeline, None);
        }
        self.init_pipeline = init;
        self.downsample_pipeline = downsample;
    }

    // Recompile the build pipelines from disk-resident source (hot-reload).
    // The MSAA mode is fixed at init, so it is reused here.
    pub(super) fn recompile_pipelines(
        &self,
        device: &Device,
        hot_reload: bool,
    ) -> Result<(vk::Pipeline, vk::Pipeline), String> {
        build_hiz_pipelines(
            device,
            self.init_pipeline_layout,
            self.downsample_pipeline_layout,
            self.sample_count,
            hot_reload,
        )
    }

    fn destroy_image_and_views(&mut self, device: &Device) {
        unsafe {
            if self.sampled_view != vk::ImageView::null() {
                device.destroy_image_view(self.sampled_view, None);
            }
            for &v in &self.mip_views {
                device.destroy_image_view(v, None);
            }
            if self.image != vk::Image::null() {
                device.destroy_image(self.image, None);
                device.free_memory(self.memory, None);
            }
        }
        self.mip_views.clear();
        self.sampled_view = vk::ImageView::null();
        self.image = vk::Image::null();
        self.memory = vk::DeviceMemory::null();
    }

    pub(super) fn destroy(&mut self, device: &Device) {
        self.destroy_image_and_views(device);
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline(self.init_pipeline, None);
            device.destroy_pipeline(self.downsample_pipeline, None);
            device.destroy_pipeline_layout(self.init_pipeline_layout, None);
            device.destroy_pipeline_layout(self.downsample_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.init_set_layout, None);
            device.destroy_descriptor_set_layout(self.downsample_set_layout, None);
            device.destroy_descriptor_set_layout(self.read_set_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            for &buf in self.cull_ubos.iter().chain(self.cull_ubos2.iter()) {
                device.destroy_buffer(buf, None);
            }
            for &mem in self
                .cull_ubo_memories
                .iter()
                .chain(self.cull_ubo2_memories.iter())
            {
                device.unmap_memory(mem);
                device.free_memory(mem, None);
            }
        }
    }
}

impl crate::vulkan::context::VkContext {
    // Encode the Hi-Z build into `cmd`. Reads this frame's main depth
    // (`depth_images[frame_idx]`) and writes the mip chain that *next* frame's
    // cull dispatch consults. A no-op when no Hi-Z resource was built (GPU-cull
    // pipeline not active). Called from `record_frame` after `execute_graph`,
    // so the Main pass has already written depth and decal / fog have restored
    // it to `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`.
    pub(in crate::vulkan) fn encode_hiz_build(&self, cmd: vk::CommandBuffer, frame_idx: usize) {
        let Some(hiz) = self.cull.hiz.as_ref() else {
            return;
        };
        if hiz.mip_count == 0 || hiz.mip_views.is_empty() {
            return;
        }
        let device = &self.device;
        let depth_image = self.depth_images[frame_idx].image;

        // 1. Transition main depth -> SHADER_READ_ONLY for the compute read,
        //    and the Hi-Z image SHADER_READ_ONLY -> GENERAL for the writes.
        let depth_to_read = depth_barrier(
            depth_image,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        let hiz_to_general = hiz_image_barrier(
            hiz.image,
            hiz.mip_count,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::SHADER_WRITE,
        );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[depth_to_read, hiz_to_general],
            );
        }

        // 2. Init: mip 0 from main depth (MAX over MSAA samples when on).
        let init_params = HizParams {
            dst_width: hiz.width,
            dst_height: hiz.height,
            src_mip: 0,
            sample_count: hiz.sample_count.max(1),
        };
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, hiz.init_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                hiz.init_pipeline_layout,
                0,
                std::slice::from_ref(&hiz.init_sets[frame_idx]),
                &[],
            );
            device.cmd_push_constants(
                cmd,
                hiz.init_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                as_bytes(&init_params),
            );
            device.cmd_dispatch(
                cmd,
                hiz.width.div_ceil(HIZ_TILE),
                hiz.height.div_ceil(HIZ_TILE),
                1,
            );
        }

        // 3. Downsample chain. Each step reads the prior mip and writes the
        //    next, with a compute write -> read barrier between dispatches.
        let mut cur_w = hiz.width;
        let mut cur_h = hiz.height;
        for mip in 1..hiz.mip_count {
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[hiz_image_barrier(
                        hiz.image,
                        hiz.mip_count,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::GENERAL,
                        vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    )],
                );
            }
            let next_w = (cur_w / 2).max(1);
            let next_h = (cur_h / 2).max(1);
            let params = HizParams {
                dst_width: next_w,
                dst_height: next_h,
                src_mip: mip - 1,
                sample_count: 0,
            };
            unsafe {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    hiz.downsample_pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    hiz.downsample_pipeline_layout,
                    0,
                    std::slice::from_ref(&hiz.downsample_sets[(mip - 1) as usize]),
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    hiz.downsample_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    as_bytes(&params),
                );
                device.cmd_dispatch(cmd, next_w.div_ceil(HIZ_TILE), next_h.div_ceil(HIZ_TILE), 1);
            }
            cur_w = next_w;
            cur_h = next_h;
        }

        // 4. Restore: main depth -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL for the
        //    next frame's main pass, and Hi-Z -> SHADER_READ_ONLY so the next
        //    frame's cull dispatch samples it (this barrier's second scope
        //    orders the writes before that cross-frame read).
        let depth_back = depth_barrier(
            depth_image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
        let hiz_back = hiz_image_barrier(
            hiz.image,
            hiz.mip_count,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[depth_back, hiz_back],
            );
        }
    }
}

fn as_bytes<T: Copy>(v: &T) -> &[u8] {
    // SAFETY: `T` is `Copy` and `repr(C)`; we read `size_of::<T>()` bytes.
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn depth_barrier(
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src: vk::AccessFlags,
    dst: vk::AccessFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .src_access_mask(src)
        .dst_access_mask(dst)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
}

fn hiz_image_barrier(
    image: vk::Image,
    mip_count: u32,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src: vk::AccessFlags,
    dst: vk::AccessFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .src_access_mask(src)
        .dst_access_mask(dst)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: mip_count,
            base_array_layer: 0,
            layer_count: 1,
        })
}

fn create_set_layout(
    device: &Device,
    bindings: &[(u32, vk::DescriptorType)],
) -> Result<vk::DescriptorSetLayout, String> {
    let binds: Vec<_> = bindings
        .iter()
        .map(|&(b, ty)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds),
            None,
        )
    }
    .map_err(|e| format!("hiz set layout: {e}"))
}

fn create_pipeline_layout(
    device: &Device,
    set_layout: vk::DescriptorSetLayout,
    push_range: vk::PushConstantRange,
) -> Result<vk::PipelineLayout, String> {
    let layouts = [set_layout];
    unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(std::slice::from_ref(&push_range)),
            None,
        )
    }
    .map_err(|e| format!("hiz pipeline layout: {e}"))
}

fn create_pool(
    device: &Device,
    frames: usize,
    two_pass: bool,
) -> Result<vk::DescriptorPool, String> {
    let f = frames as u32;
    // Two-pass occlusion adds one extra cull-read set per frame (phase 2),
    // each with a sampler + a UBO descriptor.
    let read_rings = if two_pass { 2 } else { 1 };
    let sizes = [
        // init depth (frames) + cull-read Hi-Z (frames per read ring).
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count((1 + read_rings) * f),
        // init dst (frames) + downsample src+dst (2 per step).
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(f + 2 * MAX_HIZ_MIPS as u32),
        // cull-read UBO (frames per read ring).
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(read_rings * f),
    ];
    // init (frames) + cull-read (frames per read ring) + downsample (per mip).
    let max_sets = (1 + read_rings) * f + MAX_HIZ_MIPS as u32;
    unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&sizes)
                .max_sets(max_sets),
            None,
        )
    }
    .map_err(|e| format!("hiz descriptor pool: {e}"))
}

fn create_sampler(device: &Device) -> Result<vk::Sampler, String> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .min_lod(0.0)
        .max_lod(MAX_HIZ_MIPS as f32);
    unsafe { device.create_sampler(&info, None) }.map_err(|e| format!("hiz sampler: {e}"))
}

fn write_sampler(
    device: &Device,
    set: vk::DescriptorSet,
    binding: u32,
    view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(view)
        .sampler(sampler);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

fn write_storage_image(device: &Device, set: vk::DescriptorSet, binding: u32, view: vk::ImageView) {
    let info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(view);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

fn write_uniform_buffer(
    device: &Device,
    set: vk::DescriptorSet,
    binding: u32,
    buffer: vk::Buffer,
    range: u64,
) {
    let info = vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(0)
        .range(range);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

#[cfg(test)]
mod tests {
    use super::{CullHizParams, HizParams, hiz_mip_count};
    use std::mem::{offset_of, size_of};

    #[test]
    fn hiz_params_layout() {
        // GLSL HizParams push block: four tightly packed uints (16 bytes).
        assert_eq!(size_of::<HizParams>(), 16);
        assert_eq!(offset_of!(HizParams, dst_width), 0);
        assert_eq!(offset_of!(HizParams, dst_height), 4);
        assert_eq!(offset_of!(HizParams, src_mip), 8);
        assert_eq!(offset_of!(HizParams, sample_count), 12);
    }

    #[test]
    fn cull_hiz_params_layout_matches_glsl() {
        // std140 CullHizParams in cull.comp: mat4 (64) + vec2 (8, 8-aligned) +
        // two uints. Total 80 bytes, tightly packed after the mat4.
        assert_eq!(size_of::<CullHizParams>(), 80);
        assert_eq!(offset_of!(CullHizParams, prev_view_proj), 0);
        assert_eq!(offset_of!(CullHizParams, hiz_size), 64);
        assert_eq!(offset_of!(CullHizParams, hiz_mip_count), 72);
        assert_eq!(offset_of!(CullHizParams, hiz_enabled), 76);
    }

    #[test]
    fn mip_count_power_of_two() {
        assert_eq!(hiz_mip_count(1, 1), 1);
        assert_eq!(hiz_mip_count(2, 2), 2);
        assert_eq!(hiz_mip_count(256, 256), 9);
        assert_eq!(hiz_mip_count(1024, 1024), 11);
    }

    #[test]
    fn mip_count_uses_larger_dimension() {
        assert_eq!(hiz_mip_count(1920, 1080), hiz_mip_count(1920, 1920));
        assert_eq!(hiz_mip_count(1920, 1080), 11);
        assert_eq!(hiz_mip_count(1280, 720), 11);
    }

    #[test]
    fn mip_count_clamps_zero() {
        assert_eq!(hiz_mip_count(0, 0), 1);
        assert_eq!(hiz_mip_count(0, 8), 4);
    }
}
