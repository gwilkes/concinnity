// src/vulkan/auto_exposure.rs
//
// Auto-exposure (EV adaptation) on Vulkan: a per-frame CPU readback of a
// previous frame's average log-luminance, an EMA step that updates the
// adapted EV, and the histogram build + average compute dispatches that
// produce next frame's average. The compute passes are encoded after the
// main HDR resolve (where `hdr_resolve_images[frame_idx]` carries this
// frame's scene colour in SHADER_READ_ONLY_OPTIMAL) and the result is
// copied into a per-frame HOST_VISIBLE readback buffer that the CPU reads
// at the top of a later frame, so there is `frames_in_flight` frames of
// latency between the scene's actual luminance and the exposure applied,
// invisible at human-scale eye-adaptation rates. Mirrors
// `metal/auto_exposure.rs` and `directx/auto_exposure.rs`.

use ash::{Device, vk};

use crate::gfx::auto_exposure::HISTOGRAM_BINS;

use super::context::VkContext;
use super::pipeline::{compile_glsl, shader_source, spv_module};
use super::texture::create_buffer;

pub(in crate::vulkan) const AUTO_EXPOSURE_BUILD_GLSL: &str =
    include_str!("shaders/auto_exposure_build.comp");
pub(in crate::vulkan) const AUTO_EXPOSURE_AVERAGE_GLSL: &str =
    include_str!("shaders/auto_exposure_average.comp");

// Compile the auto-exposure build + average compute kernels. Used at init
// and by shader hot-reload to rebuild the two compute pipelines.
pub(in crate::vulkan) fn compile_auto_exposure_shaders(
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let build_src = shader_source(
        hot_reload,
        "auto_exposure_build.comp",
        AUTO_EXPOSURE_BUILD_GLSL,
    );
    let average_src = shader_source(
        hot_reload,
        "auto_exposure_average.comp",
        AUTO_EXPOSURE_AVERAGE_GLSL,
    );
    let build_cs = compile_glsl(
        &build_src,
        shaderc::ShaderKind::Compute,
        "auto_exposure_build.comp",
    )?;
    let average_cs = compile_glsl(
        &average_src,
        shaderc::ShaderKind::Compute,
        "auto_exposure_average.comp",
    )?;
    Ok((build_cs, average_cs))
}

// Byte size of the `AutoExposureParams` push-constant block. Must match
// the `layout(push_constant) ... AutoExposureParams` in both compute
// shaders.
const AUTO_EXPOSURE_PUSH_BYTES: u32 = 16;

// Push-constant payload pushed at the top of every compute dispatch.
// Mirrors the HLSL / Metal struct of the same name.
#[derive(Copy, Clone)]
#[repr(C)]
struct AutoExposureParams {
    lum_log2_min: f32,
    lum_log2_range: f32,
    lum_to_bin_scale: f32,
    _pad: f32,
}

// Owns the compute pipelines + GPU buffers + per-frame readback driving
// the auto-exposure histogram path. Built only when the world's
// `PostProcessConfig` opts in; the encoder is a no-op otherwise.
pub(in crate::vulkan) struct AutoExposureResources {
    // Build kernel: one thread per HDR-resolve pixel; merges per-threadgroup
    // local histograms into the global histogram SSBO.
    build_pipeline: vk::Pipeline,
    build_pipeline_layout: vk::PipelineLayout,
    build_set_layout: vk::DescriptorSetLayout,
    // One build set per frame: binding 0 references that frame slot's
    // `hdr_resolve_images[frame_idx]` view.
    build_sets: Vec<vk::DescriptorSet>,

    // Average kernel: one threadgroup of HISTOGRAM_BINS threads reduces the
    // histogram, clears it, and writes the average log-luminance.
    average_pipeline: vk::Pipeline,
    average_pipeline_layout: vk::PipelineLayout,
    average_set_layout: vk::DescriptorSetLayout,
    // Single shared average set: both buffers are global, no per-frame
    // variation.
    average_set: vk::DescriptorSet,

    descriptor_pool: vk::DescriptorPool,

    // Device-local 256-bin u32 histogram. The build kernel atomically
    // increments bins into it; the average kernel reads and clears them.
    histogram_buffer: vk::Buffer,
    histogram_memory: vk::DeviceMemory,
    // Device-local single f32 the average kernel writes; copied into the
    // per-frame readback after each dispatch.
    output_buffer: vk::Buffer,
    output_memory: vk::DeviceMemory,

    // Per-frame HOST_VISIBLE readback buffers. Each holds 4 bytes (one
    // f32). Persistently mapped; the CPU reads `readback_ptrs[frame_idx]`
    // at the top of a later frame after the fence wait gates this slot's
    // previous copy.
    readback_buffers: Vec<vk::Buffer>,
    readback_memories: Vec<vk::DeviceMemory>,
    readback_ptrs: Vec<*const f32>,
}

impl AutoExposureResources {
    // Build all auto-exposure resources. Called from `VkContext::new` only
    // when `PostProcessConfig.auto_exposure` is enabled.
    pub(in crate::vulkan) fn new(
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        frames: usize,
        hdr_resolve_views: &[vk::ImageView],
        linear_sampler: vk::Sampler,
        hot_reload: bool,
    ) -> Result<Self, String> {
        // Build descriptor set layout: 0 = HDR combined image sampler,
        // 1 = histogram SSBO.
        let build_set_layout = create_build_set_layout(device)?;
        let average_set_layout = create_average_set_layout(device)?;

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(AUTO_EXPOSURE_PUSH_BYTES);

        let build_layouts = [build_set_layout];
        let build_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&build_layouts)
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
                None,
            )
        }
        .map_err(|e| format!("auto-exposure build pipeline layout: {e}"))?;
        let average_layouts = [average_set_layout];
        let average_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&average_layouts)
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
                None,
            )
        }
        .map_err(|e| format!("auto-exposure average pipeline layout: {e}"))?;

        let (build_spv, average_spv) = compile_auto_exposure_shaders(hot_reload)?;
        let build_pipeline = create_compute_pipeline(device, build_pipeline_layout, &build_spv)?;
        let average_pipeline =
            create_compute_pipeline(device, average_pipeline_layout, &average_spv)?;

        // Histogram + output buffers (device-local).
        let histogram_bytes = (HISTOGRAM_BINS * std::mem::size_of::<u32>()) as vk::DeviceSize;
        let (histogram_buffer, histogram_memory) = create_buffer(
            instance,
            device,
            physical_device,
            histogram_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let output_bytes = std::mem::size_of::<f32>() as vk::DeviceSize;
        let (output_buffer, output_memory) = create_buffer(
            instance,
            device,
            physical_device,
            output_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Per-frame HOST_VISIBLE readback buffers (persistently mapped).
        let mut readback_buffers = Vec::with_capacity(frames);
        let mut readback_memories = Vec::with_capacity(frames);
        let mut readback_ptrs: Vec<*const f32> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let (buf, mem) = create_buffer(
                instance,
                device,
                physical_device,
                output_bytes,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let ptr =
                unsafe { device.map_memory(mem, 0, output_bytes, vk::MemoryMapFlags::empty()) }
                    .map_err(|e| format!("map auto-exposure readback: {e}"))?
                    as *const f32;
            readback_buffers.push(buf);
            readback_memories.push(mem);
            readback_ptrs.push(ptr);
        }

        // Descriptor pool: enough for `frames` build sets + 1 average set.
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: frames as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: (frames + 2) as u32,
            },
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets((frames + 1) as u32)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|e| format!("auto-exposure descriptor pool: {e}"))?;

        // Allocate build sets (one per frame) + average set.
        let build_set_layouts: Vec<_> = (0..frames).map(|_| build_set_layout).collect();
        let build_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&build_set_layouts),
            )
        }
        .map_err(|e| format!("auto-exposure build sets: {e}"))?;
        let avg_layouts_single = [average_set_layout];
        let average_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&avg_layouts_single),
            )
        }
        .map_err(|e| format!("auto-exposure average set: {e}"))?[0];

        // Write each build set's HDR sampled image + histogram bindings.
        let last_view_idx = hdr_resolve_views.len().saturating_sub(1);
        for (i, &set) in build_sets.iter().enumerate() {
            let view = hdr_resolve_views[i.min(last_view_idx)];
            write_build_set(device, set, view, linear_sampler, histogram_buffer);
        }
        // Write the average set's histogram + output bindings.
        write_average_set(device, average_set, histogram_buffer, output_buffer);

        Ok(Self {
            build_pipeline,
            build_pipeline_layout,
            build_set_layout,
            build_sets,
            average_pipeline,
            average_pipeline_layout,
            average_set_layout,
            average_set,
            descriptor_pool,
            histogram_buffer,
            histogram_memory,
            output_buffer,
            output_memory,
            readback_buffers,
            readback_memories,
            readback_ptrs,
        })
    }

    // Pipeline layout for the build kernel. Exposed so the shader
    // hot-reload pass can rebuild the pipeline against the existing layout.
    pub(in crate::vulkan) fn build_pipeline_layout(&self) -> vk::PipelineLayout {
        self.build_pipeline_layout
    }
    // Pipeline layout for the average kernel. Same purpose as
    // [`Self::build_pipeline_layout`].
    pub(in crate::vulkan) fn average_pipeline_layout(&self) -> vk::PipelineLayout {
        self.average_pipeline_layout
    }
    // Swap the freshly-built build + average pipelines into the live
    // resources. The caller has already `device_wait_idle`'d so the old
    // pipelines are not in flight. Driven by the Vulkan shader hot-reload
    // pass after every replacement successfully compiled.
    pub(in crate::vulkan) fn swap_pipelines(
        &mut self,
        device: &Device,
        build_pipeline: vk::Pipeline,
        average_pipeline: vk::Pipeline,
    ) {
        unsafe {
            device.destroy_pipeline(self.build_pipeline, None);
            device.destroy_pipeline(self.average_pipeline, None);
        }
        self.build_pipeline = build_pipeline;
        self.average_pipeline = average_pipeline;
    }

    // Construct a compute pipeline (build or average) against the existing
    // pipeline layout. Exposed so the shader hot-reload pass can rebuild
    // either kernel without re-creating the descriptor set layout +
    // pipeline layout. Mirrors `directx::auto_exposure::create_compute_pso`.
    pub(in crate::vulkan) fn create_compute_pipeline(
        device: &Device,
        layout: vk::PipelineLayout,
        spv: &[u8],
    ) -> Result<vk::Pipeline, String> {
        create_compute_pipeline(device, layout, spv)
    }

    // Rewrite the per-frame build sets' HDR sampled-image binding after a
    // swapchain rebuild swapped the `hdr_resolve_images`. The histogram /
    // output buffers are resolution-independent and survive the rebuild
    // untouched; only binding 0 of each build set needs to point at the
    // new view.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        device: &Device,
        hdr_resolve_views: &[vk::ImageView],
        linear_sampler: vk::Sampler,
    ) {
        let last_view_idx = hdr_resolve_views.len().saturating_sub(1);
        for (i, &set) in self.build_sets.iter().enumerate() {
            let view = hdr_resolve_views[i.min(last_view_idx)];
            write_build_set(device, set, view, linear_sampler, self.histogram_buffer);
        }
    }

    // Free every owned handle. Called from `Drop for VkContext` after
    // `device_wait_idle`.
    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        unsafe {
            for (&mem, &_buf) in self
                .readback_memories
                .iter()
                .zip(self.readback_buffers.iter())
            {
                device.unmap_memory(mem);
            }
            for &buf in &self.readback_buffers {
                device.destroy_buffer(buf, None);
            }
            for &mem in &self.readback_memories {
                device.free_memory(mem, None);
            }
            device.destroy_buffer(self.output_buffer, None);
            device.free_memory(self.output_memory, None);
            device.destroy_buffer(self.histogram_buffer, None);
            device.free_memory(self.histogram_memory, None);

            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.build_pipeline, None);
            device.destroy_pipeline(self.average_pipeline, None);
            device.destroy_pipeline_layout(self.build_pipeline_layout, None);
            device.destroy_pipeline_layout(self.average_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.build_set_layout, None);
            device.destroy_descriptor_set_layout(self.average_set_layout, None);
        }
    }
}

fn create_build_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
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
        .map_err(|e| format!("auto-exposure build set layout: {e}"))
}

fn create_average_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
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
        .map_err(|e| format!("auto-exposure average set layout: {e}"))
}

fn write_build_set(
    device: &Device,
    set: vk::DescriptorSet,
    view: vk::ImageView,
    sampler: vk::Sampler,
    histogram: vk::Buffer,
) {
    let img = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(view)
        .sampler(sampler);
    let hist = vk::DescriptorBufferInfo::default()
        .buffer(histogram)
        .offset(0)
        .range(vk::WHOLE_SIZE);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&img)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&hist)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn write_average_set(
    device: &Device,
    set: vk::DescriptorSet,
    histogram: vk::Buffer,
    output: vk::Buffer,
) {
    let hist = vk::DescriptorBufferInfo::default()
        .buffer(histogram)
        .offset(0)
        .range(vk::WHOLE_SIZE);
    let out = vk::DescriptorBufferInfo::default()
        .buffer(output)
        .offset(0)
        .range(vk::WHOLE_SIZE);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&hist)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&out)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
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
    .map_err(|(_, e)| format!("create auto-exposure pipeline: {e}"))?[0];
    unsafe { device.destroy_shader_module(module, None) };
    Ok(pipeline)
}

impl VkContext {
    // Build the per-frame compute params for the auto-exposure kernels. The
    // log-luminance range and the precomputed `bins / range` scale match the
    // `gfx::auto_exposure::LUM_LOG2_*` constants exactly.
    fn auto_exposure_params(&self) -> AutoExposureParams {
        use crate::gfx::auto_exposure::{LUM_LOG2_MAX, LUM_LOG2_MIN};
        let range = LUM_LOG2_MAX - LUM_LOG2_MIN;
        AutoExposureParams {
            lum_log2_min: LUM_LOG2_MIN,
            lum_log2_range: range,
            lum_to_bin_scale: HISTOGRAM_BINS as f32 / range,
            _pad: 0.0,
        }
    }

    // Step the auto-exposure EMA from a previous frame's GPU measurement,
    // then push the new exposure multiplier into `self.post_process.exposure`.
    // A no-op when auto-exposure is disabled: the static authored EV then
    // drives `exposure` unchanged.
    //
    // Called at the top of `draw_frame` after the fence wait for this slot's
    // previous use completes, so the matching readback buffer holds a fully
    // committed GPU result (one or two frames stale, smoothed by the EMA).
    // `elapsed` is the total elapsed seconds since startup; the per-call
    // diff drives `dt` for the EMA.
    pub(in crate::vulkan) fn update_auto_exposure(&mut self, elapsed: f32, frame_idx: usize) {
        let Some(settings) = self.auto_exposure_settings else {
            return;
        };
        let Some(resources) = self.auto_exposure.as_ref() else {
            return;
        };
        let Some(state) = self.auto_exposure_state.as_mut() else {
            return;
        };
        let Some(&ptr) = resources.readback_ptrs.get(frame_idx) else {
            return;
        };

        // Read the previous frame's average log-luminance for this slot. The
        // fence wait above this call already gated the GPU work that wrote
        // it, so the HOST_COHERENT mapping reflects the committed value.
        let avg_log_lum = unsafe { ptr.read() };
        let avg_log_lum = if avg_log_lum.is_finite() {
            avg_log_lum
        } else {
            crate::gfx::auto_exposure::LUM_LOG2_MIN
        };

        let dt = (elapsed - self.auto_exposure_last_elapsed).max(0.0);
        self.auto_exposure_last_elapsed = elapsed;

        let adapted_ev = state.update(avg_log_lum, self.auto_exposure_bias_ev, &settings, dt);
        // `self.post_process.exposure` is the linear multiplier the bloom
        // prefilter and composite consume. `state.update` already folds the
        // bias into the target EV; re-adding it would double the bias.
        self.post_process.exposure = adapted_ev.exp2();
    }

    // Encode the auto-exposure histogram passes against the current frame's
    // resolved HDR scene. The build kernel runs one thread per HDR pixel;
    // the average kernel runs one threadgroup of `HISTOGRAM_BINS` threads
    // that reduces the histogram, clears it for the next frame, and writes
    // the average log-luminance to the device-local output buffer; a copy
    // then carries the value into this frame's readback buffer for the
    // CPU's EMA step at the top of a later frame. A no-op when
    // auto-exposure is disabled.
    pub(in crate::vulkan) fn encode_auto_exposure(&self, cmd: vk::CommandBuffer, frame_idx: usize) {
        let Some(resources) = self.auto_exposure.as_ref() else {
            return;
        };
        let device = &self.device;
        let params = self.auto_exposure_params();
        // SAFETY: `AutoExposureParams` is `repr(C)`, 16 bytes, push range matched.
        let push_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const AutoExposureParams as *const u8,
                std::mem::size_of::<AutoExposureParams>(),
            )
        };

        let extent = self.render_extent;
        if extent.width == 0 || extent.height == 0 {
            return;
        }

        unsafe {
            // Order Main pass's resolve color writes before our compute
            // shader sample of the HDR resolve image. The render pass's
            // exit-dep targets COLOR_ATTACHMENT_OUTPUT (for the next
            // subpass-attachment consumer); compute-shader reads need a
            // dedicated barrier.
            let pre_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&pre_barrier),
                &[],
                &[],
            );

            // Build dispatch: 16×16 threadgroups, one thread per HDR pixel.
            let build_set = resources
                .build_sets
                .get(frame_idx)
                .copied()
                .unwrap_or_else(|| resources.build_sets[0]);
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                resources.build_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                resources.build_pipeline_layout,
                0,
                std::slice::from_ref(&build_set),
                &[],
            );
            device.cmd_push_constants(
                cmd,
                resources.build_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            device.cmd_dispatch(
                cmd,
                extent.width.div_ceil(16),
                extent.height.div_ceil(16),
                1,
            );

            // Order build histogram writes before the average read+clear.
            let hist_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&hist_barrier),
                &[],
                &[],
            );

            // Average dispatch: one threadgroup of HISTOGRAM_BINS threads.
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                resources.average_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                resources.average_pipeline_layout,
                0,
                std::slice::from_ref(&resources.average_set),
                &[],
            );
            device.cmd_push_constants(
                cmd,
                resources.average_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            device.cmd_dispatch(cmd, 1, 1, 1);

            // Order the average kernel's output_buf write before the copy
            // into the readback buffer.
            let out_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&out_barrier),
                &[],
                &[],
            );

            // Copy the freshly-written average to this slot's readback buffer.
            let readback = resources
                .readback_buffers
                .get(frame_idx)
                .copied()
                .unwrap_or_else(|| resources.readback_buffers[0]);
            let copy = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: std::mem::size_of::<f32>() as vk::DeviceSize,
            };
            device.cmd_copy_buffer(
                cmd,
                resources.output_buffer,
                readback,
                std::slice::from_ref(&copy),
            );

            // Order the transfer write to the host-visible buffer before the
            // CPU read at the top of a later frame. The fence wait that
            // gates this slot's next trip provides the host-side ordering;
            // this barrier just makes the transfer write visible to the
            // host.
            let host_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&host_barrier),
                &[],
                &[],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // AutoExposureParams must match the `AutoExposureParams` push-constant
    // block in both auto-exposure compute shaders: the three luminance-mapping
    // scalars then a pad rounding to 16 bytes. Pinned by AUTO_EXPOSURE_PUSH_BYTES.
    #[test]
    fn auto_exposure_params_layout_matches_glsl() {
        assert_eq!(std::mem::size_of::<AutoExposureParams>(), 16);
        assert_eq!(
            std::mem::size_of::<AutoExposureParams>() as u32,
            AUTO_EXPOSURE_PUSH_BYTES
        );
        assert_eq!(std::mem::offset_of!(AutoExposureParams, lum_log2_min), 0);
        assert_eq!(std::mem::offset_of!(AutoExposureParams, lum_log2_range), 4);
        assert_eq!(
            std::mem::offset_of!(AutoExposureParams, lum_to_bin_scale),
            8
        );
        assert_eq!(std::mem::offset_of!(AutoExposureParams, _pad), 12);
    }
}
