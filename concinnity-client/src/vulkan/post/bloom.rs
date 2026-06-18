// src/vulkan/post/bloom.rs
//
// Bloom for the Vulkan backend. Co-locates the bloom GLSL sources, the
// prefilter / downsample / upsample pipeline builders, the bloom mip-chain
// target allocator (per frame slot), the framebuffer + descriptor wiring, and
// the per-frame `encode_bloom` encoder. Mirrors src/metal/post/bloom.rs.

use ash::{Device, vk};

use crate::gfx::render_types::PostProcessParams;

use super::super::context::*;
use super::super::pipeline::{COMPOSITE_VERT_GLSL, compile_glsl, spv_module};
use super::super::resources::alloc_descriptor_sets;
use super::super::texture::*;

// Upper bound on `bloom_mip_count` (which clamps to 4..=6). The bloom
// descriptor pool is sized for this many mips per frame so a resize that
// changes the octave count never has to resize the pool.
pub(in crate::vulkan) const MAX_BLOOM_MIPS: u32 = 6;

//  Bloom shaders

const BLOOM_PREFILTER_GLSL: &str = include_str!("../shaders/bloom_prefilter.frag");
const BLOOM_DOWNSAMPLE_GLSL: &str = include_str!("../shaders/bloom_downsample.frag");
const BLOOM_UPSAMPLE_GLSL: &str = include_str!("../shaders/bloom_upsample.frag");

// SPIR-V for the bloom chain: the shared fullscreen-triangle vertex shader
// plus the prefilter / downsample / upsample fragment shaders.
pub(in crate::vulkan) struct BloomShaders {
    pub vert: Vec<u8>,
    pub prefilter: Vec<u8>,
    pub downsample: Vec<u8>,
    pub upsample: Vec<u8>,
}

pub(in crate::vulkan) fn compile_bloom_shaders(hot_reload: bool) -> Result<BloomShaders, String> {
    use super::super::pipeline::shader_source;
    Ok(BloomShaders {
        vert: compile_glsl(
            &shader_source(hot_reload, "composite.vert", COMPOSITE_VERT_GLSL),
            shaderc::ShaderKind::Vertex,
            "bloom_vert.glsl",
        )?,
        prefilter: compile_glsl(
            &shader_source(hot_reload, "bloom_prefilter.frag", BLOOM_PREFILTER_GLSL),
            shaderc::ShaderKind::Fragment,
            "bloom_prefilter.glsl",
        )?,
        downsample: compile_glsl(
            &shader_source(hot_reload, "bloom_downsample.frag", BLOOM_DOWNSAMPLE_GLSL),
            shaderc::ShaderKind::Fragment,
            "bloom_downsample.glsl",
        )?,
        upsample: compile_glsl(
            &shader_source(hot_reload, "bloom_upsample.frag", BLOOM_UPSAMPLE_GLSL),
            shaderc::ShaderKind::Fragment,
            "bloom_upsample.glsl",
        )?,
    })
}

//  Pipeline builder

// Build a bloom-chain pipeline: a vertex-buffer-less fullscreen triangle into
// a single-sample HDR mip, no depth. With `additive` set the colour blend is
// `dst + src`, used by the upsample pass to accumulate onto the downsampled
// mip already in the target.
pub(in crate::vulkan) fn create_bloom_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
    additive: bool,
) -> Result<vk::Pipeline, String> {
    let vert_mod = spv_module(device, vert_spv)?;
    let frag_mod = spv_module(device, frag_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_mod)
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod)
            .name(&entry),
    ];

    let vert_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::ALWAYS);

    let color_blend_attach = if additive {
        vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
    } else {
        vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)
    };

    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(std::slice::from_ref(&color_blend_attach));

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vert_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create bloom pipeline: {e}"))?[0];

    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

//  Target builder

// Number of mip levels in the bloom chain for an HDR target of the given
// resolution. Clamped to 4..=6: enough octaves for a wide soft glow without
// spending a dozen render passes on sub-pixel mips. Mirrors `bloom_mip_count`
// in metal/texture.rs.
pub(in crate::vulkan) fn bloom_mip_count(width: u32, height: u32) -> u32 {
    let min_dim = width.min(height).max(1);
    // mip 0 is already half-res, so subtract one octave before clamping.
    let levels = (min_dim as f32).log2().floor() as i32 - 1;
    levels.clamp(4, 6) as u32
}

// Create the bloom mip chain for an HDR target of `width`x`height`. `mips[i]`
// has resolution `(width >> (i+1), height >> (i+1))`, floored at one texel;
// `mips[0]` is half-res. Each mip is a single-sample colour image usable as
// both a render target and a sampled texture, and is pre-transitioned to
// `SHADER_READ_ONLY_OPTIMAL` so the composite pass can bind it even when
// bloom is disabled and the bloom passes never run.
#[allow(clippy::too_many_arguments)]
pub(in crate::vulkan) fn create_bloom_mips(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    width: u32,
    height: u32,
    format: vk::Format,
    mip0_override: Option<(vk::Image, vk::ImageView)>,
) -> Result<(Vec<GpuImage>, Vec<vk::Extent2D>), String> {
    let full_w = width.max(1);
    let full_h = height.max(1);
    let count = bloom_mip_count(full_w, full_h);

    let mut mips = Vec::with_capacity(count as usize);
    let mut extents = Vec::with_capacity(count as usize);
    for i in 0..count {
        let mw = (full_w >> (i + 1)).max(1);
        let mh = (full_h >> (i + 1)).max(1);
        let gpu_image = if i == 0
            && let Some((image, view)) = mip0_override
        {
            // Pooled `bloom_top`: the transient pool owns image + view + memory.
            // Wrap it as a borrowed `GpuImage` (null memory) so the chain indexes
            // it uniformly; the prefilter re-establishes its layout from
            // UNDEFINED each frame, so no pre-transition is done here. Teardown
            // (`destroy_swapchain_resources`) skips null-memory mips.
            GpuImage {
                image,
                memory: vk::DeviceMemory::null(),
                view,
                aux_views: Vec::new(),
            }
        } else {
            let (image, memory) = create_image(
                instance,
                device,
                physical_device,
                mw,
                mh,
                format,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::SampleCountFlags::TYPE_1,
            )?;
            one_shot_submit(device, command_pool, queue, |cmd| {
                transition_image_layout(
                    device,
                    cmd,
                    image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageAspectFlags::COLOR,
                );
            })?;
            let view = create_image_view(device, image, format, vk::ImageAspectFlags::COLOR)?;
            GpuImage {
                image,
                memory,
                view,
                aux_views: Vec::new(),
            }
        };
        mips.push(gpu_image);
        extents.push(vk::Extent2D {
            width: mw,
            height: mh,
        });
    }
    Ok((mips, extents))
}

// Create the per-frame-slot bloom mip chains. Returns one chain per slot
// plus the shared mip extents (the same across all slots).
// `bloom_top` is the per-frame pooled mip 0 (one `(image, view)` per frame in
// flight) when bloom is enabled, else empty (mip 0 is committed like the rest).
#[allow(clippy::too_many_arguments)]
pub(in crate::vulkan) fn create_bloom_chain(
    instance: &ash::Instance,
    device: &Device,
    pd: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    extent: vk::Extent2D,
    frames: usize,
    bloom_top: &[(vk::Image, vk::ImageView)],
) -> Result<(Vec<Vec<GpuImage>>, Vec<vk::Extent2D>), String> {
    let mut mips = Vec::with_capacity(frames);
    let mut extents = Vec::new();
    for f in 0..frames {
        let (m, e) = create_bloom_mips(
            instance,
            device,
            pd,
            command_pool,
            queue,
            extent.width,
            extent.height,
            HDR_FORMAT,
            bloom_top.get(f).copied(),
        )?;
        if extents.is_empty() {
            extents = e;
        }
        mips.push(m);
    }
    Ok((mips, extents))
}

// Build the bloom write + blend framebuffers for every frame slot. The write
// set has one framebuffer per mip; the blend set omits the smallest mip,
// which is never upsampled into.
#[allow(clippy::type_complexity)]
pub(in crate::vulkan) fn create_bloom_framebuffers(
    device: &Device,
    write_pass: vk::RenderPass,
    blend_pass: vk::RenderPass,
    bloom_mips: &[Vec<GpuImage>],
    extents: &[vk::Extent2D],
) -> Result<(Vec<Vec<vk::Framebuffer>>, Vec<Vec<vk::Framebuffer>>), String> {
    let make_fb = |rp: vk::RenderPass, view: vk::ImageView, ext: vk::Extent2D| {
        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(rp)
            .attachments(std::slice::from_ref(&view))
            .width(ext.width)
            .height(ext.height)
            .layers(1);
        unsafe { device.create_framebuffer(&fb_info, None) }
            .map_err(|e| format!("bloom framebuffer: {e}"))
    };
    let mut write = Vec::with_capacity(bloom_mips.len());
    let mut blend = Vec::with_capacity(bloom_mips.len());
    for mips in bloom_mips {
        let mut w = Vec::with_capacity(mips.len());
        let mut b = Vec::with_capacity(mips.len().saturating_sub(1));
        for (i, mip) in mips.iter().enumerate() {
            w.push(make_fb(write_pass, mip.view, extents[i])?);
            if i + 1 < mips.len() {
                b.push(make_fb(blend_pass, mip.view, extents[i])?);
            }
        }
        write.push(w);
        blend.push(b);
    }
    Ok((write, blend))
}

// Re-point bloom input set 0's binding 0 at `view`. Used when TAA is enabled
// so the bloom prefilter thresholds the post-TAA scene image instead of the
// raw HDR resolve.
pub(in crate::vulkan) fn rebind_bloom_input0(
    device: &Device,
    set: vk::DescriptorSet,
    view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let img_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(view)
        .sampler(sampler);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(&img_info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

// Allocate + wire the bloom input descriptor sets. Per frame slot there is
// one set per distinct input image: set 0 binds that slot's HDR resolve
// image, set `1 + m` binds bloom mip `m`.
pub(in crate::vulkan) fn alloc_bloom_input_sets(
    device: &Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
    hdr_resolve_images: &[GpuImage],
    bloom_mips: &[Vec<GpuImage>],
) -> Result<Vec<Vec<vk::DescriptorSet>>, String> {
    let mut out = Vec::with_capacity(bloom_mips.len());
    for (frame, mips) in bloom_mips.iter().enumerate() {
        let layouts: Vec<_> = (0..mips.len() + 1).map(|_| layout).collect();
        let sets = alloc_descriptor_sets(device, pool, &layouts)?;
        for (idx, &set) in sets.iter().enumerate() {
            let view = if idx == 0 {
                hdr_resolve_images[frame].view
            } else {
                mips[idx - 1].view
            };
            let img_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view)
                .sampler(sampler);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&img_info));
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }
        out.push(sets);
    }
    Ok(out)
}

//  Per-frame encoder

// The bloom chain orchestration lives once in `gfx::fullscreen`; this impl binds
// + draws each sub-pass in Vulkan. `Args` is the frame-in-flight index selecting
// the per-frame framebuffers + descriptor sets (the scene input is pre-wired into
// `bloom_input_sets[frame_idx][0]`, so prefilter needs no extra argument).
impl crate::gfx::fullscreen::BloomEncoder for VkContext {
    type Rec = vk::CommandBuffer;
    type Args = usize;

    fn bloom_mip_count(&self) -> usize {
        self.bloom_mip_extents.len()
    }

    // Vulkan has no per-encode preamble; render-pass state is set per sub-pass.
    fn begin_bloom(&self, _cmd: &Self::Rec, _frame_idx: &Self::Args) {}

    // Prefilter: HDR resolve (input set 0) -> mip 0 (soft-knee + Karis).
    fn bloom_prefilter(&self, cmd: &Self::Rec, frame_idx: &Self::Args) {
        let f = *frame_idx;
        self.bloom_run_pass(
            *cmd,
            self.bloom_write_pass,
            self.bloom_write_framebuffers[f][0],
            self.bloom_mip_extents[0],
            self.bloom_pipeline_prefilter,
            self.bloom_input_sets[f][0],
        );
    }

    // Downsample: mip dst-1 -> mip dst. Input set for mip m is `m`.
    fn bloom_downsample(&self, cmd: &Self::Rec, frame_idx: &Self::Args, dst: usize) {
        let f = *frame_idx;
        self.bloom_run_pass(
            *cmd,
            self.bloom_write_pass,
            self.bloom_write_framebuffers[f][dst],
            self.bloom_mip_extents[dst],
            self.bloom_pipeline_downsample,
            self.bloom_input_sets[f][dst],
        );
    }

    // Upsample: mip dst+1 -> mip dst, additively blended. Input set is `dst + 2`.
    fn bloom_upsample(&self, cmd: &Self::Rec, frame_idx: &Self::Args, dst: usize) {
        let f = *frame_idx;
        self.bloom_run_pass(
            *cmd,
            self.bloom_blend_pass,
            self.bloom_blend_framebuffers[f][dst],
            self.bloom_mip_extents[dst],
            self.bloom_pipeline_upsample,
            self.bloom_input_sets[f][dst + 2],
        );
    }
}

impl VkContext {
    // Encode the bloom prefilter, downsample, and additive upsample passes for
    // frame slot `frame_idx` via the shared `gfx::fullscreen` driver. On return
    // `bloom_mips[frame_idx][0]` holds the accumulated bloom the composite pass
    // samples. Called only when `post_process.bloom_intensity > 0`.
    pub(in crate::vulkan) fn encode_bloom(&self, cmd: vk::CommandBuffer, frame_idx: usize) {
        crate::gfx::fullscreen::encode_bloom_chain(self, &cmd, frame_idx);
    }

    // One fullscreen-triangle bloom sub-pass: render into `framebuffer` (sized
    // `ext`) sampling `input_set`, with `pipeline` bound inside `render_pass`.
    fn bloom_run_pass(
        &self,
        cmd: vk::CommandBuffer,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        ext: vk::Extent2D,
        pipeline: vk::Pipeline,
        input_set: vk::DescriptorSet,
    ) {
        let device = &self.device;
        let push = self.post_process;
        let push_bytes = unsafe {
            std::slice::from_raw_parts(
                &push as *const PostProcessParams as *const u8,
                std::mem::size_of::<PostProcessParams>(),
            )
        };
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D::default().extent(ext));
        let vp = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: ext.width as f32,
            height: ext.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(ext);
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.bloom_pipeline_layout,
                0,
                std::slice::from_ref(&input_set),
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.bloom_pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                push_bytes,
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
        }
    }
}
