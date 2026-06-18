// src/vulkan/post/ssao.rs
//
// SSAO (GTAO) for the Vulkan backend. Owns the GTAO horizon-search kernel
// pipeline, the depth-aware blur pipeline, and the `encode_ssao` per-frame
// encoder. The depth + normal the kernel / blur sample come from the unified
// G-buffer pre-pass; the kernel / blur sets are wired to its per-frame views
// by `init.rs`.
//
// The main pass samples `SsaoResources::ao` (the blurred occlusion) at set 0
// binding 6 to modulate its ambient term; when SSAO is disabled the renderer
// binds the 1×1 `ssao_white` fallback at that slot so the multiplier is a
// pass-through 1.0. Mirrors src/directx/post/ssao.rs.

use ash::{Device, vk};

use crate::gfx::render_types::SsaoParams;

use super::super::context::VkContext;
use super::super::pipeline::*;
use super::super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::super::texture::*;

// GLSL sources
const SSAO_FULLSCREEN_VERT_GLSL: &str = include_str!("../shaders/ssao_fullscreen.vert");
const SSAO_KERNEL_FRAG_GLSL: &str = include_str!("../shaders/ssao_kernel.frag");
const SSAO_BLUR_FRAG_GLSL: &str = include_str!("../shaders/ssao_blur.frag");

// Single-channel occlusion target format. 1.0 = unoccluded; the main pass
// multiplies the ambient term by this value.
pub(in crate::vulkan) const SSAO_OCCLUSION_FORMAT: vk::Format = vk::Format::R8_UNORM;

// SSAO resources held by `VkContext` when `PostProcessConfig.ssao` is on.
// All `vk::*` handles are owned by this struct and freed on `destroy`.
pub(in crate::vulkan) struct SsaoResources {
    // Resolved authored tunables; turned into a per-frame `SsaoParams` push.
    pub(in crate::vulkan) settings: crate::gfx::ssao::SsaoSettings,

    // The kernel pass writes `ao_raw` through this render pass: it discards
    // on load (UNDEFINED) and stores SHADER_READ_ONLY so the blur can sample
    // ao_raw. `ao_raw` is SSAO-internal (not a graph resource), so it stays
    // render-pass-driven.
    pub(in crate::vulkan) fullscreen_render_pass: vk::RenderPass,

    // The blur pass writes `ao` (the graph's `ao_output`) through this render
    // pass. `ao`'s layout transitions are graph-driven, so this pass performs
    // none: it keeps `ao` in COLOR_ATTACHMENT_OPTIMAL (initial == final) and
    // the executor emits ao_output's `barriers_before` around it
    // (UNDEFINED -> COLOR_ATTACHMENT before SsaoBlur, COLOR_ATTACHMENT ->
    // SHADER_READ before Main).
    pub(in crate::vulkan) blur_render_pass: vk::RenderPass,

    // Kernel pipeline (GTAO horizon search): fullscreen triangle reading
    // the G-buffer, writing the raw R8 occlusion target.
    pub(in crate::vulkan) kernel_set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) kernel_layout: vk::PipelineLayout,
    pub(in crate::vulkan) kernel_pso: vk::Pipeline,

    // Blur pipeline: fullscreen triangle reading raw occlusion + G-buffer
    // depth, writing the final blurred occlusion target.
    pub(in crate::vulkan) blur_set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) blur_layout: vk::PipelineLayout,
    pub(in crate::vulkan) blur_pso: vk::Pipeline,

    // Per-frame kernel / blur sets. The kernel set binding 0 and the blur set
    // binding 1 sample the unified pre-pass G-buffer normal+depth (a per-frame
    // target), so these sets are per-frame too (one slot per frame in flight).
    // The blur set binding 0 samples the SSAO-internal raw AO, a single shared
    // target.
    pub(in crate::vulkan) kernel_sets: Vec<vk::DescriptorSet>,
    pub(in crate::vulkan) blur_sets: Vec<vk::DescriptorSet>,
    pub(in crate::vulkan) descriptor_pool: vk::DescriptorPool,

    // Linear-clamp sampler for the kernel/blur G-buffer / raw-AO reads.
    pub(in crate::vulkan) sampler: vk::Sampler,

    // Resolution-dependent targets (rebuilt on swapchain resize). `ao_raw` is
    // SSAO-internal (the kernel's raw occlusion, sampled by the blur). The
    // blurred `ao_output` the main pass samples is the graph's transient and is
    // owned by `VkContext::transient_pool`, per frame in flight; there is one
    // `blur_framebuffers` entry per frame, each built from that frame's pooled
    // `ao_output` view passed in at build time.
    pub(in crate::vulkan) ao_raw: GpuImage,
    pub(in crate::vulkan) kernel_framebuffer: vk::Framebuffer,
    pub(in crate::vulkan) blur_framebuffers: Vec<vk::Framebuffer>,
}

// SPIR-V blobs for every SSAO pipeline. Produced by
// [`compile_ssao_shaders`]; consumed by `SsaoResources::new` at init and by
// `rebuild_ssao_pipelines` during shader hot-reload. Mirrors
// [`crate::vulkan::post::bloom::BloomShaders`].
pub(in crate::vulkan) struct SsaoShaders {
    pub fullscreen_vs: Vec<u8>,
    pub kernel_fs: Vec<u8>,
    pub blur_fs: Vec<u8>,
}

// Compile every SSAO GLSL source. `hot_reload` routes each source resolve
// through [`crate::vulkan::pipeline::shader_source`] so dev-loop edits take
// effect on the next pipeline build. Called from `SsaoResources::new` at
// init and by the Vulkan shader hot-reload path.
pub(in crate::vulkan) fn compile_ssao_shaders(hot_reload: bool) -> Result<SsaoShaders, String> {
    use super::super::pipeline::shader_source;
    Ok(SsaoShaders {
        fullscreen_vs: compile_glsl(
            &shader_source(
                hot_reload,
                "ssao_fullscreen.vert",
                SSAO_FULLSCREEN_VERT_GLSL,
            ),
            shaderc::ShaderKind::Vertex,
            "ssao_fullscreen.vert",
        )?,
        kernel_fs: compile_glsl(
            &shader_source(hot_reload, "ssao_kernel.frag", SSAO_KERNEL_FRAG_GLSL),
            shaderc::ShaderKind::Fragment,
            "ssao_kernel.frag",
        )?,
        blur_fs: compile_glsl(
            &shader_source(hot_reload, "ssao_blur.frag", SSAO_BLUR_FRAG_GLSL),
            shaderc::ShaderKind::Fragment,
            "ssao_blur.frag",
        )?,
    })
}

// Replacement SSAO pipelines built by the hot-reload pass. Each lines up
// 1:1 with the matching field on [`SsaoResources`]. Mirrors
// `directx::post::ssao::RebuiltSsaoPipelines`.
pub(in crate::vulkan) struct RebuiltSsaoPipelines {
    pub kernel: vk::Pipeline,
    pub blur: vk::Pipeline,
}

// Rebuild every live SSAO pipeline from disk-resident GLSL source against
// the existing layouts + render pass. Returns the freshly built handles;
// the caller is responsible for destroying the displaced pipelines only
// after this call succeeds (any compile / pipeline-create failure short-
// circuits with the previous handles untouched). Called by the Vulkan
// shader hot-reload path.
pub(in crate::vulkan) fn rebuild_ssao_pipelines(
    device: &Device,
    ssao: &SsaoResources,
    hot_reload: bool,
) -> Result<RebuiltSsaoPipelines, String> {
    let shaders = compile_ssao_shaders(hot_reload)?;
    let kernel = create_fullscreen_pipeline(
        device,
        ssao.fullscreen_render_pass,
        ssao.kernel_layout,
        &shaders.fullscreen_vs,
        &shaders.kernel_fs,
    )?;
    let blur = create_fullscreen_pipeline(
        device,
        ssao.blur_render_pass,
        ssao.blur_layout,
        &shaders.fullscreen_vs,
        &shaders.blur_fs,
    )?;
    Ok(RebuiltSsaoPipelines { kernel, blur })
}

impl SsaoResources {
    // Swap the freshly-built pipelines into the live resources. The caller
    // has already `device_wait_idle`'d so the old pipelines are not in
    // flight. Driven by the Vulkan shader hot-reload pass after every
    // replacement successfully compiled.
    pub(in crate::vulkan) fn swap_pipelines(
        &mut self,
        device: &Device,
        rebuilt: RebuiltSsaoPipelines,
    ) {
        unsafe {
            device.destroy_pipeline(self.kernel_pso, None);
            device.destroy_pipeline(self.blur_pso, None);
        }
        self.kernel_pso = rebuilt.kernel;
        self.blur_pso = rebuilt.blur;
    }
}

// Kernel render pass: one R8_UNORM colour attachment, no depth. The
// fullscreen triangle overwrites every pixel so `DONT_CARE` is safe on load.
// Ends shader-readable so the blur can sample the raw occlusion it writes.
fn create_fullscreen_render_pass(device: &Device) -> Result<vk::RenderPass, String> {
    let attachment = vk::AttachmentDescription::default()
        .format(SSAO_OCCLUSION_FORMAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));
    let dep = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dep));
    unsafe { device.create_render_pass(&info, None) }
        .map_err(|e| format!("SSAO fullscreen render pass: {e}"))
}

// Blur render pass: same R8_UNORM colour attachment as the kernel pass, but
// it performs no layout transition. `ao` (the graph's `ao_output`) enters and
// leaves in COLOR_ATTACHMENT_OPTIMAL; the executor emits ao_output's
// graph-derived barriers around the SsaoBlur and Main passes. The
// SUBPASS_EXTERNAL dependency is kept identical to the kernel pass so the
// write-after-read hazard against the previous frame's main-pass sample of
// `ao` stays guarded.
fn create_blur_render_pass(device: &Device) -> Result<vk::RenderPass, String> {
    let attachment = vk::AttachmentDescription::default()
        .format(SSAO_OCCLUSION_FORMAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));
    let dep = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dep));
    unsafe { device.create_render_pass(&info, None) }
        .map_err(|e| format!("SSAO blur render pass: {e}"))
}

// Allocate an R8_UNORM target usable as both colour attachment and sampled
// texture. No pre-transition: the render pass declares an `UNDEFINED`
// initial layout.
fn create_ao_target(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
) -> Result<GpuImage, String> {
    let (image, memory) = create_image(
        instance,
        device,
        physical_device,
        width,
        height,
        SSAO_OCCLUSION_FORMAT,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::SampleCountFlags::TYPE_1,
    )?;
    let view = create_image_view(
        device,
        image,
        SSAO_OCCLUSION_FORMAT,
        vk::ImageAspectFlags::COLOR,
    )?;
    Ok(GpuImage {
        image,
        memory,
        view,
        aux_views: Vec::new(),
    })
}

// Build a fullscreen kernel/blur pipeline. No vertex input (the
// fullscreen triangle is procedural in the VS); no depth; no blend; writes
// the R8 occlusion target.
fn create_fullscreen_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
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
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::ALWAYS);
    let blend_attach = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::R)
        .blend_enable(false);
    let blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(std::slice::from_ref(&blend_attach));
    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vert_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipeline = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create ssao fullscreen pso: {e}"))?[0];
    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

#[allow(clippy::too_many_arguments)]
impl SsaoResources {
    pub(in crate::vulkan) fn new(
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        frames: usize,
        settings: crate::gfx::ssao::SsaoSettings,
        ao_views: &[vk::ImageView],
        hot_reload: bool,
    ) -> Result<Self, String> {
        let fullscreen_render_pass = create_fullscreen_render_pass(device)?;
        let blur_render_pass = create_blur_render_pass(device)?;

        // Kernel set 0: G-buffer sampler.
        let kernel_set_layout = create_descriptor_set_layout(
            device,
            &[(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
            )],
        )?;
        // Blur set 0: ao_raw + G-buffer samplers.
        let blur_set_layout = create_descriptor_set_layout(
            device,
            &[
                (
                    0,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    1,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
            ],
        )?;

        // Pipeline layouts.
        let params_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<SsaoParams>() as u32);
        let kernel_set_layouts = [kernel_set_layout];
        let kernel_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&kernel_set_layouts)
                    .push_constant_ranges(std::slice::from_ref(&params_push)),
                None,
            )
        }
        .map_err(|e| format!("ssao kernel layout: {e}"))?;

        let blur_set_layouts = [blur_set_layout];
        let blur_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&blur_set_layouts),
                None,
            )
        }
        .map_err(|e| format!("ssao blur layout: {e}"))?;

        // Pipelines.
        let shaders = compile_ssao_shaders(hot_reload)?;
        let kernel_pso = create_fullscreen_pipeline(
            device,
            fullscreen_render_pass,
            kernel_layout,
            &shaders.fullscreen_vs,
            &shaders.kernel_fs,
        )?;
        let blur_pso = create_fullscreen_pipeline(
            device,
            blur_render_pass,
            blur_layout,
            &shaders.fullscreen_vs,
            &shaders.blur_fs,
        )?;

        // Descriptor pool: `frames` kernel sets (1 sampler each) + `frames` blur
        // sets (2 samplers each). The kernel/blur sets are per-frame so each
        // binds its own frame's unified G-buffer normal+depth.
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(frames as u32 * 3)];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(frames as u32 * 2),
                None,
            )
        }
        .map_err(|e| format!("ssao descriptor pool: {e}"))?;

        let kernel_layouts: Vec<_> = (0..frames).map(|_| kernel_set_layout).collect();
        let kernel_sets = alloc_descriptor_sets(device, descriptor_pool, &kernel_layouts)?;
        let blur_layouts: Vec<_> = (0..frames).map(|_| blur_set_layout).collect();
        let blur_sets = alloc_descriptor_sets(device, descriptor_pool, &blur_layouts)?;

        // Dedicated linear-clamp sampler for kernel/blur reads.
        let sampler = create_sampler_linear_clamp(device)?;

        // Resolution-dependent targets + framebuffers.
        let mut me = Self {
            settings,
            fullscreen_render_pass,
            blur_render_pass,
            kernel_set_layout,
            kernel_layout,
            kernel_pso,
            blur_set_layout,
            blur_layout,
            blur_pso,
            kernel_sets,
            blur_sets,
            descriptor_pool,
            sampler,
            // Placeholder GpuImage; replaced by build_targets below. The
            // blurred `ao_output` lives in the transient pool, not here.
            ao_raw: GpuImage {
                image: vk::Image::null(),
                memory: vk::DeviceMemory::null(),
                view: vk::ImageView::null(),
                aux_views: Vec::new(),
            },
            kernel_framebuffer: vk::Framebuffer::null(),
            blur_framebuffers: Vec::new(),
        };
        me.build_targets(instance, device, physical_device, width, height, ao_views)?;
        // The kernel/blur normal+depth bindings are re-pointed at the unified
        // pre-pass per-frame views by the caller before the first frame; the
        // raw-AO view stands in until then so every binding is valid.
        me.wire_kernel_and_blur_sets(device, &[]);
        Ok(me)
    }

    // Allocate or re-allocate the resolution-dependent targets + framebuffers
    // at the given extent. Caller has either just constructed `self` (raw
    // fields are `NULL`) or already idled the device + destroyed the previous
    // targets via `destroy_targets`.
    fn build_targets(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        ao_views: &[vk::ImageView],
    ) -> Result<(), String> {
        let w = width.max(1);
        let h = height.max(1);
        self.ao_raw = create_ao_target(instance, device, physical_device, w, h)?;

        self.kernel_framebuffer = unsafe {
            device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(self.fullscreen_render_pass)
                    .attachments(std::slice::from_ref(&self.ao_raw.view))
                    .width(w)
                    .height(h)
                    .layers(1),
                None,
            )
        }
        .map_err(|e| format!("ssao kernel framebuffer: {e}"))?;
        // One blur framebuffer per frame in flight, each bound to that frame's
        // pooled `ao_output` view.
        let mut blur_framebuffers = Vec::with_capacity(ao_views.len());
        for &ao_view in ao_views {
            let fb = unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.blur_render_pass)
                        .attachments(std::slice::from_ref(&ao_view))
                        .width(w)
                        .height(h)
                        .layers(1),
                    None,
                )
            }
            .map_err(|e| format!("ssao blur framebuffer: {e}"))?;
            blur_framebuffers.push(fb);
        }
        self.blur_framebuffers = blur_framebuffers;
        Ok(())
    }

    // Wire the per-frame kernel + blur descriptor sets to the current G-buffer /
    // raw-AO views. Called after `build_targets` (init or resize) so the
    // descriptor-set targets stay in sync with the underlying images.
    //
    // `gbuffer_views` carries the unified pre-pass's per-frame normal+depth
    // views; kernel/blur set `i` binds slot `i`. When empty (the init pre-wire
    // before the caller re-points them) the raw-AO view stands in so the binding
    // is always a valid `SHADER_READ_ONLY` image. The blur set binding 0 always
    // samples the SSAO-internal raw AO (a single shared target).
    fn wire_kernel_and_blur_sets(&self, device: &Device, gbuffer_views: &[vk::ImageView]) {
        let raw_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(self.ao_raw.view)
            .sampler(self.sampler);
        for f in 0..self.kernel_sets.len() {
            let gb_view = if gbuffer_views.is_empty() {
                self.ao_raw.view
            } else {
                gbuffer_views[f % gbuffer_views.len()]
            };
            let gb_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(gb_view)
                .sampler(self.sampler);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.kernel_sets[f])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&gb_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.blur_sets[f])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&raw_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.blur_sets[f])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&gb_info)),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    // Re-point the per-frame kernel/blur G-buffer bindings at the unified
    // pre-pass per-frame normal+depth views. Called by the caller (init /
    // resize) when the unified G-buffer pre-pass is active. Mirrors the SSR
    // resolve / SSGI re-points.
    pub(in crate::vulkan) fn wire_kernel_and_blur_sets_gbuffer(
        &self,
        device: &Device,
        gbuffer_views: &[vk::ImageView],
    ) {
        self.wire_kernel_and_blur_sets(device, gbuffer_views);
    }

    fn destroy_targets(&mut self, device: &Device) {
        if self.kernel_framebuffer != vk::Framebuffer::null() {
            unsafe {
                device.destroy_framebuffer(self.kernel_framebuffer, None);
                for &fb in &self.blur_framebuffers {
                    device.destroy_framebuffer(fb, None);
                }
            }
            self.kernel_framebuffer = vk::Framebuffer::null();
            self.blur_framebuffers.clear();
            self.ao_raw.destroy(device);
            // `ao_output` is pool-owned (per frame); the pool frees it.
        }
    }

    // Rebuild the resolution-dependent targets at a new swapchain extent and
    // re-wire the kernel + blur descriptor sets. The caller has already
    // idled the device.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        gbuffer_views: &[vk::ImageView],
        ao_views: &[vk::ImageView],
    ) -> Result<(), String> {
        self.destroy_targets(device);
        self.build_targets(instance, device, physical_device, width, height, ao_views)?;
        self.wire_kernel_and_blur_sets(device, gbuffer_views);
        Ok(())
    }

    // Destroy every SSAO resource. The caller has already idled the device.
    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        self.destroy_targets(device);
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.kernel_pso, None);
            device.destroy_pipeline(self.blur_pso, None);
            device.destroy_pipeline_layout(self.kernel_layout, None);
            device.destroy_pipeline_layout(self.blur_layout, None);
            device.destroy_descriptor_set_layout(self.kernel_set_layout, None);
            device.destroy_descriptor_set_layout(self.blur_set_layout, None);
            device.destroy_render_pass(self.fullscreen_render_pass, None);
            device.destroy_render_pass(self.blur_render_pass, None);
        }
    }
}

// Encoder
impl VkContext {
    // Encode the GTAO horizon-search kernel and the depth-aware blur over the
    // unified pre-pass's normal+depth G-buffer. Called from `record_frame`
    // before `cmd_begin_render_pass(main_render_pass)` so the main fragment
    // shader sees the fresh blurred occlusion via set 0 binding 6. No-op when
    // SSAO is disabled.
    pub(in crate::vulkan) fn encode_ssao(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
    ) {
        let ssao = match &self.ssao {
            Some(s) => s,
            None => return,
        };
        let device = &self.device;
        let extent = self.render_extent;

        let params = ssao.settings.params(fov_y_radians, aspect);
        let scissor = vk::Rect2D::default().extent(extent);

        // Kernel: GTAO horizon search over the G-buffer → raw R8 AO
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(ssao.fullscreen_render_pass)
            .framebuffer(ssao.kernel_framebuffer)
            .render_area(vk::Rect2D::default().extent(extent));
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            // Fullscreen kernel uses a positive-height viewport; the kernel
            // shader's UV map ((pos+1)/2) lines up with the upright G-buffer.
            let fs_vp = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&fs_vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, ssao.kernel_pso);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                ssao.kernel_layout,
                0,
                std::slice::from_ref(&ssao.kernel_sets[frame_idx]),
                &[],
            );
            device.cmd_push_constants(
                cmd,
                ssao.kernel_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                std::slice::from_raw_parts(
                    &params as *const SsaoParams as *const u8,
                    std::mem::size_of::<SsaoParams>(),
                ),
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
        }

        // Blur: depth-aware smoothing of raw AO → final blurred AO. `ao`'s
        // layout transitions are graph-driven: the executor emits ao_output's
        // barriers_before (UNDEFINED -> COLOR_ATTACHMENT before this pass,
        // COLOR_ATTACHMENT -> SHADER_READ before Main), and this render pass
        // keeps `ao` in COLOR_ATTACHMENT_OPTIMAL throughout.
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(ssao.blur_render_pass)
            .framebuffer(ssao.blur_framebuffers[frame_idx])
            .render_area(vk::Rect2D::default().extent(extent));
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            let fs_vp = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&fs_vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, ssao.blur_pso);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                ssao.blur_layout,
                0,
                std::slice::from_ref(&ssao.blur_sets[frame_idx]),
                &[],
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
        }
    }
}
