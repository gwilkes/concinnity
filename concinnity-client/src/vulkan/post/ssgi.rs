// src/vulkan/post/ssgi.rs
//
// Screen-space global illumination for the Vulkan backend: a refinement of SSR.
// It reuses the SSR depth + normal pre-pass G-buffer (so turning SSGI on forces
// that pre-pass to run even when the SSR resolve is off) and runs two fullscreen
// passes on the hdr_resolve RMW chain after the main pass:
//
//   * gather:    per pixel, a cone of cosine-weighted hemisphere rays marched
//                against the G-buffer, accumulating the lit scene colour at
//                each on-screen hit into an off-screen `gi` target.
//   * composite: a depth-aware blur of that noisy `gi` target, additively
//                blended (ONE / ONE) into `hdr_resolve` so the near-field
//                indirect bounce layers on top of the IBL ambient.
//
// Mirrors src/directx/post/ssgi.rs and src/metal/post/ssgi.rs. Unlike DirectX,
// the 32-byte `SsgiParams` rides as a push constant (the same shape the SSR
// resolve uses for `SsrParams`), so there is no per-frame param UBO: the
// descriptor sets carry only image samplers.

use ash::{Device, vk};

use crate::gfx::fullscreen::{FullscreenPass, encode_fullscreen};
use crate::gfx::render_types::SsgiParams;
use crate::gfx::ssgi::SsgiSettings;

use super::super::context::{HDR_FORMAT, VkContext};
use super::super::pipeline::*;
use super::super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::super::texture::*;

// GLSL sources
const SSGI_FULLSCREEN_VERT_GLSL: &str = include_str!("../shaders/ssgi_fullscreen.vert");
const SSGI_GATHER_FRAG_GLSL: &str = include_str!("../shaders/ssgi_gather.frag");
const SSGI_COMPOSITE_FRAG_GLSL: &str = include_str!("../shaders/ssgi_composite.frag");

// SPIR-V blobs for the SSGI pipelines. Produced by [`compile_ssgi_shaders`];
// consumed by `SsgiResources::new` at init and by `rebuild_ssgi_pipelines`
// during shader hot-reload. Mirrors the matching SSR struct.
pub(in crate::vulkan) struct SsgiShaders {
    pub vs: Vec<u8>,
    pub gather_fs: Vec<u8>,
    pub composite_fs: Vec<u8>,
}

// Compile every SSGI GLSL source. `hot_reload` routes each source resolve
// through [`crate::vulkan::pipeline::shader_source`].
pub(in crate::vulkan) fn compile_ssgi_shaders(hot_reload: bool) -> Result<SsgiShaders, String> {
    use super::super::pipeline::shader_source;
    Ok(SsgiShaders {
        vs: compile_glsl(
            &shader_source(
                hot_reload,
                "ssgi_fullscreen.vert",
                SSGI_FULLSCREEN_VERT_GLSL,
            ),
            shaderc::ShaderKind::Vertex,
            "ssgi_fullscreen.vert",
        )?,
        gather_fs: compile_glsl(
            &shader_source(hot_reload, "ssgi_gather.frag", SSGI_GATHER_FRAG_GLSL),
            shaderc::ShaderKind::Fragment,
            "ssgi_gather.frag",
        )?,
        composite_fs: compile_glsl(
            &shader_source(hot_reload, "ssgi_composite.frag", SSGI_COMPOSITE_FRAG_GLSL),
            shaderc::ShaderKind::Fragment,
            "ssgi_composite.frag",
        )?,
    })
}

// SSGI resources held by `VkContext` when `PostProcessConfig.indirect_lighting`
// is `ssgi`. All `vk::*` handles are owned here and freed on `destroy`.
pub(in crate::vulkan) struct SsgiResources {
    // Resolved authored tunables; turned into a per-frame `SsgiParams` push.
    pub(in crate::vulkan) settings: SsgiSettings,

    // Render passes: gather writes the `gi` target, composite LOAD/blends into
    // the HDR resolve.
    gather_render_pass: vk::RenderPass,
    composite_render_pass: vk::RenderPass,

    // One set layout shared by both passes (binding 0 = scene/gi, binding 1 =
    // gbuffer) + one pipeline layout (that set + the SsgiParams push range).
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    gather_pso: vk::Pipeline,
    composite_pso: vk::Pipeline,

    descriptor_pool: vk::DescriptorPool,
    // Per-frame gather sets: each binds that frame's HDR resolve as the scene
    // input (binding 0) + that frame's G-buffer (binding 1).
    gather_sets: Vec<vk::DescriptorSet>,
    // Per-frame composite sets: the gi target (binding 0, shared) + that frame's
    // G-buffer (binding 1). Per-frame because the unified G-buffer is a per-frame
    // target, so the composite must sample its own frame's normal+depth.
    composite_sets: Vec<vk::DescriptorSet>,

    // Linear-clamp sampler the passes read scene / gi / G-buffer through.
    sampler: vk::Sampler,

    // Resolution-dependent targets (rebuilt on swapchain resize).
    gi: GpuImage,
    // The `gi_scale`-reduced gather extent (the gi target + gather framebuffer
    // size). The gather pass rasterizes at this extent; the composite stays at
    // full render resolution and bilateral-upsamples the gi target.
    gi_extent: vk::Extent2D,
    gather_framebuffer: vk::Framebuffer,
    // One per HDR-resolve slot: the composite LOAD/blends into each frame's
    // resolved scene in place.
    composite_framebuffers: Vec<vk::Framebuffer>,
}

// Broad SUBPASS_EXTERNAL dependencies shared by both SSGI render passes. The
// `dep_in` synchronises every prior colour write *and* shader read (the main
// pass's hdr_resolve, the SSR pre-pass's G-buffer, the gather's gi) against this
// pass's reads + writes; the `dep_out` makes this pass's colour write available
// to the next pass's fragment sample. Same shape as the SSR resolve + decal
// render-pass dependencies that already smoke clean under the validation layer.
fn ssgi_external_deps() -> [vk::SubpassDependency; 2] {
    let dep_in = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::SHADER_READ,
        );
    let dep_out = vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    [dep_in, dep_out]
}

// Gather render pass: one HDR-format colour attachment (`gi`). The gather
// overwrites every pixel so `DONT_CARE` is safe on load; ends shader-readable
// for the composite to sample.
fn create_gather_render_pass(device: &Device) -> Result<vk::RenderPass, String> {
    let attachment = vk::AttachmentDescription::default()
        .format(HDR_FORMAT)
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
    let deps = ssgi_external_deps();
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&deps);
    unsafe { device.create_render_pass(&info, None) }
        .map_err(|e| format!("SSGI gather render pass: {e}"))
}

// Composite render pass: LOAD the HDR resolve, additively blend the denoised
// indirect term, STORE. Stays in SHADER_READ_ONLY_OPTIMAL in + out so the next
// RMW pass (Decals / Fog / SSR resolve) samples it unchanged. Mirrors
// `create_decal_render_pass`.
fn create_composite_render_pass(device: &Device) -> Result<vk::RenderPass, String> {
    let attachment = vk::AttachmentDescription::default()
        .format(HDR_FORMAT)
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
    let deps = ssgi_external_deps();
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&deps);
    unsafe { device.create_render_pass(&info, None) }
        .map_err(|e| format!("SSGI composite render pass: {e}"))
}

// Allocate a single-format colour render target usable as both attachment and
// sampled texture. Mirrors the SSR `create_color_target`.
fn create_gi_target(
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
        HDR_FORMAT,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::SampleCountFlags::TYPE_1,
    )?;
    let view = create_image_view(device, image, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
    Ok(GpuImage {
        image,
        memory,
        view,
        aux_views: Vec::new(),
    })
}

// Build one fullscreen SSGI pipeline. No vertex input (the fullscreen triangle
// is procedural in the VS); no depth. `additive` configures an `ONE / ONE` add
// blend (the composite blends into the scene) vs. a plain write (the gather
// fills its own `gi` target).
fn create_ssgi_pipeline(
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
    let blend_attach = if additive {
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
    .map_err(|(_, e)| format!("create ssgi pso: {e}"))?[0];
    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

// Replacement SSGI pipelines built by the hot-reload pass.
pub(in crate::vulkan) struct RebuiltSsgiPipelines {
    pub gather: vk::Pipeline,
    pub composite: vk::Pipeline,
}

// Rebuild both SSGI pipelines from disk-resident GLSL source against the
// existing layout + render passes. Same shape as `rebuild_ssr_pipelines`.
pub(in crate::vulkan) fn rebuild_ssgi_pipelines(
    device: &Device,
    ssgi: &SsgiResources,
    hot_reload: bool,
) -> Result<RebuiltSsgiPipelines, String> {
    let shaders = compile_ssgi_shaders(hot_reload)?;
    let gather = create_ssgi_pipeline(
        device,
        ssgi.gather_render_pass,
        ssgi.pipeline_layout,
        &shaders.vs,
        &shaders.gather_fs,
        false,
    )?;
    let composite = create_ssgi_pipeline(
        device,
        ssgi.composite_render_pass,
        ssgi.pipeline_layout,
        &shaders.vs,
        &shaders.composite_fs,
        true,
    )?;
    Ok(RebuiltSsgiPipelines { gather, composite })
}

impl SsgiResources {
    // Build every SSGI resource. `hdr_resolve_views` feeds the per-frame gather
    // scene input + the composite framebuffers; `gbuffer_view` is the SSR
    // pre-pass G-buffer (view normal + linear depth) every set samples.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vulkan) fn new(
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        frames: usize,
        settings: SsgiSettings,
        hdr_resolve_views: &[vk::ImageView],
        gbuffer_view: vk::ImageView,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let gather_render_pass = create_gather_render_pass(device)?;
        let composite_render_pass = create_composite_render_pass(device)?;

        // set 0: binding 0 = scene/gi, binding 1 = gbuffer. Shared by both passes.
        let set_layout = create_descriptor_set_layout(
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

        let push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<SsgiParams>() as u32);
        let set_layouts = [set_layout];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(std::slice::from_ref(&push)),
                None,
            )
        }
        .map_err(|e| format!("ssgi pipeline layout: {e}"))?;

        let shaders = compile_ssgi_shaders(hot_reload)?;
        let gather_pso = create_ssgi_pipeline(
            device,
            gather_render_pass,
            pipeline_layout,
            &shaders.vs,
            &shaders.gather_fs,
            false,
        )?;
        let composite_pso = create_ssgi_pipeline(
            device,
            composite_render_pass,
            pipeline_layout,
            &shaders.vs,
            &shaders.composite_fs,
            true,
        )?;

        // Pool: `frames` gather sets + `frames` composite sets, 2 samplers each.
        // Both are per-frame so each binds its own frame's unified G-buffer.
        let sampler_count = frames as u32 * 2 * 2;
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(sampler_count);
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(std::slice::from_ref(&pool_size))
                    .max_sets(frames as u32 * 2),
                None,
            )
        }
        .map_err(|e| format!("ssgi descriptor pool: {e}"))?;

        let gather_layouts: Vec<_> = (0..frames).map(|_| set_layout).collect();
        let gather_sets = alloc_descriptor_sets(device, descriptor_pool, &gather_layouts)?;
        let composite_layouts: Vec<_> = (0..frames).map(|_| set_layout).collect();
        let composite_sets = alloc_descriptor_sets(device, descriptor_pool, &composite_layouts)?;

        let sampler = create_sampler_linear_clamp(device)?;

        let mut me = Self {
            settings,
            gather_render_pass,
            composite_render_pass,
            set_layout,
            pipeline_layout,
            gather_pso,
            composite_pso,
            descriptor_pool,
            gather_sets,
            composite_sets,
            sampler,
            // Placeholder; replaced by build_targets below.
            gi: GpuImage {
                image: vk::Image::null(),
                memory: vk::DeviceMemory::null(),
                view: vk::ImageView::null(),
                aux_views: Vec::new(),
            },
            gi_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            gather_framebuffer: vk::Framebuffer::null(),
            composite_framebuffers: Vec::new(),
        };
        me.build_targets(
            instance,
            device,
            physical_device,
            width,
            height,
            hdr_resolve_views,
        )?;
        me.wire_sets(
            device,
            hdr_resolve_views,
            std::slice::from_ref(&gbuffer_view),
        );
        Ok(me)
    }

    // Allocate (or re-allocate) the resolution-dependent `gi` target + the
    // gather / composite framebuffers at the given extent.
    fn build_targets(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        hdr_resolve_views: &[vk::ImageView],
    ) -> Result<(), String> {
        let w = width.max(1);
        let h = height.max(1);
        // The gather runs at `gi_scale`-reduced resolution; the composite (which
        // LOAD/blends into the full-res HDR resolve) bilateral-upsamples it back,
        // reading the gi texture's own dimensions for the tap stride. So the gi
        // target + the gather framebuffer shrink while the composite framebuffers
        // stay at full render resolution. Mirrors metal/post/ssgi.
        let (gw, gh) = self.settings.gi_dimensions(w, h);
        self.gi_extent = vk::Extent2D {
            width: gw,
            height: gh,
        };
        self.gi = create_gi_target(instance, device, physical_device, gw, gh)?;
        self.gather_framebuffer = unsafe {
            device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(self.gather_render_pass)
                    .attachments(std::slice::from_ref(&self.gi.view))
                    .width(gw)
                    .height(gh)
                    .layers(1),
                None,
            )
        }
        .map_err(|e| format!("ssgi gather framebuffer: {e}"))?;
        let mut fbs = Vec::with_capacity(hdr_resolve_views.len());
        for &view in hdr_resolve_views {
            let fb = unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.composite_render_pass)
                        .attachments(std::slice::from_ref(&view))
                        .width(w)
                        .height(h)
                        .layers(1),
                    None,
                )
            }
            .map_err(|e| format!("ssgi composite framebuffer: {e}"))?;
            fbs.push(fb);
        }
        self.composite_framebuffers = fbs;
        Ok(())
    }

    // Wire the per-frame gather sets (scene = HDR resolve, gbuffer) and the
    // per-frame composite sets (gi, gbuffer). Called after `build_targets` (init
    // or resize) so the sets see the current images.
    //
    // When the unified G-buffer pre-pass is active, `gbuffer_views` carries its
    // per-frame normal+depth views and set `i` binds slot `i`; when the slice
    // has a single entry it is shared across frames (the legacy SSR pre-pass
    // G-buffer, a single image). The gather scene input + composite gi are
    // per-frame / shared respectively as before.
    pub(in crate::vulkan) fn wire_sets(
        &self,
        device: &Device,
        hdr_resolve_views: &[vk::ImageView],
        gbuffer_views: &[vk::ImageView],
    ) {
        let gb_view = |i: usize| gbuffer_views[i % gbuffer_views.len().max(1)];
        for (i, &set) in self.gather_sets.iter().enumerate() {
            let gb_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(gb_view(i))
                .sampler(self.sampler);
            let scene_view = hdr_resolve_views[i % hdr_resolve_views.len().max(1)];
            let scene_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(scene_view)
                .sampler(self.sampler);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&scene_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&gb_info)),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        let gi_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(self.gi.view)
            .sampler(self.sampler);
        for (i, &set) in self.composite_sets.iter().enumerate() {
            let gb_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(gb_view(i))
                .sampler(self.sampler);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&gi_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&gb_info)),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    // Re-point every gather + composite set's G-buffer binding at the unified
    // pre-pass per-frame normal+depth views. Called by the caller (init /
    // resize) when the unified G-buffer pre-pass is active. The scene input + gi
    // bindings are left untouched (already wired by `wire_sets`).
    pub(in crate::vulkan) fn wire_sets_gbuffer(
        &self,
        device: &Device,
        hdr_resolve_views: &[vk::ImageView],
        gbuffer_views: &[vk::ImageView],
    ) {
        self.wire_sets(device, hdr_resolve_views, gbuffer_views);
    }

    fn destroy_targets(&mut self, device: &Device) {
        unsafe {
            if self.gather_framebuffer != vk::Framebuffer::null() {
                device.destroy_framebuffer(self.gather_framebuffer, None);
                self.gather_framebuffer = vk::Framebuffer::null();
            }
            for &fb in &self.composite_framebuffers {
                device.destroy_framebuffer(fb, None);
            }
        }
        self.composite_framebuffers.clear();
        if self.gi.image != vk::Image::null() {
            self.gi.destroy(device);
            self.gi = GpuImage {
                image: vk::Image::null(),
                memory: vk::DeviceMemory::null(),
                view: vk::ImageView::null(),
                aux_views: Vec::new(),
            };
        }
    }

    // Rebuild the resolution-dependent targets at a new swapchain extent and
    // re-wire the descriptor sets. The caller has already idled the device and
    // rebuilt the SSR pre-pass, so `gbuffer_view` is current.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        hdr_resolve_views: &[vk::ImageView],
        gbuffer_views: &[vk::ImageView],
    ) -> Result<(), String> {
        self.destroy_targets(device);
        self.build_targets(
            instance,
            device,
            physical_device,
            width,
            height,
            hdr_resolve_views,
        )?;
        self.wire_sets(device, hdr_resolve_views, gbuffer_views);
        Ok(())
    }

    // Swap freshly-built pipelines into the live resources after a hot-reload.
    // The caller has already `device_wait_idle`'d.
    pub(in crate::vulkan) fn swap_pipelines(
        &mut self,
        device: &Device,
        rebuilt: RebuiltSsgiPipelines,
    ) {
        unsafe {
            device.destroy_pipeline(self.gather_pso, None);
            device.destroy_pipeline(self.composite_pso, None);
        }
        self.gather_pso = rebuilt.gather;
        self.composite_pso = rebuilt.composite;
    }

    // Destroy every SSGI resource. The caller has already idled the device.
    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        self.destroy_targets(device);
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.gather_pso, None);
            device.destroy_pipeline(self.composite_pso, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_render_pass(self.gather_render_pass, None);
            device.destroy_render_pass(self.composite_render_pass, None);
        }
    }
}

impl VkContext {
    // Encode the SSGI gather + composite. The gather marches hemisphere rays
    // over the SSR pre-pass G-buffer and writes the noisy indirect radiance into
    // the `gi` target; the composite depth-aware-blurs it and additively blends
    // it into this frame's HDR resolve. Runs on the hdr_resolve RMW chain after
    // the main pass; only dispatched when SSGI is on (and the SSR pre-pass
    // G-buffer therefore exists).
    //
    // Both sub-passes are single-draw fullscreen passes, so each runs through the
    // shared `gfx::fullscreen` driver; the render-pass bracket + viewport /
    // scissor live once in `VkContext::begin/end_fullscreen_pass`.
    pub(in crate::vulkan) fn encode_ssgi(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
    ) {
        // Resolve the pass's resources up front, before constructing either
        // encoder, so the driver never leaves a render pass half-open. SSGI
        // gathers against the SSR pre-pass G-buffer; if it is absent there is
        // nothing to gather against, so skip rather than read a stale set.
        let Some(ssgi) = &self.ssgi else { return };
        if self.ssr.is_none() {
            return;
        }
        // The view params are shared by both sub-passes; the positive-height
        // viewport (set by begin_fullscreen_pass) matches the negative-height
        // pre-pass G-buffer, the convention ssgi_view_pos / ssgi_project assume.
        let params = ssgi.settings.params(fov_y_radians, aspect);

        // Gather: hemisphere ray-march over the G-buffer -> gi target, at the
        // `gi_scale`-reduced gather extent.
        encode_fullscreen(
            &SsgiFullscreenPass {
                ctx: self,
                ssgi,
                render_pass: ssgi.gather_render_pass,
                framebuffer: ssgi.gather_framebuffer,
                extent: ssgi.gi_extent,
                pso: ssgi.gather_pso,
                set: ssgi.gather_sets[frame_idx],
                params: &params,
            },
            &cmd,
        );

        // Composite: depth-aware blur + upsample of gi, additively blended into
        // the scene at full render resolution.
        encode_fullscreen(
            &SsgiFullscreenPass {
                ctx: self,
                ssgi,
                render_pass: ssgi.composite_render_pass,
                framebuffer: ssgi.composite_framebuffers[frame_idx],
                extent: self.render_extent,
                pso: ssgi.composite_pso,
                set: ssgi.composite_sets[frame_idx],
                params: &params,
            },
            &cmd,
        );
    }
}

// Encoder for one SSGI fullscreen sub-pass (gather or composite): the render
// pass + framebuffer + pipeline + descriptor set that distinguish the two, plus
// the shared per-frame SsgiParams push. The render-pass bracket + viewport /
// scissor live in `VkContext::begin/end_fullscreen_pass` (post/fullscreen.rs);
// only the SSGI-specific bind + draw is here. Constructed + driven by
// `encode_ssgi` through `gfx::fullscreen::encode_fullscreen`.
struct SsgiFullscreenPass<'a> {
    ctx: &'a VkContext,
    ssgi: &'a SsgiResources,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    // Target extent: the reduced gi extent for the gather, the full render
    // extent for the composite.
    extent: vk::Extent2D,
    pso: vk::Pipeline,
    set: vk::DescriptorSet,
    params: &'a SsgiParams,
}

impl FullscreenPass for SsgiFullscreenPass<'_> {
    type Rec = vk::CommandBuffer;

    fn begin(&self, cmd: &Self::Rec) {
        self.ctx
            .begin_fullscreen_pass_sized(*cmd, self.render_pass, self.framebuffer, self.extent);
    }

    fn draw(&self, cmd: &Self::Rec) {
        let cmd = *cmd;
        let device = &self.ctx.device;
        let push = unsafe {
            std::slice::from_raw_parts(
                self.params as *const SsgiParams as *const u8,
                std::mem::size_of::<SsgiParams>(),
            )
        };
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pso);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.ssgi.pipeline_layout,
                0,
                std::slice::from_ref(&self.set),
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.ssgi.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                push,
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }

    fn end(&self, cmd: &Self::Rec) {
        self.ctx.end_fullscreen_pass(*cmd);
    }
}

#[cfg(test)]
mod tests {
    // The SSGI gather + composite + fullscreen GLSL all compile to SPIR-V. The
    // CPU<->GPU `SsgiParams` std140 layout is guarded by
    // `ssgi_params_layout_matches_shaders` in gfx::render_types.
    #[test]
    fn ssgi_shaders_compile() {
        super::compile_ssgi_shaders(false).expect("ssgi shaders compile");
    }
}
