// src/vulkan/post/reflection_composite.rs
//
// Roughness-aware reflection composite for the Vulkan backend. The SSR and RT
// resolves now write reflected radiance (.rgb) + a Fresnel/gloss weight (.a) into
// their output target instead of compositing inline; this two-pass effect blurs
// that reflection by surface roughness and composites it over the base HDR scene
// into `output`, the scene-with-reflections the TAA / bloom / composite / glass
// passes consume. Mirrors src/metal/post/ssr.rs (the composite half) +
// src/directx/post/reflection_composite.rs.
//
//   pass 1 (blur, reduced resolution): weight-averages the reflection over a
//       roughness-scaled cone into `blur`. The expensive multi-tap part, run at a
//       fraction of the pixels.
//   pass 2 (composite, full resolution): lerps the sharp full-res reflection
//       against the upsampled half-res blur by roughness, then composites over the
//       scene into `output`.
//
// Both reflection paths feed one composite: `encode_ssr_resolve` /
// `encode_rt_reflections` each render their resolve target, then call
// `encode_reflection_composite` with that target's view; the composite's reflection
// binding is re-pointed to it per encode (the two paths are mutually exclusive).

use ash::{Device, vk};

use super::super::context::{HDR_FORMAT, VkContext};
use super::super::pipeline::*;
use super::super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::super::texture::*;

// GLSL sources. The fullscreen vertex shader is shared with the SSR resolve (same
// no-flip [0,1] UV convention, so the composite taps line up with the resolve's).
const COMPOSITE_VERT_GLSL: &str = include_str!("../shaders/ssr_fullscreen.vert");
const REFLECTION_BLUR_FRAG_GLSL: &str = include_str!("../shaders/reflection_blur.frag");
const REFLECTION_COMPOSITE_FRAG_GLSL: &str = include_str!("../shaders/reflection_composite.frag");

// Reflection-composite resources, held by `VkContext` when the SSR resolve or RT
// reflections are active (both feed this composite). All `vk::*` handles are owned
// here and freed on `destroy`.
pub(in crate::vulkan) struct ReflectionCompositeResources {
    // Composited scene (full render resolution): the scene image the post stack
    // consumes in place of the raw SSR / RT resolve output.
    pub(in crate::vulkan) output: GpuImage,
    output_framebuffer: vk::Framebuffer,

    // Reduced-resolution roughness blur of the reflection target (pass 1 writes it,
    // the composite upsamples it). Sized at render / `blur_scale`.
    blur: GpuImage,
    blur_framebuffer: vk::Framebuffer,
    blur_extent: vk::Extent2D,

    // One render pass (RGBA16F colour, DONT_CARE load, ends shader-readable) shared
    // by both passes; the framebuffer selects the target.
    render_pass: vk::RenderPass,

    // Pass 1 (blur) reads reflection + roughness; pass 2 (composite) reads those
    // plus scene + G-buffer normal+depth + blur.
    blur_set_layout: vk::DescriptorSetLayout,
    composite_set_layout: vk::DescriptorSetLayout,
    blur_pipeline_layout: vk::PipelineLayout,
    composite_pipeline_layout: vk::PipelineLayout,
    blur_pso: vk::Pipeline,
    composite_pso: vk::Pipeline,

    descriptor_pool: vk::DescriptorPool,
    // Per-frame sets. Binding 0 (the reflection target) is re-pointed each encode to
    // the resolve that just ran; the rest are wired at init / resize.
    blur_sets: Vec<vk::DescriptorSet>,
    composite_sets: Vec<vk::DescriptorSet>,

    sampler: vk::Sampler,

    // Per-axis divisor the blur target is sized by (from the world's
    // `reflection_blur_resolution`). Held so `rebuild` reuses the same scale.
    blur_scale: u32,
}

// Raw per-frame sets are render-thread-only; the struct lives inside `VkContext`,
// already `unsafe impl Send`.
unsafe impl Send for ReflectionCompositeResources {}

// SPIR-V blobs for the composite pipelines.
pub(in crate::vulkan) struct ReflectionCompositeShaders {
    pub vs: Vec<u8>,
    pub blur_fs: Vec<u8>,
    pub composite_fs: Vec<u8>,
}

// Compile the shared fullscreen vertex shader + the blur + composite fragments.
pub(in crate::vulkan) fn compile_reflection_composite_shaders(
    hot_reload: bool,
) -> Result<ReflectionCompositeShaders, String> {
    use super::super::pipeline::shader_source;
    Ok(ReflectionCompositeShaders {
        vs: compile_glsl(
            &shader_source(hot_reload, "ssr_fullscreen.vert", COMPOSITE_VERT_GLSL),
            shaderc::ShaderKind::Vertex,
            "ssr_fullscreen.vert",
        )?,
        blur_fs: compile_glsl(
            &shader_source(
                hot_reload,
                "reflection_blur.frag",
                REFLECTION_BLUR_FRAG_GLSL,
            ),
            shaderc::ShaderKind::Fragment,
            "reflection_blur.frag",
        )?,
        composite_fs: compile_glsl(
            &shader_source(
                hot_reload,
                "reflection_composite.frag",
                REFLECTION_COMPOSITE_FRAG_GLSL,
            ),
            shaderc::ShaderKind::Fragment,
            "reflection_composite.frag",
        )?,
    })
}

// Replacement composite pipelines from a shader hot-reload.
pub(in crate::vulkan) struct RebuiltReflectionComposite {
    pub blur: vk::Pipeline,
    pub composite: vk::Pipeline,
}

pub(in crate::vulkan) fn rebuild_reflection_composite_pipelines(
    device: &Device,
    rc: &ReflectionCompositeResources,
    hot_reload: bool,
) -> Result<RebuiltReflectionComposite, String> {
    let shaders = compile_reflection_composite_shaders(hot_reload)?;
    let blur = create_composite_pipeline(
        device,
        rc.render_pass,
        rc.blur_pipeline_layout,
        &shaders.vs,
        &shaders.blur_fs,
    )?;
    let composite = create_composite_pipeline(
        device,
        rc.render_pass,
        rc.composite_pipeline_layout,
        &shaders.vs,
        &shaders.composite_fs,
    )?;
    Ok(RebuiltReflectionComposite { blur, composite })
}

// Composite render pass: one HDR-format colour attachment, no depth. The fullscreen
// triangle overwrites every pixel so DONT_CARE is safe on load. Ends shader-readable
// for the next pass (composite -> bloom/TAA; blur -> composite). Mirrors the SSR
// resolve render pass.
fn create_composite_render_pass(device: &Device) -> Result<vk::RenderPass, String> {
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
    // Synchronise every prior colour write + shader read (the resolve output, the
    // pass-1 blur write, the scene + G-buffer) against this pass's reads + write.
    let dep = vk::SubpassDependency::default()
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
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ);
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dep));
    unsafe { device.create_render_pass(&info, None) }
        .map_err(|e| format!("reflection composite render pass: {e}"))
}

// One full-screen colour target pre-transitioned to SHADER_READ_ONLY_OPTIMAL so the
// descriptor sets bound to it at init see a valid layout before the first encode.
#[allow(clippy::too_many_arguments)]
fn create_target(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
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
        // TRANSFER_SRC so the transparent (glass) pass can snapshot the
        // post-reflection scene for its refraction tap, the same usage the SSR
        // output carried before the composite owned the scene image.
        vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC,
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
    let view = create_image_view(device, image, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
    Ok(GpuImage {
        image,
        memory,
        view,
        aux_views: Vec::new(),
    })
}

// Fullscreen pipeline: no vertex input, no depth, no blend; writes one HDR target.
// Mirrors the SSR resolve pipeline.
fn create_composite_pipeline(
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
        .color_write_mask(vk::ColorComponentFlags::RGBA)
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
    .map_err(|(_, e)| format!("create reflection composite pso: {e}"))?[0];
    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

impl ReflectionCompositeResources {
    // Build every composite resource. `hdr_resolve_views` (per-frame scene),
    // `normal_depth_views` + `roughness_views` (the unified G-buffer pre-pass's
    // per-frame views) feed the composite's static bindings; the reflection binding
    // is re-pointed per encode. `blur_scale` is the per-axis blur divisor.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vulkan) fn new(
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
        frames: usize,
        blur_scale: u32,
        hdr_resolve_views: &[vk::ImageView],
        normal_depth_views: &[vk::ImageView],
        roughness_views: &[vk::ImageView],
        hot_reload: bool,
    ) -> Result<Self, String> {
        let blur_scale = blur_scale.max(1);
        let render_pass = create_composite_render_pass(device)?;

        // Blur set 0: reflection + roughness. Composite set 0: reflection + scene +
        // gbuffer normal+depth + roughness + blur.
        let sampler_binding = |b: u32| {
            (
                b,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
            )
        };
        let blur_set_layout =
            create_descriptor_set_layout(device, &[sampler_binding(0), sampler_binding(1)])?;
        let composite_set_layout = create_descriptor_set_layout(
            device,
            &[
                sampler_binding(0),
                sampler_binding(1),
                sampler_binding(2),
                sampler_binding(3),
                sampler_binding(4),
            ],
        )?;

        let make_layout = |set_layout: vk::DescriptorSetLayout, name: &str| -> Result<_, String> {
            let layouts = [set_layout];
            unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                    None,
                )
            }
            .map_err(|e| format!("{name}: {e}"))
        };
        let blur_pipeline_layout = make_layout(blur_set_layout, "reflection blur layout")?;
        let composite_pipeline_layout =
            make_layout(composite_set_layout, "reflection composite layout")?;

        let shaders = compile_reflection_composite_shaders(hot_reload)?;
        let blur_pso = create_composite_pipeline(
            device,
            render_pass,
            blur_pipeline_layout,
            &shaders.vs,
            &shaders.blur_fs,
        )?;
        let composite_pso = create_composite_pipeline(
            device,
            render_pass,
            composite_pipeline_layout,
            &shaders.vs,
            &shaders.composite_fs,
        )?;

        // Pool: per-frame blur sets (2 samplers) + composite sets (5 samplers).
        let f = frames as u32;
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(f * 7)];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(f * 2),
                None,
            )
        }
        .map_err(|e| format!("reflection composite descriptor pool: {e}"))?;
        let blur_layouts: Vec<_> = (0..frames).map(|_| blur_set_layout).collect();
        let blur_sets = alloc_descriptor_sets(device, descriptor_pool, &blur_layouts)?;
        let composite_layouts: Vec<_> = (0..frames).map(|_| composite_set_layout).collect();
        let composite_sets = alloc_descriptor_sets(device, descriptor_pool, &composite_layouts)?;

        let sampler = create_sampler_linear_clamp(device)?;

        let mut me = Self {
            output: GpuImage::null(),
            output_framebuffer: vk::Framebuffer::null(),
            blur: GpuImage::null(),
            blur_framebuffer: vk::Framebuffer::null(),
            blur_extent: vk::Extent2D::default(),
            render_pass,
            blur_set_layout,
            composite_set_layout,
            blur_pipeline_layout,
            composite_pipeline_layout,
            blur_pso,
            composite_pso,
            descriptor_pool,
            blur_sets,
            composite_sets,
            sampler,
            blur_scale,
        };
        me.build_targets(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            width,
            height,
        )?;
        me.wire_sets(
            device,
            hdr_resolve_views,
            normal_depth_views,
            roughness_views,
        );
        Ok(me)
    }

    // Allocate / re-allocate the resolution-dependent output + blur targets and
    // their framebuffers at the given extent.
    #[allow(clippy::too_many_arguments)]
    fn build_targets(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let w = width.max(1);
        let h = height.max(1);
        let bw = (w / self.blur_scale).max(1);
        let bh = (h / self.blur_scale).max(1);
        self.output = create_target(instance, device, physical_device, command_pool, queue, w, h)?;
        self.blur = create_target(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            bw,
            bh,
        )?;
        self.blur_extent = vk::Extent2D {
            width: bw,
            height: bh,
        };

        let make_fb = |view: vk::ImageView, fw: u32, fh: u32| -> Result<vk::Framebuffer, String> {
            unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.render_pass)
                        .attachments(std::slice::from_ref(&view))
                        .width(fw)
                        .height(fh)
                        .layers(1),
                    None,
                )
            }
            .map_err(|e| format!("reflection composite framebuffer: {e}"))
        };
        self.output_framebuffer = make_fb(self.output.view, w, h)?;
        self.blur_framebuffer = make_fb(self.blur.view, bw, bh)?;
        Ok(())
    }

    // Wire the per-frame static bindings: blur set binding 1 = roughness; composite
    // set bindings 1..4 = scene / normal+depth / roughness / blur. Binding 0 (the
    // reflection target) is left at a valid placeholder and re-pointed per encode.
    // Single-entry G-buffer slices are shared across frames (the legacy pre-pass
    // produced one view); per-frame slices index by frame.
    fn wire_sets(
        &self,
        device: &Device,
        hdr_resolve_views: &[vk::ImageView],
        normal_depth_views: &[vk::ImageView],
        roughness_views: &[vk::ImageView],
    ) {
        let img = |view: vk::ImageView| {
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view)
                .sampler(self.sampler)
        };
        let write = |set: vk::DescriptorSet, binding: u32, info: &vk::DescriptorImageInfo| {
            let w = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(info));
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&w), &[]) };
        };
        let pick = |views: &[vk::ImageView], i: usize| views[i % views.len().max(1)];
        let blur_info = img(self.blur.view);
        for i in 0..self.blur_sets.len() {
            let rough = img(pick(roughness_views, i));
            let placeholder = img(pick(hdr_resolve_views, i));
            // Blur set: 0 = reflection placeholder, 1 = roughness.
            write(self.blur_sets[i], 0, &placeholder);
            write(self.blur_sets[i], 1, &rough);
            // Composite set: 0 = reflection placeholder, 1 = scene, 2 = normal+depth,
            // 3 = roughness, 4 = blur.
            let scene = img(pick(hdr_resolve_views, i));
            let nd = img(pick(normal_depth_views, i));
            write(self.composite_sets[i], 0, &placeholder);
            write(self.composite_sets[i], 1, &scene);
            write(self.composite_sets[i], 2, &nd);
            write(self.composite_sets[i], 3, &rough);
            write(self.composite_sets[i], 4, &blur_info);
        }
    }

    fn destroy_targets(&mut self, device: &Device) {
        unsafe {
            if self.output_framebuffer != vk::Framebuffer::null() {
                device.destroy_framebuffer(self.output_framebuffer, None);
                self.output_framebuffer = vk::Framebuffer::null();
            }
            if self.blur_framebuffer != vk::Framebuffer::null() {
                device.destroy_framebuffer(self.blur_framebuffer, None);
                self.blur_framebuffer = vk::Framebuffer::null();
            }
        }
        if self.output.image != vk::Image::null() {
            self.output.destroy(device);
            self.output = GpuImage::null();
        }
        if self.blur.image != vk::Image::null() {
            self.blur.destroy(device);
            self.blur = GpuImage::null();
        }
    }

    // Rebuild the resolution-dependent targets at a new extent and re-wire the
    // per-frame static bindings (the scene / G-buffer / blur views all moved). The
    // caller has already idled the device.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
        hdr_resolve_views: &[vk::ImageView],
        normal_depth_views: &[vk::ImageView],
        roughness_views: &[vk::ImageView],
    ) -> Result<(), String> {
        self.destroy_targets(device);
        self.build_targets(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            width,
            height,
        )?;
        self.wire_sets(
            device,
            hdr_resolve_views,
            normal_depth_views,
            roughness_views,
        );
        Ok(())
    }

    // Swap freshly-built pipelines into the live resources after a hot-reload.
    pub(in crate::vulkan) fn swap_pipelines(
        &mut self,
        device: &Device,
        rebuilt: RebuiltReflectionComposite,
    ) {
        unsafe {
            device.destroy_pipeline(self.blur_pso, None);
            device.destroy_pipeline(self.composite_pso, None);
        }
        self.blur_pso = rebuilt.blur;
        self.composite_pso = rebuilt.composite;
    }

    // Destroy every composite resource. The caller has already idled the device.
    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        self.destroy_targets(device);
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.blur_pso, None);
            device.destroy_pipeline(self.composite_pso, None);
            device.destroy_pipeline_layout(self.blur_pipeline_layout, None);
            device.destroy_pipeline_layout(self.composite_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.blur_set_layout, None);
            device.destroy_descriptor_set_layout(self.composite_set_layout, None);
            device.destroy_render_pass(self.render_pass, None);
        }
    }
}

impl VkContext {
    // Blur the reflection target by surface roughness and composite it over the base
    // HDR scene into `reflection_composite.output`. `reflection_view` is the resolve
    // target the SSR / RT pass just wrote (radiance + weight), in
    // SHADER_READ_ONLY_OPTIMAL after its render pass. Encoded inline at the tail of
    // `encode_ssr_resolve` / `encode_rt_reflections`. No-op when the composite is
    // absent (no reflection path active).
    pub(in crate::vulkan) fn encode_reflection_composite(
        &self,
        cmd: vk::CommandBuffer,
        reflection_view: vk::ImageView,
        frame_idx: usize,
    ) {
        let Some(rc) = &self.reflection_composite else {
            return;
        };
        let device = &self.device;

        // Re-point binding 0 (reflection) of this frame's blur + composite sets at
        // the resolve that just ran. This frame's sets are fence-gated (the previous
        // submission for this slot completed at the top of the frame), so the write
        // is safe; the SSR and RT paths are mutually exclusive, so only one feeds a
        // given frame.
        let refl = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(reflection_view)
            .sampler(rc.sampler);
        let repoint = [
            vk::WriteDescriptorSet::default()
                .dst_set(rc.blur_sets[frame_idx])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&refl)),
            vk::WriteDescriptorSet::default()
                .dst_set(rc.composite_sets[frame_idx])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&refl)),
        ];
        unsafe { device.update_descriptor_sets(&repoint, &[]) };

        // Pass 1: roughness blur into the reduced-resolution blur target.
        self.begin_fullscreen_pass_sized(cmd, rc.render_pass, rc.blur_framebuffer, rc.blur_extent);
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, rc.blur_pso);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                rc.blur_pipeline_layout,
                0,
                std::slice::from_ref(&rc.blur_sets[frame_idx]),
                &[],
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
        self.end_fullscreen_pass(cmd);

        // Pass 2: lerp sharp vs upsampled blur by roughness, composite over scene.
        self.begin_fullscreen_pass(cmd, rc.render_pass, rc.output_framebuffer);
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, rc.composite_pso);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                rc.composite_pipeline_layout,
                0,
                std::slice::from_ref(&rc.composite_sets[frame_idx]),
                &[],
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
        self.end_fullscreen_pass(cmd);
    }
}

#[cfg(test)]
mod tests {
    // The composite vert + blur + composite fragments compile to SPIR-V. Guards the
    // GLSL so a shader error fails a test instead of only an init failure on the GPU
    // host. The composite passes carry no push constant / UBO, so there is no
    // CPU<->GPU layout to assert.
    #[test]
    fn reflection_composite_shaders_compile() {
        let shaders = super::compile_reflection_composite_shaders(false)
            .expect("reflection composite shaders compile");
        assert!(super::is_spirv(&shaders.vs));
        assert!(super::is_spirv(&shaders.blur_fs));
        assert!(super::is_spirv(&shaders.composite_fs));
    }
}
