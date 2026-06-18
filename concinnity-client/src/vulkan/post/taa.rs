// src/vulkan/post/taa.rs
//
// Temporal anti-aliasing resources for the Vulkan backend: the history-resolve
// render pass, pipeline, targets, and the per-frame `encode_taa` encoder. The
// per-pixel motion vector the resolve samples comes from the unified G-buffer
// pre-pass's velocity channel; the TAA sets are wired to its per-frame views
// by `init.rs`.
use ash::{Device, vk};

use crate::gfx::fullscreen::{FullscreenPass, encode_fullscreen};

use super::super::context::*;
use super::super::pipeline::*;
use super::super::resources::*;
use super::super::texture::*;

//  Temporal anti-aliasing

// Temporal anti-aliasing GPU resources. Built only when the world's
// `PostProcessConfig` set `taa: true`. The TAA resolve blends the HDR scene
// with a reprojected, neighbourhood-clipped history buffer; per-pixel motion
// comes from the unified pre-pass's velocity channel. Mirrors the Metal TAA
// path (`encode_taa` in metal/draw.rs).
//
// History feedback across `frames_in_flight` slots: `taa_out_images` holds
// one image per slot. Frame in slot `f` reads slot `(f + n - 1) % n` (the
// previous frame's output) as history and writes slot `f`. The TAA render
// pass's `EXTERNAL` subpass dependency orders this frame's history sample
// after the previous frame's write, and this frame's overwrite after the
// previous frame's read, so no manual cross-frame barrier is needed.
pub(in crate::vulkan) struct TaaResources {
    // Resolution-independent pass / pipeline / layout.
    pub(in crate::vulkan) taa_render_pass: vk::RenderPass,
    pub(in crate::vulkan) taa_pipeline: vk::Pipeline,
    pub(in crate::vulkan) taa_pipeline_layout: vk::PipelineLayout,
    pub(in crate::vulkan) taa_set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) descriptor_pool: vk::DescriptorPool,
    // Resolution-dependent targets (rebuilt on swapchain resize).
    pub(in crate::vulkan) taa_out_images: Vec<GpuImage>,
    pub(in crate::vulkan) taa_framebuffers: Vec<vk::Framebuffer>,
    pub(in crate::vulkan) taa_sets: Vec<vk::DescriptorSet>,
    // Drives the Halton jitter sequence; also gates history validity
    // (`taa_frame == 0` on the first frame and after a resize).
    pub(in crate::vulkan) taa_frame: u32,
}

// TAA resolve render pass: one `HDR_FORMAT` colour target. The fullscreen
// triangle overwrites every pixel, so the attachment is discarded on load and
// ends shader-readable for the bloom + composite passes. The `EXTERNAL`
// dependency orders the subpass after the main / velocity writes it samples
// *and* after the previous frame's read of the slot it is about to overwrite.
fn create_taa_render_pass(device: &Device) -> Result<vk::RenderPass, String> {
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
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ);
    let rp_info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dependency));
    unsafe { device.create_render_pass(&rp_info, None) }
        .map_err(|e| format!("TAA render pass: {e}"))
}

impl TaaResources {
    // Build every TAA resource. `hdr_resolve_images` feed the per-frame TAA
    // resolve sets as the initial scene input; the velocity binding is
    // re-pointed at the unified pre-pass per-frame views by the caller.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vulkan) fn new(
        instance: &ash::Instance,
        device: &Device,
        pd: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        frames: usize,
        extent: vk::Extent2D,
        hdr_resolve_images: &[GpuImage],
        sampler: vk::Sampler,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let taa_render_pass = create_taa_render_pass(device)?;

        // set 0 for the TAA resolve: scene / velocity / history samplers.
        let taa_set_layout = create_descriptor_set_layout(
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
                (
                    2,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
            ],
        )?;

        // TAA resolve layout: the sampler set + a 4-byte history-valid flag.
        let taa_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(4);
        let taa_layouts = [taa_set_layout];
        let taa_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&taa_layouts)
                    .push_constant_ranges(std::slice::from_ref(&taa_push)),
                None,
            )
        }
        .map_err(|e| format!("TAA pipeline layout: {e}"))?;

        // Pipeline.
        let (taa_vert, taa_frag) = compile_taa_shaders(hot_reload)?;
        let taa_pipeline = create_taa_pipeline(
            device,
            taa_render_pass,
            taa_pipeline_layout,
            &taa_vert,
            &taa_frag,
        )?;

        // Descriptor pool: one TAA set (3 samplers) per frame slot.
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(frames as u32 * 3)];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(frames as u32),
                None,
            )
        }
        .map_err(|e| format!("TAA descriptor pool: {e}"))?;

        let mut taa = TaaResources {
            taa_render_pass,
            taa_pipeline,
            taa_pipeline_layout,
            taa_set_layout,
            descriptor_pool,
            taa_out_images: Vec::new(),
            taa_framebuffers: Vec::new(),
            taa_sets: Vec::new(),
            taa_frame: 0,
        };
        taa.build_targets(instance, device, pd, command_pool, queue, extent, frames)?;
        taa.wire_sets(device, hdr_resolve_images, sampler);
        Ok(taa)
    }

    // (Re)build the resolution-dependent targets + framebuffers. Allocates the
    // descriptor sets from the (reset) pool; the caller then calls `wire_sets`.
    #[allow(clippy::too_many_arguments)]
    fn build_targets(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        pd: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        extent: vk::Extent2D,
        frames: usize,
    ) -> Result<(), String> {
        // The history ring is at least two images deep even with a single
        // frame in flight, so the TAA pass never samples the slot it is
        // writing (a same-image read+write is a validation error). With
        // `frames == 1` the second image is never written, so TAA degrades to
        // a pass-through; `frames >= 2` (the default) gets true accumulation.
        let n_out = frames.max(2);
        for _ in 0..n_out {
            self.taa_out_images.push(create_taa_history_image(
                instance,
                device,
                pd,
                command_pool,
                queue,
                extent.width,
                extent.height,
                HDR_FORMAT,
            )?);
        }
        for f in 0..frames {
            let taa_fb = unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.taa_render_pass)
                        .attachments(std::slice::from_ref(&self.taa_out_images[f].view))
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                    None,
                )
            }
            .map_err(|e| format!("TAA framebuffer: {e}"))?;
            self.taa_framebuffers.push(taa_fb);
        }
        let taa_layouts: Vec<_> = (0..frames).map(|_| self.taa_set_layout).collect();
        self.taa_sets = alloc_descriptor_sets(device, self.descriptor_pool, &taa_layouts)?;
        Ok(())
    }

    // Wire the TAA sets to the scene (HDR resolve), velocity, and history
    // (previous slot's TAA output) images. The velocity binding is re-pointed
    // at the unified pre-pass's per-frame views by `rewire_velocity` before the
    // first frame; the scene image stands in until then so the binding is a
    // valid `SHADER_READ_ONLY` image.
    fn wire_sets(&self, device: &Device, hdr_resolve_images: &[GpuImage], sampler: vk::Sampler) {
        for (f, scene_img) in hdr_resolve_images.iter().enumerate() {
            // History = the previous frame's TAA output. The ring may be
            // deeper than the frame count (see `build_targets`), so index it
            // by its own length.
            let n_out = self.taa_out_images.len();
            let prev = (f + n_out - 1) % n_out;
            let scene = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(scene_img.view)
                .sampler(sampler);
            // Placeholder until `rewire_velocity` points this at the unified
            // pre-pass's per-frame velocity view.
            let velocity = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(scene_img.view)
                .sampler(sampler);
            let history = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.taa_out_images[prev].view)
                .sampler(sampler);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.taa_sets[f])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&scene)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.taa_sets[f])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&velocity)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.taa_sets[f])
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&history)),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    // The TAA output image view for frame slot `f`: what the bloom prefilter
    // and composite pass sample instead of the raw HDR resolve image.
    pub(in crate::vulkan) fn output_view(&self, frame: usize) -> vk::ImageView {
        self.taa_out_images[frame].view
    }

    // Re-point binding 0 of every `taa_sets[f]` (the TAA resolve's scene
    // input) at a single shared scene view. Used when SSR is on: the SSR
    // resolve output replaces the per-frame HDR resolve as the TAA's input.
    pub(in crate::vulkan) fn rewire_scene(
        &self,
        device: &Device,
        scene_view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
        let scene = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(scene_view)
            .sampler(sampler);
        for &set in &self.taa_sets {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&scene));
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }
    }

    // Re-point binding 1 of every `taa_sets[f]` (the TAA resolve's velocity
    // input) at the unified G-buffer pre-pass's per-frame velocity views. Used
    // when the merged pre-pass is active: its velocity channel replaces TAA's
    // own velocity pre-pass output. The TAA resolve still runs; only its motion
    // input moves. Mirrors the SSR resolve / SSAO / SSGI re-points.
    pub(in crate::vulkan) fn rewire_velocity(
        &self,
        device: &Device,
        velocity_views: &[vk::ImageView],
        sampler: vk::Sampler,
    ) {
        for (f, &set) in self.taa_sets.iter().enumerate() {
            let velocity = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(velocity_views[f % velocity_views.len().max(1)])
                .sampler(sampler);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&velocity));
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }
    }

    // Destroy the resolution-dependent targets + framebuffers and reset the
    // descriptor pool. Called before `build_targets` on resize and from
    // `destroy` at teardown.
    fn destroy_targets(&mut self, device: &Device) {
        for &fb in &self.taa_framebuffers {
            unsafe { device.destroy_framebuffer(fb, None) };
        }
        for img in &self.taa_out_images {
            img.destroy(device);
        }
        self.taa_framebuffers.clear();
        self.taa_out_images.clear();
        unsafe {
            let _ = device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty());
        }
        self.taa_sets.clear();
    }

    // Rebuild the resolution-dependent targets at a new swapchain extent and
    // re-wire all descriptor sets. The caller (`rebuild_swapchain`) has
    // already idled the device.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        pd: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        extent: vk::Extent2D,
        frames: usize,
        hdr_resolve_images: &[GpuImage],
        sampler: vk::Sampler,
    ) -> Result<(), String> {
        self.destroy_targets(device);
        self.build_targets(instance, device, pd, command_pool, queue, extent, frames)?;
        self.wire_sets(device, hdr_resolve_images, sampler);
        // Stale history cannot be reprojected onto the new resolution.
        self.taa_frame = 0;
        Ok(())
    }

    // Destroy every TAA resource. The caller has already idled the device.
    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        self.destroy_targets(device);
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.taa_pipeline, None);
            device.destroy_pipeline_layout(self.taa_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.taa_set_layout, None);
            device.destroy_render_pass(self.taa_render_pass, None);
        }
    }
}

//  Per-frame encoders (moved from draw.rs)

// Push constant for the TAA resolve pass (4 bytes): history-valid flag.
#[derive(Copy, Clone)]
#[repr(C)]
struct TaaPush {
    history_valid: f32,
}

impl VkContext {
    // Encode the TAA resolve pass: one fullscreen-triangle draw blending the
    // HDR scene with the reprojected, neighbourhood-clipped history into the
    // per-frame TAA output image. Per-pixel motion comes from the unified
    // pre-pass's velocity channel. Runs before bloom. Only called when TAA is
    // on.
    pub(in crate::vulkan) fn encode_taa(&self, cmd: vk::CommandBuffer, frame_idx: usize) {
        let Some(taa) = &self.taa else { return };
        encode_fullscreen(
            &TaaResolvePass {
                ctx: self,
                taa,
                frame_idx,
            },
            &cmd,
        );
    }
}

// Encoder for the TAA resolve fullscreen pass: the resolved resources + the frame
// slot. The render-pass bracket + viewport / scissor live in
// `VkContext::begin/end_fullscreen_pass` (post/fullscreen.rs); only the
// TAA-specific bind + draw is here. Constructed + driven by `encode_taa` through
// `gfx::fullscreen::encode_fullscreen`.
struct TaaResolvePass<'a> {
    ctx: &'a VkContext,
    taa: &'a TaaResources,
    frame_idx: usize,
}

impl FullscreenPass for TaaResolvePass<'_> {
    type Rec = vk::CommandBuffer;

    fn begin(&self, cmd: &Self::Rec) {
        self.ctx.begin_fullscreen_pass(
            *cmd,
            self.taa.taa_render_pass,
            self.taa.taa_framebuffers[self.frame_idx],
        );
    }

    fn draw(&self, cmd: &Self::Rec) {
        let cmd = *cmd;
        let device = &self.ctx.device;
        // History is invalid on the first frame; the scene then passes straight
        // through.
        let push = TaaPush {
            history_valid: if self.taa.taa_frame > 0 { 1.0 } else { 0.0 },
        };
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.taa.taa_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.taa.taa_pipeline_layout,
                0,
                std::slice::from_ref(&self.taa.taa_sets[self.frame_idx]),
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.taa.taa_pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                std::slice::from_raw_parts(
                    &push as *const TaaPush as *const u8,
                    std::mem::size_of::<TaaPush>(),
                ),
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }

    fn end(&self, cmd: &Self::Rec) {
        self.ctx.end_fullscreen_pass(*cmd);
    }
}

//  TAA shaders (moved from pipeline.rs)

// TAA resolve pass. A fullscreen triangle (COMPOSITE_VERT_GLSL) blends the
// current HDR scene with a reprojected, neighbourhood-clipped history buffer.
// Per-pixel motion comes from the unified pre-pass's velocity channel. Mirrors
// `taa_fragment_main` in metal/pipeline.rs (YCoCg variance clip + non-finite
// sanitisation).
const TAA_FRAG_GLSL: &str = include_str!("../shaders/taa.frag");

// SPIR-V for the TAA resolve pass: the shared fullscreen-triangle vertex
// shader plus the history-blend fragment shader.
pub(in crate::vulkan) fn compile_taa_shaders(
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    use super::super::pipeline::shader_source;
    let vert = compile_glsl(
        &shader_source(hot_reload, "composite.vert", COMPOSITE_VERT_GLSL),
        shaderc::ShaderKind::Vertex,
        "taa_vert.glsl",
    )?;
    let frag = compile_glsl(
        &shader_source(hot_reload, "taa.frag", TAA_FRAG_GLSL),
        shaderc::ShaderKind::Fragment,
        "taa_frag.glsl",
    )?;
    Ok((vert, frag))
}

// Replacement TAA resolve pipeline built by the hot-reload pass.
pub(in crate::vulkan) struct RebuiltTaaPipelines {
    pub taa: vk::Pipeline,
}

// Rebuild the live TAA resolve pipeline from disk-resident GLSL source against
// the existing layout + render pass.
pub(in crate::vulkan) fn rebuild_taa_pipelines(
    device: &Device,
    taa: &TaaResources,
    hot_reload: bool,
) -> Result<RebuiltTaaPipelines, String> {
    let (taa_vs, taa_fs) = compile_taa_shaders(hot_reload)?;
    let taa_pipeline = create_taa_pipeline(
        device,
        taa.taa_render_pass,
        taa.taa_pipeline_layout,
        &taa_vs,
        &taa_fs,
    )?;
    Ok(RebuiltTaaPipelines { taa: taa_pipeline })
}

impl TaaResources {
    // Swap the freshly-built pipeline into the live resources. The caller
    // has already `device_wait_idle`'d so the old pipeline is not in
    // flight. Driven by the Vulkan shader hot-reload pass.
    pub(in crate::vulkan) fn swap_pipelines(
        &mut self,
        device: &Device,
        rebuilt: RebuiltTaaPipelines,
    ) {
        unsafe {
            device.destroy_pipeline(self.taa_pipeline, None);
        }
        self.taa_pipeline = rebuilt.taa;
    }
}

//  Pipeline builders (moved from pipeline.rs)

// Build the TAA resolve pipeline: a vertex-buffer-less fullscreen triangle
// that blends the HDR scene with the reprojected history into a single-sample
// `R16G16B16A16_SFLOAT` target. No depth; same shape as the composite pipeline.
fn create_taa_pipeline(
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

    let color_blend_attach = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false);
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
    .map_err(|(_, e)| format!("create TAA pipeline: {e}"))?[0];

    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

//  Target builders (moved from texture.rs)

// Create the TAA history / output image: a single-sample `HDR_FORMAT` colour
// image usable as both a render target and a sampled texture. Pre-transitioned
// to `SHADER_READ_ONLY_OPTIMAL` so the first frame's TAA resolve can bind a
// neighbouring slot as history before that slot has ever been rendered to.
#[allow(clippy::too_many_arguments)]
fn create_taa_history_image(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
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
    Ok(GpuImage {
        image,
        memory,
        view,
        aux_views: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    // The TAA resolve GLSL (the shared fullscreen-triangle vertex shader plus
    // the history-blend fragment shader) compiles to SPIR-V. Per-pixel motion
    // comes from the unified pre-pass's velocity channel, so this module no
    // longer owns a velocity pre-pass.
    #[test]
    fn taa_shaders_compile() {
        super::compile_taa_shaders(false).expect("taa shaders compile");
    }

    // TaaPush must match the `TaaBlock` push constant in taa.frag: a single
    // history-valid flag (4 bytes).
    #[test]
    fn taa_push_layout_matches_glsl() {
        assert_eq!(std::mem::size_of::<super::TaaPush>(), 4);
        assert_eq!(std::mem::offset_of!(super::TaaPush, history_valid), 0);
    }
}
