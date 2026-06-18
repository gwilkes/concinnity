// src/vulkan/post/gbuffer.rs
//
// Unified geometry G-buffer pre-pass for the Vulkan backend. One jittered
// traversal of the visible set (static + instanced + skinned) rasterises into a
// single MRT:
//
//   target 0  RGBA16F  view-space normal (rgb) + positive linear view depth (a)
//   target 1  R8       perceptual roughness
//   target 2  RG16F    screen-space motion (prev_uv - cur_uv)
//
// plus a private single-sample depth buffer. Every screen-space consumer (SSR
// resolve, SSAO, SSGI, TAA, FSR) reads this one output instead of
// re-rasterising, replacing the separate SSR pre-pass + SSAO pre-pass +
// velocity pre-pass. Rasterisation uses the jittered VP (matching the main pass
// coverage); the motion vector derives from the un-jittered current / previous
// VPs in-shader so projection jitter never contaminates motion. Fuses the
// former SSR depth+normal pre-pass and TAA velocity pre-pass into one node;
// mirrors src/directx/post/gbuffer.rs.
//
// Unlike DirectX's single-resource G-buffer, the Vulkan unified buffer holds a
// per-frame `Vec<GpuImage>` for every MRT target (and per-frame framebuffers),
// because TAA reads `velocity_images[frame_idx]` and the engine pipelines
// frames-in-flight deep; this follows the per-frame `Vec` shape of taa.rs.

use ash::{Device, vk};

use super::super::context::VkContext;
use super::super::math::IDENTITY4;
use super::super::pipeline::*;
use super::super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::super::texture::*;

// GLSL sources
const GBUFFER_PREPASS_VERT_GLSL: &str = include_str!("../shaders/gbuffer_prepass.vert");
const GBUFFER_PREPASS_INSTANCED_VERT_GLSL: &str =
    include_str!("../shaders/gbuffer_prepass_instanced.vert");
const GBUFFER_PREPASS_SKINNED_VERT_GLSL: &str =
    include_str!("../shaders/gbuffer_prepass_skinned.vert");
const GBUFFER_PREPASS_FRAG_GLSL: &str = include_str!("../shaders/gbuffer_prepass.frag");

// GPU-driven (bindless) G-buffer pre-pass shaders. The VS reads model +
// roughness from the per-frame GpuObjectData SSBO by gl_InstanceIndex and the
// previous-frame model from a parallel SSBO; the FS mirrors gbuffer_prepass.frag
// but sources roughness from a flat VS varying. Drive the same MRT, reusing the
// main pass's GPU-culled indirect buffer.
const GBUFFER_BINDLESS_VERT_GLSL: &str = include_str!("../shaders/gbuffer_bindless.vert");
const GBUFFER_BINDLESS_FRAG_GLSL: &str = include_str!("../shaders/gbuffer_bindless.frag");

// Normal+depth target: rgb = unit view-space normal, a = positive linear view
// depth (-view_z). Alpha 0 (cleared background) marks "no geometry". Matches
// the SSR G-buffer so the resolve maths is byte-identical.
pub(in crate::vulkan) const GBUFFER_NORMAL_DEPTH_FORMAT: vk::Format =
    vk::Format::R16G16B16A16_SFLOAT;

// Single-channel perceptual roughness. 1.0 (cleared background) = no reflection;
// 0.0 = mirror.
pub(in crate::vulkan) const GBUFFER_ROUGHNESS_FORMAT: vk::Format = vk::Format::R8_UNORM;

// Screen-space motion (prev_uv - cur_uv). Cleared to 0 (no motion).
pub(in crate::vulkan) const GBUFFER_VELOCITY_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;

// Size of the per-frame view UBO: jittered_vp + cur_vp + prev_vp + view_mat
// (four std140 mat4 = 256 B). Matches the `GbView` UBO in every pre-pass VS.
pub(in crate::vulkan) const GBUFFER_VIEW_UBO_SIZE: vk::DeviceSize = 256;

// Size of the prepass push-constant block: cur_model (64) + prev_model (64) +
// roughness (4) + 12 B pad so the block is 16-byte aligned. Both stages see the
// full block; the vertex shaders reference cur/prev model, the fragment shader
// only roughness.
const GBUFFER_PREPASS_PUSH_BYTES: u32 = 144;

// std140 view block uploaded to the G-buffer pre-pass vertex shaders. Matches
// the `GbView` UBO (set 0, binding 0): the jittered VP rasterises, the
// un-jittered cur/prev VPs drive the motion vector, the view matrix transforms
// the normal + depth.
#[derive(Copy, Clone)]
#[repr(C)]
struct GbViewUniforms {
    jittered_vp: [[f32; 4]; 4],
    cur_vp: [[f32; 4]; 4],
    prev_vp: [[f32; 4]; 4],
    view_mat: [[f32; 4]; 4],
}

// Push constant the prepass pipelines see. Layout-matched to the shared GLSL
// `PushBlock` (`mat4 cur_model; mat4 prev_model; float roughness;`) plus
// trailing pad. The motion vector reads cur/prev model; the fragment reads
// roughness.
#[derive(Copy, Clone)]
#[repr(C)]
struct GbModelPush {
    cur_model: [[f32; 4]; 4],
    prev_model: [[f32; 4]; 4],
    roughness: f32,
    _pad: [f32; 3],
}

// SPIR-V blobs for every G-buffer pre-pass pipeline. Produced by
// [`compile_gbuffer_shaders`]; consumed by `GbufferResources::new` at init and
// by the hot-reload pass. Mirrors the matching SSR struct.
pub(in crate::vulkan) struct GbufferShaders {
    pub prepass_vs: Vec<u8>,
    pub prepass_instanced_vs: Vec<u8>,
    pub prepass_skinned_vs: Vec<u8>,
    pub prepass_fs: Vec<u8>,
}

// Compile every G-buffer pre-pass GLSL source. `hot_reload` routes each source
// resolve through [`crate::vulkan::pipeline::shader_source`].
pub(in crate::vulkan) fn compile_gbuffer_shaders(
    hot_reload: bool,
) -> Result<GbufferShaders, String> {
    use super::super::pipeline::shader_source;
    Ok(GbufferShaders {
        prepass_vs: compile_glsl(
            &shader_source(
                hot_reload,
                "gbuffer_prepass.vert",
                GBUFFER_PREPASS_VERT_GLSL,
            ),
            shaderc::ShaderKind::Vertex,
            "gbuffer_prepass.vert",
        )?,
        prepass_instanced_vs: compile_glsl(
            &shader_source(
                hot_reload,
                "gbuffer_prepass_instanced.vert",
                GBUFFER_PREPASS_INSTANCED_VERT_GLSL,
            ),
            shaderc::ShaderKind::Vertex,
            "gbuffer_prepass_instanced.vert",
        )?,
        prepass_skinned_vs: compile_glsl(
            &shader_source(
                hot_reload,
                "gbuffer_prepass_skinned.vert",
                GBUFFER_PREPASS_SKINNED_VERT_GLSL,
            ),
            shaderc::ShaderKind::Vertex,
            "gbuffer_prepass_skinned.vert",
        )?,
        prepass_fs: compile_glsl(
            &shader_source(
                hot_reload,
                "gbuffer_prepass.frag",
                GBUFFER_PREPASS_FRAG_GLSL,
            ),
            shaderc::ShaderKind::Fragment,
            "gbuffer_prepass.frag",
        )?,
    })
}

// Pre-pass render pass: an RGBA16F normal+depth target, an R8 roughness target,
// and an RG16F velocity target, plus a private depth buffer. All colour
// attachments clear and end shader-readable so the consumers can sample them
// without an extra barrier. The depth is STORE'd because the temporal upscaler
// (FSR) consumes this render-resolution single-sample depth alongside the
// motion vectors.
fn create_prepass_render_pass(device: &Device) -> Result<vk::RenderPass, String> {
    let attachments = [
        vk::AttachmentDescription::default()
            .format(GBUFFER_NORMAL_DEPTH_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(GBUFFER_ROUGHNESS_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(GBUFFER_VELOCITY_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];
    let color_refs = [
        vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
    ];
    let depth_ref = vk::AttachmentReference::default()
        .attachment(3)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs)
        .depth_stencil_attachment(&depth_ref);
    let dep = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::SHADER_READ)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dep));
    unsafe { device.create_render_pass(&info, None) }
        .map_err(|e| format!("gbuffer prepass render pass: {e}"))
}

// Allocate a single-format colour render target usable as both attachment and
// sampled texture. No pre-transition: the render pass declares an `UNDEFINED`
// initial layout.
fn create_color_target(
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
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
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

// Build a pre-pass pipeline. Three MRT colour targets (normal+depth, roughness,
// velocity) over a private depth buffer; same no-cull / LESS depth as the main
// pass.
#[allow(clippy::too_many_arguments)]
fn create_prepass_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
    bindings: &[vk::VertexInputBindingDescription],
    attrs: &[vk::VertexInputAttributeDescription],
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
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(bindings)
        .vertex_attribute_descriptions(attrs);
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
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);
    // All three attachments must be byte-identical without `independentBlend`
    // enabled at device creation. The R8 roughness target stores only R, so a
    // uniform RGBA write-mask is the smallest-diff way to satisfy the spec.
    let blend_attaches = [
        vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false),
        vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false),
        vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false),
    ];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attaches);
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
    .map_err(|(_, e)| format!("create gbuffer prepass pso: {e}"))?[0];
    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

// Vertex input for the static / instanced G-buffer pre-pass over the 56-byte
// `Vertex`. Declares only the attributes the pre-pass vertex shaders consume:
// position (0), normal (1), and the skybox-sentinel colour (3); the instanced
// variant slices `[..2]` (position + normal, model comes from the instance
// SSBO). Stride stays the full 56 bytes; tangent (2) + uv (4) are not fetched.
fn vertex_56_input() -> (
    [vk::VertexInputBindingDescription; 1],
    [vk::VertexInputAttributeDescription; 3],
) {
    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(56)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(36),
    ];
    ([binding], attrs)
}

// Vertex input for the skinned G-buffer pre-pass over the 80-byte
// `SkinnedVertex`. `gbuffer_prepass_skinned.vert` consumes position (0), normal
// (1), and the skinning joints (5) + weights (6); tangent (2), colour (3), and
// uv (4) are omitted so the pipeline matches the shader interface. Stride stays
// 80.
fn skinned_vertex_input() -> (
    [vk::VertexInputBindingDescription; 1],
    [vk::VertexInputAttributeDescription; 4],
) {
    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(80)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(5)
            .format(vk::Format::R16G16B16A16_UINT)
            .offset(56),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(6)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(64),
    ];
    ([binding], attrs)
}

// Vertex input for the GPU-driven (bindless) G-buffer pre-pass: the current
// attributes the VS reads (position 0, normal 1, skybox-sentinel colour 3) on
// binding 0, plus the previous-frame position (location 5) on binding 1. Both
// bindings carry the 56-byte `Vertex`; the static prefix binds the static VB to
// both (prev_pos == cur_pos), the skinned tail binds the current deformed buffer
// to binding 0 and the previous-frame deformed buffer to binding 1. Tangent + UV
// are unused (the pre-pass samples no textures).
fn vertex_56_dual_input() -> (
    [vk::VertexInputBindingDescription; 2],
    [vk::VertexInputAttributeDescription; 4],
) {
    let bindings = [
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(56)
            .input_rate(vk::VertexInputRate::VERTEX),
        vk::VertexInputBindingDescription::default()
            .binding(1)
            .stride(56)
            .input_rate(vk::VertexInputRate::VERTEX),
    ];
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(36),
        vk::VertexInputAttributeDescription::default()
            .binding(1)
            .location(5)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
    ];
    (bindings, attrs)
}

// GPU-driven G-buffer pre-pass resources, built when the bindless cull path is
// active AND the G-buffer is enabled. Stored on `VkCull`. The pipeline reuses the
// G-buffer render pass; the per-frame `prev_model` SSBOs supply the velocity
// history (instance region init-written, static + skinned rewritten each frame);
// the per-frame set 0 binds the G-buffer view UBO + that frame's prev_model SSBO,
// and set 1 reuses the bindless GpuObjectData set.
pub(in crate::vulkan) struct GbufferBindless {
    pub(in crate::vulkan) pipeline: vk::Pipeline,
    pub(in crate::vulkan) pipeline_layout: vk::PipelineLayout,
    pub(in crate::vulkan) set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) sets: Vec<vk::DescriptorSet>,
    pub(in crate::vulkan) prev_model_buffers: Vec<vk::Buffer>,
    pub(in crate::vulkan) prev_model_memories: Vec<vk::DeviceMemory>,
    pub(in crate::vulkan) prev_model_ptrs: Vec<*mut u8>,
}

// Build the GPU-driven G-buffer pre-pass pipeline + its per-frame previous-frame
// model SSBOs + descriptor sets. The previous-frame model buffers' instance
// region `[n_objects, n_objects + n_instances)` is written once here (immutable,
// camera-only motion); the static + skinned regions are rewritten each frame by
// `build_gbuffer_prev_models`. Set 0 = G-buffer view UBO + prev_model SSBO; set 1
// = the shared bindless GpuObjectData set (object id via gl_InstanceIndex).
#[allow(clippy::too_many_arguments)]
pub(in crate::vulkan) fn build_gbuffer_bindless(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    descriptor_pool: vk::DescriptorPool,
    bindless_set_layout: vk::DescriptorSetLayout,
    gb: &GbufferResources,
    instance_models: &[[[f32; 4]; 4]],
    n_objects: usize,
    n_cull: usize,
    frames: usize,
    hot_reload: bool,
) -> Result<GbufferBindless, String> {
    use super::super::pipeline::shader_source;

    let vs = compile_glsl(
        &shader_source(
            hot_reload,
            "gbuffer_bindless.vert",
            GBUFFER_BINDLESS_VERT_GLSL,
        ),
        shaderc::ShaderKind::Vertex,
        "gbuffer_bindless.vert",
    )?;
    let fs = compile_glsl(
        &shader_source(
            hot_reload,
            "gbuffer_bindless.frag",
            GBUFFER_BINDLESS_FRAG_GLSL,
        ),
        shaderc::ShaderKind::Fragment,
        "gbuffer_bindless.frag",
    )?;

    // Set 0: GbView UBO (binding 0) + prev_model SSBO (binding 1), both VERTEX.
    let set_layout = create_descriptor_set_layout(
        device,
        &[
            (
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::VERTEX,
            ),
            (
                1,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::VERTEX,
            ),
        ],
    )?;
    let layouts = [set_layout, bindless_set_layout];
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
            None,
        )
    }
    .map_err(|e| format!("gbuffer bindless pipeline layout: {e}"))?;

    let (bindings, attrs) = vertex_56_dual_input();
    let pipeline = create_prepass_pipeline(
        device,
        gb.prepass_render_pass,
        pipeline_layout,
        &vs,
        &fs,
        &bindings,
        &attrs,
    )?;

    // Per-frame prev_model SSBOs (host-visible, persistently mapped), sized for
    // `n_cull` column-major `float4x4` records, parallel to the object buffer.
    let buf_size = (n_cull * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
    let mut prev_model_buffers = Vec::with_capacity(frames);
    let mut prev_model_memories = Vec::with_capacity(frames);
    let mut prev_model_ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
    for _ in 0..frames {
        let (buf, mem) = create_buffer(
            instance,
            device,
            physical_device,
            buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let ptr = unsafe { device.map_memory(mem, 0, buf_size, vk::MemoryMapFlags::empty()) }
            .map_err(|e| format!("map prev_model buffer: {e}"))? as *mut u8;
        // Instance region: the instances' current models (immutable, camera-only
        // motion). Written once into every frame buffer after the static prefix;
        // the per-frame fill rewrites only the static + skinned regions.
        if !instance_models.is_empty() {
            let stride = std::mem::size_of::<[[f32; 4]; 4]>();
            // SAFETY: the buffer holds `n_cull >= n_objects + instance_models.len()`
            // records, so writing past the `n_objects` offset stays in bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    instance_models.as_ptr() as *const u8,
                    ptr.add(n_objects * stride),
                    std::mem::size_of_val(instance_models),
                );
            }
        }
        prev_model_buffers.push(buf);
        prev_model_memories.push(mem);
        prev_model_ptrs.push(ptr);
    }

    // One set 0 per frame: binding 0 = that frame's GbView UBO, binding 1 = that
    // frame's prev_model SSBO. Both buffers are stable for the world's lifetime,
    // so the sets are written once here.
    let set_layouts: Vec<_> = (0..frames).map(|_| set_layout).collect();
    let sets = alloc_descriptor_sets(device, descriptor_pool, &set_layouts)?;
    for (f, &set) in sets.iter().enumerate() {
        let view_info = vk::DescriptorBufferInfo::default()
            .buffer(gb.view_ubo_buffers[f])
            .offset(0)
            .range(GBUFFER_VIEW_UBO_SIZE);
        let pm_info = vk::DescriptorBufferInfo::default()
            .buffer(prev_model_buffers[f])
            .offset(0)
            .range(buf_size);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&view_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&pm_info)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    Ok(GbufferBindless {
        pipeline,
        pipeline_layout,
        set_layout,
        sets,
        prev_model_buffers,
        prev_model_memories,
        prev_model_ptrs,
    })
}

// Unified G-buffer pre-pass resources held by `VkContext` when any screen-space
// consumer is enabled. All `vk::*` handles are owned by this struct and freed
// on `destroy`. Holds per-frame MRT images / framebuffers because the velocity
// target is read per-frame-in-flight by TAA, mirroring taa.rs's `Vec` shape.
pub(in crate::vulkan) struct GbufferResources {
    // Render pass.
    pub(in crate::vulkan) prepass_render_pass: vk::RenderPass,

    // Pre-pass pipelines (static always, instanced / skinned conditional).
    pub(in crate::vulkan) prepass_set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) prepass_layout_static: vk::PipelineLayout,
    pub(in crate::vulkan) prepass_layout_instanced: Option<vk::PipelineLayout>,
    pub(in crate::vulkan) prepass_layout_skinned: Option<vk::PipelineLayout>,
    pub(in crate::vulkan) prepass_pso_static: vk::Pipeline,
    pub(in crate::vulkan) prepass_pso_instanced: Option<vk::Pipeline>,
    pub(in crate::vulkan) prepass_pso_skinned: Option<vk::Pipeline>,

    // Per-frame view UBO (jittered_vp + cur_vp + prev_vp + view_mat),
    // host-mapped + descriptor set.
    pub(in crate::vulkan) view_ubo_buffers: Vec<vk::Buffer>,
    pub(in crate::vulkan) view_ubo_memories: Vec<vk::DeviceMemory>,
    pub(in crate::vulkan) view_ubo_ptrs: Vec<*mut u8>,
    pub(in crate::vulkan) prepass_sets: Vec<vk::DescriptorSet>,
    pub(in crate::vulkan) descriptor_pool: vk::DescriptorPool,

    // Per-frame MRT targets + private depth + framebuffers (rebuilt on resize).
    // One slot per frame in flight: TAA reads `velocity_images[frame_idx]`.
    pub(in crate::vulkan) normal_depth_images: Vec<GpuImage>,
    pub(in crate::vulkan) roughness_images: Vec<GpuImage>,
    pub(in crate::vulkan) velocity_images: Vec<GpuImage>,
    pub(in crate::vulkan) depth_images: Vec<GpuImage>,
    pub(in crate::vulkan) framebuffers: Vec<vk::Framebuffer>,

    // Previous-frame motion state, owned here so the velocity channel works for
    // any consumer (TAA or FSR) independent of whether engine-TAA is on.
    // `prev_view_proj` is last frame's un-jittered VP; `prev_models` is each
    // draw's previous transform. Both advance once per frame.
    pub(in crate::vulkan) prev_view_proj: [[f32; 4]; 4],
    pub(in crate::vulkan) prev_models: Vec<[[f32; 4]; 4]>,

    // True only under `cn debug`. Stored so the lazy
    // `ensure_skinned_gbuffer_pso` path and the shader hot-reload pass read
    // every GLSL source through the disk-first helper. Mirrors
    // `SsrResources::hot_reload`.
    pub(in crate::vulkan) hot_reload: bool,
}

// Replacement G-buffer pre-pass pipelines built by the hot-reload pass.
// Conditional variants are `Some` exactly when the corresponding
// `prepass_pso_*` is `Some` on the live resource.
pub(in crate::vulkan) struct RebuiltGbufferPipelines {
    pub prepass_static: vk::Pipeline,
    pub prepass_instanced: Option<vk::Pipeline>,
    pub prepass_skinned: Option<vk::Pipeline>,
}

// Rebuild every live G-buffer pre-pass pipeline from disk-resident GLSL source
// against the existing layouts + render pass. Same shape as
// [`rebuild_ssr_pipelines`].
pub(in crate::vulkan) fn rebuild_gbuffer_pipelines(
    device: &Device,
    gbuffer: &GbufferResources,
    hot_reload: bool,
) -> Result<RebuiltGbufferPipelines, String> {
    let shaders = compile_gbuffer_shaders(hot_reload)?;
    let (vbindings, vattrs) = vertex_56_input();
    let prepass_static = create_prepass_pipeline(
        device,
        gbuffer.prepass_render_pass,
        gbuffer.prepass_layout_static,
        &shaders.prepass_vs,
        &shaders.prepass_fs,
        &vbindings,
        &vattrs,
    )?;
    let prepass_instanced = if let (Some(layout), Some(_)) = (
        gbuffer.prepass_layout_instanced,
        gbuffer.prepass_pso_instanced,
    ) {
        // Instanced pre-pass reads only position + normal (model comes from the
        // instance SSBO), so bind just those two attributes.
        Some(create_prepass_pipeline(
            device,
            gbuffer.prepass_render_pass,
            layout,
            &shaders.prepass_instanced_vs,
            &shaders.prepass_fs,
            &vbindings,
            &vattrs[..2],
        )?)
    } else {
        None
    };
    let prepass_skinned = if let (Some(layout), Some(_)) =
        (gbuffer.prepass_layout_skinned, gbuffer.prepass_pso_skinned)
    {
        let (sbindings, sattrs) = skinned_vertex_input();
        Some(create_prepass_pipeline(
            device,
            gbuffer.prepass_render_pass,
            layout,
            &shaders.prepass_skinned_vs,
            &shaders.prepass_fs,
            &sbindings,
            &sattrs,
        )?)
    } else {
        None
    };
    Ok(RebuiltGbufferPipelines {
        prepass_static,
        prepass_instanced,
        prepass_skinned,
    })
}

impl GbufferResources {
    // Build every G-buffer pre-pass resource. `instance_ssbo_set_layout` /
    // `skinned_ssbo_set_layout` are the existing per-instance / per-object joint
    // storage-buffer layouts the main pass uses; the pre-pass reuses those
    // buffers directly.
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
        instance_ssbo_set_layout: Option<vk::DescriptorSetLayout>,
        skinned_ssbo_set_layout: Option<vk::DescriptorSetLayout>,
        object_count: usize,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let prepass_render_pass = create_prepass_render_pass(device)?;

        // Pre-pass set 0: GbView UBO. Set 1 (instance/joint SSBO) is supplied by
        // the caller from the existing main-pass / skinned pipeline so the
        // pre-pass reuses those buffers directly.
        let prepass_set_layout = create_descriptor_set_layout(
            device,
            &[(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::VERTEX,
            )],
        )?;

        // Pipeline layouts. Both stages see the full prepass push block; the VS
        // reads cur/prev model, the FS only reads roughness.
        let prepass_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(GBUFFER_PREPASS_PUSH_BYTES);
        let static_layouts = [prepass_set_layout];
        let prepass_layout_static = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&static_layouts)
                    .push_constant_ranges(std::slice::from_ref(&prepass_push)),
                None,
            )
        }
        .map_err(|e| format!("gbuffer prepass static layout: {e}"))?;

        let prepass_layout_instanced = if let Some(isl) = instance_ssbo_set_layout {
            let layouts = [prepass_set_layout, isl];
            Some(
                unsafe {
                    device.create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&layouts)
                            .push_constant_ranges(std::slice::from_ref(&prepass_push)),
                        None,
                    )
                }
                .map_err(|e| format!("gbuffer prepass instanced layout: {e}"))?,
            )
        } else {
            None
        };

        let prepass_layout_skinned = if let Some(jsl) = skinned_ssbo_set_layout {
            // Set 1 = current joint palette, set 2 = previous joint palette.
            // Both reuse the single main-pass joint set layout.
            let layouts = [prepass_set_layout, jsl, jsl];
            Some(
                unsafe {
                    device.create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&layouts)
                            .push_constant_ranges(std::slice::from_ref(&prepass_push)),
                        None,
                    )
                }
                .map_err(|e| format!("gbuffer prepass skinned layout: {e}"))?,
            )
        } else {
            None
        };

        // Pipelines.
        let shaders = compile_gbuffer_shaders(hot_reload)?;
        let (vbindings, vattrs) = vertex_56_input();
        let prepass_pso_static = create_prepass_pipeline(
            device,
            prepass_render_pass,
            prepass_layout_static,
            &shaders.prepass_vs,
            &shaders.prepass_fs,
            &vbindings,
            &vattrs,
        )?;
        let prepass_pso_instanced = if let Some(layout) = prepass_layout_instanced {
            // Instanced pre-pass reads only position + normal (model comes from
            // the instance SSBO), so bind just those two attributes.
            Some(create_prepass_pipeline(
                device,
                prepass_render_pass,
                layout,
                &shaders.prepass_instanced_vs,
                &shaders.prepass_fs,
                &vbindings,
                &vattrs[..2],
            )?)
        } else {
            None
        };
        let prepass_pso_skinned = if let Some(layout) = prepass_layout_skinned {
            let (sbindings, sattrs) = skinned_vertex_input();
            Some(create_prepass_pipeline(
                device,
                prepass_render_pass,
                layout,
                &shaders.prepass_skinned_vs,
                &shaders.prepass_fs,
                &sbindings,
                &sattrs,
            )?)
        } else {
            None
        };

        // Per-frame view UBO (jittered_vp + cur_vp + prev_vp + view_mat).
        let mut view_ubo_buffers = Vec::with_capacity(frames);
        let mut view_ubo_memories = Vec::with_capacity(frames);
        let mut view_ubo_ptrs = Vec::with_capacity(frames);
        for _ in 0..frames {
            let (buf, mem) = create_buffer(
                instance,
                device,
                physical_device,
                GBUFFER_VIEW_UBO_SIZE,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let ptr = unsafe {
                device.map_memory(mem, 0, GBUFFER_VIEW_UBO_SIZE, vk::MemoryMapFlags::empty())
            }
            .map_err(|e| format!("map gbuffer view UBO: {e}"))? as *mut u8;
            view_ubo_buffers.push(buf);
            view_ubo_memories.push(mem);
            view_ubo_ptrs.push(ptr);
        }

        // Descriptor pool: `frames` prepass sets (1 UBO each).
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(frames as u32)];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(frames as u32),
                None,
            )
        }
        .map_err(|e| format!("gbuffer descriptor pool: {e}"))?;

        let prepass_layouts: Vec<_> = (0..frames).map(|_| prepass_set_layout).collect();
        let prepass_sets = alloc_descriptor_sets(device, descriptor_pool, &prepass_layouts)?;
        for (i, &set) in prepass_sets.iter().enumerate() {
            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(view_ubo_buffers[i])
                .offset(0)
                .range(GBUFFER_VIEW_UBO_SIZE);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }

        let mut me = Self {
            prepass_render_pass,
            prepass_set_layout,
            prepass_layout_static,
            prepass_layout_instanced,
            prepass_layout_skinned,
            prepass_pso_static,
            prepass_pso_instanced,
            prepass_pso_skinned,
            view_ubo_buffers,
            view_ubo_memories,
            view_ubo_ptrs,
            prepass_sets,
            descriptor_pool,
            normal_depth_images: Vec::new(),
            roughness_images: Vec::new(),
            velocity_images: Vec::new(),
            depth_images: Vec::new(),
            framebuffers: Vec::new(),
            prev_view_proj: IDENTITY4,
            prev_models: vec![IDENTITY4; object_count],
            hot_reload,
        };
        me.build_targets(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            width,
            height,
            frames,
        )?;
        Ok(me)
    }

    // Allocate the per-frame MRT targets + private depth + framebuffers at the
    // given extent. One slot per frame in flight.
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
        frames: usize,
    ) -> Result<(), String> {
        let w = width.max(1);
        let h = height.max(1);
        for _ in 0..frames {
            let normal_depth = create_color_target(
                instance,
                device,
                physical_device,
                w,
                h,
                GBUFFER_NORMAL_DEPTH_FORMAT,
            )?;
            let roughness = create_color_target(
                instance,
                device,
                physical_device,
                w,
                h,
                GBUFFER_ROUGHNESS_FORMAT,
            )?;
            let velocity = create_color_target(
                instance,
                device,
                physical_device,
                w,
                h,
                GBUFFER_VELOCITY_FORMAT,
            )?;
            let depth = create_depth_image(
                instance,
                device,
                physical_device,
                command_pool,
                queue,
                w,
                h,
                vk::SampleCountFlags::TYPE_1,
            )?;
            let attachments = [normal_depth.view, roughness.view, velocity.view, depth.view];
            let framebuffer = unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.prepass_render_pass)
                        .attachments(&attachments)
                        .width(w)
                        .height(h)
                        .layers(1),
                    None,
                )
            }
            .map_err(|e| format!("gbuffer prepass framebuffer: {e}"))?;
            self.normal_depth_images.push(normal_depth);
            self.roughness_images.push(roughness);
            self.velocity_images.push(velocity);
            self.depth_images.push(depth);
            self.framebuffers.push(framebuffer);
        }
        Ok(())
    }

    // The per-frame normal+depth view a reader (SSR resolve, SSAO, SSGI) binds.
    pub(in crate::vulkan) fn normal_depth_view(&self, frame: usize) -> vk::ImageView {
        self.normal_depth_images[frame].view
    }

    // The per-frame roughness view a reader binds.
    pub(in crate::vulkan) fn roughness_view(&self, frame: usize) -> vk::ImageView {
        self.roughness_images[frame].view
    }

    // The per-frame velocity view the TAA resolve / FSR binds.
    pub(in crate::vulkan) fn velocity_view(&self, frame: usize) -> vk::ImageView {
        self.velocity_images[frame].view
    }

    // The per-frame depth view FSR consumes alongside the motion vectors. FSR
    // binds the `GpuImage` (image + view) directly from `depth_images`, so this
    // view-only accessor is part of the symmetric reader API but currently
    // unused; kept for parity with the other channel accessors.
    #[allow(dead_code)]
    pub(in crate::vulkan) fn depth_view(&self, frame: usize) -> vk::ImageView {
        self.depth_images[frame].view
    }

    // Per-frame normal+depth views, one per frame in flight. The readers that
    // bind a per-frame descriptor set (SSR resolve, SSAO kernel/blur, SSGI, RT)
    // slice this so each set samples its own frame's unified G-buffer.
    pub(in crate::vulkan) fn normal_depth_views(&self) -> Vec<vk::ImageView> {
        (0..self.normal_depth_images.len())
            .map(|f| self.normal_depth_view(f))
            .collect()
    }

    // Per-frame roughness views, one per frame in flight.
    pub(in crate::vulkan) fn roughness_views(&self) -> Vec<vk::ImageView> {
        (0..self.roughness_images.len())
            .map(|f| self.roughness_view(f))
            .collect()
    }

    // Per-frame velocity views, one per frame in flight. The TAA resolve binds
    // its frame's slot; FSR reads `velocity_images[frame]` directly.
    pub(in crate::vulkan) fn velocity_views(&self) -> Vec<vk::ImageView> {
        (0..self.velocity_images.len())
            .map(|f| self.velocity_view(f))
            .collect()
    }

    fn destroy_targets(&mut self, device: &Device) {
        for &fb in &self.framebuffers {
            unsafe { device.destroy_framebuffer(fb, None) };
        }
        for img in self
            .normal_depth_images
            .iter()
            .chain(&self.roughness_images)
            .chain(&self.velocity_images)
            .chain(&self.depth_images)
        {
            img.destroy(device);
        }
        self.framebuffers.clear();
        self.normal_depth_images.clear();
        self.roughness_images.clear();
        self.velocity_images.clear();
        self.depth_images.clear();
    }

    // Rebuild the per-frame targets at a new swapchain extent. The caller has
    // already idled the device. The descriptor sets and UBOs are
    // resolution-independent and untouched.
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
        frames: usize,
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
            frames,
        )?;
        Ok(())
    }

    // Build (or rebuild) the skinned G-buffer pre-pass pipeline lazily, once a
    // `SkinnedMesh` has been uploaded and the joint descriptor set layout
    // exists. Idempotent: re-calling replaces the existing pipeline.
    pub(in crate::vulkan) fn ensure_skinned_gbuffer_pso(
        &mut self,
        device: &Device,
        joint_set_layout: vk::DescriptorSetLayout,
    ) -> Result<(), String> {
        if let Some(p) = self.prepass_pso_skinned.take() {
            unsafe { device.destroy_pipeline(p, None) };
        }
        if let Some(l) = self.prepass_layout_skinned.take() {
            unsafe { device.destroy_pipeline_layout(l, None) };
        }
        let prepass_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(GBUFFER_PREPASS_PUSH_BYTES);
        // Set 1 = current joint palette, set 2 = previous joint palette; the
        // skinned VS deforms both poses to emit a real deformation motion
        // vector. Both reuse the single main-pass joint set layout.
        let layouts = [self.prepass_set_layout, joint_set_layout, joint_set_layout];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(std::slice::from_ref(&prepass_push)),
                None,
            )
        }
        .map_err(|e| format!("gbuffer prepass skinned layout: {e}"))?;
        use super::super::pipeline::shader_source;
        let sk_vs = compile_glsl(
            &shader_source(
                self.hot_reload,
                "gbuffer_prepass_skinned.vert",
                GBUFFER_PREPASS_SKINNED_VERT_GLSL,
            ),
            shaderc::ShaderKind::Vertex,
            "gbuffer_prepass_skinned.vert",
        )?;
        let prepass_fs = compile_glsl(
            &shader_source(
                self.hot_reload,
                "gbuffer_prepass.frag",
                GBUFFER_PREPASS_FRAG_GLSL,
            ),
            shaderc::ShaderKind::Fragment,
            "gbuffer_prepass.frag",
        )?;
        let (sbindings, sattrs) = skinned_vertex_input();
        let pso = create_prepass_pipeline(
            device,
            self.prepass_render_pass,
            layout,
            &sk_vs,
            &prepass_fs,
            &sbindings,
            &sattrs,
        )?;
        self.prepass_layout_skinned = Some(layout);
        self.prepass_pso_skinned = Some(pso);
        Ok(())
    }

    // Swap the freshly-built pipelines into the live resources. The caller has
    // already `device_wait_idle`'d so the old pipelines are not in flight.
    pub(in crate::vulkan) fn swap_pipelines(
        &mut self,
        device: &Device,
        rebuilt: RebuiltGbufferPipelines,
    ) {
        unsafe {
            device.destroy_pipeline(self.prepass_pso_static, None);
            if let Some(p) = self.prepass_pso_instanced.take() {
                device.destroy_pipeline(p, None);
            }
            if let Some(p) = self.prepass_pso_skinned.take() {
                device.destroy_pipeline(p, None);
            }
        }
        self.prepass_pso_static = rebuilt.prepass_static;
        self.prepass_pso_instanced = rebuilt.prepass_instanced;
        self.prepass_pso_skinned = rebuilt.prepass_skinned;
    }

    // Destroy every G-buffer pre-pass resource. The caller has already idled the
    // device.
    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        self.destroy_targets(device);
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.prepass_pso_static, None);
            if let Some(p) = self.prepass_pso_instanced.take() {
                device.destroy_pipeline(p, None);
            }
            if let Some(p) = self.prepass_pso_skinned.take() {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.prepass_layout_static, None);
            if let Some(l) = self.prepass_layout_instanced.take() {
                device.destroy_pipeline_layout(l, None);
            }
            if let Some(l) = self.prepass_layout_skinned.take() {
                device.destroy_pipeline_layout(l, None);
            }
            device.destroy_descriptor_set_layout(self.prepass_set_layout, None);
            device.destroy_render_pass(self.prepass_render_pass, None);
            for (&buf, &mem) in self.view_ubo_buffers.iter().zip(&self.view_ubo_memories) {
                device.unmap_memory(mem);
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
        }
    }
}

impl VkContext {
    // Encode the unified G-buffer pre-pass: one jittered traversal of the
    // visible set (static + GPU-instanced + skinned) into the per-frame
    // normal+depth / roughness / velocity MRT plus a private depth buffer. Runs
    // before the main pass. `velocity_active` is true when a consumer (TAA or
    // FSR) reads motion; when false, prev == cur so the motion channel is a
    // harmless zero. Fuses the former SSR depth+normal and TAA velocity
    // pre-passes.
    //
    // `gb` is borrowed from the owning `self.gbuffer` field by the caller,
    // matching how the SSR / TAA encoders take their resources.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vulkan) fn encode_gbuffer_prepass(
        &self,
        gb: &GbufferResources,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        jittered_vp: [[f32; 4]; 4],
        cur_vp: [[f32; 4]; 4],
        visible: &[u32],
        frustum: &crate::gfx::frustum::Frustum,
        cam_pos: [f32; 3],
        velocity_active: bool,
    ) {
        let device = &self.device;
        let extent = self.render_extent;

        // Upload this frame's view UBO. When velocity is inactive the previous
        // VP equals the current one, so instanced + sky motion is zero.
        let prev_vp = if velocity_active {
            gb.prev_view_proj
        } else {
            cur_vp
        };
        let view_uni = GbViewUniforms {
            jittered_vp,
            cur_vp,
            prev_vp,
            view_mat: self.view_matrix,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &view_uni as *const GbViewUniforms as *const u8,
                gb.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<GbViewUniforms>(),
            );
        }

        // Clears: alpha-0 normal+depth = "no geometry"; roughness 1.0 = no SSR;
        // velocity 0 = no motion.
        let clears = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(gb.prepass_render_pass)
            .framebuffer(gb.framebuffers[frame_idx])
            .render_area(vk::Rect2D::default().extent(extent))
            .clear_values(&clears);
        unsafe { device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE) };

        // Negative-height viewport: matches the main pass so the G-buffer lines
        // up with the main pass at pixel coordinates; the fragment shader's
        // upright-UV math expects this orientation.
        let vp = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        unsafe {
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
        }

        // When the bindless GPU-cull path is active, the pre-pass is GPU-driven:
        // it reuses the main pass's per-frame indirect buffer (same camera frustum
        // + active LOD) with two `cmd_draw_indexed_indirect` draws (static +
        // instance prefix, then the skinned tail over the deformed VB) instead of
        // the CPU per-object loops, plus a legacy extra loop for streamed chunks /
        // runtime clones not in the cull records. A non-bindless world (custom
        // shader) keeps the legacy path below. Both write the same MRT.
        if self.cull.gbuffer_bindless_pipeline.is_some() && self.cull_count() > 0 {
            self.encode_gbuffer_prepass_gpu_driven(
                gb,
                cmd,
                frame_idx,
                visible,
                cam_pos,
                velocity_active,
            );
            unsafe { device.cmd_end_render_pass(cmd) };
            return;
        }

        unsafe {
            device.cmd_bind_vertex_buffers(
                cmd,
                0,
                std::slice::from_ref(&self.geometry.vertex_buffer),
                &[0],
            );
            device.cmd_bind_index_buffer(cmd, self.geometry.index_buffer, 0, vk::IndexType::UINT32);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, gb.prepass_pso_static);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                gb.prepass_layout_static,
                0,
                std::slice::from_ref(&gb.prepass_sets[frame_idx]),
                &[],
            );
        }

        // Static geometry: same visible set + LOD pick as the main pass so the
        // G-buffer covers exactly what main rasterised.
        let last_obj = self.draw_objects.len().saturating_sub(1);
        for &draw_idx in visible {
            let i = (draw_idx as usize).min(last_obj);
            let obj = match self.draw_objects.get(i) {
                Some(o) => o,
                None => continue,
            };
            if !obj.visible || !obj.resident {
                continue;
            }
            let d = crate::gfx::lod::camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let prev_model = if velocity_active {
                gb.prev_models.get(i).copied().unwrap_or(obj.model)
            } else {
                obj.model
            };
            let push = GbModelPush {
                cur_model: obj.model,
                prev_model,
                roughness: obj.material.roughness,
                _pad: [0.0; 3],
            };
            unsafe {
                device.cmd_push_constants(
                    cmd,
                    gb.prepass_layout_static,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    std::slice::from_raw_parts(
                        &push as *const GbModelPush as *const u8,
                        std::mem::size_of::<GbModelPush>(),
                    ),
                );
                device.cmd_draw_indexed(
                    cmd,
                    index_count as u32,
                    1,
                    index_offset as u32,
                    obj.base_vertex,
                    0,
                );
            }
        }

        // GPU-instanced clusters: instance transforms never change, so the
        // motion is camera-only (the instanced VS feeds the same matrix to cur
        // and prev clip). Reuses the per-cluster instance SSBO the main
        // instanced pass already filled this frame.
        if let (Some(inst_pso), Some(inst_layout)) =
            (gb.prepass_pso_instanced, gb.prepass_layout_instanced)
            && !self.instanced.clusters.is_empty()
            && !self.instanced.sets.is_empty()
        {
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, inst_pso);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    inst_layout,
                    0,
                    std::slice::from_ref(&gb.prepass_sets[frame_idx]),
                    &[],
                );
            }
            for (cluster_idx, cluster) in self.instanced.clusters.iter().enumerate() {
                if cluster.instances.is_empty() {
                    continue;
                }
                if cluster.cullable() {
                    if !frustum.intersects_aabb(cluster.cluster_bb_min, cluster.cluster_bb_max) {
                        continue;
                    }
                    if cluster.cull_distance > 0.0 {
                        let d2 = crate::gfx::frustum::aabb_distance_sq(
                            cam_pos,
                            cluster.cluster_bb_min,
                            cluster.cluster_bb_max,
                        );
                        if d2 > cluster.cull_distance * cluster.cull_distance {
                            continue;
                        }
                    }
                }
                let Some(buckets) = self.instanced.lod_buckets.get(cluster_idx) else {
                    continue;
                };
                let inst_set = self.instanced.sets[frame_idx][cluster_idx];
                let push = GbModelPush {
                    cur_model: [[0.0; 4]; 4],  // ignored by instanced VS
                    prev_model: [[0.0; 4]; 4], // ignored by instanced VS
                    roughness: cluster.material.roughness,
                    _pad: [0.0; 3],
                };
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        inst_layout,
                        1,
                        std::slice::from_ref(&inst_set),
                        &[],
                    );
                    device.cmd_push_constants(
                        cmd,
                        inst_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        std::slice::from_raw_parts(
                            &push as *const GbModelPush as *const u8,
                            std::mem::size_of::<GbModelPush>(),
                        ),
                    );
                    // One draw per LOD bucket, matching the Main pass partition
                    // so the G-buffer stays pixel-aligned with the scene.
                    let mut first_instance: u32 = 0;
                    for bucket in buckets {
                        let count = bucket.instances.len() as u32;
                        device.cmd_draw_indexed(
                            cmd,
                            bucket.index_count as u32,
                            count,
                            bucket.index_offset as u32,
                            0,
                            first_instance,
                        );
                        first_instance += count;
                    }
                }
            }
        }

        // Skinned meshes: drawn last so the G-buffer reflects animated
        // characters too. The current palette (set 1) and the previous-frame
        // palette (set 2) deform the two poses so per-vertex skinned motion
        // produces a correct motion vector. The model matrix is static (skinned
        // meshes are self-placing), so cur and prev model are usually identical;
        // it is threaded through identically to the static path. The previous
        // palette lives at the prior slot of the joint-set ring; with fewer than
        // two frames in flight there is no distinct prior slot, so prev = cur
        // (it cannot ghost without a second in-flight frame anyway).
        if let (Some(sk_pso), Some(sk_layout)) = (gb.prepass_pso_skinned, gb.prepass_layout_skinned)
            && !self.skinned.draw_objects.is_empty()
        {
            let frames = self.frames_in_flight.max(1);
            let prev_frame_idx = if velocity_active && frames >= 2 {
                (frame_idx + frames - 1) % frames
            } else {
                frame_idx
            };
            let (sk_vbuf, sk_ibuf) = self.skinned_geometry();
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, sk_pso);
                device.cmd_bind_vertex_buffers(cmd, 0, std::slice::from_ref(&sk_vbuf), &[0]);
                device.cmd_bind_index_buffer(cmd, sk_ibuf, 0, vk::IndexType::UINT16);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    sk_layout,
                    0,
                    std::slice::from_ref(&gb.prepass_sets[frame_idx]),
                    &[],
                );
            }
            for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
                if !obj.visible {
                    continue;
                }
                let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
                let (index_offset, index_count) = obj.active_lod(d);
                // Skinned meshes are self-placing, so cur and prev model are
                // identical; the deformation motion comes from the current vs
                // previous joint palettes bound at sets 1 / 2. Threaded through
                // identically to the static path. `prev_models` is keyed by
                // static draw-object index, so it is not consulted here.
                let push = GbModelPush {
                    cur_model: obj.model,
                    prev_model: obj.model,
                    roughness: obj.material.roughness,
                    _pad: [0.0; 3],
                };
                unsafe {
                    // Set 1 = current palette, set 2 = previous-frame palette.
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        sk_layout,
                        1,
                        std::slice::from_ref(&self.skinned.joint_sets[frame_idx][i]),
                        &[],
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        sk_layout,
                        2,
                        std::slice::from_ref(&self.skinned.joint_sets[prev_frame_idx][i]),
                        &[],
                    );
                    device.cmd_push_constants(
                        cmd,
                        sk_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        std::slice::from_raw_parts(
                            &push as *const GbModelPush as *const u8,
                            std::mem::size_of::<GbModelPush>(),
                        ),
                    );
                    device.cmd_draw_indexed(cmd, index_count as u32, 1, index_offset as u32, 0, 0);
                }
            }
            // Restore the static vertex/index buffers for any later pass that
            // does not rebind them itself.
            unsafe {
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    std::slice::from_ref(&self.geometry.vertex_buffer),
                    &[0],
                );
                device.cmd_bind_index_buffer(
                    cmd,
                    self.geometry.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
            }
        }

        unsafe { device.cmd_end_render_pass(cmd) };
    }

    // GPU-driven G-buffer pre-pass raster (inside the render pass the caller
    // began). Reuses the main pass's per-frame indirect buffer (the camera-frustum
    // cull already produced it, so no extra cull dispatch) with two indirect draws:
    // the static + instance prefix `[0, skinned_record_base())` over the static VB
    // (bound to BOTH vertex bindings, so prev_pos == cur_pos and the motion is the
    // per-object model delta plus camera), then the skinned tail over the current
    // deformed VB (binding 0) + the previous-frame deformed VB (binding 1) for
    // per-vertex deformation motion. model + roughness ride the per-frame
    // GpuObjectData SSBO (gl_InstanceIndex); the previous-frame model a parallel
    // SSBO. Streamed chunks / runtime clones keep a legacy per-object loop.
    fn encode_gbuffer_prepass_gpu_driven(
        &self,
        gb: &GbufferResources,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        visible: &[u32],
        cam_pos: [f32; 3],
        velocity_active: bool,
    ) {
        let device = &self.device;
        let (Some(pipeline), Some(layout)) = (
            self.cull.gbuffer_bindless_pipeline,
            self.cull.gbuffer_bindless_pipeline_layout,
        ) else {
            return;
        };
        let Some(indirect) = self.cull.indirect_buffers.get(frame_idx).copied() else {
            return;
        };
        let Some(&gset) = self.cull.gbuffer_sets.get(frame_idx) else {
            return;
        };
        let stride = std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32;
        let prefix = self.skinned_record_base() as u32;

        // Build this frame's previous-frame model buffer (static + skinned regions;
        // the instance region is init-written + immutable). Honours velocity_active.
        self.build_gbuffer_prev_models(gb, frame_idx, velocity_active);

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            // set 0 = GbView UBO + prev_model SSBO; set 1 = bindless GpuObjectData.
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[gset, self.cull.bindless_sets[frame_idx]],
                &[],
            );

            // Static + instance prefix: the static VB bound to BOTH vertex bindings
            // (prev_pos == cur_pos) + the static u32 IB.
            device.cmd_bind_vertex_buffers(
                cmd,
                0,
                &[self.geometry.vertex_buffer, self.geometry.vertex_buffer],
                &[0, 0],
            );
            device.cmd_bind_index_buffer(cmd, self.geometry.index_buffer, 0, vk::IndexType::UINT32);
            if prefix > 0 {
                device.cmd_draw_indexed_indirect(cmd, indirect, 0, prefix, stride);
                self.inc_draw_calls(1);
            }
        }

        // Skinned tail: the current deformed VB (binding 0) + the previous-frame
        // deformed VB (binding 1) + the skinned u16 IB. Records carry base_vertex
        // = 0 (global skinned indexing). The previous deformed buffer is read only
        // once the ring is primed (a prior frame posed that slot); before then (or
        // when velocity is inactive) it is the current buffer, so prev_pos ==
        // cur_pos gives a harmless zero skinned motion vector.
        if self.n_skinned > 0
            && let Some(cur) = self.skinned.deformed.get(frame_idx)
        {
            let frames = self.frames_in_flight.max(1);
            let use_prev = velocity_active
                && frames >= 2
                && self
                    .skinned
                    .deformed_primed
                    .load(std::sync::atomic::Ordering::Relaxed);
            let prev_idx = if use_prev {
                (frame_idx + frames - 1) % frames
            } else {
                frame_idx
            };
            let prev = self.skinned.deformed.get(prev_idx).unwrap_or(cur);
            unsafe {
                device.cmd_bind_vertex_buffers(cmd, 0, &[cur.buffer, prev.buffer], &[0, 0]);
                device.cmd_bind_index_buffer(
                    cmd,
                    self.skinned.index_buffer,
                    0,
                    vk::IndexType::UINT16,
                );
                device.cmd_draw_indexed_indirect(
                    cmd,
                    indirect,
                    (self.skinned_record_base() * stride as usize) as u64,
                    self.n_skinned as u32,
                    stride,
                );
            }
            self.inc_draw_calls(1);
            // The current deformed buffer is posed this frame, so next frame's
            // history slot (this slot) is valid -- prime the ring.
            self.skinned
                .deformed_primed
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }

        // Legacy extra: streamed chunks + runtime clones (records past `n_objects`)
        // are not in the GpuObjectData buffer, so draw them with the legacy
        // per-object pipeline into the same MRT.
        self.encode_gbuffer_legacy_extra(gb, cmd, frame_idx, visible, cam_pos, velocity_active);
    }

    // Legacy per-object G-buffer draws for runtime clones past the bindless range
    // (`i >= n_objects` AND in `clone_slot_by_draw_idx`). Streamed VoxelWorld chunks
    // now fold into the GPU-driven cull records (drawn by the prefix indirect draw),
    // so they are skipped here. Mirrors the legacy static loop, appending into the
    // same MRT after the indirect draws. A no-op for worlds with no clones.
    fn encode_gbuffer_legacy_extra(
        &self,
        gb: &GbufferResources,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        visible: &[u32],
        cam_pos: [f32; 3],
        velocity_active: bool,
    ) {
        if self.clone_slot_by_draw_idx.is_empty() {
            return;
        }
        let device = &self.device;
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, gb.prepass_pso_static);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                gb.prepass_layout_static,
                0,
                std::slice::from_ref(&gb.prepass_sets[frame_idx]),
                &[],
            );
            device.cmd_bind_vertex_buffers(
                cmd,
                0,
                std::slice::from_ref(&self.geometry.vertex_buffer),
                &[0],
            );
            device.cmd_bind_index_buffer(cmd, self.geometry.index_buffer, 0, vk::IndexType::UINT32);
        }
        for &draw_idx in visible {
            let i = draw_idx as usize;
            if i < self.n_objects {
                continue; // build-time object, already drawn via indirect
            }
            if !self.clone_slot_by_draw_idx.contains_key(&i) {
                continue; // streamed chunk -> folded into the cull records
            }
            let Some(obj) = self.draw_objects.get(i) else {
                continue;
            };
            if !obj.visible || !obj.resident {
                continue;
            }
            let d = crate::gfx::lod::camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let prev_model = if velocity_active {
                gb.prev_models.get(i).copied().unwrap_or(obj.model)
            } else {
                obj.model
            };
            let push = GbModelPush {
                cur_model: obj.model,
                prev_model,
                roughness: obj.material.roughness,
                _pad: [0.0; 3],
            };
            unsafe {
                device.cmd_push_constants(
                    cmd,
                    gb.prepass_layout_static,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    std::slice::from_raw_parts(
                        &push as *const GbModelPush as *const u8,
                        std::mem::size_of::<GbModelPush>(),
                    ),
                );
                device.cmd_draw_indexed(
                    cmd,
                    index_count as u32,
                    1,
                    index_offset as u32,
                    obj.base_vertex,
                    0,
                );
            }
        }
    }

    // Fill this frame's previous-frame model SSBO for the GPU-driven G-buffer
    // velocity. Indexed by cull record id, parallel to the GpuObjectData buffer:
    // the static prefix `[0, n_objects)` gets last frame's model (so a moving
    // static object reprojects correctly), the skinned tail gets the current model
    // (skinned deformation motion comes from the previous-frame deformed buffer,
    // not the model matrix). The instance region is init-written + immutable
    // (camera-only motion), so it is left untouched. When velocity is inactive
    // every written record gets its current model, so the motion stays zero (GbView
    // prev_vp also equals cur_vp). Mirrors build_object_buffer's record indexing.
    fn build_gbuffer_prev_models(
        &self,
        gb: &GbufferResources,
        frame_idx: usize,
        velocity_active: bool,
    ) {
        let Some(&ptr) = self.cull.prev_model_ptrs.get(frame_idx) else {
            return;
        };
        let stride = std::mem::size_of::<[[f32; 4]; 4]>();
        for (i, obj) in self.draw_objects.iter().take(self.n_objects).enumerate() {
            let prev = if velocity_active {
                gb.prev_models.get(i).copied().unwrap_or(obj.model)
            } else {
                obj.model
            };
            // SAFETY: the buffer was sized for `cull_count()` records and the loop
            // is bounded by `take(n_objects)`, so `i * stride` is in range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &prev as *const [[f32; 4]; 4] as *const u8,
                    ptr.add(i * stride),
                    stride,
                );
            }
        }
        // Streamed chunks: current model -> camera-only velocity (chunk terrain is
        // static-in-world; the unused reserve slots keep stale prev_models but their
        // draw-args are disabled, so the gbuffer never rasterises them).
        let chunk_base = self.chunk_record_base();
        self.for_each_chunk_record(|k, obj| {
            let prev = obj.model;
            // SAFETY: `for_each_chunk_record` caps `k < n_chunk`, so
            // `chunk_base + k < skinned_record_base()`, in range for `cull_count()`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &prev as *const [[f32; 4]; 4] as *const u8,
                    ptr.add((chunk_base + k) * stride),
                    stride,
                );
            }
        });
        let base = self.skinned_record_base();
        for (k, obj) in self
            .skinned
            .draw_objects
            .iter()
            .take(self.n_skinned)
            .enumerate()
        {
            // Skinned motion is per-vertex (previous deformed buffer), so the model
            // matrix is the current one (cur == prev model, like the legacy path).
            let prev = obj.model;
            // SAFETY: the buffer reserved `n_skinned` records past
            // `skinned_record_base()` at init; the loop is bounded by
            // `self.skinned.draw_objects.len() == self.n_skinned`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &prev as *const [[f32; 4]; 4] as *const u8,
                    ptr.add((base + k) * stride),
                    stride,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // GbViewUniforms must match the `GbView` UBO (set 0, binding 0) in every
    // pre-pass VS: four std140 column-major mat4 at offsets 0, 64, 128, 192
    // (256 B total).
    #[test]
    fn gb_view_uniforms_layout_matches_glsl() {
        assert_eq!(size_of::<GbViewUniforms>(), 256);
        assert_eq!(offset_of!(GbViewUniforms, jittered_vp), 0);
        assert_eq!(offset_of!(GbViewUniforms, cur_vp), 64);
        assert_eq!(offset_of!(GbViewUniforms, prev_vp), 128);
        assert_eq!(offset_of!(GbViewUniforms, view_mat), 192);
        // Upload size must not exceed the UBO allocation.
        assert!(size_of::<GbViewUniforms>() as u64 <= GBUFFER_VIEW_UBO_SIZE);
    }

    // GbModelPush is pushed as the shared `PushBlock`: cur_model then prev_model
    // (two column-major mat4) then roughness at offset 128, plus pad. The total
    // must match the push-constant range size.
    #[test]
    fn gb_model_push_layout_matches_glsl() {
        assert_eq!(size_of::<GbModelPush>(), 144);
        assert_eq!(offset_of!(GbModelPush, cur_model), 0);
        assert_eq!(offset_of!(GbModelPush, prev_model), 64);
        assert_eq!(offset_of!(GbModelPush, roughness), 128);
        assert_eq!(size_of::<GbModelPush>() as u32, GBUFFER_PREPASS_PUSH_BYTES);
    }

    // Every G-buffer pre-pass GLSL (static + instanced + skinned vertex shaders
    // and the shared fragment) compiles to SPIR-V. Exercises the fused
    // ssr_prepass + velocity contract: the vertex shaders emit cur_clip /
    // prev_clip the fragment consumes for the motion vector.
    #[test]
    fn gbuffer_shaders_compile() {
        compile_gbuffer_shaders(false).expect("gbuffer shaders compile");
    }
}
