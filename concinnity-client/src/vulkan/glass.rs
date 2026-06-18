// src/vulkan/glass.rs
//
// GlassPanel: the generic producer for the engine's transparent pass on the
// Vulkan backend. Each panel is a flat world-space quad (built once at init)
// drawn in the `PassId::Transparent` slot after SSR resolve and before TAA. The
// pass snapshots the pre-transparent scene, sorts the panels back-to-front by
// camera distance, and draws each one; the fragment shader refracts the
// snapshot, tints it, and adds a Fresnel rim (see shaders/glass.frag).
//
// GLSL/Vulkan port of `src/directx/glass.rs`: same uniform layouts, same
// back-to-front ordering, same manual depth-occlusion test. The pass writes
// into the post-SSR scene image (the same image the post stack samples:
// `SsrResources::output` when SSR is on, else `hdr_resolve_images[frame]`),
// alpha-blending over it; downstream TAA / bloom / composite pick the
// translucent geometry up unchanged. Water is a separate (Metal-only) producer
// and is not ported here; the transparent slot on Vulkan is glass-only.

use ash::{Device, vk};

use crate::assets::GlassPanel;
use crate::geometry::glass_quad::build_glass_quad;
use crate::gfx::mesh_payload::Vertex;

use super::context::{HDR_FORMAT, VkContext};
use super::pipeline::{compile_glsl, inject_define, shader_source, spv_module};
use super::texture::{
    GpuImage, create_buffer, create_image, create_image_view, one_shot_submit,
    transition_image_layout_range,
};

const GLASS_VERT: &str = include_str!("shaders/glass.vert");
const GLASS_FRAG: &str = include_str!("shaders/glass.frag");

// Per-frame view UBO bound at set 0 binding 0. Layout matches the
// `TransparentViewBlock` std140 block in `shaders/glass.{vert,frag}` and the
// DirectX / Metal `TransparentView`. 160 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub(in crate::vulkan) struct TransparentView {
    pub(in crate::vulkan) vp: [[f32; 4]; 4],
    pub(in crate::vulkan) inv_vp: [[f32; 4]; 4],
    pub(in crate::vulkan) camera_pos: [f32; 4],
    pub(in crate::vulkan) viewport: [f32; 2],
    pub(in crate::vulkan) time: f32,
    pub(in crate::vulkan) _pad: f32,
}

// Per-panel UBO bound at set 1 binding 0. Layout matches the `GlassParamsBlock`
// std140 block + the DirectX `GlassParamsGpu`. 64 bytes. Vec3 fields ride in
// vec4s (.w unused) so the layout is byte-identical regardless of std140
// packing.
#[derive(Copy, Clone)]
#[repr(C)]
struct GlassParams {
    centre: [f32; 4],
    normal: [f32; 4],
    tint: [f32; 4],
    opacity: f32,
    refraction_strength: f32,
    fresnel_power: f32,
    _pad1: f32,
}

// Build the per-panel `GlassParams` from an authored panel. Pure; unit tested.
// Mirrors `directx::glass::glass_params_from`.
fn glass_params_from(panel: &GlassPanel) -> GlassParams {
    let n = panel.normal; // already unit-length from GlassPanel::from_args
    GlassParams {
        centre: [panel.centre[0], panel.centre[1], panel.centre[2], 0.0],
        normal: [n[0], n[1], n[2], 0.0],
        tint: [panel.tint[0], panel.tint[1], panel.tint[2], 0.0],
        opacity: panel.opacity,
        refraction_strength: panel.refraction_strength,
        fresnel_power: panel.fresnel_power,
        _pad1: 0.0,
    }
}

// World-space distance from the camera to a panel centre. Larger = farther =
// drawn first. Pure; unit tested.
fn sort_distance(centre: [f32; 3], cam: [f32; 3]) -> f32 {
    let dx = centre[0] - cam[0];
    let dy = centre[1] - cam[1];
    let dz = centre[2] - cam[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// Indices of the visible panels, ordered farthest-camera-distance first. Pure;
// unit tested. Invisible panels are excluded; the visible set is sorted via the
// shared `gfx::transparent::back_to_front_order`.
fn ordered_visible(centres: &[[f32; 3]], visible: &[bool], cam: [f32; 3]) -> Vec<usize> {
    let live: Vec<usize> = (0..centres.len()).filter(|&i| visible[i]).collect();
    let dists: Vec<f32> = live
        .iter()
        .map(|&i| sort_distance(centres[i], cam))
        .collect();
    crate::gfx::transparent::back_to_front_order(&dists)
        .into_iter()
        .map(|oi| live[oi])
        .collect()
}

// Compile the glass vertex + fragment shaders, injecting the MSAA define so the
// depth sampler type matches the main-depth resource's sample count.
fn compile_glass_shaders(hot_reload: bool, msaa: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let define = if msaa {
        "#define USE_MSAA 1\n"
    } else {
        "#define USE_MSAA 0\n"
    };
    let vert_src = inject_define(&shader_source(hot_reload, "glass.vert", GLASS_VERT), define);
    let frag_src = inject_define(&shader_source(hot_reload, "glass.frag", GLASS_FRAG), define);
    let vert = compile_glsl(&vert_src, shaderc::ShaderKind::Vertex, "glass.vert")?;
    let frag = compile_glsl(&frag_src, shaderc::ShaderKind::Fragment, "glass.frag")?;
    Ok((vert, frag))
}

// Per-panel GPU state: the static world-space quad VB + IB, the per-panel
// `GlassParams` UBO + its descriptor set, and the visibility flag.
struct GlassPanelRecord {
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    index_count: u32,
    params_ubo: vk::Buffer,
    params_ubo_memory: vk::DeviceMemory,
    params_set: vk::DescriptorSet,
    visible: bool,
    // World-space centre, used for the back-to-front camera-distance sort.
    centre: [f32; 3],
}

// Engine-side glass resources. Built only when the world declared at least one
// `GlassPanel`; `VkContext::glass` stays `None` otherwise and the Transparent
// pass is omitted from the frame graph.
pub(in crate::vulkan) struct GlassResources {
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    view_set_layout: vk::DescriptorSetLayout,
    params_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,

    // Per-frame `TransparentView` UBO ring. Persistently mapped; the encoder
    // memcpys this frame's view into `view_ubo_ptrs[frame_idx]` before binding.
    view_ubos: Vec<vk::Buffer>,
    view_ubo_memories: Vec<vk::DeviceMemory>,
    view_ubo_ptrs: Vec<*mut u8>,
    view_sets: Vec<vk::DescriptorSet>,

    // Per-frame scene target the pass blends into: `SsrResources::output`
    // (repeated for every frame slot) when SSR is on, else this slot's
    // `hdr_resolve_images[i]`. The framebuffer targets the view; the snapshot
    // copy reads the image.
    scene_images: Vec<vk::Image>,
    framebuffers: Vec<vk::Framebuffer>,

    // Pre-transparent HDR scene snapshot for the refraction tap. The encoder
    // copies the scene image into this at the head of the pass; sized to render
    // dims, recreated by `rebuild` on resize. Single image shared across frames
    // (the same single-shared-snapshot pattern as the raymarch pass).
    snapshot: GpuImage,
    // Linear sampler bound alongside the snapshot (binding 1) and the main
    // depth (binding 2). Borrowed from `VkContext`; not owned, never destroyed
    // here.
    sampler: vk::Sampler,

    panels: Vec<GlassPanelRecord>,
}

// The transparent render pass: load + store the single-sample scene image (the
// post-SSR scene rests in SHADER_READ_ONLY) with no depth attachment (the
// fragment shader does the manual occlusion test). Mirrors the decal render
// pass shape.
fn create_glass_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass, String> {
    let color = vk::AttachmentDescription::default()
        .format(format)
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
    // The encoder's explicit barrier (scene back to SHADER_READ_ONLY after the
    // snapshot copy) makes the load available; this dependency orders the load
    // after it.
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::TRANSFER,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        );
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&color))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dependency));
    unsafe { device.create_render_pass(&info, None) }.map_err(|e| format!("glass render pass: {e}"))
}

fn create_view_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
    let frag = vk::ShaderStageFlags::FRAGMENT;
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(frag),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(frag),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("glass view set layout: {e}"))
}

fn create_params_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
    let info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding));
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("glass params set layout: {e}"))
}

fn create_descriptor_pool(
    device: &Device,
    frames: usize,
    panels: usize,
) -> Result<vk::DescriptorPool, String> {
    let f = frames as u32;
    let p = panels as u32;
    let sizes = [
        // view UBO per frame + params UBO per panel.
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: f + p,
        },
        // snapshot + depth per per-frame view set.
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 2 * f,
        },
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(f + p)
        .pool_sizes(&sizes);
    unsafe { device.create_descriptor_pool(&info, None) }
        .map_err(|e| format!("glass descriptor pool: {e}"))
}

fn alloc_sets(
    device: &Device,
    pool: vk::DescriptorPool,
    layouts: &[vk::DescriptorSetLayout],
) -> Result<Vec<vk::DescriptorSet>, String> {
    let info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(layouts);
    unsafe { device.allocate_descriptor_sets(&info) }
        .map_err(|e| format!("glass descriptor sets: {e}"))
}

// Write one per-frame view set: the view UBO (binding 0), the shared scene
// snapshot (binding 1), and this frame's main-depth view (binding 2).
fn write_view_set(
    device: &Device,
    set: vk::DescriptorSet,
    view_ubo: vk::Buffer,
    snapshot_view: vk::ImageView,
    depth_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let view_info = vk::DescriptorBufferInfo::default()
        .buffer(view_ubo)
        .offset(0)
        .range(std::mem::size_of::<TransparentView>() as u64);
    let img = |view: vk::ImageView| {
        vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(sampler)
    };
    let snapshot_info = img(snapshot_view);
    let depth_info = img(depth_view);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&view_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&snapshot_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&depth_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn write_params_set(device: &Device, set: vk::DescriptorSet, params_ubo: vk::Buffer) {
    let info = vk::DescriptorBufferInfo::default()
        .buffer(params_ubo)
        .offset(0)
        .range(std::mem::size_of::<GlassParams>() as u64);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

// Build the glass graphics pipeline. No face culling (the shader is two-sided),
// no depth attachment / test (the fragment does the manual occlusion test), and
// SRC_ALPHA / ONE_MINUS_SRC_ALPHA blending into the single-sample scene target.
// The standard engine `Vertex` stride is bound with only the position attribute
// (location 0) fetched. Negative-height viewport applied dynamically at encode.
fn create_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert = spv_module(device, vert_spv)?;
    let frag = spv_module(device, frag_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag)
            .name(&entry),
    ];

    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attribute = vk::VertexInputAttributeDescription::default()
        .location(0)
        .binding(0)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset(0);
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(std::slice::from_ref(&binding))
        .vertex_attribute_descriptions(std::slice::from_ref(&attribute));

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    // The scene target is single-sample regardless of the main pass's MSAA.
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // No depth attachment: the fragment shader does the manual occlusion test.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false);
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend_attachments = [blend_attachment];
    let blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&blend_state)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass);
    let pipeline = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create glass pipeline: {e}"))?[0];
    unsafe {
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
    }
    Ok(pipeline)
}

// Create the pre-transparent HDR scene snapshot (SAMPLED | TRANSFER_DST,
// GPU-local) and rest it in SHADER_READ_ONLY so the first frame's snapshot
// barrier (SHADER_READ_ONLY -> TRANSFER_DST) matches. Mirrors the raymarch
// scene snapshot.
fn create_snapshot(
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
        width.max(1),
        height.max(1),
        HDR_FORMAT,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::SampleCountFlags::TYPE_1,
    )?;
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
            1,
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

// Upload one panel's static quad VB + IB (host-visible, written once) and its
// per-panel `GlassParams` UBO; allocate + write the panel's descriptor set.
type PanelBuffers = (
    vk::Buffer,
    vk::DeviceMemory,
    vk::Buffer,
    vk::DeviceMemory,
    u32,
);
fn build_panel_buffers(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    panel: &GlassPanel,
) -> Result<PanelBuffers, String> {
    let (verts, idxs) = build_glass_quad(panel.centre, panel.normal, panel.half_size);

    // Flatten into the standard engine `Vertex` layout. Tangent is a
    // placeholder (the glass shader rebuilds its frame from the panel normal)
    // and per-vertex colour is unused.
    let mut packed: Vec<Vertex> = Vec::with_capacity(verts.len());
    for (pos, normal, color, uv) in verts {
        packed.push(Vertex {
            pos,
            normal,
            tangent: [1.0, 0.0, 0.0],
            color,
            uv,
        });
    }

    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let vb_bytes = std::mem::size_of_val(packed.as_slice()) as u64;
    let ib_bytes = std::mem::size_of_val(idxs.as_slice()) as u64;
    let (vb, vb_mem) = create_buffer(
        instance,
        device,
        physical_device,
        vb_bytes,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        host,
    )?;
    let (ib, ib_mem) = create_buffer(
        instance,
        device,
        physical_device,
        ib_bytes,
        vk::BufferUsageFlags::INDEX_BUFFER,
        host,
    )?;
    unsafe {
        let p = device
            .map_memory(vb_mem, 0, vb_bytes, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("glass vb map: {e}"))?;
        std::ptr::copy_nonoverlapping(
            packed.as_ptr() as *const u8,
            p as *mut u8,
            vb_bytes as usize,
        );
        device.unmap_memory(vb_mem);

        let p = device
            .map_memory(ib_mem, 0, ib_bytes, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("glass ib map: {e}"))?;
        std::ptr::copy_nonoverlapping(idxs.as_ptr() as *const u8, p as *mut u8, ib_bytes as usize);
        device.unmap_memory(ib_mem);
    }
    Ok((vb, vb_mem, ib, ib_mem, idxs.len() as u32))
}

impl GlassResources {
    // Build the glass pipeline + per-panel quad buffers + per-panel uniform
    // UBOs + the per-frame view ring + the scene snapshot + the per-frame
    // framebuffers. Called from `VkContext::new` when the world declares any
    // `GlassPanel`. `scene_views` / `scene_images` are the post-SSR scene
    // target per frame slot (SSR output repeated, or `hdr_resolve_images[i]`);
    // `depth_views` are the per-frame main-depth views the manual occlusion
    // test samples.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vulkan) fn new(
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        frames: usize,
        msaa_samples: vk::SampleCountFlags,
        width: u32,
        height: u32,
        scene_views: &[vk::ImageView],
        scene_images: &[vk::Image],
        depth_views: &[vk::ImageView],
        sampler: vk::Sampler,
        panels: &[GlassPanel],
        hot_reload: bool,
    ) -> Result<Self, String> {
        let msaa = msaa_samples != vk::SampleCountFlags::TYPE_1;
        let render_pass = create_glass_render_pass(device, HDR_FORMAT)?;
        let view_set_layout = create_view_set_layout(device)?;
        let params_set_layout = create_params_set_layout(device)?;
        let set_layouts = [view_set_layout, params_set_layout];
        let pipeline_layout = {
            let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            unsafe { device.create_pipeline_layout(&info, None) }
                .map_err(|e| format!("glass pipeline layout: {e}"))?
        };

        let (vert_spv, frag_spv) = compile_glass_shaders(hot_reload, msaa)?;
        let pipeline = create_pipeline(device, render_pass, pipeline_layout, &vert_spv, &frag_spv)?;

        let snapshot = create_snapshot(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            width,
            height,
        )?;

        // Per-frame view UBO ring (HOST_VISIBLE | HOST_COHERENT, mapped).
        let view_size = std::mem::size_of::<TransparentView>() as u64;
        let mut view_ubos = Vec::with_capacity(frames);
        let mut view_ubo_memories = Vec::with_capacity(frames);
        let mut view_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let (buf, mem) = create_buffer(
                instance,
                device,
                physical_device,
                view_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let ptr = unsafe { device.map_memory(mem, 0, view_size, vk::MemoryMapFlags::empty()) }
                .map_err(|e| format!("map glass view ubo: {e}"))? as *mut u8;
            view_ubos.push(buf);
            view_ubo_memories.push(mem);
            view_ubo_ptrs.push(ptr);
        }

        let descriptor_pool = create_descriptor_pool(device, frames, panels.len())?;
        let view_layouts: Vec<_> = (0..frames).map(|_| view_set_layout).collect();
        let view_sets = alloc_sets(device, descriptor_pool, &view_layouts)?;
        for (i, &set) in view_sets.iter().enumerate() {
            write_view_set(
                device,
                set,
                view_ubos[i],
                snapshot.view,
                depth_views[i.min(depth_views.len().saturating_sub(1))],
                sampler,
            );
        }

        // Per-frame framebuffers targeting the scene image for that slot.
        let framebuffers = create_framebuffers(device, render_pass, scene_views, width, height)?;

        // Per-panel records: quad buffers + static params UBO + descriptor set.
        let mut records: Vec<GlassPanelRecord> = Vec::with_capacity(panels.len());
        for panel in panels {
            let (vertex_buffer, vertex_memory, index_buffer, index_memory, index_count) =
                build_panel_buffers(instance, device, physical_device, panel)?;

            let params = glass_params_from(panel);
            let (params_ubo, params_ubo_memory) = create_buffer(
                instance,
                device,
                physical_device,
                std::mem::size_of::<GlassParams>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            unsafe {
                let p = device
                    .map_memory(
                        params_ubo_memory,
                        0,
                        std::mem::size_of::<GlassParams>() as u64,
                        vk::MemoryMapFlags::empty(),
                    )
                    .map_err(|e| format!("map glass params ubo: {e}"))?;
                std::ptr::copy_nonoverlapping(
                    &params as *const GlassParams as *const u8,
                    p as *mut u8,
                    std::mem::size_of::<GlassParams>(),
                );
                device.unmap_memory(params_ubo_memory);
            }
            let params_set = alloc_sets(device, descriptor_pool, &[params_set_layout])?[0];
            write_params_set(device, params_set, params_ubo);

            records.push(GlassPanelRecord {
                vertex_buffer,
                vertex_memory,
                index_buffer,
                index_memory,
                index_count,
                params_ubo,
                params_ubo_memory,
                params_set,
                visible: panel.visible,
                centre: panel.centre,
            });
        }

        Ok(Self {
            render_pass,
            pipeline,
            pipeline_layout,
            view_set_layout,
            params_set_layout,
            descriptor_pool,
            view_ubos,
            view_ubo_memories,
            view_ubo_ptrs,
            view_sets,
            scene_images: scene_images.to_vec(),
            framebuffers,
            snapshot,
            sampler,
            panels: records,
        })
    }

    // True when any panel is currently visible. Drives
    // `FrameGraphInputs::transparent_enabled` and the encoder early-out.
    pub(in crate::vulkan) fn any_visible(&self) -> bool {
        self.panels.iter().any(|p| p.visible)
    }

    // Recreate the scene snapshot + per-frame framebuffers at new render dims +
    // re-point the snapshot (binding 1) and per-frame depth (binding 2) of every
    // view set. The pipeline, layouts, UBOs, panel buffers, and render pass all
    // survive. Called from the swapchain-resize handler after the SSR / HDR
    // resolve targets have been rebuilt (so `scene_views` / `scene_images` carry
    // the new handles).
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
        scene_views: &[vk::ImageView],
        scene_images: &[vk::Image],
        depth_views: &[vk::ImageView],
    ) -> Result<(), String> {
        let old = std::mem::replace(
            &mut self.snapshot,
            create_snapshot(
                instance,
                device,
                physical_device,
                command_pool,
                queue,
                width,
                height,
            )?,
        );
        old.destroy(device);

        unsafe {
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
        }
        self.framebuffers =
            create_framebuffers(device, self.render_pass, scene_views, width, height)?;
        self.scene_images = scene_images.to_vec();

        for (i, &set) in self.view_sets.iter().enumerate() {
            write_view_set(
                device,
                set,
                self.view_ubos[i],
                self.snapshot.view,
                depth_views[i.min(depth_views.len().saturating_sub(1))],
                self.sampler,
            );
        }
        Ok(())
    }

    // Destroy every owned GPU resource. The `sampler` is borrowed from
    // `VkContext` and is not destroyed here.
    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        unsafe {
            for p in &self.panels {
                device.destroy_buffer(p.vertex_buffer, None);
                device.free_memory(p.vertex_memory, None);
                device.destroy_buffer(p.index_buffer, None);
                device.free_memory(p.index_memory, None);
                device.destroy_buffer(p.params_ubo, None);
                device.free_memory(p.params_ubo_memory, None);
            }
            for (&buf, &mem) in self.view_ubos.iter().zip(self.view_ubo_memories.iter()) {
                device.unmap_memory(mem);
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            self.snapshot.destroy(device);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.view_set_layout, None);
            device.destroy_descriptor_set_layout(self.params_set_layout, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);
        }
        self.panels.clear();
        self.view_ubos.clear();
        self.view_ubo_memories.clear();
        self.view_ubo_ptrs.clear();
        self.framebuffers.clear();
        self.scene_images.clear();
    }
}

// One framebuffer per frame slot, each binding that slot's scene image view as
// the sole colour attachment.
fn create_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    scene_views: &[vk::ImageView],
    width: u32,
    height: u32,
) -> Result<Vec<vk::Framebuffer>, String> {
    let mut out = Vec::with_capacity(scene_views.len());
    for &view in scene_views {
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(std::slice::from_ref(&view))
            .width(width.max(1))
            .height(height.max(1))
            .layers(1);
        let fb = unsafe { device.create_framebuffer(&info, None) }
            .map_err(|e| format!("glass framebuffer: {e}"))?;
        out.push(fb);
    }
    Ok(out)
}

impl VkContext {
    // Assemble the per-frame transparent view from the frame's jittered VP (the
    // matrix the main pass rasterised the depth buffer with, so the glass quad's
    // clip-space depth matches the stored main-depth) + camera position. Mirrors
    // `directx::graph_exec::build_transparent_view`.
    pub(in crate::vulkan) fn build_transparent_view(
        &self,
        vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        time: f32,
    ) -> TransparentView {
        TransparentView {
            vp,
            inv_vp: super::math::mat4_inverse(vp),
            camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], 0.0],
            viewport: [
                self.render_extent.width as f32,
                self.render_extent.height as f32,
            ],
            time,
            _pad: 0.0,
        }
    }

    // Encode the transparent (glass) pass. Runs after `SsrResolve` and before
    // `TaaResolve` / `Upscale`. Snapshots the post-SSR scene into `snapshot`
    // for refractive taps, then draws every visible panel back-to-front into the
    // scene image with SRC_ALPHA blending; the manual occlusion test samples the
    // main depth. No-op when no glass / no visible panels. Leaves the scene image
    // SHADER_READ_ONLY and the main depth DEPTH_STENCIL_ATTACHMENT_OPTIMAL for the
    // downstream stack.
    pub(in crate::vulkan) fn encode_transparent(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        view: &TransparentView,
    ) -> Result<(), String> {
        let Some(glass) = self.glass.as_ref() else {
            return Ok(());
        };
        let cam = [view.camera_pos[0], view.camera_pos[1], view.camera_pos[2]];
        let centres: Vec<[f32; 3]> = glass.panels.iter().map(|p| p.centre).collect();
        let visible: Vec<bool> = glass.panels.iter().map(|p| p.visible).collect();
        let order = ordered_visible(&centres, &visible, cam);
        if order.is_empty() {
            return Ok(());
        }

        let device = &self.device;
        let extent = self.render_extent;
        let scene_image = *glass
            .scene_images
            .get(frame_idx)
            .ok_or("glass: scene image index OOB")?;
        let snapshot = glass.snapshot.image;
        let depth_image = self.depth_images[frame_idx].image;

        // Upload this frame's view UBO.
        let view_ptr = *glass
            .view_ubo_ptrs
            .get(frame_idx)
            .ok_or("glass: view_ubo_ptrs index OOB")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                view as *const TransparentView as *const u8,
                view_ptr,
                std::mem::size_of::<TransparentView>(),
            );
        }

        let color_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let depth_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let color_barrier = |image: vk::Image,
                             old: vk::ImageLayout,
                             new: vk::ImageLayout,
                             src: vk::AccessFlags,
                             dst: vk::AccessFlags| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range)
        };

        // 1) Open the scene image + snapshot for the refraction snapshot copy.
        // The src scopes order the scene's last writer (SSR resolve / particles
        // colour write) and the prior frame's snapshot read ahead of the
        // transfer.
        let scene_to_src = color_barrier(
            scene_image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_READ,
        );
        let snapshot_to_dst = color_barrier(
            snapshot,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[scene_to_src, snapshot_to_dst],
            );
            let region = vk::ImageCopy::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                });
            device.cmd_copy_image(
                cmd,
                scene_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                snapshot,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );
        }

        // 2) Close the snapshot for the fragment read, restore the scene image
        // to SHADER_READ_ONLY (so the render pass's colour LOAD matches its
        // declared initial layout), and flip the main depth to SHADER_READ_ONLY
        // for the manual occlusion test.
        let snapshot_to_read = color_barrier(
            snapshot,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        let scene_to_read = color_barrier(
            scene_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::COLOR_ATTACHMENT_READ,
        );
        let depth_to_read = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(depth_image)
            .subresource_range(depth_range);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[snapshot_to_read, scene_to_read, depth_to_read],
            );
        }

        // 3) The render pass: LOAD the scene colour, draw each visible panel
        // back-to-front, STORE. The negative-height viewport matches the main
        // pass so the manual depth test + refraction taps line up at pixel
        // coordinates.
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(glass.render_pass)
            .framebuffer(glass.framebuffers[frame_idx])
            .render_area(vk::Rect2D::default().extent(extent));
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
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, glass.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                glass.pipeline_layout,
                0,
                std::slice::from_ref(&glass.view_sets[frame_idx]),
                &[],
            );
            for &i in &order {
                let p = &glass.panels[i];
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    glass.pipeline_layout,
                    1,
                    std::slice::from_ref(&p.params_set),
                    &[],
                );
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    std::slice::from_ref(&p.vertex_buffer),
                    &[0],
                );
                device.cmd_bind_index_buffer(cmd, p.index_buffer, 0, vk::IndexType::UINT16);
                device.cmd_draw_indexed(cmd, p.index_count, 1, 0, 0, 0);
                self.inc_draw_calls(1);
            }
            device.cmd_end_render_pass(cmd);
        }

        // 4) Restore the main depth to DEPTH_STENCIL_ATTACHMENT for the next
        // frame's main pass. The scene image already rests in SHADER_READ_ONLY
        // (render-pass final layout) for TAA / bloom / composite.
        let depth_to_attach = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(depth_image)
            .subresource_range(depth_range);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&depth_to_attach),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // The GLSL `TransparentViewBlock` std140 layout is 160 bytes; pin both the
    // size and every field offset so a Rust-side reorder fails the suite
    // without a GPU (mirrors the render_types `*_layout_matches_*` guards).
    #[test]
    fn transparent_view_layout_matches_glsl() {
        assert_eq!(size_of::<TransparentView>(), 160);
        assert_eq!(offset_of!(TransparentView, vp), 0);
        assert_eq!(offset_of!(TransparentView, inv_vp), 64);
        assert_eq!(offset_of!(TransparentView, camera_pos), 128);
        assert_eq!(offset_of!(TransparentView, viewport), 144);
        assert_eq!(offset_of!(TransparentView, time), 152);
        assert_eq!(offset_of!(TransparentView, _pad), 156);
    }

    // The GLSL `GlassParamsBlock` std140 layout is 64 bytes.
    #[test]
    fn glass_params_layout_matches_glsl() {
        assert_eq!(size_of::<GlassParams>(), 64);
        assert_eq!(offset_of!(GlassParams, centre), 0);
        assert_eq!(offset_of!(GlassParams, normal), 16);
        assert_eq!(offset_of!(GlassParams, tint), 32);
        assert_eq!(offset_of!(GlassParams, opacity), 48);
        assert_eq!(offset_of!(GlassParams, refraction_strength), 52);
        assert_eq!(offset_of!(GlassParams, fresnel_power), 56);
        assert_eq!(offset_of!(GlassParams, _pad1), 60);
    }

    #[test]
    fn glass_params_from_maps_fields() {
        let panel = GlassPanel {
            centre: [1.0, 2.0, 3.0],
            normal: [0.0, 0.0, 1.0],
            tint: [0.6, 0.85, 0.9],
            opacity: 0.45,
            refraction_strength: 0.04,
            fresnel_power: 4.0,
            ..Default::default()
        };
        let p = glass_params_from(&panel);
        assert_eq!(p.centre, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(p.normal, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(p.tint, [0.6, 0.85, 0.9, 0.0]);
        assert_eq!(p.opacity, 0.45);
        assert_eq!(p.refraction_strength, 0.04);
        assert_eq!(p.fresnel_power, 4.0);
        assert_eq!(p._pad1, 0.0);
    }

    #[test]
    fn sort_distance_is_euclidean_and_monotone() {
        let cam = [0.0, 0.0, 0.0];
        let near = sort_distance([0.0, 0.0, 1.0], cam);
        let far = sort_distance([0.0, 0.0, 5.0], cam);
        assert!((near - 1.0).abs() < 1e-5);
        assert!((far - 5.0).abs() < 1e-5);
        assert!(far > near);
    }

    #[test]
    fn ordered_visible_excludes_hidden_and_sorts_back_to_front() {
        // Panel 1 is hidden; 0 (dist 5) and 2 (dist 3) are visible. Farthest
        // first => [0, 2]; the hidden panel never appears.
        let centres = [[0.0, 0.0, 5.0], [0.0, 0.0, 9.0], [0.0, 0.0, 3.0]];
        let visible = [true, false, true];
        let order = ordered_visible(&centres, &visible, [0.0, 0.0, 0.0]);
        assert_eq!(order, vec![0, 2]);
    }

    // Compile the glass vertex + fragment shaders (both MSAA variants) so a GLSL
    // regression fails the suite without a GPU. Mirrors the decal / fog compile
    // guards.
    #[test]
    fn glass_shaders_compile() {
        super::compile_glass_shaders(false, true).expect("glass shaders compile (msaa)");
        super::compile_glass_shaders(false, false).expect("glass shaders compile (no msaa)");
    }
}
