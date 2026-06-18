// src/vulkan/raymarch.rs
//
// Raymarched SDF volume pass for the Vulkan backend. Runs at `PassId::Raymarch`
// between `AutoExposure` and `Decals` on the hdr_resolve RMW chain. Each
// `SdfVolume` rasterises the back faces of its world-space bounding box and runs
// a user-authored GLSL fragment shader that sphere-traces the SDF inside the
// box. GLSL/Vulkan port of `src/directx/raymarch.rs`: same shader interface,
// same depth-compositing rules.
//
// MSAA depth write-back. MSAA is on by default, so the pass renders the proxy
// into the multisampled HDR colour + the writable scene depth (the shader
// writes hit depth via `gl_FragDepth` redeclared `depth_less`), then the render
// pass resolves the combined colour into `hdr_resolve` so the single-sample
// post stack picks up the raymarched pixels and the raymarched-surface depth.
// This reuses the two-pass-occlusion main render passes (`load = false` STOREs
// the MSAA colour at the main pass so this pass can `load = true` it back), and
// the existing main framebuffers, which are render-pass-compatible. The main
// pass selects the STORE-colour variant whenever raymarch is active (see
// `vulkan/main.rs`). When single-sampled the main pass already leaves the scene
// in `hdr_resolve`, so the pass loads it directly and re-stores it (no resolve).
//
// Backend filter. The asset's `fragment_shader` path picks the backend: Vulkan
// consumes `.glsl` payloads; `.metal` / `.hlsl` SDFs are skipped at init with a
// logged warning and the rest of the world renders unchanged.

use std::ffi::CString;

use ash::{Device, vk};

use crate::assets::SdfVolume;
use crate::assets::sdf_volume::SDF_PARAMS_LEN;
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::{LightUniforms, ShadowUniforms};

use super::context::{HDR_FORMAT, VkContext};
use super::pipeline::{compile_glsl, shader_source, spv_module};
use super::render_pass::create_main_render_pass_two_pass;
use super::texture::{
    GpuImage, create_buffer, create_image, create_image_view, one_shot_submit,
    transition_image_layout_range,
};

// Engine-shipped GLSL: the proxy vertex shader (standalone) and the helpers +
// opaque template the per-volume user fragment shader is sandwiched between.
const RAYMARCH_PROXY_VERT: &str = include_str!("shaders/raymarch_proxy.vert");
const RAYMARCH_HELPERS_GLSL: &str = include_str!("shaders/raymarch_helpers.glsl");
const RAYMARCH_TEMPLATE_GLSL: &str = include_str!("shaders/raymarch_template.glsl");

// Volumetric template: appended after the helpers + the user's `sampleVolume`
// for `volumetric` volumes (participating media). Marches the bounding box and
// integrates Beer-Lambert transmittance instead of sphere-tracing a surface.
const RAYMARCH_VOLUMETRIC_TEMPLATE_GLSL: &str =
    include_str!("shaders/raymarch_volumetric_template.glsl");

// Depth-only shadow-caster shaders: a proxy vertex that projects through the
// active cascade's light VP, and the depth-only fragment template appended after
// the helpers + the user's `map` / `shade`.
const RAYMARCH_SHADOW_PROXY_VERT: &str = include_str!("shaders/raymarch_shadow_proxy.vert");
const RAYMARCH_SHADOW_TEMPLATE_GLSL: &str = include_str!("shaders/raymarch_shadow_template.glsl");

// 36 indices for the unit-cube proxy: front faces are culled so each pixel
// inside the bounding box gets exactly one back-face fragment.
const CUBE_INDEX_COUNT: u32 = 36;

// Per-frame view UBO bound at set 0 binding 0. Layout matches the
// `RaymarchViewBlock` std140 block in `shaders/raymarch_helpers.glsl` and the
// DirectX / Metal `RaymarchView`. 160 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub(in crate::vulkan) struct RaymarchView {
    pub(in crate::vulkan) vp: [[f32; 4]; 4],
    pub(in crate::vulkan) inv_vp: [[f32; 4]; 4],
    pub(in crate::vulkan) cam_pos: [f32; 3],
    pub(in crate::vulkan) _pad0: f32,
    pub(in crate::vulkan) viewport: [f32; 2],
    pub(in crate::vulkan) time: f32,
    pub(in crate::vulkan) prefilter_mip_count: f32,
}

// Per-volume UBO bound at set 1 binding 0. Layout matches the `SdfVolumeBlock`
// std140 block + the DirectX `RaymarchVolumeUniforms`. 176 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub(in crate::vulkan) struct RaymarchVolumeUniforms {
    pub(in crate::vulkan) centre: [f32; 3],
    pub(in crate::vulkan) _pad0: f32,
    pub(in crate::vulkan) extent: [f32; 3],
    pub(in crate::vulkan) _pad1: f32,
    pub(in crate::vulkan) cone_ratio: f32,
    pub(in crate::vulkan) max_distance: f32,
    pub(in crate::vulkan) max_steps: i32,
    pub(in crate::vulkan) receive_shadows: i32,
    pub(in crate::vulkan) params: [f32; SDF_PARAMS_LEN],
}

pub(in crate::vulkan) fn volume_uniforms_from(v: &SdfVolume) -> RaymarchVolumeUniforms {
    RaymarchVolumeUniforms {
        centre: v.centre,
        _pad0: 0.0,
        extent: v.extent,
        _pad1: 0.0,
        cone_ratio: v.cone_ratio(),
        max_distance: v.max_distance,
        max_steps: v.max_steps as i32,
        receive_shadows: if v.receive_shadows { 1 } else { 0 },
        params: v.params,
    }
}

// Per-`SdfVolume` GPU state: the compiled render pipeline, the static per-volume
// UBO (uploaded once at init), its descriptor set, and the visibility flag the
// encoder + `any_visible` read.
struct RaymarchVolumeRecord {
    pipeline: vk::Pipeline,
    // Depth-only shadow-caster pipeline. `Some` when the asset's `cast_shadows`
    // is set; the shadow encoder iterates only the volumes where this is `Some`
    // and `visible`. Targets `shadow_render_pass`.
    shadow_pipeline: Option<vk::Pipeline>,
    volume_ubo: vk::Buffer,
    volume_ubo_memory: vk::DeviceMemory,
    volume_set: vk::DescriptorSet,
    visible: bool,
}

// Engine-side raymarch resources. Built only when at least one `.glsl`
// `SdfVolume` landed at init; `VkContext::raymarch` stays `None` otherwise and
// the pass is omitted from the frame graph.
pub(in crate::vulkan) struct RaymarchResources {
    // The raymarch render pass. MSAA: the two-pass `load = true` main pass
    // (loads the stored MSAA colour + scene depth, draws, resolves into
    // hdr_resolve). Single-sample: a dedicated load+store pass on hdr_resolve.
    render_pass: vk::RenderPass,
    // STORE-colour main render pass the main pass switches to while raymarch is
    // active (MSAA only) so the MSAA samples survive for `render_pass` to load.
    // `None` when single-sampled (the main pass already keeps the resolve).
    pub(in crate::vulkan) main_store_color_pass: Option<vk::RenderPass>,
    pipeline_layout: vk::PipelineLayout,
    view_set_layout: vk::DescriptorSetLayout,
    volume_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,

    // Per-frame `RaymarchView` UBO ring. Persistently mapped; the encoder
    // memcpys this frame's view into `view_ubo_ptrs[frame_idx]` before binding.
    view_ubos: Vec<vk::Buffer>,
    view_ubo_memories: Vec<vk::DeviceMemory>,
    view_ubo_ptrs: Vec<*mut u8>,
    view_sets: Vec<vk::DescriptorSet>,

    // Shared unit-cube proxy geometry (positions at +/-1; the vertex shader
    // scales by `vol_extent` + offsets by `vol_centre`).
    cube_vb: vk::Buffer,
    cube_vb_memory: vk::DeviceMemory,
    cube_ib: vk::Buffer,
    cube_ib_memory: vk::DeviceMemory,

    // Pre-raymarch HDR scene snapshot for the refraction tap (`scene_color`).
    // The encoder copies hdr_resolve into this at the head of the pass; sized to
    // render dims, recreated by `rebuild` on resize.
    snapshot: GpuImage,
    // Sampler bound alongside the snapshot at set 0 binding 6. Borrowed from
    // `VkContext::linear_sampler`; not owned, never destroyed here.
    scene_sampler: vk::Sampler,

    // Shadow-caster resources. Built only when at least one volume opts into
    // `cast_shadows`; null / empty otherwise. The shadow view set is a minimal
    // 3-UBO set (RaymarchView for `view_time`, lights, shadow VPs) with its own
    // per-frame `RaymarchView` ring written by the shadow pass, so the shared
    // main view ring (written by `encode_raymarch`) is never touched from the
    // concurrently-recorded Shadow pass. The pipeline layout carries a
    // `cascade_idx` push constant.
    shadow_pipeline_layout: vk::PipelineLayout,
    shadow_view_set_layout: vk::DescriptorSetLayout,
    shadow_view_ubos: Vec<vk::Buffer>,
    shadow_view_ubo_memories: Vec<vk::DeviceMemory>,
    shadow_view_ubo_ptrs: Vec<*mut u8>,
    shadow_view_sets: Vec<vk::DescriptorSet>,

    msaa: bool,
    volumes: Vec<RaymarchVolumeRecord>,
}

// Push constant for the shadow-caster pipeline: which CSM cascade this draw
// targets, selecting `light_vps[cascade_idx]` in the proxy vertex + the fragment
// reprojection. Matches the `ShadowCascade` push block in the shadow shaders.
#[derive(Copy, Clone)]
#[repr(C)]
struct ShadowCascadePush {
    cascade_idx: u32,
}

// Assemble the per-volume fragment source: engine helpers, then the user's
// `map` / `shade`, then the opaque template's `main`. The single `#version`
// lives at the top of the helpers; the user source + template must not declare
// their own. Mirrors the DirectX `wrap_user_source` (helpers -> user ->
// template), adapted for GLSL's separate vertex/fragment compilation.
fn wrap_user_fragment(user_source: &str, hot_reload: bool) -> String {
    let helpers = shader_source(hot_reload, "raymarch_helpers.glsl", RAYMARCH_HELPERS_GLSL);
    let template = shader_source(hot_reload, "raymarch_template.glsl", RAYMARCH_TEMPLATE_GLSL);
    format!(
        "{helpers}\n// === user SdfVolume fragment shader ===\n{user_source}\n// === engine raymarch template ===\n{template}\n"
    )
}

// Compile the proxy vertex shader + the assembled per-volume fragment shader to
// SPIR-V. Returns `(vertex_spv, fragment_spv)`.
fn compile_raymarch_shaders(
    user_source: &str,
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let vert_src = shader_source(hot_reload, "raymarch_proxy.vert", RAYMARCH_PROXY_VERT);
    let vert = compile_glsl(
        &vert_src,
        shaderc::ShaderKind::Vertex,
        "raymarch_proxy.vert",
    )?;
    let frag_src = wrap_user_fragment(user_source, hot_reload);
    let frag = compile_glsl(
        &frag_src,
        shaderc::ShaderKind::Fragment,
        "raymarch_fragment",
    )?;
    Ok((vert, frag))
}

// Assemble the per-volume fragment source for a volumetric volume: engine
// helpers, then the user's `sampleVolume`, then the volumetric template's
// `main`. glslang prunes the unreachable surface helpers (`sdfNormal`,
// `coneRaymarch`) and their forward-declared `map` calls, so the volumetric
// author needs no surface stubs. Mirrors the DirectX `wrap_user_source_volumetric`.
fn wrap_user_fragment_volumetric(user_source: &str, hot_reload: bool) -> String {
    let helpers = shader_source(hot_reload, "raymarch_helpers.glsl", RAYMARCH_HELPERS_GLSL);
    let template = shader_source(
        hot_reload,
        "raymarch_volumetric_template.glsl",
        RAYMARCH_VOLUMETRIC_TEMPLATE_GLSL,
    );
    format!(
        "{helpers}\n// === user SdfVolume fragment shader (volumetric) ===\n{user_source}\n// === engine raymarch volumetric template ===\n{template}\n"
    )
}

// Compile the proxy vertex shader (shared with the surface pass) + the assembled
// volumetric fragment shader to SPIR-V. Returns `(vertex_spv, fragment_spv)`.
fn compile_raymarch_volumetric_shaders(
    user_source: &str,
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let vert_src = shader_source(hot_reload, "raymarch_proxy.vert", RAYMARCH_PROXY_VERT);
    let vert = compile_glsl(
        &vert_src,
        shaderc::ShaderKind::Vertex,
        "raymarch_proxy.vert",
    )?;
    let frag_src = wrap_user_fragment_volumetric(user_source, hot_reload);
    let frag = compile_glsl(
        &frag_src,
        shaderc::ShaderKind::Fragment,
        "raymarch_volumetric_fragment",
    )?;
    Ok((vert, frag))
}

// Assemble + compile the depth-only shadow-caster shaders: the standalone shadow
// proxy vertex + the (helpers + user + shadow template) fragment. spirv-opt DCEs
// the unused `shade` / IBL / scene-colour bindings, so the compiled fragment
// references only the view / lights / shadow UBOs.
fn compile_raymarch_shadow_shaders(
    user_source: &str,
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let vert_src = shader_source(
        hot_reload,
        "raymarch_shadow_proxy.vert",
        RAYMARCH_SHADOW_PROXY_VERT,
    );
    let vert = compile_glsl(
        &vert_src,
        shaderc::ShaderKind::Vertex,
        "raymarch_shadow_proxy.vert",
    )?;
    let helpers = shader_source(hot_reload, "raymarch_helpers.glsl", RAYMARCH_HELPERS_GLSL);
    let template = shader_source(
        hot_reload,
        "raymarch_shadow_template.glsl",
        RAYMARCH_SHADOW_TEMPLATE_GLSL,
    );
    let frag_src = format!(
        "{helpers}\n// === user SdfVolume fragment shader ===\n{user_source}\n// === engine raymarch shadow template ===\n{template}\n"
    );
    let frag = compile_glsl(
        &frag_src,
        shaderc::ShaderKind::Fragment,
        "raymarch_shadow_fragment",
    )?;
    Ok((vert, frag))
}

// One corner of the proxy cube. Only the position is fetched (location 0); the
// rest of the 56-byte engine `Vertex` is zeroed.
fn cube_vertex(pos: [f32; 3]) -> Vertex {
    Vertex {
        pos,
        normal: [0.0; 3],
        tangent: [0.0; 3],
        color: [0.0; 3],
        uv: [0.0; 2],
    }
}

// Build the shared unit-cube proxy VB + IB. 8 corners at +/-1; 36 indices (the
// pipeline culls front faces so only back faces fire). Host-visible buffers,
// written once. Mirrors `directx::raymarch::build_cube_buffers`.
type CubeBuffers = (vk::Buffer, vk::DeviceMemory, vk::Buffer, vk::DeviceMemory);
fn build_cube_buffers(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
) -> Result<CubeBuffers, String> {
    #[rustfmt::skip]
    let corners: [Vertex; 8] = [
        cube_vertex([-1.0, -1.0, -1.0]),
        cube_vertex([ 1.0, -1.0, -1.0]),
        cube_vertex([ 1.0,  1.0, -1.0]),
        cube_vertex([-1.0,  1.0, -1.0]),
        cube_vertex([-1.0, -1.0,  1.0]),
        cube_vertex([ 1.0, -1.0,  1.0]),
        cube_vertex([ 1.0,  1.0,  1.0]),
        cube_vertex([-1.0,  1.0,  1.0]),
    ];
    #[rustfmt::skip]
    let indices: [u16; 36] = [
        0, 2, 1,  0, 3, 2, // -Z
        4, 5, 6,  4, 6, 7, // +Z
        0, 4, 7,  0, 7, 3, // -X
        1, 2, 6,  1, 6, 5, // +X
        0, 1, 5,  0, 5, 4, // -Y
        3, 7, 6,  3, 6, 2, // +Y
    ];

    let vb_bytes = std::mem::size_of_val(&corners) as u64;
    let ib_bytes = std::mem::size_of_val(&indices) as u64;
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

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
            .map_err(|e| format!("raymarch cube vb map: {e}"))?;
        std::ptr::copy_nonoverlapping(
            corners.as_ptr() as *const u8,
            p as *mut u8,
            vb_bytes as usize,
        );
        device.unmap_memory(vb_mem);

        let p = device
            .map_memory(ib_mem, 0, ib_bytes, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("raymarch cube ib map: {e}"))?;
        std::ptr::copy_nonoverlapping(
            indices.as_ptr() as *const u8,
            p as *mut u8,
            ib_bytes as usize,
        );
        device.unmap_memory(ib_mem);
    }
    Ok((vb, vb_mem, ib, ib_mem))
}

// The single-sample raymarch render pass: load + store hdr_resolve directly
// (the main pass already left the scene there in SHADER_READ_ONLY) and the
// scene depth, with no resolve. The MSAA path reuses the two-pass main passes
// instead.
fn create_raymarch_render_pass_single(
    device: &Device,
    format: vk::Format,
) -> Result<vk::RenderPass, String> {
    let attachments = [
        vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref))
        .depth_stencil_attachment(&depth_ref);
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
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
        .dependencies(std::slice::from_ref(&dependency));
    unsafe { device.create_render_pass(&info, None) }
        .map_err(|e| format!("raymarch render pass: {e}"))
}

fn create_view_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
    let vert_frag = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let frag = vk::ShaderStageFlags::FRAGMENT;
    let ubo = |b: u32, stages: vk::ShaderStageFlags| {
        vk::DescriptorSetLayoutBinding::default()
            .binding(b)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(stages)
    };
    let tex = |b: u32| {
        vk::DescriptorSetLayoutBinding::default()
            .binding(b)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(frag)
    };
    let bindings = [
        ubo(0, vert_frag), // RaymarchView
        ubo(1, frag),      // RaymarchLights
        ubo(2, frag),      // RaymarchShadow
        tex(3),            // shadow_map (sampler2DArrayShadow)
        tex(4),            // irradiance cube
        tex(5),            // prefilter cube
        tex(6),            // scene_color snapshot
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("raymarch view set layout: {e}"))
}

fn create_volume_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
    let info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding));
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("raymarch volume set layout: {e}"))
}

fn create_descriptor_pool(
    device: &Device,
    frames: usize,
    volumes: usize,
    has_shadow: bool,
) -> Result<vk::DescriptorPool, String> {
    let f = frames as u32;
    let v = volumes as u32;
    // Shadow view sets (when any volume casts shadows): 3 UBOs each per frame.
    let shadow_sets = if has_shadow { f } else { 0 };
    let sizes = [
        // view: RaymarchView + Lights + Shadow (3) per frame; volume: 1 each;
        // shadow view: 3 per frame.
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 3 * f + v + 3 * shadow_sets,
        },
        // view: shadow_map + irradiance + prefilter + scene_color (4) per frame.
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 4 * f,
        },
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(f + v + shadow_sets)
        .pool_sizes(&sizes);
    unsafe { device.create_descriptor_pool(&info, None) }
        .map_err(|e| format!("raymarch descriptor pool: {e}"))
}

// Minimal 3-UBO descriptor set layout for the shadow-caster pass: RaymarchView
// (view_time), lights (sun direction), and the cascade light VPs. No texture
// bindings (the shadow march never samples), so the shadow map being written
// this pass is never also bound as a descriptor.
fn create_shadow_view_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
    let frag = vk::ShaderStageFlags::FRAGMENT;
    let vert_frag = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let ubo = |b: u32, stages: vk::ShaderStageFlags| {
        vk::DescriptorSetLayoutBinding::default()
            .binding(b)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(stages)
    };
    let bindings = [
        ubo(0, frag),      // RaymarchView (view_time)
        ubo(1, frag),      // RaymarchLights (sun direction)
        ubo(2, vert_frag), // RaymarchShadow (light VPs)
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("raymarch shadow view set layout: {e}"))
}

fn write_shadow_view_set(
    device: &Device,
    set: vk::DescriptorSet,
    view_ubo: vk::Buffer,
    light_ubo: vk::Buffer,
    shadow_ubo: vk::Buffer,
) {
    let view_info = vk::DescriptorBufferInfo::default()
        .buffer(view_ubo)
        .offset(0)
        .range(std::mem::size_of::<RaymarchView>() as u64);
    let light_info = vk::DescriptorBufferInfo::default()
        .buffer(light_ubo)
        .offset(0)
        .range(std::mem::size_of::<LightUniforms>() as u64);
    let shadow_info = vk::DescriptorBufferInfo::default()
        .buffer(shadow_ubo)
        .offset(0)
        .range(std::mem::size_of::<ShadowUniforms>() as u64);
    let ubo = |b: u32| {
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(b)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
    };
    let writes = [
        ubo(0).buffer_info(std::slice::from_ref(&view_info)),
        ubo(1).buffer_info(std::slice::from_ref(&light_info)),
        ubo(2).buffer_info(std::slice::from_ref(&shadow_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
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
        .map_err(|e| format!("raymarch descriptor sets: {e}"))
}

// Write every binding of one per-frame view set. The shadow / IBL / light /
// shadow-UBO bindings are shared engine resources; the snapshot is the
// per-frame-independent scene tap (its contents are refreshed by the encoder).
#[allow(clippy::too_many_arguments)]
fn write_view_set(
    device: &Device,
    set: vk::DescriptorSet,
    view_ubo: vk::Buffer,
    light_ubo: vk::Buffer,
    shadow_ubo: vk::Buffer,
    shadow_map_view: vk::ImageView,
    shadow_sampler: vk::Sampler,
    irradiance_view: vk::ImageView,
    prefilter_view: vk::ImageView,
    cube_sampler: vk::Sampler,
    snapshot_view: vk::ImageView,
    scene_sampler: vk::Sampler,
) {
    let view_info = vk::DescriptorBufferInfo::default()
        .buffer(view_ubo)
        .offset(0)
        .range(std::mem::size_of::<RaymarchView>() as u64);
    let light_info = vk::DescriptorBufferInfo::default()
        .buffer(light_ubo)
        .offset(0)
        .range(std::mem::size_of::<LightUniforms>() as u64);
    let shadow_info = vk::DescriptorBufferInfo::default()
        .buffer(shadow_ubo)
        .offset(0)
        .range(std::mem::size_of::<ShadowUniforms>() as u64);
    let img = |view: vk::ImageView, sampler: vk::Sampler| {
        vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(sampler)
    };
    let shadow_map_info = img(shadow_map_view, shadow_sampler);
    let irradiance_info = img(irradiance_view, cube_sampler);
    let prefilter_info = img(prefilter_view, cube_sampler);
    let snapshot_info = img(snapshot_view, scene_sampler);

    let ubo = |b: u32| {
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(b)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
    };
    let tex = |b: u32| {
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(b)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
    };
    let writes = [
        ubo(0).buffer_info(std::slice::from_ref(&view_info)),
        ubo(1).buffer_info(std::slice::from_ref(&light_info)),
        ubo(2).buffer_info(std::slice::from_ref(&shadow_info)),
        tex(3).image_info(std::slice::from_ref(&shadow_map_info)),
        tex(4).image_info(std::slice::from_ref(&irradiance_info)),
        tex(5).image_info(std::slice::from_ref(&prefilter_info)),
        tex(6).image_info(std::slice::from_ref(&snapshot_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn write_volume_set(device: &Device, set: vk::DescriptorSet, volume_ubo: vk::Buffer) {
    let info = vk::DescriptorBufferInfo::default()
        .buffer(volume_ubo)
        .offset(0)
        .range(std::mem::size_of::<RaymarchVolumeUniforms>() as u64);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

// Build a per-volume raymarch graphics pipeline. Front-face culled (back faces
// of the proxy cube rasterise regardless of camera position), depth-tested
// LESS_OR_EQUAL with depth write (the fragment writes `gl_FragDepth`), opaque
// (no blend). Negative-height viewport is applied dynamically at encode time.
fn create_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    msaa_samples: vk::SampleCountFlags,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert = spv_module(device, vert_spv)?;
    let frag = spv_module(device, frag_spv)?;
    let entry = CString::new("main").unwrap();
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

    // Cube proxy VB: the 56-byte engine `Vertex`, position (location 0) only.
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
    // Front-face cull. The main pass renders with a negative-height (Y-flipped)
    // viewport; under that flip the proxy's near faces wind CCW, so culling them
    // as the front face leaves the back faces to rasterise (matches the DirectX
    // CULL_FRONT path).
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::FRONT)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(msaa_samples);
    // Depth test against the existing scene depth; write hit depth (the fragment
    // overrides `gl_FragDepth`) so downstream passes see the raymarched surface.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
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
    .map_err(|(_, e)| format!("create raymarch pipeline: {e}"))?[0];
    unsafe {
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
    }
    Ok(pipeline)
}

// Build the volumetric variant of the per-volume pipeline. Same cube proxy +
// front cull as the opaque pass, but the colour output alpha-blends over the
// existing scene (SRC_ALPHA / ONE_MINUS_SRC_ALPHA) and the depth state keeps the
// LESS_OR_EQUAL early-z test without writing: the medium is translucent and
// never updates the depth buffer downstream passes read. Mirrors the DirectX
// `create_raymarch_volumetric_pso`.
fn create_volumetric_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    msaa_samples: vk::SampleCountFlags,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert = spv_module(device, vert_spv)?;
    let frag = spv_module(device, frag_spv)?;
    let entry = CString::new("main").unwrap();
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
        .cull_mode(vk::CullModeFlags::FRONT)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(msaa_samples);
    // Early-z against the existing scene depth, but no depth write: the medium
    // doesn't occlude itself or update SSR / decal depth.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
    // Alpha-blend the in-scattered luminance over the rasterised scene.
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
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
    .map_err(|(_, e)| format!("create raymarch volumetric pipeline: {e}"))?[0];
    unsafe {
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
    }
    Ok(pipeline)
}

// Build a per-volume depth-only shadow-caster pipeline. Same cube proxy + front
// cull as the main pass, but no colour attachment (single-sample shadow map),
// depth-test LESS (matching the rasterised CSM casters) with depth write, and
// the fragment writes hit depth via `gl_FragDepth`. Targets `shadow_render_pass`.
fn create_shadow_pipeline(
    device: &Device,
    shadow_render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert = spv_module(device, vert_spv)?;
    let frag = spv_module(device, frag_spv)?;
    let entry = CString::new("main").unwrap();
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
    // Same front-face cull as the main pass: under the shadow pass's
    // negative-height viewport the proxy's near faces wind CCW, so culling them
    // leaves the back faces to seed the from-light ray.
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::FRONT)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);
    // No colour attachment in the shadow render pass.
    let blend_state = vk::PipelineColorBlendStateCreateInfo::default().logic_op_enable(false);
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
        .render_pass(shadow_render_pass);
    let pipeline = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create raymarch shadow pipeline: {e}"))?[0];
    unsafe {
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
    }
    Ok(pipeline)
}

// Create the pre-raymarch HDR scene snapshot (SAMPLED | TRANSFER_DST, GPU-local)
// and rest it in SHADER_READ_ONLY so the first frame's snapshot barrier
// (SHADER_READ_ONLY -> TRANSFER_DST) matches.
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

impl RaymarchResources {
    // Build every raymarch resource + the per-volume records. `sdf_volumes` is
    // the drained-and-payload-paired list from `graphics_system::init`; each
    // volume's `fragment_shader` path is checked here: `.glsl` payloads compile,
    // anything else (Metal-first `.metal` / DirectX `.hlsl`) is skipped with a
    // logged warning. Returns `Ok(None)` when no volume survived the filter so
    // the engine omits the pass.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vulkan) fn try_new(
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        frames: usize,
        msaa_samples: vk::SampleCountFlags,
        width: u32,
        height: u32,
        shadow_map_view: vk::ImageView,
        shadow_sampler: vk::Sampler,
        irradiance_view: vk::ImageView,
        prefilter_view: vk::ImageView,
        cube_sampler: vk::Sampler,
        linear_sampler: vk::Sampler,
        light_ubo: vk::Buffer,
        shadow_ubo: vk::Buffer,
        shadow_render_pass: vk::RenderPass,
        sdf_volumes: &[(SdfVolume, Vec<u8>, String)],
        hot_reload: bool,
    ) -> Result<Option<Self>, String> {
        // Filter `.glsl` volumes; Metal/DirectX-first SDFs get dropped with a
        // warning so the rest of the world keeps rendering.
        let active: Vec<&(SdfVolume, Vec<u8>, String)> = sdf_volumes
            .iter()
            .filter(|(v, _, label)| {
                if v.fragment_shader.to_ascii_lowercase().ends_with(".glsl") {
                    true
                } else {
                    tracing::warn!(
                        "SdfVolume '{}': fragment shader '{}' is not .glsl; skipping on \
                         Vulkan (the rest of the world still renders)",
                        label,
                        v.fragment_shader
                    );
                    false
                }
            })
            .collect();
        if active.is_empty() {
            return Ok(None);
        }

        let msaa = msaa_samples != vk::SampleCountFlags::TYPE_1;
        let (render_pass, main_store_color_pass) = if msaa {
            (
                create_main_render_pass_two_pass(device, HDR_FORMAT, msaa_samples, true)?,
                Some(create_main_render_pass_two_pass(
                    device,
                    HDR_FORMAT,
                    msaa_samples,
                    false,
                )?),
            )
        } else {
            (
                create_raymarch_render_pass_single(device, HDR_FORMAT)?,
                None,
            )
        };

        let view_set_layout = create_view_set_layout(device)?;
        let volume_set_layout = create_volume_set_layout(device)?;
        let set_layouts = [view_set_layout, volume_set_layout];
        let pipeline_layout = {
            let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            unsafe { device.create_pipeline_layout(&info, None) }
                .map_err(|e| format!("raymarch pipeline layout: {e}"))?
        };

        let (cube_vb, cube_vb_memory, cube_ib, cube_ib_memory) =
            build_cube_buffers(instance, device, physical_device)?;

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
        let view_size = std::mem::size_of::<RaymarchView>() as u64;
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
                .map_err(|e| format!("map raymarch view ubo: {e}"))?
                as *mut u8;
            view_ubos.push(buf);
            view_ubo_memories.push(mem);
            view_ubo_ptrs.push(ptr);
        }

        let has_shadow = active.iter().any(|(v, _, _)| v.cast_shadows);
        let descriptor_pool = create_descriptor_pool(device, frames, active.len(), has_shadow)?;
        let view_layouts: Vec<_> = (0..frames).map(|_| view_set_layout).collect();
        let view_sets = alloc_sets(device, descriptor_pool, &view_layouts)?;
        for (i, &set) in view_sets.iter().enumerate() {
            write_view_set(
                device,
                set,
                view_ubos[i],
                light_ubo,
                shadow_ubo,
                shadow_map_view,
                shadow_sampler,
                irradiance_view,
                prefilter_view,
                cube_sampler,
                snapshot.view,
                linear_sampler,
            );
        }

        // Shadow-caster infrastructure: a minimal 3-UBO view set with its own
        // per-frame `RaymarchView` ring (written by the Shadow pass, never the
        // concurrently-recorded Raymarch pass) + a pipeline layout carrying the
        // `cascade_idx` push constant. Built only when a volume casts shadows.
        let mut shadow_pipeline_layout = vk::PipelineLayout::null();
        let mut shadow_view_set_layout = vk::DescriptorSetLayout::null();
        let mut shadow_view_ubos: Vec<vk::Buffer> = Vec::new();
        let mut shadow_view_ubo_memories: Vec<vk::DeviceMemory> = Vec::new();
        let mut shadow_view_ubo_ptrs: Vec<*mut u8> = Vec::new();
        let mut shadow_view_sets: Vec<vk::DescriptorSet> = Vec::new();
        if has_shadow {
            shadow_view_set_layout = create_shadow_view_set_layout(device)?;
            let set_layouts = [shadow_view_set_layout, volume_set_layout];
            let push = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size(std::mem::size_of::<ShadowCascadePush>() as u32);
            let info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(std::slice::from_ref(&push));
            shadow_pipeline_layout = unsafe { device.create_pipeline_layout(&info, None) }
                .map_err(|e| format!("raymarch shadow pipeline layout: {e}"))?;

            for _ in 0..frames {
                let (buf, mem) = create_buffer(
                    instance,
                    device,
                    physical_device,
                    view_size,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?;
                let ptr =
                    unsafe { device.map_memory(mem, 0, view_size, vk::MemoryMapFlags::empty()) }
                        .map_err(|e| format!("map raymarch shadow view ubo: {e}"))?
                        as *mut u8;
                shadow_view_ubos.push(buf);
                shadow_view_ubo_memories.push(mem);
                shadow_view_ubo_ptrs.push(ptr);
            }
            let shadow_layouts: Vec<_> = (0..frames).map(|_| shadow_view_set_layout).collect();
            shadow_view_sets = alloc_sets(device, descriptor_pool, &shadow_layouts)?;
            for (i, &set) in shadow_view_sets.iter().enumerate() {
                write_shadow_view_set(device, set, shadow_view_ubos[i], light_ubo, shadow_ubo);
            }
        }

        // Build per-volume records. A compile error in an active volume is a
        // developer-time bug, so it aborts init (unlike the .glsl filter above).
        let mut volumes: Vec<RaymarchVolumeRecord> = Vec::with_capacity(active.len());
        for (vol, bytes, label) in &active {
            let user_source = std::str::from_utf8(bytes).map_err(|e| {
                format!("SdfVolume '{label}': fragment shader payload is not valid UTF-8: {e}")
            })?;
            // Volumetric volumes (participating media) author `sampleVolume` and
            // render alpha-blended without a depth write; surface volumes author
            // `map` / `shade` and sphere-trace an opaque surface. The asset's
            // `volumetric` flag selects which template + pipeline state is built.
            let pipeline = if vol.volumetric {
                let (vert_spv, frag_spv) =
                    compile_raymarch_volumetric_shaders(user_source, hot_reload)
                        .map_err(|e| format!("SdfVolume '{label}' (volumetric): {e}"))?;
                create_volumetric_pipeline(
                    device,
                    render_pass,
                    pipeline_layout,
                    msaa_samples,
                    &vert_spv,
                    &frag_spv,
                )?
            } else {
                let (vert_spv, frag_spv) = compile_raymarch_shaders(user_source, hot_reload)
                    .map_err(|e| format!("SdfVolume '{label}': {e}"))?;
                create_pipeline(
                    device,
                    render_pass,
                    pipeline_layout,
                    msaa_samples,
                    &vert_spv,
                    &frag_spv,
                )?
            };

            // Depth-only shadow-caster pipeline when the asset opts in. The
            // shadow template is engine-shipped, so the only realistic compile
            // failure is a user `map` that already failed for the main pipeline.
            let shadow_pipeline = if vol.cast_shadows {
                let (sh_vert, sh_frag) =
                    compile_raymarch_shadow_shaders(user_source, hot_reload)
                        .map_err(|e| format!("SdfVolume '{label}' (shadow): {e}"))?;
                Some(create_shadow_pipeline(
                    device,
                    shadow_render_pass,
                    shadow_pipeline_layout,
                    &sh_vert,
                    &sh_frag,
                )?)
            } else {
                None
            };

            let uniforms = volume_uniforms_from(vol);
            let (volume_ubo, volume_ubo_memory) = create_buffer(
                instance,
                device,
                physical_device,
                std::mem::size_of::<RaymarchVolumeUniforms>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            unsafe {
                let p = device
                    .map_memory(
                        volume_ubo_memory,
                        0,
                        std::mem::size_of::<RaymarchVolumeUniforms>() as u64,
                        vk::MemoryMapFlags::empty(),
                    )
                    .map_err(|e| format!("map raymarch volume ubo: {e}"))?;
                std::ptr::copy_nonoverlapping(
                    &uniforms as *const RaymarchVolumeUniforms as *const u8,
                    p as *mut u8,
                    std::mem::size_of::<RaymarchVolumeUniforms>(),
                );
                device.unmap_memory(volume_ubo_memory);
            }
            let volume_set = alloc_sets(device, descriptor_pool, &[volume_set_layout])?[0];
            write_volume_set(device, volume_set, volume_ubo);

            volumes.push(RaymarchVolumeRecord {
                pipeline,
                shadow_pipeline,
                volume_ubo,
                volume_ubo_memory,
                volume_set,
                visible: vol.visible,
            });
        }

        Ok(Some(Self {
            render_pass,
            main_store_color_pass,
            pipeline_layout,
            view_set_layout,
            volume_set_layout,
            descriptor_pool,
            view_ubos,
            view_ubo_memories,
            view_ubo_ptrs,
            view_sets,
            cube_vb,
            cube_vb_memory,
            cube_ib,
            cube_ib_memory,
            snapshot,
            scene_sampler: linear_sampler,
            shadow_pipeline_layout,
            shadow_view_set_layout,
            shadow_view_ubos,
            shadow_view_ubo_memories,
            shadow_view_ubo_ptrs,
            shadow_view_sets,
            msaa,
            volumes,
        }))
    }

    // True when any volume in the world is currently visible. Drives
    // `FrameGraphInputs::raymarch_enabled` and the encoder early-out.
    pub(in crate::vulkan) fn any_visible(&self) -> bool {
        self.volumes.iter().any(|v| v.visible)
    }

    // True when at least one visible volume opted into shadow casting (so its
    // `shadow_pipeline` was built). Gates the shadow-pass injection.
    fn any_shadow_casters(&self) -> bool {
        self.volumes
            .iter()
            .any(|v| v.visible && v.shadow_pipeline.is_some())
    }

    // Recreate the scene snapshot at new render dims + re-point the `scene_color`
    // binding of every view set. Called from the swapchain-resize handler; the
    // pipelines, layouts, UBOs, cube buffers, and render passes all survive.
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
        for &set in &self.view_sets {
            let info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.snapshot.view)
                .sampler(self.scene_sampler);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&info));
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }
        Ok(())
    }

    // Re-point the irradiance + prefilter cube bindings (set 0 bindings 4 + 5)
    // of every view set after an EnvironmentMap hot-reload swapped the IBL
    // cubes for fresh image views. The light / shadow / snapshot bindings and
    // the cube-less shadow-caster view sets are left untouched. Reached only
    // through the bin's `cn debug` env-map hot-reload path (dead in the FFI
    // lib, live in the bin).
    #[allow(dead_code)]
    pub(in crate::vulkan) fn rewire_ibl_cubes(
        &self,
        device: &Device,
        irradiance_view: vk::ImageView,
        prefilter_view: vk::ImageView,
        cube_sampler: vk::Sampler,
    ) {
        let irr_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(irradiance_view)
            .sampler(cube_sampler);
        let pre_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(prefilter_view)
            .sampler(cube_sampler);
        for &set in &self.view_sets {
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&irr_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&pre_info)),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    // Destroy every owned GPU resource. The `scene_sampler` is borrowed from
    // `VkContext` and is not destroyed here.
    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        unsafe {
            for vol in &self.volumes {
                device.destroy_pipeline(vol.pipeline, None);
                if let Some(p) = vol.shadow_pipeline {
                    device.destroy_pipeline(p, None);
                }
                device.destroy_buffer(vol.volume_ubo, None);
                device.free_memory(vol.volume_ubo_memory, None);
            }
            for (&buf, &mem) in self.view_ubos.iter().zip(self.view_ubo_memories.iter()) {
                device.unmap_memory(mem);
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
            for (&buf, &mem) in self
                .shadow_view_ubos
                .iter()
                .zip(self.shadow_view_ubo_memories.iter())
            {
                device.unmap_memory(mem);
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
            self.snapshot.destroy(device);
            device.destroy_buffer(self.cube_vb, None);
            device.free_memory(self.cube_vb_memory, None);
            device.destroy_buffer(self.cube_ib, None);
            device.free_memory(self.cube_ib_memory, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.view_set_layout, None);
            device.destroy_descriptor_set_layout(self.volume_set_layout, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            if self.shadow_view_set_layout != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(self.shadow_view_set_layout, None);
            }
            if self.shadow_pipeline_layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.shadow_pipeline_layout, None);
            }
            device.destroy_render_pass(self.render_pass, None);
            if let Some(rp) = self.main_store_color_pass {
                device.destroy_render_pass(rp, None);
            }
        }
        self.volumes.clear();
        self.view_ubos.clear();
        self.view_ubo_memories.clear();
        self.view_ubo_ptrs.clear();
        self.shadow_view_ubos.clear();
        self.shadow_view_ubo_memories.clear();
        self.shadow_view_ubo_ptrs.clear();
    }
}

impl VkContext {
    // Assemble the per-frame raymarch view from the frame's jittered VP (the
    // matrix the main pass rasterised the depth buffer with) + camera position.
    pub(in crate::vulkan) fn build_raymarch_view(
        &self,
        vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        time: f32,
    ) -> RaymarchView {
        RaymarchView {
            vp,
            inv_vp: super::math::mat4_inverse(vp),
            cam_pos,
            _pad0: 0.0,
            viewport: [
                self.render_extent.width as f32,
                self.render_extent.height as f32,
            ],
            time,
            prefilter_mip_count: self.prefilter_mip_count as f32,
        }
    }

    // Upload this frame's `view_time` into the shadow-caster view ring so the
    // from-light SDF march samples the same animation time as the live pass. A
    // no-op when no volume casts shadows. Called once per frame from the Shadow
    // pass, before the cascade loop; the dedicated ring (not the main view ring)
    // keeps this write off the concurrently-recorded Raymarch pass's buffer.
    pub(in crate::vulkan) fn upload_raymarch_shadow_view(&self, frame_idx: usize, elapsed: f32) {
        let Some(rm) = self.raymarch.as_ref() else {
            return;
        };
        if !rm.any_shadow_casters() {
            return;
        }
        let Some(&ptr) = rm.shadow_view_ubo_ptrs.get(frame_idx) else {
            return;
        };
        // Only `time` is read by the shadow shaders; the rest is inert padding.
        let view = RaymarchView {
            vp: [[0.0; 4]; 4],
            inv_vp: [[0.0; 4]; 4],
            cam_pos: [0.0; 3],
            _pad0: 0.0,
            viewport: [0.0, 0.0],
            time: elapsed,
            prefilter_mip_count: 0.0,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &view as *const RaymarchView as *const u8,
                ptr,
                std::mem::size_of::<RaymarchView>(),
            );
        }
    }

    // Draw the visible SDF shadow casters into one CSM cascade. Called from the
    // Shadow pass inside each cascade's depth-only render pass, after the
    // rasterised casters: the cascade's LESS depth test keeps the nearer of the
    // rasterised vs raymarched occluder per texel. The viewport / scissor set by
    // the shadow pass persist (dynamic state), so this only rebinds the cube
    // geometry, the shadow pipeline, the shadow view + per-volume sets, and the
    // cascade push constant. A no-op when no volume casts shadows.
    pub(in crate::vulkan) fn encode_sdf_shadow_cascade(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        cascade_idx: usize,
    ) {
        let Some(rm) = self.raymarch.as_ref() else {
            return;
        };
        if !rm.any_shadow_casters() || rm.shadow_view_sets.is_empty() {
            return;
        }
        let device = &self.device;
        let push = ShadowCascadePush {
            cascade_idx: cascade_idx as u32,
        };
        unsafe {
            device.cmd_bind_vertex_buffers(cmd, 0, std::slice::from_ref(&rm.cube_vb), &[0]);
            device.cmd_bind_index_buffer(cmd, rm.cube_ib, 0, vk::IndexType::UINT16);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                rm.shadow_pipeline_layout,
                0,
                std::slice::from_ref(&rm.shadow_view_sets[frame_idx]),
                &[],
            );
            device.cmd_push_constants(
                cmd,
                rm.shadow_pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                std::slice::from_raw_parts(
                    &push as *const ShadowCascadePush as *const u8,
                    std::mem::size_of::<ShadowCascadePush>(),
                ),
            );
            for vol in &rm.volumes {
                let Some(shadow_pipeline) = vol.shadow_pipeline else {
                    continue;
                };
                if !vol.visible {
                    continue;
                }
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, shadow_pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    rm.shadow_pipeline_layout,
                    1,
                    std::slice::from_ref(&vol.volume_set),
                    &[],
                );
                device.cmd_draw_indexed(cmd, CUBE_INDEX_COUNT, 1, 0, 0, 0);
                self.inc_draw_calls(1);
            }
        }
    }

    // Encode the raymarched SDF volume pass. Runs after `AutoExposure` (which
    // sampled the pre-raymarch hdr_resolve) and before `Decals`. Snapshots the
    // resolved scene into `snapshot` for refractive taps, draws each visible
    // volume's proxy back faces into the MSAA colour + scene depth, and the
    // render pass resolves the combined colour into hdr_resolve (single-sample:
    // renders into hdr_resolve directly). Leaves hdr_resolve SHADER_READ_ONLY
    // and depth DEPTH_STENCIL_ATTACHMENT_OPTIMAL for the downstream stack.
    pub(in crate::vulkan) fn encode_raymarch(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        view: &RaymarchView,
    ) -> Result<(), String> {
        let Some(rm) = self.raymarch.as_ref() else {
            return Ok(());
        };
        if !rm.any_visible() {
            return Ok(());
        }
        let device = &self.device;
        let extent = self.render_extent;
        let hdr_resolve = self
            .hdr_resolve_images
            .get(frame_idx)
            .ok_or("raymarch: hdr_resolve index OOB")?
            .image;
        let snapshot = rm.snapshot.image;

        // Upload this frame's view.
        let view_ptr = *rm
            .view_ubo_ptrs
            .get(frame_idx)
            .ok_or("raymarch: view_ubo_ptrs index OOB")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                view as *const RaymarchView as *const u8,
                view_ptr,
                std::mem::size_of::<RaymarchView>(),
            );
        }

        let color_aspect = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let image_barrier = |image: vk::Image,
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
                .subresource_range(color_aspect)
        };

        // 1) Open hdr_resolve + snapshot for the refraction snapshot copy. The
        // src scopes order AutoExposure's compute read + the previous frame's
        // fragment read ahead of the transfer.
        let to_src = image_barrier(
            hdr_resolve,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_READ,
        );
        let to_dst = image_barrier(
            snapshot,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_src, to_dst],
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
                hdr_resolve,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                snapshot,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );
        }

        // 2) Close the snapshot for the fragment read, order the main pass's
        // colour + depth writes (and the copy read of hdr_resolve) ahead of the
        // render pass's attachment load + resolve, and (single-sample only)
        // restore hdr_resolve to SHADER_READ_ONLY so the render pass's colour
        // load matches its declared initial layout.
        let snapshot_to_read = image_barrier(
            snapshot,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        let load_barrier = vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags::TRANSFER_READ,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );
        // Single-sample also restores hdr_resolve to SHADER_READ_ONLY for the
        // render pass's colour load; MSAA leaves it as the resolve target, so
        // only the snapshot barrier applies. Build both on the stack and slice
        // off the second when MSAA is on, avoiding a per-frame heap allocation.
        let hdr_to_read = image_barrier(
            hdr_resolve,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::COLOR_ATTACHMENT_READ,
        );
        let image_barriers = [snapshot_to_read, hdr_to_read];
        let image_barriers = if rm.msaa {
            &image_barriers[..1]
        } else {
            &image_barriers[..]
        };
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&load_barrier),
                &[],
                image_barriers,
            );
        }

        // 3) The render pass: LOAD the scene colour + depth, draw each visible
        // volume, then resolve (MSAA) / store (single-sample) into hdr_resolve.
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(rm.render_pass)
            .framebuffer(self.framebuffers[frame_idx])
            .render_area(vk::Rect2D::default().extent(extent));
        // Negative-height viewport: matches the main pass so the proxy rasterises
        // into identical pixels and the reprojected hit depth shares its space.
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
            device.cmd_bind_vertex_buffers(cmd, 0, std::slice::from_ref(&rm.cube_vb), &[0]);
            device.cmd_bind_index_buffer(cmd, rm.cube_ib, 0, vk::IndexType::UINT16);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                rm.pipeline_layout,
                0,
                std::slice::from_ref(&rm.view_sets[frame_idx]),
                &[],
            );
            for vol in &rm.volumes {
                if !vol.visible {
                    continue;
                }
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, vol.pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    rm.pipeline_layout,
                    1,
                    std::slice::from_ref(&vol.volume_set),
                    &[],
                );
                device.cmd_draw_indexed(cmd, CUBE_INDEX_COUNT, 1, 0, 0, 0);
                self.inc_draw_calls(1);
            }
            device.cmd_end_render_pass(cmd);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // Representative user shaders, inlined so the compile guards are
    // self-contained (no dependency on any file outside this crate). The
    // helpers prepended ahead of these (`sdSphere`, `sdTorus`, `sdfParamValue`,
    // `SdfSurface`, `VolumeSample`, ...) are declared in raymarch_helpers.glsl.

    // Surface volume: a sphere smooth-unioned with a spinning torus, shaded as
    // chrome. Defines the V1 `map` + `shade` pair.
    const DEMO_SURFACE_GLSL: &str = r#"
float map(vec3 p, SdfParams params, float time) {
    float speed = sdfParamValue(params, 5u);
    float angle = time * speed;
    float s = sin(angle);
    float c = cos(angle);
    vec3 rp = vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
    float d_sphere = sdSphere(rp, sdfParamValue(params, 6u));
    float d_torus = sdTorus(rp, vec2(sdfParamValue(params, 7u), sdfParamValue(params, 8u)));
    float k = max(sdfParamValue(params, 9u), 1e-3);
    return opSmoothUnion(d_sphere, d_torus, k);
}

SdfSurface shade(vec3 p, vec3 normal, SdfParams params, float time, vec2 frag_uv) {
    SdfSurface s;
    s.albedo = vec3(sdfParamValue(params, 0u), sdfParamValue(params, 1u), sdfParamValue(params, 2u));
    s.roughness = clamp(sdfParamValue(params, 3u), 0.02, 1.0);
    s.metallic = clamp(sdfParamValue(params, 4u), 0.0, 1.0);
    s.emissive = vec3(0.0);
    s.transmitted = vec3(0.0);
    return s;
}
"#;

    // Volumetric volume: a drifting fbm cloud. Defines `sampleVolume` instead
    // of `map` / `shade`.
    const DEMO_VOLUMETRIC_GLSL: &str = r#"
float cloud_hash(vec3 p) {
    vec3 q = fract(p * 0.1031);
    q += dot(q, q.yzx + 19.19);
    return fract((q.x + q.y) * q.z);
}

float cloud_noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);

    float n000 = cloud_hash(i + vec3(0.0, 0.0, 0.0));
    float n100 = cloud_hash(i + vec3(1.0, 0.0, 0.0));
    float n010 = cloud_hash(i + vec3(0.0, 1.0, 0.0));
    float n110 = cloud_hash(i + vec3(1.0, 1.0, 0.0));
    float n001 = cloud_hash(i + vec3(0.0, 0.0, 1.0));
    float n101 = cloud_hash(i + vec3(1.0, 0.0, 1.0));
    float n011 = cloud_hash(i + vec3(0.0, 1.0, 1.0));
    float n111 = cloud_hash(i + vec3(1.0, 1.0, 1.0));

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx0 = mix(nx00, nx10, u.y);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nx1 = mix(nx01, nx11, u.y);
    return mix(nx0, nx1, u.z);
}

float cloud_fbm(vec3 p) {
    float v = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 4; ++i) {
        v += amp * cloud_noise(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return v;
}

VolumeSample sampleVolume(vec3 p, SdfParams params, float time) {
    float cloud_scale = max(sdfParamValue(params, 0u), 0.01);
    vec3 flow = vec3(sdfParamValue(params, 1u), sdfParamValue(params, 2u), sdfParamValue(params, 3u));
    float base_density = sdfParamValue(params, 4u);
    float albedo = sdfParamValue(params, 5u);

    vec3 sample_pos = (p + flow * time) / cloud_scale;
    float n = cloud_fbm(sample_pos);
    float density = max(0.0, n - 0.45) * 2.0 * base_density;

    VolumeSample vs;
    vs.density = density;
    vs.scattering = vec3(albedo, albedo, albedo);
    vs.emission = vec3(0.0, 0.0, 0.0);
    return vs;
}
"#;

    // The GLSL `RaymarchViewBlock` std140 layout is 160 bytes; pin both the
    // size and every field offset so a Rust-side reorder fails the suite
    // without a GPU (mirrors the render_types `*_layout_matches_*` tests).
    #[test]
    fn raymarch_view_layout_matches_glsl() {
        assert_eq!(size_of::<RaymarchView>(), 160);
        assert_eq!(offset_of!(RaymarchView, vp), 0);
        assert_eq!(offset_of!(RaymarchView, inv_vp), 64);
        assert_eq!(offset_of!(RaymarchView, cam_pos), 128);
        assert_eq!(offset_of!(RaymarchView, viewport), 144);
        assert_eq!(offset_of!(RaymarchView, time), 152);
        assert_eq!(offset_of!(RaymarchView, prefilter_mip_count), 156);
    }

    // The GLSL `SdfVolumeBlock` std140 layout is 176 bytes.
    #[test]
    fn sdf_volume_uniforms_layout_matches_glsl() {
        assert_eq!(size_of::<RaymarchVolumeUniforms>(), 176);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, centre), 0);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, extent), 16);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, cone_ratio), 32);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, max_distance), 36);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, max_steps), 40);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, receive_shadows), 44);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, params), 48);
    }

    // Compile the proxy vertex + the assembled fragment (helpers + the demo
    // chrome-blob user shader + template) so a GLSL regression fails the suite
    // without a GPU. Mirrors the fog `fog_shaders_compile` guard.
    #[test]
    fn raymarch_shaders_compile() {
        super::compile_raymarch_shaders(DEMO_SURFACE_GLSL, false)
            .expect("raymarch shaders compile");
    }

    // Compile the depth-only shadow-caster shaders (proxy vertex + helpers +
    // demo user shader + shadow template) so a GLSL regression in the shadow
    // path fails the suite without a GPU.
    #[test]
    fn raymarch_shadow_shaders_compile() {
        super::compile_raymarch_shadow_shaders(DEMO_SURFACE_GLSL, false)
            .expect("raymarch shadow shaders compile");
    }

    // Compile the volumetric shaders (proxy vertex + helpers + the demo cloud
    // `sampleVolume` shader + volumetric template). Guards both the GLSL
    // volumetric template and the assumption that glslang prunes the unused
    // surface helpers (`map` / `shade` are forward-declared but never defined by
    // a volumetric author), which would otherwise fail with a missing-body link
    // error. Fails the suite without a GPU on any regression.
    #[test]
    fn raymarch_volumetric_shaders_compile() {
        super::compile_raymarch_volumetric_shaders(DEMO_VOLUMETRIC_GLSL, false)
            .expect("raymarch volumetric shaders compile");
    }
}
