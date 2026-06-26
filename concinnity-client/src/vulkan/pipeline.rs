// Vulkan pipeline creation for the main, shadow, and text render passes.
// GLSL shader sources are defined inline here and compiled to SPIR-V at context
// init time via shaderc, unless the caller supplies valid SPIR-V bytes directly.

use ash::{Device, vk};

//  GLSL source strings

// Uniform and push-constant layouts are designed to match the #[repr(C)] Rust
// structs in gfx::render_types byte-for-byte under std140/std430 rules:
//
//  - ViewUniforms (160 bytes, std140 UBO): mat4 vp, mat4 view, float elapsed,
//    float _pad0, then cam_pos as 3 individual floats + 3 pad floats.
//  - LightUniforms (400 bytes, std140 UBO): DirLight and PointLight each
//    represented as two vec4s so their size is 32 bytes (matching Rust [f32;3]+f32).
//  - ShadowUniforms (272 bytes, std140 UBO): mat4 light_vps[4] (256) +
//    vec4 cascade_splits (16). Holds the cascaded shadow map VPs and the
//    view-space far-depth threshold for each cascade.
//  - Push constants (112 bytes, std430): mat4 model (64) + MaterialUniforms (48).
//    MaterialUniforms uses vec3 tint/emissive which in std430 have alignment 16;
//    the Rust struct places them at offsets 16 and 32 (both 16-byte aligned) ✓.

const VERT_GLSL: &str = include_str!("shaders/main.vert");

const FRAG_GLSL: &str = include_str!("shaders/main.frag");

// Bindless siblings of VERT_GLSL / FRAG_GLSL for the bindless static
// main pass. Instead of a per-draw push constant + per-object descriptor set,
// every object's transform + material + texture-pool indices live in one
// per-frame `GpuObjectData` storage buffer (set 1, binding 0), indexed by the
// object id the draw call passes as `gl_InstanceIndex` (Vulkan's
// `gl_InstanceIndex` includes `firstInstance`). Albedo + normal maps come from
// a bindless `sampler2D tex_pool[]` (set 1, binding 1). Only build-time static
// objects render through these; streamed VoxelWorld chunks keep the legacy
// per-draw pipeline (VERT_GLSL / FRAG_GLSL, also used by the instanced +
// skinned passes). The fragment BRDF mirrors FRAG_GLSL; only the binding
// model differs. `{POOL_SIZE}` in the fragment source is substituted with the
// texture-pool size at compile time.
const VERT_BINDLESS_GLSL: &str = include_str!("shaders/main_bindless.vert");

const FRAG_BINDLESS_GLSL: &str = include_str!("shaders/main_bindless.frag");

// Shared reflection-probe sampling (box-parallax partition-of-unity blend),
// substituted into the bindless fragment shader at its `{PROBE_COMMON}` marker
// (shaderc has no #include). `{MAX_PROBES}` inside it is replaced with the bind
// count so the GLSL array sizes stay locked to `probe_uniforms::MAX_PROBES`.
pub(in crate::vulkan) const PROBE_COMMON_GLSL: &str = include_str!("shaders/probe_common.glsl");

// GPU-instanced sibling of VERT_GLSL. Reads per-instance world matrices from a
// storage buffer at set=2,binding=0 indexed by gl_InstanceIndex instead of the
// push-constant model field (which is ignored here). Paired with FRAG_GLSL.
const VERT_INSTANCED_GLSL: &str = include_str!("shaders/instanced.vert");

const SHADOW_VERT_GLSL: &str = include_str!("shaders/shadow.vert");

// Depth-only bindless sibling of SHADOW_VERT_GLSL for the GPU-driven shadow
// pass: reads `model` from the per-frame GpuObjectData SSBO (set 1) by
// gl_InstanceIndex and projects through light_vps[cascade_idx] (cascade index =
// a push constant). Consumes the cull-written per-cascade indirect buffers.
const SHADOW_VERT_BINDLESS_GLSL: &str = include_str!("shaders/shadow_bindless.vert");

// Skeletally animated sibling of VERT_GLSL. Each vertex carries four joint
// indices + blend weights; the shader blends up to four joint matrices from the
// per-object storage buffer at set=2,binding=0 (linear blend skinning), applies
// the blended matrix to position/normal/tangent, then proceeds exactly like
// VERT_GLSL. Paired with FRAG_GLSL.
const SKINNED_VERT_GLSL: &str = include_str!("shaders/skinned.vert");

// Skeletally animated sibling of SHADOW_VERT_GLSL. Blends the joint matrices so
// a skinned mesh casts a correctly deformed shadow. Reads the per-object joint
// storage buffer at set=1,binding=0 (the shadow pass has no per-object texture
// set, so set 1 is free).
const SKINNED_SHADOW_VERT_GLSL: &str = include_str!("shaders/skinned_shadow.vert");

const TEXT_VERT_GLSL: &str = include_str!("shaders/text.vert");

const TEXT_FRAG_GLSL: &str = include_str!("shaders/text.frag");

// Composite (post-process) pass. A fullscreen triangle samples the resolved
// HDR scene image, composites bloom, applies the Narkowicz ACES tonemap +
// gamma 2.2 encode, a single FXAA 3.11-style edge pass, a 3D-LUT colour grade,
// and a radial vignette. Mirrors `post_fragment_main` in metal/pipeline.rs.
pub(in crate::vulkan) const COMPOSITE_VERT_GLSL: &str = include_str!("shaders/composite.vert");

const COMPOSITE_FRAG_GLSL: &str = include_str!("shaders/composite.frag");

//  Shader compilation

pub(super) fn is_spirv(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == 0x07230203
}

// Resolve a built-in shader's source. With `hot_reload` off (production
// `cn run`) the `include_str!`-baked GLSL passed in `embedded` is returned
// directly. With it on (`cn debug`), the matching `<crate>/src/vulkan/shaders/`
// file is read from disk first, so dev-loop edits take effect on the next
// pipeline build. A missing or unreadable disk file falls back to the
// embedded source: a typo in the path can never crash the running session.
// Mirrors `directx::pipeline::shader_source` and `metal::pipeline::shader_source`.
pub(in crate::vulkan) fn shader_source(
    hot_reload: bool,
    name: &str,
    embedded: &'static str,
) -> std::borrow::Cow<'static, str> {
    if hot_reload {
        let path = format!("{}/src/vulkan/shaders/{}", env!("CARGO_MANIFEST_DIR"), name);
        match std::fs::read_to_string(&path) {
            Ok(s) => return std::borrow::Cow::Owned(s),
            Err(e) => {
                tracing::debug!(
                    "hot-reload: falling back to embedded source for {} ({})",
                    name,
                    e
                );
            }
        }
    }
    std::borrow::Cow::Borrowed(embedded)
}

// Compile the bindless static-pass shaders (bindless). `pool_size` is
// the bindless texture-pool length, substituted into the fragment source's
// `sampler2D tex_pool[]` array declaration. Always built from the inline
// GLSL: the bindless path only drives the built-in shader.
pub(super) fn compile_bindless_shaders(
    hot_reload: bool,
    pool_size: usize,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let vert_src = shader_source(hot_reload, "main_bindless.vert", VERT_BINDLESS_GLSL);
    let vert = compile_glsl(&vert_src, shaderc::ShaderKind::Vertex, "vert_bindless.glsl")?;
    let frag_src_template = shader_source(hot_reload, "main_bindless.frag", FRAG_BINDLESS_GLSL);
    // Inject the shared probe sampling first (it contains its own {MAX_PROBES}),
    // then substitute the bind counts. `{MAX_PROBES}` is locked to the Rust
    // `probe_uniforms::MAX_PROBES` so the GLSL array sizes match the descriptor
    // layout's probe cube array + the ProbeSet UBO byte layout.
    let probe_common = shader_source(hot_reload, "probe_common.glsl", PROBE_COMMON_GLSL);
    let frag_src = frag_src_template
        .replace("{PROBE_COMMON}", &probe_common)
        .replace(
            "{MAX_PROBES}",
            &super::probe_uniforms::MAX_PROBES.to_string(),
        )
        // The global set IS set 0 in the forward bindless pass.
        .replace("{PROBE_DESC_SET}", "0")
        .replace("{POOL_SIZE}", &pool_size.to_string());
    let frag = compile_glsl(
        &frag_src,
        shaderc::ShaderKind::Fragment,
        "frag_bindless.glsl",
    )?;
    Ok((vert, frag))
}

// Compute cull compute kernel. One invocation per build-time `DrawObject`
// frustum/distance-tests the object's `GpuObjectData` AABB against the six
// CPU-extracted frustum planes and writes one `VkDrawIndexedIndirectCommand`
// into the per-frame indirect buffer: survivors get `instance_count = 1`,
// culled or disabled objects get `instance_count = 0` (a no-op draw). The main
// bindless pass then issues the whole buffer with a single
// `cmd_draw_indexed_indirect`, so the CPU never walks the static draw list.
//
// The frustum and distance maths mirror `gfx::frustum` exactly (the six
// planes are extracted CPU-side already normalised) so the GPU path culls
// identically to the CPU BVH path it replaces. `GpuObjectData` / `GpuDrawArgs`
// mirror `gfx::render_types` under std430; the command struct mirrors
// `VkDrawIndexedIndirectCommand`. The object id rides `first_instance` (the
// bindless vertex shader reads it as `gl_InstanceIndex`).
const CULL_COMPUTE_GLSL: &str = include_str!("shaders/cull.comp");

// Byte size of the cull kernel's `CullParams` push-constant block: six
// `vec4` planes (96) + `vec3 cam_pos` + `uint object_count` (the trailing
// scalar shares the camera position's 16-byte std430 slot). Within the
// 128-byte minimum guaranteed push-constant range.
pub(super) const CULL_PUSH_CONSTANT_BYTES: u32 = 112;

// Compile the Compute cull compute kernel to SPIR-V.
pub(super) fn compile_cull_shader(hot_reload: bool) -> Result<Vec<u8>, String> {
    let src = shader_source(hot_reload, "cull.comp", CULL_COMPUTE_GLSL);
    compile_glsl(&src, shaderc::ShaderKind::Compute, "cull_compute.glsl")
}

// Compile the phase-2 (two-pass occlusion) variant of the cull kernel. Same
// source as `compile_cull_shader`, with a `CULL_PHASE2` define injected after
// `#version` to select the `main_phase2` body (re-test the phase-1
// Hi-Z-occluded objects against the rebuilt pyramid). Mirrors the MSAA
// `#define` split the Hi-Z init kernel uses.
pub(super) fn compile_cull_shader_phase2(hot_reload: bool) -> Result<Vec<u8>, String> {
    let src = shader_source(hot_reload, "cull.comp", CULL_COMPUTE_GLSL);
    let src = inject_define(&src, "#define CULL_PHASE2 1\n");
    compile_glsl(
        &src,
        shaderc::ShaderKind::Compute,
        "cull_compute_phase2.glsl",
    )
}

// Compile the GPU-driven shadow cull kernel: the same cull source with a
// `SHADOW_CULL` define, which drops the Hi-Z (set 1) + status (binding 3)
// bindings and does a frustum + distance test against each cascade's light
// frustum. Paired with the lean 3-SSBO shadow cull set layout.
pub(super) fn compile_shadow_cull_shader(hot_reload: bool) -> Result<Vec<u8>, String> {
    let src = shader_source(hot_reload, "cull.comp", CULL_COMPUTE_GLSL);
    let src = inject_define(&src, "#define SHADOW_CULL 1\n");
    compile_glsl(
        &src,
        shaderc::ShaderKind::Compute,
        "cull_compute_shadow.glsl",
    )
}

// Compile the GPU-driven shadow pass's depth-only bindless vertex shader.
pub(super) fn compile_shadow_bindless_vs(hot_reload: bool) -> Result<Vec<u8>, String> {
    let src = shader_source(
        hot_reload,
        "shadow_bindless.vert",
        SHADOW_VERT_BINDLESS_GLSL,
    );
    compile_glsl(&src, shaderc::ShaderKind::Vertex, "shadow_bindless.vert")
}

// Inject a `#define` line immediately after the `#version` directive.
pub(in crate::vulkan) fn inject_define(src: &str, define: &str) -> String {
    if let Some(pos) = src.find('\n') {
        let (head, tail) = src.split_at(pos + 1);
        format!("{head}{define}{tail}")
    } else {
        format!("{define}{src}")
    }
}

// Create the GPU-cull compute pipeline. `layout` must include the cull
// descriptor set (set 0: object SSBO, draw-args SSBO, indirect-command SSBO)
// and the `CullParams` push-constant range.
pub(super) fn create_cull_pipeline(
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
    .map_err(|(_, e)| format!("create cull pipeline: {e}"))?[0];
    unsafe { device.destroy_shader_module(module, None) };
    Ok(pipeline)
}

pub(in crate::vulkan) fn compile_glsl(
    source: &str,
    kind: shaderc::ShaderKind,
    label: &str,
) -> Result<Vec<u8>, String> {
    let compiler = shaderc::Compiler::new().map_err(|e| format!("shaderc init failed: {e}"))?;
    let mut opts =
        shaderc::CompileOptions::new().map_err(|e| format!("shaderc options failed: {e}"))?;
    opts.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_0 as u32,
    );
    opts.set_optimization_level(shaderc::OptimizationLevel::Performance);
    let artifact = compiler
        .compile_into_spirv(source, kind, label, "main", Some(&opts))
        .map_err(|e| format!("compile {label}: {e}"))?;
    Ok(artifact.as_binary_u8().to_vec())
}

// Compile GLSL that uses `GL_EXT_ray_query` (the hardware ray-traced reflection
// fragment shader). Ray query needs SPIR-V 1.4 + the Vulkan-1.2 target
// environment (the `RayQueryKHR` capability is invalid under the default
// Vulkan-1.0 target `compile_glsl` uses); the engine's instance is already 1.2,
// so the resulting module loads fine. Kept separate from `compile_glsl` so every
// other built-in shader keeps the conservative 1.0 target.
pub(in crate::vulkan) fn compile_glsl_rt(
    source: &str,
    kind: shaderc::ShaderKind,
    label: &str,
) -> Result<Vec<u8>, String> {
    let compiler = shaderc::Compiler::new().map_err(|e| format!("shaderc init failed: {e}"))?;
    let mut opts =
        shaderc::CompileOptions::new().map_err(|e| format!("shaderc options failed: {e}"))?;
    opts.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_2 as u32,
    );
    opts.set_target_spirv(shaderc::SpirvVersion::V1_4);
    opts.set_optimization_level(shaderc::OptimizationLevel::Performance);
    let artifact = compiler
        .compile_into_spirv(source, kind, label, "main", Some(&opts))
        .map_err(|e| format!("compile {label}: {e}"))?;
    Ok(artifact.as_binary_u8().to_vec())
}

pub(in crate::vulkan) fn spv_module(
    device: &Device,
    spv: &[u8],
) -> Result<vk::ShaderModule, String> {
    // ash requires 4-byte aligned SPIR-V; copy into aligned Vec<u32>.
    let len = spv.len() / 4;
    let mut code = vec![0u32; len];
    unsafe { std::ptr::copy_nonoverlapping(spv.as_ptr(), code.as_mut_ptr() as *mut u8, spv.len()) };
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.create_shader_module(&info, None) }.map_err(|e| format!("shader module: {e}"))
}

// Resolve vertex/fragment/shadow SPIR-V bytes: use caller bytes if they are
// valid SPIR-V, otherwise compile the built-in GLSL fallback.
pub(super) fn resolve_main_shaders(
    hot_reload: bool,
    vert_bytes: &[u8],
    frag_bytes: &[u8],
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let vert = if is_spirv(vert_bytes) {
        vert_bytes.to_vec()
    } else {
        let src = shader_source(hot_reload, "main.vert", VERT_GLSL);
        compile_glsl(&src, shaderc::ShaderKind::Vertex, "vert.glsl")?
    };
    let frag = if is_spirv(frag_bytes) {
        frag_bytes.to_vec()
    } else {
        let src = shader_source(hot_reload, "main.frag", FRAG_GLSL);
        compile_glsl(&src, shaderc::ShaderKind::Fragment, "frag.glsl")?
    };
    Ok((vert, frag))
}

// Resolve the GPU-instanced vertex shader bytes. Returns None when no
// instancing was requested AND no caller bytes are present.
pub(super) fn resolve_instanced_shader(
    hot_reload: bool,
    vert_instanced_bytes: &[u8],
    need_instanced: bool,
) -> Result<Option<Vec<u8>>, String> {
    if !need_instanced && !is_spirv(vert_instanced_bytes) {
        return Ok(None);
    }
    let spv = if is_spirv(vert_instanced_bytes) {
        vert_instanced_bytes.to_vec()
    } else {
        let src = shader_source(hot_reload, "instanced.vert", VERT_INSTANCED_GLSL);
        compile_glsl(&src, shaderc::ShaderKind::Vertex, "vert_instanced.glsl")?
    };
    Ok(Some(spv))
}

// SPIR-V for the skinned-mesh shader stages: the main skinned VS, the
// depth-only skinned shadow VS, and the fragment shader (shared with the
// static path; `frag_bytes`, when valid SPIR-V, is used directly, otherwise
// the built-in `FRAG_GLSL` is compiled).
#[allow(clippy::type_complexity)]
pub(super) fn compile_skinned_shaders(
    hot_reload: bool,
    frag_bytes: &[u8],
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), String> {
    let main_vs_src = shader_source(hot_reload, "skinned.vert", SKINNED_VERT_GLSL);
    let main_vs = compile_glsl(
        &main_vs_src,
        shaderc::ShaderKind::Vertex,
        "skinned_vert.glsl",
    )?;
    let shadow_vs_src = shader_source(hot_reload, "skinned_shadow.vert", SKINNED_SHADOW_VERT_GLSL);
    let shadow_vs = compile_glsl(
        &shadow_vs_src,
        shaderc::ShaderKind::Vertex,
        "skinned_shadow_vert.glsl",
    )?;
    let frag = if is_spirv(frag_bytes) {
        frag_bytes.to_vec()
    } else {
        let src = shader_source(hot_reload, "main.frag", FRAG_GLSL);
        compile_glsl(&src, shaderc::ShaderKind::Fragment, "frag.glsl")?
    };
    Ok((main_vs, shadow_vs, frag))
}

pub(super) fn resolve_shadow_shader(
    hot_reload: bool,
    shadow_bytes: &[u8],
) -> Result<Option<Vec<u8>>, String> {
    // The shadow vertex shader is engine-internal: a non-SPIR-V or empty
    // `shadow_bytes` selects the baked SHADOW_VERT_GLSL; only a real SPIR-V
    // override is used verbatim. Whether the shadow pass runs at all is gated by
    // `effective_shadow_size` at the call site, not by this function.
    let spv = if is_spirv(shadow_bytes) {
        shadow_bytes.to_vec()
    } else {
        let src = shader_source(hot_reload, "shadow.vert", SHADOW_VERT_GLSL);
        compile_glsl(&src, shaderc::ShaderKind::Vertex, "shadow_vert.glsl")?
    };
    Ok(Some(spv))
}

pub(super) fn compile_text_shaders(hot_reload: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let vert_src = shader_source(hot_reload, "text.vert", TEXT_VERT_GLSL);
    let vert = compile_glsl(&vert_src, shaderc::ShaderKind::Vertex, "text_vert.glsl")?;
    let frag_src = shader_source(hot_reload, "text.frag", TEXT_FRAG_GLSL);
    let frag = compile_glsl(&frag_src, shaderc::ShaderKind::Fragment, "text_frag.glsl")?;
    Ok((vert, frag))
}

pub(super) fn compile_composite_shaders(hot_reload: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let vert_src = shader_source(hot_reload, "composite.vert", COMPOSITE_VERT_GLSL);
    let vert = compile_glsl(
        &vert_src,
        shaderc::ShaderKind::Vertex,
        "composite_vert.glsl",
    )?;
    let frag_src = shader_source(hot_reload, "composite.frag", COMPOSITE_FRAG_GLSL);
    let frag = compile_glsl(
        &frag_src,
        shaderc::ShaderKind::Fragment,
        "composite_frag.glsl",
    )?;
    Ok((vert, frag))
}

//  Pipeline creation

// Vertex binding and attribute descriptions for the full Vertex struct (56 bytes).
fn main_vertex_input() -> (
    [vk::VertexInputBindingDescription; 1],
    [vk::VertexInputAttributeDescription; 5],
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
            .location(2)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(24),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(36),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(4)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(48),
    ];
    ([binding], attrs)
}

// Vertex binding + attributes for the SkinnedVertex struct (80 bytes): the
// 56-byte static attributes plus uvec4 joint indices (offset 56) and vec4
// blend weights (offset 64).
fn skinned_vertex_input() -> (
    [vk::VertexInputBindingDescription; 1],
    [vk::VertexInputAttributeDescription; 7],
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
            .location(2)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(24),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(36),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(4)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(48),
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

// Reduced vertex input for the depth-only skinned shadow pipeline: only the
// position + joint indices + blend weights the skinned shadow VS consumes
// (binding stride stays 80, the same SkinnedVertex buffer is bound).
fn skinned_shadow_vertex_input() -> (
    [vk::VertexInputBindingDescription; 1],
    [vk::VertexInputAttributeDescription; 3],
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

// TextVertex binding (32 bytes): pos(vec2) + uv(vec2) + color(vec3) + pad.
fn text_vertex_input() -> (
    [vk::VertexInputBindingDescription; 1],
    [vk::VertexInputAttributeDescription; 3],
) {
    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(8),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(16),
    ];
    ([binding], attrs)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn create_main_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
    msaa: vk::SampleCountFlags,
    _surface_format: vk::Format,
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

    let (bindings, attrs) = main_vertex_input();
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);

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
        // Match Metal's default + DirectX (no back-face culling) so meshes
        // with mixed winding (particularly procedural floor / ceiling planes
        // whose triangles have a -Y normal under the unsigned plane order)
        // render from both sides. Vulkan's pipeline-default was BACK, which
        // hid the showcase floor while leaving every solid mesh visible.
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(msaa);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

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
    .map_err(|(_, e)| format!("create main pipeline: {e}"))?[0];

    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

// Same as `create_main_pipeline` but takes an instanced vertex shader. The
// caller is responsible for using a pipeline layout that includes the
// per-instance storage buffer descriptor set (set=2).
#[allow(clippy::too_many_arguments)]
pub(super) fn create_instanced_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
    msaa: vk::SampleCountFlags,
    surface_format: vk::Format,
) -> Result<vk::Pipeline, String> {
    create_main_pipeline(
        device,
        render_pass,
        layout,
        vert_spv,
        frag_spv,
        msaa,
        surface_format,
    )
}

pub(super) fn create_shadow_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert_mod = spv_module(device, vert_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_mod)
        .name(&entry)];

    // `shadow.vert` only reads position (it writes depth-only NDC), so the
    // optimizer strips the other attributes from its interface. Bind just
    // location 0 so the pipeline matches the shader and the validation layer
    // does not warn about unconsumed attributes. The binding keeps the full
    // 56-byte `Vertex` stride; the omitted attributes are simply not fetched.
    let (bindings, attrs) = main_vertex_input();
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs[..1]);

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
        // Match Metal's default + DirectX (no back-face culling) so meshes
        // with mixed winding (particularly procedural floor / ceiling planes
        // whose triangles have a -Y normal under the unsigned plane order)
        // render from both sides. Vulkan's pipeline-default was BACK, which
        // hid the showcase floor while leaving every solid mesh visible.
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(true)
        .depth_bias_constant_factor(0.005)
        .depth_bias_slope_factor(1.0)
        // A non-zero clamp needs the optional depthBiasClamp device feature;
        // 0.0 (unclamped) keeps the constant + slope bias without it.
        .depth_bias_clamp(0.0);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

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
    .map_err(|(_, e)| format!("create shadow pipeline: {e}"))?[0];

    unsafe { device.destroy_shader_module(vert_mod, None) };
    Ok(pipeline)
}

// Main-pass pipeline for skinned geometry: the skinned vertex shader (80-byte
// layout) paired with the standard fragment shader. The caller passes a
// pipeline layout that includes the joint storage-buffer descriptor set.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_skinned_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
    msaa: vk::SampleCountFlags,
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

    let (bindings, attrs) = skinned_vertex_input();
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);

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
        // Match Metal's default + DirectX (no back-face culling) so meshes
        // with mixed winding (particularly procedural floor / ceiling planes
        // whose triangles have a -Y normal under the unsigned plane order)
        // render from both sides. Vulkan's pipeline-default was BACK, which
        // hid the showcase floor while leaving every solid mesh visible.
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(msaa);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

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
    .map_err(|(_, e)| format!("create skinned pipeline: {e}"))?[0];

    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

// Shadow-pass pipeline for skinned geometry: the skinned shadow vertex shader
// (80-byte layout, depth-only).
pub(super) fn create_skinned_shadow_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert_mod = spv_module(device, vert_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_mod)
        .name(&entry)];

    let (bindings, attrs) = skinned_shadow_vertex_input();
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);

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
        // Match Metal's default + DirectX (no back-face culling) so meshes
        // with mixed winding (particularly procedural floor / ceiling planes
        // whose triangles have a -Y normal under the unsigned plane order)
        // render from both sides. Vulkan's pipeline-default was BACK, which
        // hid the showcase floor while leaving every solid mesh visible.
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(true)
        .depth_bias_constant_factor(0.005)
        .depth_bias_slope_factor(1.0)
        .depth_bias_clamp(0.0);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

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
    .map_err(|(_, e)| format!("create skinned shadow pipeline: {e}"))?[0];

    unsafe { device.destroy_shader_module(vert_mod, None) };
    Ok(pipeline)
}

pub(super) fn create_text_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
    msaa: vk::SampleCountFlags,
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

    let (bindings, attrs) = text_vertex_input();
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);

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
        .rasterization_samples(msaa);

    // No depth test for text overlay; always draws on top.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::ALWAYS);

    // Standard over-compositing alpha blend.
    let blend_attach = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD);

    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(std::slice::from_ref(&blend_attach));

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
    .map_err(|(_, e)| format!("create text pipeline: {e}"))?[0];

    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

// Build the composite (post-process) pipeline: a vertex-buffer-less fullscreen
// triangle that samples the resolved HDR target and applies ACES + gamma +
// FXAA. Targets the single-sample swapchain backbuffer; no depth attachment.
pub(super) fn create_composite_pipeline(
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

    // No vertex input: the fullscreen triangle is generated from gl_VertexIndex.
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

    // The composite pass always renders to the single-sample swapchain image.
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
    .map_err(|(_, e)| format!("create composite pipeline: {e}"))?[0];

    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::{
        FRAG_BINDLESS_GLSL, FRAG_GLSL, PROBE_COMMON_GLSL, VERT_GLSL, compile_bindless_shaders,
        compile_cull_shader, compile_cull_shader_phase2, compile_shadow_bindless_vs,
        compile_shadow_cull_shader, compile_skinned_shaders, is_spirv, resolve_instanced_shader,
        resolve_main_shaders,
    };

    // The phase-1 cull kernel, its two-pass `CULL_PHASE2` variant, and the
    // GPU-driven shadow `SHADOW_CULL` variant all compile to valid SPIR-V from
    // the embedded source. Guards the `#ifdef` split in `cull.comp`, which the
    // Vulkan-on-Windows runtime cannot currently smoke-test.
    #[test]
    fn cull_shaders_compile_both_phases() {
        let phase1 = compile_cull_shader(false).expect("phase-1 cull compiles");
        let phase2 = compile_cull_shader_phase2(false).expect("phase-2 cull compiles");
        let shadow = compile_shadow_cull_shader(false).expect("shadow cull compiles");
        assert!(is_spirv(&phase1), "phase-1 cull is valid SPIR-V");
        assert!(is_spirv(&phase2), "phase-2 cull is valid SPIR-V");
        assert!(is_spirv(&shadow), "shadow cull is valid SPIR-V");
        // Each define selects a different kernel body, so the modules differ.
        assert_ne!(phase1, phase2);
        assert_ne!(phase1, shadow);
    }

    // The GPU-driven shadow pass's depth-only bindless vertex shader compiles to
    // valid SPIR-V from the embedded source.
    #[test]
    fn shadow_bindless_vs_compiles() {
        let vs = compile_shadow_bindless_vs(false).expect("shadow bindless VS compiles");
        assert!(is_spirv(&vs), "shadow bindless VS is valid SPIR-V");
    }

    // The bindless main shaders compile to valid SPIR-V from the embedded source,
    // including the reflection-probe sampling injected from `probe_common.glsl` at
    // the `{PROBE_COMMON}` marker + the `{MAX_PROBES}` / `{POOL_SIZE}` substitutions.
    // Guards the probe forward path (the box-parallax partition-of-unity blend +
    // the ProbeSet UBO / probe cube array declarations) offline: a GLSL error in
    // the injection fails here without needing a GPU.
    #[test]
    fn bindless_shaders_compile() {
        let (vs, fs) = compile_bindless_shaders(false, 4).expect("bindless shaders compile");
        assert!(is_spirv(&vs), "bindless vertex is valid SPIR-V");
        assert!(is_spirv(&fs), "bindless fragment is valid SPIR-V");
        // The probe markers must be fully substituted (no literal token survives).
        let frag_src = FRAG_BINDLESS_GLSL
            .replace("{PROBE_COMMON}", PROBE_COMMON_GLSL)
            .replace(
                "{MAX_PROBES}",
                &crate::vulkan::probe_uniforms::MAX_PROBES.to_string(),
            )
            .replace("{PROBE_DESC_SET}", "0")
            .replace("{POOL_SIZE}", "4");
        assert!(!frag_src.contains("{PROBE_COMMON}"));
        assert!(!frag_src.contains("{MAX_PROBES}"));
        assert!(!frag_src.contains("{PROBE_DESC_SET}"));
        assert!(!frag_src.contains("{POOL_SIZE}"));
    }

    // The shader-resolution helpers that `update_world_shader_pipelines`
    // composes when hot-swapping a world's `ShaderStage` pipelines: valid
    // SPIR-V (the bytes the hot-reload recompile always produces) is passed
    // through verbatim, while non-SPIR-V selects the built-in GLSL fallback.
    // No device is needed, so this guards the world-shader hot-swap path the
    // Vulkan-on-Windows runtime cannot unit-test end to end.
    #[test]
    fn world_shader_resolution_passes_spirv_and_falls_back_to_glsl() {
        // Build real SPIR-V from the bundled GLSL, then confirm
        // `resolve_main_shaders` returns it unchanged (the hot-swap's main
        // pipeline reuses these bytes directly).
        let vert_spv = super::compile_glsl(VERT_GLSL, shaderc::ShaderKind::Vertex, "vert").unwrap();
        let frag_spv =
            super::compile_glsl(FRAG_GLSL, shaderc::ShaderKind::Fragment, "frag").unwrap();
        let (v, f) = resolve_main_shaders(false, &vert_spv, &frag_spv).unwrap();
        assert_eq!(v, vert_spv, "SPIR-V vertex bytes pass through unchanged");
        assert_eq!(f, frag_spv, "SPIR-V fragment bytes pass through unchanged");

        // Non-SPIR-V bytes fall back to the engine GLSL, which compiles to
        // valid SPIR-V.
        let (v2, f2) = resolve_main_shaders(false, b"not spirv", b"still not spirv").unwrap();
        assert!(is_spirv(&v2), "GLSL fallback vertex compiles to SPIR-V");
        assert!(is_spirv(&f2), "GLSL fallback fragment compiles to SPIR-V");

        // The instanced helper, forced on (as the hot-swap does when an
        // instanced pipeline is live), yields valid SPIR-V from the fallback.
        let inst = resolve_instanced_shader(false, b"not spirv", true)
            .unwrap()
            .expect("forced instanced resolve yields Some");
        assert!(is_spirv(&inst), "instanced fallback compiles to SPIR-V");

        // The skinned helper compiles its engine-internal vertex + shadow
        // stages from inline GLSL and passes the supplied SPIR-V fragment
        // through, matching what the hot-swap feeds the skinned pipeline.
        let (skinned_vs, skinned_shadow_vs, skinned_frag) =
            compile_skinned_shaders(false, &frag_spv).unwrap();
        assert!(is_spirv(&skinned_vs), "skinned VS compiles to SPIR-V");
        assert!(is_spirv(&skinned_shadow_vs), "skinned shadow VS compiles");
        assert_eq!(skinned_frag, frag_spv, "skinned fragment passes through");
    }
}
