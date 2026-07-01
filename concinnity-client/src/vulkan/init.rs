// src/vulkan/init.rs
//
// VkContext construction: GLFW window creation and the one-time GPU
// resource setup performed by VkContext::new.
use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};

use ash::vk;
use ash::vk::Handle;

use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::*;

use super::context::*;
use super::device::*;
use super::draw::*;
use super::math::*;
use super::pipeline::*;
use super::post::bloom::{
    MAX_BLOOM_MIPS, alloc_bloom_input_sets, compile_bloom_shaders, create_bloom_chain,
    create_bloom_framebuffers, create_bloom_pipeline, rebind_bloom_input0,
};
use super::post::taa::*;
use super::render_pass::*;
use super::resources::*;
use super::swapchain::*;
use super::texture::{self, *};

//  Construction

#[allow(clippy::too_many_arguments)]
impl VkContext {
    pub fn new(
        title: &str,
        width: u32,
        height: u32,
        validation: bool,
        frames_in_flight: usize,
        vsync: bool,
        clear_color: [f32; 4],
        vertices: &[Vertex],
        indices: &[u32],
        draw_objects: Vec<DrawObject>,
        instanced_clusters: Vec<InstancedCluster>,
        // Skinned draw-object count, threaded to size the shared GPU-cull buffers'
        // reserved skinned tail at init (`n_objects + n_instances + n_skinned`).
        // The skinned geometry is uploaded later by `upload_skinned`, which sets
        // the live `self.n_skinned`; this only reserves capacity.
        n_skinned: usize,
        // Worst-case resident chunk count for a streaming VoxelWorld (0 otherwise).
        // Reserves a chunk record region in the shared cull buffers at init
        // (`[n_objects + n_instances, +n_chunk_max)`); resident chunks fold into the
        // indirect path each frame. Sets the live `self.n_chunk`.
        n_chunk_max: usize,
        vert_bytes: &[u8],
        frag_bytes: &[u8],
        shadow_bytes: &[u8],
        vert_instanced_bytes: &[u8],
        textures: &[(u32, u32, Vec<u8>)],
        normal_maps: &[(u32, u32, Vec<u8>)],
        light_uniforms: LightUniforms,
        shadow_map_size: u32,
        // Shadow-cascade re-render policy: hybrid amortizes the far cascades
        // across frames, every_frame refreshes all cascades every frame.
        shadow_update: crate::assets::ShadowUpdate,
        // Shadow distance (GraphicsConfig.shadow_distance, world units). The
        // per-frame cascade split reads it, capped at the camera far plane.
        shadow_distance: u32,
        // Shadow cascade count (GraphicsConfig.shadow_cascades, 1..=4). The
        // per-frame split + schedule read it; applies at the next launch.
        shadow_cascades: u32,
        // Scene-sampler max anisotropy (GraphicsConfig.anisotropy), clamped to the
        // device limit where the sampler is built below.
        anisotropy: u32,
        text_atlases: Vec<(u32, u32, Vec<u8>)>,
        // Serialised `EnvironmentMap` payload; `None` binds 1×1 grey fallback
        // cubes and disables IBL via `prefilter_mip_count = 0`.
        env_map_bytes: Option<&[u8]>,
        // Post-process tunables. `bloom_intensity` / `bloom_threshold` /
        // `bloom_knee` drive the bloom chain; `exposure` / `vignette` /
        // `lut_strength` feed the composite pass.
        mut post_process: crate::gfx::render_types::PostProcessParams,
        // Serialised `ColorLut` payload (3D grading LUT) baked into the
        // composite pass. `None` binds a 2x2x2 identity LUT, so the grade is a
        // no-op at any `lut_strength`.
        color_lut_bytes: Option<&[u8]>,
        // Temporal anti-aliasing toggle, resolved from `PostProcessConfig.aa_mode`.
        // When set, a velocity pre-pass + history-resolve pass run and the
        // projection is sub-pixel jittered; when false all of that is skipped.
        taa_enabled: bool,
        // SSAO (GTAO) settings. `Some` builds the pre-pass + kernel + blur
        // pipelines and a per-frame VP UBO; the main pass then samples the
        // blurred occlusion at set 0 binding 6 to modulate its ambient term.
        // `None` skips SSAO entirely and a 1×1 white fallback is bound at
        // binding 6 so the multiplier collapses to a pass-through.
        ssao_settings: Option<crate::gfx::ssao::SsaoSettings>,
        // SSR settings. `Some` builds the depth + normal + roughness pre-pass
        // and the fullscreen ray-march resolve; the resolve output then
        // replaces the raw HDR resolve as the scene the bloom / composite /
        // TAA passes consume (`scene_view_for_post`). `None` skips SSR.
        ssr_settings: Option<crate::gfx::ssr::SsrSettings>,
        // SSGI settings. `Some` (the world selected `indirect_lighting: ssgi`)
        // builds the hemisphere-gather + depth-aware-blur GI pass, which reuses
        // the SSR depth + normal pre-pass G-buffer. Turning SSGI on therefore
        // forces that pre-pass to be built even when the SSR resolve is off.
        // `None` leaves the indirect-diffuse term as the IBL ambient alone.
        ssgi_settings: Option<crate::gfx::ssgi::SsgiSettings>,
        // Hardware ray-traced reflection settings from `PostProcessConfig`.
        // `Some` (the world set `ray_traced_reflections: true`) builds the scene
        // acceleration structure + the inline `rayQueryEXT` reflection pass when
        // the GPU exposes the ray-query extensions; the pass then replaces the
        // SSR resolve in the frame graph (it reuses the SSR depth + normal + roughness
        // pre-pass G-buffer, so that pre-pass is forced on like SSGI). `None`, an
        // unsupported GPU, or any build failure leaves RT off and SSR remains the
        // fallback. Mirrors the DirectX / Metal `rt_reflection_settings`.
        rt_settings: Option<crate::gfx::rt_reflections::RtReflectionSettings>,
        // Per-axis render-resolution divisor for the reflection composite's roughness
        // blur target, resolved from `PostProcessConfig.reflection_blur_resolution`
        // (Half=2 default). The blur is low-frequency so running it reduced and
        // bilinear-upsampling in the composite is visually free.
        reflection_blur_scale: u32,
        // Authored projected decals resolved from the world's `Decal`
        // components. The decal pipeline + per-frame uniforms are always
        // built (so runtime `add_decal` works from an empty world); the
        // encoder simply skips when every slot is `None`.
        decals: Vec<crate::gfx::decal::DecalRecord>,
        // Authored particle emitters resolved from the world's
        // `ParticleEmitter` components. The compute + render pipelines and
        // per-emitter GPU pool buffers are built only when at least one
        // emitter is declared (or when runtime `add_particle_emitter`
        // fires); the encoder skips the passes when `particle_resources`
        // is `None`. Mirrors `directx/init/mod.rs`.
        particles: Vec<crate::gfx::particles::ParticleEmitterRecord>,
        // Volumetric-fog settings resolved from the world's `VolumetricFog`.
        // `Some` builds the fog pipeline + per-frame uniform ring; `None`
        // skips the fog pass entirely.
        fog_settings: Option<crate::gfx::volumetric_fog::FogSettings>,
        // Auto-exposure settings resolved from `PostProcessConfig`. `Some`
        // builds the histogram + average compute pipelines and drives the
        // EMA; `None` disables auto-exposure entirely (the authored
        // `exposure_ev` remains the only input to `post_process.exposure`).
        auto_exposure_settings: Option<crate::gfx::auto_exposure::AutoExposureSettings>,
        // Authored `exposure_ev` carried through as a bias on the adapted
        // EV when auto-exposure is on; ignored when it is off.
        auto_exposure_bias_ev: f32,
        // World-side HDR display request from `PostProcessConfig.hdr_display`.
        // When true, the constructor enables `VK_EXT_swapchain_colorspace`
        // (instance extension; gated on availability) and probes the surface
        // for `R16G16B16A16_SFLOAT` + `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT`.
        // If found, the swapchain runs in scRGB linear and the composite
        // shader's `hdr_output > 0.5` branch lights up; otherwise the
        // request falls back to SDR with a logged warning. Mirrors
        // `DxContext::new`'s `hdr_display`.
        hdr_display: bool,
        // PQ-encoded HDR output request from `PostProcessConfig.hdr_pq`. Honoured
        // only when `hdr_display` also resolves on AND the surface advertises an
        // `HDR10_ST2084_EXT` colour-space pair: the swapchain is then created in
        // that colour space and the composite shader PQ-encodes (SMPTE ST 2084)
        // in-shader. When PQ is unavailable the renderer falls back to the
        // scRGB-linear extended-range path. Mirrors `DxContext::new`.
        hdr_pq: bool,
        // Temporal upscaling toggle from `PostProcessConfig.temporal_upscaling`.
        // When on (and the FFX VK runtime loads), the scene renders at the
        // reduced `render_extent` and an FSR pass reconstructs the swapchain
        // resolution; falls back to native rendering when FFX is unavailable.
        temporal_upscaling: bool,
        // Per-axis input-to-output ratio from `PostProcessConfig.upscale_quality`
        // (e.g. 2/3 Quality, 0.5 Performance). Drives the `render_extent` split
        // when `temporal_upscaling` is on.
        upscale_scale: f32,
        // Upscaler backend selector from `PostProcessConfig.upscale_backend`.
        // Only FSR is available on Vulkan (DLSS / XeSS are DirectX-only); a
        // DX-only request logs a note and uses FSR.
        upscale_backend: crate::assets::UpscalerBackend,
        // Two-pass Hi-Z occlusion toggle from `PostProcessConfig.occlusion_two_pass`.
        // When set (and the bindless GPU-cull path is active), the constructor
        // builds the phase-2 cull pipeline + second indirect buffers + the
        // phase-1/phase-2 main render passes, and the frame graph inserts
        // HizBuild -> Cull2 -> Main2 after Main. Inert without bindless cull.
        occlusion_two_pass: bool,
        // Raymarched SDF volumes drained from the world's `SdfVolume`
        // components, paired with their compiled-payload fragment shader source
        // bytes + asset label. `.glsl` payloads build a per-volume raymarch
        // pipeline; `.metal` / `.hlsl` SDFs are skipped with a warning.
        sdf_volumes: Vec<(crate::assets::SdfVolume, Vec<u8>, String)>,
        // Translucent glass panels drained from the world's `GlassPanel`
        // components. Each becomes one back-to-front-sorted draw in the shared
        // transparent pass. Empty leaves `glass` None and the pass skipped.
        glass_panels: Vec<crate::assets::GlassPanel>,
        // True only under `cn debug` (set via `crate::app::dev_flags`).
        // Routes every built-in shader compile through the disk-first
        // `shader_source` helper and spawns the `vulkan/shaders/`
        // filesystem watcher. `cn run` leaves it false; the
        // include_str!-baked GLSL sources continue to drive every pipeline.
        hot_reload: bool,
    ) -> Result<Self, String> {
        // Record this (main) thread so the `RenderBackend` mutation entry points
        // can `debug_assert_main_thread` against it; the Send invariant rests on
        // the context being touched from this thread alone.
        super::context::record_main_thread();

        // Temporal upscaling (FSR) consumes the velocity pre-pass's
        // render-resolution motion + depth, which TaaResources owns, so force
        // the TAA stack built when upscaling is on (the TAA *resolve* is still
        // dropped from the frame graph; only the velocity pre-pass is reused).
        let taa_enabled = taa_enabled || temporal_upscaling;
        let frames = frames_in_flight.max(1);

        //  GLFW window
        let mut window = crate::vulkan::window::GlfwWindow::new(
            title,
            width,
            height,
            &crate::assets::WindowMode::Windowed,
            true,
        )?;

        //  Vulkan entry
        let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("load vulkan: {e}"))?;

        // Resolve which (if any) upscaler SDK needs Vulkan instance / device
        // extensions enabled at creation time (DLSS / XeSS). Queried before
        // instance creation (it needs at most the loaded SDK), then threaded
        // into `create_logical_device` for the device extensions / features.
        // Inert (`choice == Native`) when upscaling is off or the backend needs
        // nothing; held in scope until after device creation so its
        // instance-ext pointers + XeSS feature chain stay valid. Resolved before
        // `app_info` so its `min_api_version` can raise the instance apiVersion.
        let upscale_sdk = super::post::UpscaleSdk::prepare(temporal_upscaling, upscale_backend);

        //  Instance
        let app_name = CString::new(title).unwrap_or_default();
        let engine_name = CString::new("Concinnity").unwrap();
        // Vulkan 1.2 baseline: FidelityFX FSR's precompiled shaders are SPIR-V
        // 1.5, valid only under a 1.2+ instance. XeSS 3.x raises the floor to
        // 1.3 (its shaders use SPV_KHR_integer_dot_product, a 1.3 capability),
        // reported via `min_api_version`. Take the max, clamped to what the
        // loader actually supports so an unsupported request can't fail instance
        // creation (the backend then falls back). The engine's own shaders are
        // unaffected by the bump.
        let loader_version = unsafe { entry.try_enumerate_instance_version() }
            .ok()
            .flatten()
            .unwrap_or(vk::API_VERSION_1_2);
        let api_version = vk::API_VERSION_1_2
            .max(upscale_sdk.min_api_version())
            .min(loader_version);
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(api_version);

        // Hold the windowing extension name CStrings in scope so their pointers
        // stay valid through instance creation, then drop with the rest of init
        // (mirrors `device.rs`'s `enabled`/`ext_names` pairing). The later
        // pushes are all `'static` NAME pointers, so they need no backing store.
        let instance_ext_cstrings: Vec<CString> = window
            .required_instance_extensions()
            .iter()
            .map(|s| CString::new(s.as_str()).unwrap())
            .collect();
        let mut ext_names_raw: Vec<*const c_char> =
            instance_ext_cstrings.iter().map(|c| c.as_ptr()).collect();

        let debug_ext = ash::ext::debug_utils::NAME.as_ptr();
        if validation {
            ext_names_raw.push(debug_ext);
        }

        // HDR display: enable `VK_EXT_swapchain_colorspace` (instance
        // extension) when the world asked for HDR and the loader exposes
        // it. With the extension enabled, the surface formats query
        // includes the extended-range colour spaces; without it, the
        // scRGB-linear pair we look for is unreachable. The extension
        // landed pre-Vulkan-1.0.43 and is supported on every recent
        // desktop driver; the availability check makes a missing-loader
        // case (older Linux distros, headless CI) degrade to SDR rather
        // than failing instance creation.
        let swapchain_colorspace_ext_available = hdr_display
            && unsafe { entry.enumerate_instance_extension_properties(None) }
                .map(|exts| {
                    exts.iter().any(|p| {
                        let name = unsafe { std::ffi::CStr::from_ptr(p.extension_name.as_ptr()) };
                        name == ash::ext::swapchain_colorspace::NAME
                    })
                })
                .unwrap_or(false);
        if swapchain_colorspace_ext_available {
            ext_names_raw.push(ash::ext::swapchain_colorspace::NAME.as_ptr());
        } else if hdr_display {
            tracing::warn!(
                "HDR display requested but VK_EXT_swapchain_colorspace is not exposed by the \
                 Vulkan loader; falling back to SDR (BGRA8 sRGB) output"
            );
        }

        // Instance extensions the chosen upscaler SDK requires (DLSS / XeSS).
        // The pointers borrow from `upscale_sdk`, which outlives this scope.
        for ptr in upscale_sdk.instance_extension_ptrs() {
            ext_names_raw.push(ptr);
        }

        let layer_names_raw: Vec<*const c_char> = if validation {
            let layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
            let ptr = layer.as_ptr();
            std::mem::forget(layer);
            vec![ptr]
        } else {
            vec![]
        };

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&ext_names_raw)
            .enabled_layer_names(&layer_names_raw);

        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(|e| format!("create instance: {e}"))?;

        //  Debug messenger
        // Budget the messenger callback consumes to drop benign DLSS first-frame
        // layout errors; set after `build_upscaler` resolves to DLSS. Heap-boxed
        // so its address stays stable when this `VkContext` is returned by value,
        // and kept as a context field that outlives the messenger (destroyed in
        // `Drop` before fields). `None` when validation (the messenger) is off.
        let debug_filter: Option<Box<std::sync::atomic::AtomicU32>> =
            validation.then(|| Box::new(std::sync::atomic::AtomicU32::new(0)));
        let (debug_utils, debug_messenger) = if validation {
            let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let user_data = debug_filter
                .as_ref()
                .map(|b| &**b as *const std::sync::atomic::AtomicU32 as *mut std::ffi::c_void)
                .unwrap_or(std::ptr::null_mut());
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback))
                .user_data(user_data);
            let messenger = unsafe { du.create_debug_utils_messenger(&info, None) }
                .map_err(|e| format!("debug messenger: {e}"))?;
            (Some(du), Some(messenger))
        } else {
            (None, None)
        };

        //  Surface
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let surface_handle = window.create_surface(instance.handle().as_raw() as usize)?;
        let surface = vk::SurfaceKHR::from_raw(surface_handle as u64);

        //  Physical device
        let (physical_device, graphics_family, present_family) =
            pick_physical_device(&instance, &surface_loader, surface)?;

        //  Logical device. `rt_capable` comes back true when the device exposes
        //  the ray-query extension set (and XeSS is not the active backend); the
        //  RT extensions are enabled whenever capable so a live RT toggle works,
        //  independent of whether the world wants RT at launch. The
        //  acceleration-structure build + RT pass below are gated on
        //  `rt_settings.is_some() && rt_capable` (everything falls back to SSR
        //  when RT is off or the device is incapable).
        let (device, memory_budget_supported, rt_capable) = create_logical_device(
            &instance,
            physical_device,
            graphics_family,
            present_family,
            validation,
            &upscale_sdk,
        )?;

        let graphics_queue = unsafe { device.get_device_queue(graphics_family, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family, 0) };

        //  Timestamp support: the per-frame GPU-time chip uses a query pool
        //  with `2 * frames` slots, a pair per in-flight frame. `timestamp_period`
        //  is nanoseconds-per-tick; `timestamp_valid_bits` on the graphics queue
        //  family must be non-zero for `cmd_write_timestamp` to be valid. Without
        //  either the renderer leaves `gpu_frame_us` at zero. Mirrors
        //  `directx::build_timestamp_resources`.
        let device_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let queue_family_props =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let timestamp_period = device_props.limits.timestamp_period;
        let timestamp_valid_bits = queue_family_props
            .get(graphics_family as usize)
            .map(|f| f.timestamp_valid_bits)
            .unwrap_or(0);
        let timestamps_supported = timestamp_period > 0.0 && timestamp_valid_bits > 0;
        let timestamp_query_pool = if timestamps_supported {
            // One per-frame block of `SLOTS_PER_FRAME` slots (whole-frame pair +
            // one pair per render pass) per frame in flight.
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count((super::pass_timing::SLOTS_PER_FRAME * frames) as u32);
            match unsafe { device.create_query_pool(&info, None) } {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!("timestamp query pool create failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        //  Device-local heap indices for the VRAM-residency chip. Sums
        //  `heap_usage` on every DEVICE_LOCAL heap when `VK_EXT_memory_budget`
        //  is supported; otherwise the field stays empty and the chip reports
        //  zero (matching DirectX's adapter-without-QueryVideoMemoryInfo
        //  fallback).
        let memory_props =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let device_local_heaps: Vec<u32> = if memory_budget_supported {
            (0..memory_props.memory_heap_count as usize)
                .filter(|i| {
                    memory_props.memory_heaps[*i]
                        .flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                })
                .map(|i| i as u32)
                .collect()
        } else {
            Vec::new()
        };

        //  MSAA sample count
        let msaa_samples = get_max_usable_sample_count(&instance, physical_device);

        // HDR-output resolve. The world's `hdr_display` toggle is the
        // gate; even on a capable display, no HDR unless the asset opts
        // in. The reverse (`hdr_display = true` on an SDR-only surface,
        // or with the colour-space loader extension missing) falls back
        // to SDR with a logged warning. Vulkan has no portable max-EDR
        // query: when the surface advertises the scRGB-linear colour
        // space we synthesise a placeholder `max_edr = 2.0` (the
        // HDR400-class minimum) so the shared `HdrOutputMode::resolve`
        // logic stays uniform across backends.
        // Probe which HDR colour-space pairs the surface advertises. An
        // advertised HDR colour space is Vulkan's "HDR available" signal (there
        // is no portable max-EDR query), so we synthesise the placeholder
        // `max_edr` from it. scRGB-linear drives the extended-linear path; an
        // `HDR10_ST2084_EXT` pair (float or 10-bit packed) drives the PQ path.
        let surface_formats =
            unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) }
                .unwrap_or_default();
        let advertises = |fmt: vk::Format, cs: vk::ColorSpaceKHR| {
            surface_formats
                .iter()
                .any(|f| f.format == fmt && f.color_space == cs)
        };
        let scrgb_advertises = swapchain_colorspace_ext_available
            && advertises(
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
            );
        let pq_advertises = swapchain_colorspace_ext_available
            && (advertises(
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ColorSpaceKHR::HDR10_ST2084_EXT,
            ) || advertises(
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::ColorSpaceKHR::HDR10_ST2084_EXT,
            ));
        // PQ needs the HDR10 colour space. When `hdr_pq` is requested but only
        // scRGB is advertised, fall back to the extended-linear path so the
        // shader encode and the swapchain colour space never diverge (sending
        // PQ-encoded values to an scRGB-linear swapchain would look wrong).
        let pq_capable = hdr_pq && pq_advertises;
        if hdr_display && hdr_pq && !pq_advertises {
            tracing::warn!(
                "HDR display + hdr_pq:true requested but no surface format advertises HDR10 PQ \
                 (RGBA16F / A2B10G10R10_UNORM_PACK32 + HDR10_ST2084_EXT); falling back to \
                 scRGB-linear extended-range output"
            );
        }
        let max_edr = if scrgb_advertises || pq_advertises {
            2.0
        } else {
            1.0
        };
        let hdr_mode =
            crate::gfx::hdr_output::HdrOutputMode::resolve(hdr_display, pq_capable, max_edr);
        if hdr_display && !hdr_mode.is_hdr() {
            tracing::warn!(
                "HDR display requested but no surface format advertises an HDR colour space \
                 (scRGB linear or HDR10 PQ): falling back to SDR (BGRA8 sRGB) output"
            );
        } else if hdr_mode.pq_flag() > 0.5 {
            tracing::info!("HDR display output enabled: HDR10 PQ swapchain (SMPTE ST 2084)");
        } else if hdr_mode.is_hdr() {
            tracing::info!(
                "HDR display output enabled: scRGB-linear swapchain (RGBA16F + \
                 EXTENDED_SRGB_LINEAR_EXT)"
            );
        }
        // Drive the composite shader's `hdr_output > 0.5` branch and its
        // in-branch `pq_output` encode flag from the resolved mode. Mirrors
        // `DxContext::new`.
        post_process.hdr_output = hdr_mode.shader_flag();
        post_process.pq_output = hdr_mode.pq_flag();

        //  Swapchain
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let (swapchain, swapchain_images, swapchain_format, swapchain_extent) =
            create_swapchain_inner(
                &instance,
                &device,
                physical_device,
                &surface_loader,
                surface,
                &swapchain_loader,
                width,
                height,
                graphics_family,
                present_family,
                vk::SwapchainKHR::null(),
                hdr_mode,
                vsync,
            )?;
        let swapchain_image_views =
            create_swapchain_image_views(&device, &swapchain_images, swapchain_format)?;

        //  Command pool
        let command_pool = {
            let info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(graphics_family);
            unsafe { device.create_command_pool(&info, None) }
                .map_err(|e| format!("command pool: {e}"))?
        };

        // Temporal upscaling (FSR / DLSS / XeSS). Built here, before the
        // off-screen attachments, because its render dims drive `render_extent`:
        // when an upscaler builds, the whole scene pipeline renders at
        // `round(swapchain_extent * upscale_scale)` and the upscaler
        // reconstructs the swapchain resolution. When `temporal_upscaling` is
        // off (or no backend is available) `render_extent == swapchain_extent`
        // and the pipeline collapses to native-resolution rendering. Bloom /
        // composite / swapchain always stay at `swapchain_extent`.
        // `build_upscaler` resolves `upscale_backend` against availability with
        // a DLSS -> XeSS -> FSR -> native fallback (the DLSS / XeSS device
        // extensions were enabled above via `upscale_sdk`).
        let upscale = if temporal_upscaling {
            let (built, resolved) = super::post::build_upscaler(
                &instance,
                &device,
                physical_device,
                command_pool,
                graphics_queue,
                swapchain_extent.width,
                swapchain_extent.height,
                upscale_scale,
                upscale_backend,
            )?;
            // Arm the messenger's benign-error budget for DLSS (see
            // `DLSS_FIRST_FRAME_LAYOUT_SUPPRESS`); a no-op for other backends.
            if resolved == super::post::ResolvedBackend::Dlss
                && let Some(f) = &debug_filter
            {
                f.store(
                    DLSS_FIRST_FRAME_LAYOUT_SUPPRESS,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            built
        } else {
            None
        };
        let render_extent = match &upscale {
            Some(u) => {
                let (w, h) = u.render_dims();
                vk::Extent2D {
                    width: w,
                    height: h,
                }
            }
            None => swapchain_extent,
        };

        //  Initial reset of every timestamp query slot. Without this the first
        //  `vkGetQueryPoolResults` call on each slot (before that slot has
        //  ever been written) hits an uninitialised query and the validation
        //  layer emits a "query not reset" error. After the reset, the slot
        //  is in "unavailable" state, so `get_query_pool_results` returns
        //  NOT_READY → 0 cleanly until `record_frame` writes the first pair.
        if let Some(pool) = timestamp_query_pool {
            super::texture::one_shot_submit(&device, command_pool, graphics_queue, |cmd| unsafe {
                device.cmd_reset_query_pool(
                    cmd,
                    pool,
                    0,
                    (super::pass_timing::SLOTS_PER_FRAME * frames) as u32,
                );
            })?;
        }

        //  Shadow map (4-layer D32_SFLOAT array image, one slice per cascade)
        // CSM is gated on `shadow_map_size` (from GraphicsConfig; 0 disables
        // shadows). The shadow vertex shader is engine-internal (the baked
        // SHADOW_VERT_GLSL), so an empty `shadow_bytes` override no longer means
        // "no shadows": it just selects the built-in shader. Mirrors the Metal
        // internal-shadow path.
        let effective_shadow_size = shadow_map_size;
        let shadow_map = create_shadow_map_array(
            &instance,
            &device,
            physical_device,
            command_pool,
            graphics_queue,
            effective_shadow_size,
            NUM_SHADOW_CASCADES as u32,
        )?;

        //  Textures
        let gpu_textures: Vec<GpuImage> = if textures.is_empty() {
            vec![texture::create_fallback_white(
                &instance,
                &device,
                physical_device,
                command_pool,
                graphics_queue,
            )?]
        } else {
            textures
                .iter()
                .enumerate()
                .map(|(i, (w, h, px))| {
                    upload_texture(
                        &instance,
                        &device,
                        physical_device,
                        command_pool,
                        graphics_queue,
                        *w,
                        *h,
                        px,
                    )
                    .map_err(|e| format!("texture[{i}]: {e}"))
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        let flat_normal = texture::create_fallback_flat_normal(
            &instance,
            &device,
            physical_device,
            command_pool,
            graphics_queue,
        )?;
        let mut gpu_normal_maps = vec![flat_normal];
        for (i, (w, h, px)) in normal_maps.iter().enumerate() {
            gpu_normal_maps.push(
                upload_texture(
                    &instance,
                    &device,
                    physical_device,
                    command_pool,
                    graphics_queue,
                    *w,
                    *h,
                    px,
                )
                .map_err(|e| format!("normal_map[{i}]: {e}"))?,
            );
        }

        let gpu_text_atlases: Vec<GpuImage> = text_atlases
            .iter()
            .enumerate()
            .map(|(i, (w, h, px))| {
                upload_texture(
                    &instance,
                    &device,
                    physical_device,
                    command_pool,
                    graphics_queue,
                    *w,
                    *h,
                    px,
                )
                .map_err(|e| format!("text_atlas[{i}]: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        //  Samplers
        // Anisotropic degree for the scene sampler: enabled only when the device
        // supports `samplerAnisotropy` (the matching feature is turned on in
        // `device.rs`). Clamp the requested degree (GraphicsConfig.anisotropy) to
        // the GPU's 1..16 range and then to the device limit.
        let scene_aniso = {
            let feats = unsafe { instance.get_physical_device_features(physical_device) };
            if feats.sampler_anisotropy != 0 {
                let limit = unsafe { instance.get_physical_device_properties(physical_device) }
                    .limits
                    .max_sampler_anisotropy;
                (anisotropy.clamp(1, 16) as f32).min(limit)
            } else {
                1.0
            }
        };
        let linear_sampler = create_sampler_linear_repeat(&device, scene_aniso)?;
        let shadow_sampler = create_sampler_shadow(&device)?;
        let text_sampler = create_sampler_linear_clamp(&device)?;
        // Linear-clamp sampler the composite pass reads the HDR resolve with;
        // clamp keeps the FXAA neighbour taps from wrapping at screen edges.
        let composite_sampler = create_sampler_linear_clamp(&device)?;

        //  Render passes
        let main_render_pass = create_main_render_pass(&device, HDR_FORMAT, msaa_samples)?;
        let shadow_render_pass = create_shadow_render_pass(&device)?;
        let composite_render_pass = create_composite_render_pass(&device, swapchain_format)?;
        let bloom_write_pass = create_bloom_render_pass(&device, HDR_FORMAT, false)?;
        let bloom_blend_pass = create_bloom_render_pass(&device, HDR_FORMAT, true)?;

        //  Off-screen HDR attachments (one set per frame-in-flight slot)
        let (color_images, depth_images, hdr_resolve_images) = create_attachments(
            &instance,
            &device,
            physical_device,
            command_pool,
            graphics_queue,
            render_extent.width,
            render_extent.height,
            msaa_samples,
            frames,
        )?;

        //  Framebuffers
        let framebuffers = create_main_framebuffers(
            &device,
            main_render_pass,
            &color_images,
            &depth_images,
            &hdr_resolve_images,
            render_extent,
            msaa_samples,
        )?;
        let composite_framebuffers = create_composite_framebuffers(
            &device,
            composite_render_pass,
            &swapchain_image_views,
            swapchain_extent,
        )?;

        //  Transient image pool: the graph-owned transients (`ao_output`,
        //  `bloom_top`). Built before the bloom chain so bloom mip 0 binds the
        //  pooled `bloom_top` image, and before SSAO (below) so its blur
        //  framebuffers + the main pass binding 6 bind the pooled `ao_output`.
        let bloom_on = post_process.bloom_intensity > 0.0;
        let transient_pool = super::transient_pool::TransientImagePool::build(
            &instance,
            &device,
            physical_device,
            frames,
            &super::transient_pool::transient_slots(
                ssao_settings.is_some(),
                bloom_on,
                render_extent,
                swapchain_extent,
            ),
        )?;
        let bloom_top_pairs = transient_pool.pairs_for_frames("bloom_top", frames);

        //  Bloom chain (per frame-in-flight slot)
        let (bloom_mips, bloom_mip_extents) = create_bloom_chain(
            &instance,
            &device,
            physical_device,
            command_pool,
            graphics_queue,
            swapchain_extent,
            frames,
            &bloom_top_pairs,
        )?;
        let (bloom_write_framebuffers, bloom_blend_framebuffers) = create_bloom_framebuffers(
            &device,
            bloom_write_pass,
            bloom_blend_pass,
            &bloom_mips,
            &bloom_mip_extents,
        )?;

        //  Geometry buffers. When ray-traced reflections are live the shared
        //  vertex / index buffers double as acceleration-structure build inputs
        //  (device-addressed) and as storage buffers the RT fragment shader
        //  fetches hit-triangle attributes from, so they carry the extra usage.
        //  These flags require the ray-query extensions enabled at device
        //  creation, so they are added whenever the device is RT-capable (not
        //  only when RT is on at launch) -- a later live toggle needs the shared
        //  buffers already AS-build-input + device-addressable, and the buffers
        //  cannot gain usage flags after creation. Inert when RT is never built.
        let rt_geo_usage = if rt_capable {
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER
        } else {
            vk::BufferUsageFlags::empty()
        };
        let (vertex_buffer, vertex_buffer_memory) = upload_geometry_buffer(
            &instance,
            &device,
            physical_device,
            command_pool,
            graphics_queue,
            vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER | rt_geo_usage,
        )?;
        let (index_buffer, index_buffer_memory) = upload_geometry_buffer(
            &instance,
            &device,
            physical_device,
            command_pool,
            graphics_queue,
            indices,
            vk::BufferUsageFlags::INDEX_BUFFER | rt_geo_usage,
        )?;
        // Empty geometry still allocates a 4-byte buffer (see
        // `upload_geometry_buffer_raw`); track the real allocation size so
        // `setup_chunk_streaming` copies the right prefix when it grows them.
        let vertex_buffer_bytes = (std::mem::size_of_val(vertices) as u64).max(4);
        let index_buffer_bytes = (std::mem::size_of_val(indices) as u64).max(4);

        //  Uniform buffers
        let view_ubo_size = std::mem::size_of::<super::draw::ViewUniforms>() as u64;
        let light_ubo_size = std::mem::size_of::<LightUniforms>() as u64;
        let shadow_ubo_size = std::mem::size_of::<ShadowUniforms>() as u64;

        let mut view_ubo_buffers = Vec::with_capacity(frames);
        let mut view_ubo_memories = Vec::with_capacity(frames);
        let mut view_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let (buf, mem) = create_buffer(
                &instance,
                &device,
                physical_device,
                view_ubo_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let ptr = unsafe {
                device
                    .map_memory(mem, 0, view_ubo_size, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("map view ubo: {e}"))? as *mut u8
            };
            view_ubo_buffers.push(buf);
            view_ubo_memories.push(mem);
            view_ubo_ptrs.push(ptr);
        }

        // Per-frame `ProbeSet` UBO ring (global set 0 binding 7): the
        // reflection-probe count + per-probe parallax boxes. Persistently mapped;
        // `record_frame` writes `self.probe_set` here each frame.
        let probe_set_ubo_size = std::mem::size_of::<super::probe_uniforms::ProbeSet>() as u64;
        let mut probe_set_ubo_buffers = Vec::with_capacity(frames);
        let mut probe_set_ubo_memories = Vec::with_capacity(frames);
        let mut probe_set_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let (buf, mem) = create_buffer(
                &instance,
                &device,
                physical_device,
                probe_set_ubo_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let ptr = unsafe {
                device
                    .map_memory(mem, 0, probe_set_ubo_size, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("map probe set ubo: {e}"))? as *mut u8
            };
            probe_set_ubo_buffers.push(buf);
            probe_set_ubo_memories.push(mem);
            probe_set_ubo_ptrs.push(ptr);
        }

        let (light_ubo, light_ubo_memory) = create_buffer(
            &instance,
            &device,
            physical_device,
            light_ubo_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let (shadow_ubo, shadow_ubo_memory) = create_buffer(
            &instance,
            &device,
            physical_device,
            shadow_ubo_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Per-frame CSM updates use the first directional light's direction;
        // we cache it here at init so subsequent frames don't have to look it
        // up. Matches the Metal/DirectX pattern.
        let shadow_light_dir = if light_uniforms.num_directional > 0 {
            light_uniforms.directional[0].direction
        } else {
            // Match LightUniforms::DEFAULT.
            [-0.3, 0.85, 0.4]
        };
        // Sun direction + intensity-weighted colour for the volumetric-fog
        // encoder. The Vulkan backend uploads LightUniforms once at init
        // (no runtime light mutation), so the sun colour fed into the fog
        // ray-march is fixed. Mirrors `directx/init`.
        let fog_sun_dir = shadow_light_dir;
        let fog_sun_color = if light_uniforms.num_directional > 0 {
            let l = &light_uniforms.directional[0];
            [
                l.color[0] * l.intensity,
                l.color[1] * l.intensity,
                l.color[2] * l.intensity,
            ]
        } else {
            [1.0, 1.0, 1.0]
        };
        let shadow_uniforms = crate::gfx::csm::empty_shadow_uniforms();
        upload_shadow_uniforms(&device, shadow_ubo_memory, &shadow_uniforms)?;
        upload_light_uniforms(&device, light_ubo_memory, &light_uniforms)?;

        //  IBL resources (always created so descriptor bindings 4/5 are valid)
        let cube_sampler = create_sampler_cube_linear(&device)?;
        let env_map = if let Some(bytes) = env_map_bytes {
            let view = crate::build::environment_map::deserialise(bytes)
                .map_err(|e| format!("EnvironmentMap payload malformed: {}", e))?;
            upload_environment_map(
                &instance,
                &device,
                physical_device,
                command_pool,
                graphics_queue,
                view.irradiance_face,
                view.irradiance_bytes,
                view.prefilter_face,
                &view.prefilter_mip_bytes,
            )?
        } else {
            EnvironmentMapTextures {
                irradiance: texture::create_fallback_cubemap(
                    &instance,
                    &device,
                    physical_device,
                    command_pool,
                    graphics_queue,
                    [0.05, 0.05, 0.05, 1.0],
                )?,
                prefilter: texture::create_fallback_cubemap(
                    &instance,
                    &device,
                    physical_device,
                    command_pool,
                    graphics_queue,
                    [0.05, 0.05, 0.05, 1.0],
                )?,
                prefilter_mip_count: 0,
            }
        };

        // Colour-grading LUT: upload the declared `ColorLut` payload, or build a
        // 2x2x2 identity LUT so the composite pass always binds a valid 3D
        // texture. With the identity LUT the grade is a no-op at any strength.
        let color_lut = if let Some(bytes) = color_lut_bytes {
            let (size, data) = crate::build::color_lut::deserialise(bytes)
                .map_err(|e| format!("ColorLut payload malformed: {e}"))?;
            upload_color_lut(
                &instance,
                &device,
                physical_device,
                command_pool,
                graphics_queue,
                size,
                data,
            )?
        } else {
            create_fallback_color_lut(
                &instance,
                &device,
                physical_device,
                command_pool,
                graphics_queue,
            )?
        };

        //  Descriptor set layouts
        // Global set (set 0): view UBO, light UBO, shadow UBO, shadow array
        // sampler (binding 3), IBL irradiance cube (4), IBL prefilter cube (5),
        // SSAO occlusion (binding 6, bound to the live `ssao.ao` image when
        // SSAO is enabled, otherwise to the 1×1 `ssao_white` fallback so the
        // main pass's `ambient *= ao` multiplier collapses to a pass-through).
        // Global set (set 0): the geometry path's view / light / shadow UBOs +
        // shadow-map + IBL cubes + SSAO sampler + ProbeSet UBO (binding 7) + the
        // reflection-probe cube array (binding 8). Binding table + lock-down test
        // live in `descriptor_layout.rs`. Built inline (not via the count-1
        // `create_descriptor_set_layout` helper) because binding 8 is a
        // `descriptorCount = MAX_PROBES` cube array; the count-1 bindings come from
        // the locked `global_set()` table, then the array binding is appended (the
        // same shape as the bindless texture pool's array binding below).
        let global_set_layout = {
            let mut bindings: Vec<vk::DescriptorSetLayoutBinding> =
                super::descriptor_layout::global_set()
                    .iter()
                    .map(|&(b, ty, stage)| {
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(b)
                            .descriptor_type(ty)
                            .descriptor_count(1)
                            .stage_flags(stage)
                    })
                    .collect();
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(super::descriptor_layout::PROBE_CUBE_ARRAY_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(super::probe_uniforms::MAX_PROBES as u32)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            );
            unsafe {
                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
            }
            .map_err(|e| format!("global set layout: {e}"))?
        };
        // Per-object set (set 1): albedo + normal map.
        let object_set_layout =
            create_descriptor_set_layout(&device, &super::descriptor_layout::object_set())?;
        // Text set (set 0 for text pass): atlas sampler.
        let text_set_layout = create_descriptor_set_layout(
            &device,
            &[(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
            )],
        )?;
        // Shadow global set (set 0 for shadow pass): ShadowUniforms UBO.
        let shadow_global_set_layout =
            create_descriptor_set_layout(&device, &super::descriptor_layout::shadow_global_set())?;
        // Composite set (set 0 for composite pass): HDR resolve image at
        // binding 0, bloom mip 0 at binding 1, the 3D colour LUT at binding 2.
        let composite_set_layout = create_descriptor_set_layout(
            &device,
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
        // Bloom set (set 0 for every bloom pass): the single input image.
        let bloom_set_layout = create_descriptor_set_layout(
            &device,
            &[(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
            )],
        )?;

        //  Pipeline layouts
        // Main push constants: 112 bytes for model (64) + material (48).
        let main_pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(112);
        let main_set_layouts = [global_set_layout, object_set_layout];
        let main_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&main_set_layouts)
                    .push_constant_ranges(std::slice::from_ref(&main_pc_range)),
                None,
            )
        }
        .map_err(|e| format!("main pipeline layout: {e}"))?;

        let shadow_pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            // 64 bytes for model + 16 bytes for cascade_idx + padding.
            .size(80);
        let shadow_set_layouts = [shadow_global_set_layout];
        let shadow_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&shadow_set_layouts)
                    .push_constant_ranges(std::slice::from_ref(&shadow_pc_range)),
                None,
            )
        }
        .map_err(|e| format!("shadow pipeline layout: {e}"))?;

        let text_pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(16);
        let text_set_layouts = [text_set_layout];
        let text_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&text_set_layouts)
                    .push_constant_ranges(std::slice::from_ref(&text_pc_range)),
                None,
            )
        }
        .map_err(|e| format!("text pipeline layout: {e}"))?;

        // Post-process push constant: the full `PostProcessParams` struct,
        // fragment-stage. Shared by the composite + bloom-prefilter shaders.
        let post_pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<crate::gfx::render_types::PostProcessParams>() as u32);

        // Composite layout: one descriptor set (HDR resolve + bloom mip 0).
        let composite_set_layouts = [composite_set_layout];
        let composite_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&composite_set_layouts)
                    .push_constant_ranges(std::slice::from_ref(&post_pc_range)),
                None,
            )
        }
        .map_err(|e| format!("composite pipeline layout: {e}"))?;

        // Bloom layout: one descriptor set (the input image) + the shared
        // post-process push constant (read only by the prefilter).
        let bloom_set_layouts = [bloom_set_layout];
        let bloom_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&bloom_set_layouts)
                    .push_constant_ranges(std::slice::from_ref(&post_pc_range)),
                None,
            )
        }
        .map_err(|e| format!("bloom pipeline layout: {e}"))?;

        //  Pipelines
        let (vert_spv, frag_spv) = resolve_main_shaders(hot_reload, vert_bytes, frag_bytes)?;
        let main_pipeline = create_main_pipeline(
            &device,
            main_render_pass,
            main_pipeline_layout,
            &vert_spv,
            &frag_spv,
            msaa_samples,
            swapchain_format,
        )?;

        //  Instanced pipeline (optional)
        // Set 2 binding 0 is a storage buffer of per-instance world matrices.
        let need_instanced = !instanced_clusters.is_empty();
        let instance_set_layout_opt = if need_instanced {
            Some(create_descriptor_set_layout(
                &device,
                &[(
                    0,
                    vk::DescriptorType::STORAGE_BUFFER,
                    vk::ShaderStageFlags::VERTEX,
                )],
            )?)
        } else {
            None
        };

        let (instanced_pipeline_opt, instanced_pipeline_layout_opt) = if need_instanced {
            let instance_set_layout = instance_set_layout_opt.unwrap();
            let instanced_set_layouts = [global_set_layout, object_set_layout, instance_set_layout];
            let instanced_pl = unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&instanced_set_layouts)
                        .push_constant_ranges(std::slice::from_ref(&main_pc_range)),
                    None,
                )
            }
            .map_err(|e| format!("instanced pipeline layout: {e}"))?;

            let inst_spv_opt =
                resolve_instanced_shader(hot_reload, vert_instanced_bytes, need_instanced)?
                    .ok_or("instanced shader payload missing")?;
            let pipeline = create_instanced_pipeline(
                &device,
                main_render_pass,
                instanced_pl,
                &inst_spv_opt,
                &frag_spv,
                msaa_samples,
                swapchain_format,
            )?;
            (Some(pipeline), Some(instanced_pl))
        } else {
            (None, None)
        };

        let (shadow_pipeline_opt, shadow_framebuffers_vec) = if effective_shadow_size > 0
            && let Ok(Some(shadow_spv)) = resolve_shadow_shader(hot_reload, shadow_bytes)
        {
            let pl = create_shadow_pipeline(
                &device,
                shadow_render_pass,
                shadow_pipeline_layout,
                &shadow_spv,
            )?;
            let fbs = create_shadow_framebuffers(
                &device,
                shadow_render_pass,
                &shadow_map,
                effective_shadow_size,
            )?;
            (Some(pl), fbs)
        } else {
            // No shadow pipeline: per-frame DEPTH_STENCIL→SHADER_READ_ONLY
            // transition in draw.rs is skipped, so move the (1×1 fallback)
            // shadow_map into the layout the main-pass descriptor expects.
            one_shot_submit(&device, command_pool, graphics_queue, |cmd| {
                transition_image_layout_array(
                    &device,
                    cmd,
                    shadow_map.image,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageAspectFlags::DEPTH,
                    1,
                );
            })?;
            (None, Vec::new())
        };

        // Text renders in the composite pass (post-tonemap, single-sample), so
        // its pipeline targets the composite render pass.
        let text_pipeline_opt = if !gpu_text_atlases.is_empty() {
            let (tv, tf) = compile_text_shaders(hot_reload)?;
            let tp = create_text_pipeline(
                &device,
                composite_render_pass,
                text_pipeline_layout,
                &tv,
                &tf,
                vk::SampleCountFlags::TYPE_1,
            )?;
            Some(tp)
        } else {
            None
        };

        //  Composite (post-process) pipeline
        let composite_pipeline = {
            let (cv, cf) = compile_composite_shaders(hot_reload)?;
            create_composite_pipeline(
                &device,
                composite_render_pass,
                composite_pipeline_layout,
                &cv,
                &cf,
            )?
        };

        //  Bloom pipelines (prefilter / downsample / upsample)
        let (bloom_pipeline_prefilter, bloom_pipeline_downsample, bloom_pipeline_upsample) = {
            let bs = compile_bloom_shaders(hot_reload)?;
            let prefilter = create_bloom_pipeline(
                &device,
                bloom_write_pass,
                bloom_pipeline_layout,
                &bs.vert,
                &bs.prefilter,
                false,
            )?;
            let downsample = create_bloom_pipeline(
                &device,
                bloom_write_pass,
                bloom_pipeline_layout,
                &bs.vert,
                &bs.downsample,
                false,
            )?;
            // The upsample pipeline targets the LOAD blend pass and blends
            // additively onto the mip already there.
            let upsample = create_bloom_pipeline(
                &device,
                bloom_blend_pass,
                bloom_pipeline_layout,
                &bs.vert,
                &bs.upsample,
                true,
            )?;
            (prefilter, downsample, upsample)
        };

        //  SSAO (GTAO): pre-pass + kernel + blur, plus a 1×1 white fallback
        //  that is always bound at set 0 binding 6 when SSAO is off so the
        //  main pass's `ambient *= ao` multiplier collapses to a pass-through.
        let ssao_white = texture::create_fallback_white(
            &instance,
            &device,
            physical_device,
            command_pool,
            graphics_queue,
        )?;
        // The transient image pool was built above (before the bloom chain); it
        // already holds this frame's pooled `ao_output` views when SSAO is on.
        let ssao_opt = if let Some(settings) = ssao_settings {
            let ao_views = transient_pool.views_for_frames("ao_output", frames);
            Some(super::post::ssao::SsaoResources::new(
                &instance,
                &device,
                physical_device,
                render_extent.width,
                render_extent.height,
                frames,
                settings,
                &ao_views,
                hot_reload,
            )?)
        } else {
            None
        };

        //  SSR (screen-space reflections): depth + normal + roughness pre-pass
        //  and a fullscreen ray-march resolve. The pre-pass G-buffer is shared
        //  with SSGI, so `SsrResources` is built whenever SSR *or* SSGI is on;
        //  the resolve half only does its work (and owns the post-stack scene
        //  image) when SSR reflections are actually enabled (`ssr_resolve_on`).
        //  When the resolve is on, the bloom prefilter + composite + (optional)
        //  TAA scene input is re-pointed at `SsrResources::output` further down
        //  so the post stack consumes the HDR scene with reflections composited
        //  in; a SSGI-only build leaves those pointed at the raw HDR resolve
        //  (SSGI composites its bounce into it earlier on the RMW chain).
        // The SSR resolve runs only when SSR is authored AND ray-traced
        // reflections are not live (RT replaces the resolve in the same graph
        // slot; resolved below once `rt_wanted` + the AS build are known).
        let ssr_authored = ssr_settings.is_some();
        // For the pre-pass build the resolve settings drive the resolve
        // pipeline's tunables; a SSGI-only / RT-only build has no authored SSR
        // settings, so fall back to the defaults (the resolve never runs, so the
        // values are inert, but `SsrResources::new` needs a concrete `SsrSettings`).
        let ssr_build_settings =
            ssr_settings.unwrap_or_else(|| crate::gfx::ssr::SsrSettings::resolve(0.0, 0.0));
        // RT reflections reuse the SSR depth + normal + roughness pre-pass
        // G-buffer (like SSGI), so the pre-pass half is built whenever SSR, SSGI,
        // *or* RT (and the device supports it) is on.
        let rt_wanted = rt_settings.is_some() && rt_capable;
        let ssr_opt = if ssr_settings.is_some() || ssgi_settings.is_some() || rt_wanted {
            let settings = ssr_build_settings;
            let hdr_views: Vec<vk::ImageView> =
                hdr_resolve_images.iter().map(|img| img.view).collect();
            Some(super::post::ssr::SsrResources::new(
                &instance,
                &device,
                physical_device,
                command_pool,
                graphics_queue,
                render_extent.width,
                render_extent.height,
                frames,
                settings,
                &hdr_views,
                env_map.prefilter.view,
                cube_sampler,
                global_set_layout,
                hot_reload,
            )?)
        } else {
            None
        };

        //  Unified geometry G-buffer pre-pass. Built whenever any screen-space
        //  consumer of the merged buffer is on: SSR resolve / SSGI / RT (all
        //  fold into `ssr_opt`), SSAO, or the velocity channel a TAA / upscale
        //  consumer needs (`taa_enabled`). One jittered traversal rasterises the
        //  normal+depth / roughness / velocity MRT every reader then samples,
        //  replacing the separate SSR / SSAO / velocity pre-passes. The skinned
        //  variant is built lazily by `upload_skinned` once the joint-set layout
        //  exists (it doesn't at init). Mirrors the DirectX `self.gbuffer` build.
        let gbuffer_opt = if ssr_opt.is_some() || ssao_opt.is_some() || taa_enabled {
            Some(super::post::gbuffer::GbufferResources::new(
                &instance,
                &device,
                physical_device,
                command_pool,
                graphics_queue,
                render_extent.width,
                render_extent.height,
                frames,
                instance_set_layout_opt,
                // Skinned variant built lazily by `upload_skinned` via
                // `ensure_skinned_gbuffer_pso` (the joint-set layout does not
                // exist yet at init time), matching the gbuffer / TAA.
                None,
                draw_objects.len(),
                hot_reload,
            )?)
        } else {
            None
        };

        //  SSGI (screen-space global illumination): the hemisphere-gather +
        //  depth-aware-blur GI pass. Built only when the world selected
        //  `indirect_lighting: ssgi`; it samples the unified pre-pass G-buffer
        //  (`gbuffer_opt` is guaranteed `Some` here because SSGI forces `ssr_opt`
        //  on, which the gbuffer gate ORs in). The gather samples each frame's
        //  HDR resolve as the bounce-radiance source and the composite additively
        //  blends the denoised indirect term back into the same image on the RMW
        //  chain. Its G-buffer binding is re-pointed at the unified per-frame
        //  views further down; the first view is the valid init placeholder.
        let ssgi_opt = if let Some(settings) = ssgi_settings {
            let gb = gbuffer_opt
                .as_ref()
                .expect("SSGI build forces the unified G-buffer pre-pass to exist");
            let nd_views = gb.normal_depth_views();
            let hdr_views: Vec<vk::ImageView> =
                hdr_resolve_images.iter().map(|img| img.view).collect();
            Some(super::post::ssgi::SsgiResources::new(
                &instance,
                &device,
                physical_device,
                render_extent.width,
                render_extent.height,
                frames,
                settings,
                &hdr_views,
                nd_views[0],
                hot_reload,
            )?)
        } else {
            None
        };

        // Instanced props fold into the GPU-driven bindless cull buffers: each
        // instance becomes a `GpuObjectData` record appended after the `n_objects`
        // static records (written once at init below), so the object / draw-args /
        // indirect / cull-status buffers size for the combined `n_cull` count and
        // the cull kernel tests every instance independently. Skinned objects fold
        // in after the instances (a per-frame-rebuilt tail of `n_skinned` records),
        // so `n_cull` reserves their slots too. Mirrors `directx/init`.
        let n_instances: usize = instanced_clusters.iter().map(|c| c.instances.len()).sum();
        // Streamed-chunk record reserve (`[n_objects + n_instances, +n_chunk_max)`),
        // between the instances and the skinned tail; resident chunks fold in per
        // frame. 0 for a non-voxel world.
        let n_cull = draw_objects.len() + n_instances + n_chunk_max + n_skinned;

        // Bindless static pass: active when the world uses the built-in shader AND
        // there is ANYTHING to GPU-drive -- build-time static geometry, instances,
        // streamed chunks, or skinned meshes (`n_cull > 0`). A pure-voxel world has
        // no build-time geometry but folds its chunks here. Its texture pool is the
        // deduplicated [albedo..] ++ [normal-map..] image set.
        let bindless_active = !is_spirv(vert_bytes) && !is_spirv(frag_bytes) && n_cull > 0;
        let bindless_pool_size = if bindless_active {
            gpu_textures.len() + gpu_normal_maps.len()
        } else {
            0
        };

        //  Descriptor pool
        let n_obj = draw_objects.len().max(1) as u32;
        let n_cluster = instanced_clusters.len() as u32;
        let n_atlas = gpu_text_atlases.len().max(1) as u32;
        let n_frames = frames as u32;
        let bindless_sets_count = if bindless_active { n_frames } else { 0 };
        // GPU-driven G-buffer pre-pass: one set 0 per frame (1 UBO + 1 SSBO),
        // allocated only when the bindless cull path is active AND the G-buffer is
        // enabled. The depth/MRT draw reuses the bindless GpuObjectData set (set 1),
        // so it adds no further sets here.
        let gbuffer_active = bindless_active && gbuffer_opt.is_some();
        let gbuffer_sets_count = if gbuffer_active { n_frames } else { 0 };

        // A pool size with descriptorCount 0 is invalid, so the storage-buffer
        // entry is only added when there are instanced clusters / bindless sets
        // to size it.
        let mut pool_sizes = vec![
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                // global (4 per frame: view + light + shadow + ProbeSet) + shadow
                // global (1 per frame) + gbuffer bindless GbView UBO (1 per frame).
                .descriptor_count(n_frames * 4 + n_frames + gbuffer_sets_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                // per-obj(2) + per-frame {shadow + IBL irradiance + IBL
                // prefilter + SSAO occlusion} + per-frame probe cube array
                // (MAX_PROBES) + text atlas + per-cluster(2) + per-frame
                // composite(3: HDR resolve + bloom mip 0 + 3D colour LUT) +
                // per-frame bindless texture pool.
                .descriptor_count(
                    n_obj * 2
                        + n_frames * 4
                        + n_frames * super::probe_uniforms::MAX_PROBES as u32
                        + n_atlas
                        + n_cluster * 2
                        + n_frames * 3
                        + bindless_pool_size as u32 * bindless_sets_count,
                ),
        ];
        // GPU-driven shadow: one cull set per (frame, cascade), each with 3
        // STORAGE_BUFFER descriptors (objects + draw-args + that cascade's
        // indirect-command buffer). Allocated only when the bindless cull path is
        // active AND shadows are enabled. The depth-only shadow draw reuses the
        // shadow-global + bindless sets, so it adds no sets here.
        let shadow_cull_set_count = if bindless_active && shadow_pipeline_opt.is_some() {
            n_frames * crate::gfx::render_types::NUM_SHADOW_CASCADES as u32
        } else {
            0
        };
        // Storage buffers: one per cluster per frame (instance matrices) + one
        // per frame for the bindless GpuObjectData buffer + four per frame for
        // the GPU-cull set (object + draw-args + indirect-command + cull-status
        // SSBOs) + three per (frame, cascade) for the shadow cull sets. The
        // phase-2 cull sets (two-pass occlusion) draw from their own dedicated
        // pool, so they don't enter this count.
        let storage_count = n_cluster * n_frames
            + bindless_sets_count
            + 4 * bindless_sets_count
            + 3 * shadow_cull_set_count
            // GPU-driven G-buffer: one prev_model SSBO per frame.
            + gbuffer_sets_count;
        if storage_count > 0 {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(storage_count),
            );
        }
        // total sets: global (n_frames) + shadow global (n_frames) + per-obj +
        // per-cluster object set + atlas + per-frame×cluster instance sets +
        // per-frame composite sets + per-frame bindless sets + per-frame
        // GPU-cull sets.
        let total_sets = n_frames
            + n_obj
            + n_frames
            + n_atlas
            + n_cluster
            + n_frames * n_cluster
            + n_frames
            + bindless_sets_count
            + bindless_sets_count
            + shadow_cull_set_count
            + gbuffer_sets_count;
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(total_sets);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| format!("descriptor pool: {e}"))?;

        //  Descriptor sets
        // Global sets (one per frame).
        let global_layouts: Vec<_> = (0..frames).map(|_| global_set_layout).collect();
        let global_sets = alloc_descriptor_sets(&device, descriptor_pool, &global_layouts)?;
        // Update global sets.
        for (i, &set) in global_sets.iter().enumerate() {
            let view_info = vk::DescriptorBufferInfo::default()
                .buffer(view_ubo_buffers[i])
                .offset(0)
                .range(view_ubo_size);
            let light_info = vk::DescriptorBufferInfo::default()
                .buffer(light_ubo)
                .offset(0)
                .range(light_ubo_size);
            let shadow_info = vk::DescriptorBufferInfo::default()
                .buffer(shadow_ubo)
                .offset(0)
                .range(shadow_ubo_size);
            // Layout must match the post-cascade transition in draw.rs, which
            // flips the shadow array to SHADER_READ_ONLY_OPTIMAL.
            let shadow_img_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(shadow_map.view)
                .sampler(shadow_sampler);
            let irr_img_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(env_map.irradiance.view)
                .sampler(cube_sampler);
            let pre_img_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(env_map.prefilter.view)
                .sampler(cube_sampler);
            // SSAO occlusion: this frame's blurred occlusion when SSAO is on
            // (per frame in flight, pooled), or the 1×1 white fallback when it
            // is off. Either way the descriptor is bound so the main pass's
            // `ambient *= ao` always samples a valid texture.
            let ssao_view = transient_pool
                .view_for("ao_output", i)
                .unwrap_or(ssao_white.view);
            let ssao_img_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(ssao_view)
                .sampler(linear_sampler);
            // ProbeSet UBO (binding 7): this frame's reflection-probe set.
            let probe_set_info = vk::DescriptorBufferInfo::default()
                .buffer(probe_set_ubo_buffers[i])
                .offset(0)
                .range(probe_set_ubo_size);
            // Probe cube array (binding 8): every slot points at the IBL prefilter
            // cube until a probe bakes. No descriptor-indexing extension is
            // enabled, so every one of the MAX_PROBES descriptors must hold a valid
            // cube (an unwritten slot is UB); the EMPTY ProbeSet (count 0) keeps the
            // shader on the sky path, so these are never actually sampled yet.
            let probe_cube_infos: Vec<vk::DescriptorImageInfo> = (0
                ..super::probe_uniforms::MAX_PROBES)
                .map(|_| {
                    vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(env_map.prefilter.view)
                        .sampler(cube_sampler)
                })
                .collect();
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&view_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&light_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&shadow_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&shadow_img_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&irr_img_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&pre_img_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&ssao_img_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(7)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&probe_set_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(super::descriptor_layout::PROBE_CUBE_ARRAY_BINDING)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&probe_cube_infos),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        // Shadow global sets (one per frame).
        let shadow_global_layouts: Vec<_> = (0..frames).map(|_| shadow_global_set_layout).collect();
        let shadow_global_sets =
            alloc_descriptor_sets(&device, descriptor_pool, &shadow_global_layouts)?;
        for &set in &shadow_global_sets {
            let su_info = vk::DescriptorBufferInfo::default()
                .buffer(shadow_ubo)
                .offset(0)
                .range(shadow_ubo_size);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&su_info));
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }

        // Per-object sets.
        let object_set_layouts: Vec<_> = draw_objects.iter().map(|_| object_set_layout).collect();
        let object_sets = if object_set_layouts.is_empty() {
            vec![]
        } else {
            let sets = alloc_descriptor_sets(&device, descriptor_pool, &object_set_layouts)?;
            let last_tex = gpu_textures.len().saturating_sub(1);
            let last_nm = gpu_normal_maps.len().saturating_sub(1);
            for (&set, obj) in sets.iter().zip(draw_objects.iter()) {
                let tex_slot = obj.texture_slot.min(last_tex);
                let nm_slot = obj.normal_map_slot.min(last_nm);
                let albedo_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(gpu_textures[tex_slot].view)
                    .sampler(linear_sampler);
                let nm_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(gpu_normal_maps[nm_slot].view)
                    .sampler(linear_sampler);
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(&albedo_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(&nm_info)),
                ];
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }
            sets
        };

        // Bindless static pass: bindless static main pass resources. A dedicated
        // set layout (set 1: SSBO + bindless texture pool), pipeline layout,
        // pipeline, per-frame GpuObjectData storage buffers, and one descriptor
        // set per frame. `None`/empty when the bindless pass is inactive.
        let (
            bindless_pipeline,
            bindless_pipeline_layout,
            bindless_set_layout,
            bindless_sets,
            object_buffers,
            object_buffer_memories,
            object_buffer_ptrs,
        ) = if bindless_active {
            let set_bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(bindless_pool_size as u32)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ];
            let set_layout = unsafe {
                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&set_bindings),
                    None,
                )
            }
            .map_err(|e| format!("bindless set layout: {e}"))?;

            let layouts = [global_set_layout, set_layout];
            let pipeline_layout = unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                    None,
                )
            }
            .map_err(|e| format!("bindless pipeline layout: {e}"))?;

            let (bvs, bfs) = compile_bindless_shaders(hot_reload, bindless_pool_size)?;
            let pipeline = create_main_pipeline(
                &device,
                main_render_pass,
                pipeline_layout,
                &bvs,
                &bfs,
                msaa_samples,
                swapchain_format,
            )?;

            // Per-frame GpuObjectData storage buffers, persistently mapped.
            // Sized for `n_cull` so the instanced merge's records fit past the
            // `n_objects` static prefix.
            let object_buffer_size =
                (n_cull * std::mem::size_of::<crate::gfx::render_types::GpuObjectData>()) as u64;
            let mut buffers = Vec::with_capacity(frames);
            let mut memories = Vec::with_capacity(frames);
            let mut ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
            for _ in 0..frames {
                let (buf, mem) = create_buffer(
                    &instance,
                    &device,
                    physical_device,
                    object_buffer_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?;
                let ptr = unsafe {
                    device
                        .map_memory(mem, 0, object_buffer_size, vk::MemoryMapFlags::empty())
                        .map_err(|e| format!("map object buffer: {e}"))?
                        as *mut u8
                };
                buffers.push(buf);
                memories.push(mem);
                ptrs.push(ptr);
            }

            // One bindless set per frame: binding 0 = that frame's SSBO,
            // binding 1 = the shared pool ([albedo views..] ++ [normal..]).
            let set_layouts: Vec<_> = (0..frames).map(|_| set_layout).collect();
            let sets = alloc_descriptor_sets(&device, descriptor_pool, &set_layouts)?;
            let pool_infos: Vec<vk::DescriptorImageInfo> = gpu_textures
                .iter()
                .chain(gpu_normal_maps.iter())
                .map(|img| {
                    vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(img.view)
                        .sampler(linear_sampler)
                })
                .collect();
            for (i, &set) in sets.iter().enumerate() {
                let buf_info = vk::DescriptorBufferInfo::default()
                    .buffer(buffers[i])
                    .offset(0)
                    .range(object_buffer_size);
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&buf_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&pool_infos),
                ];
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }

            (
                Some(pipeline),
                Some(pipeline_layout),
                Some(set_layout),
                sets,
                buffers,
                memories,
                ptrs,
            )
        } else {
            (
                None,
                None,
                None,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        };

        // Hardware ray-traced reflections: the scene acceleration structure +
        // the inline-`rayQueryEXT` reflection pass. Built only when the world
        // requested it AND the device exposed the ray-query extensions
        // (`rt_wanted`). Reuses the SSR pre-pass G-buffer (forced on above) for
        // the per-pixel surface point + normal, and the bindless pool (when live)
        // for textured hit shading. Graceful-fallback throughout: no resident
        // geometry, an AS build error, or a shader compile failure leaves both
        // `None` and the graph keeps `SsrResolve`. RT takes precedence over the
        // SSR resolve in the shared graph slot, so `ssr_resolve_on` is ANDed with
        // `!rt_active` once the build outcome is known.
        let (rt_accel_opt, rt_opt) = if rt_wanted {
            match crate::vulkan::raytrace::build_rt_accel(
                &instance,
                &device,
                physical_device,
                command_pool,
                graphics_queue,
                vertex_buffer,
                index_buffer,
                &draw_objects,
                &instanced_clusters,
                gpu_textures.len(),
                gpu_normal_maps.len(),
                vertices.len(),
                frames,
                hot_reload,
            ) {
                Ok(Some(accel)) => {
                    let hdr_views: Vec<vk::ImageView> =
                        hdr_resolve_images.iter().map(|i| i.view).collect();
                    // RT reads the unified G-buffer pre-pass's per-frame
                    // normal+depth + roughness (built above whenever any consumer
                    // is on); `gbuffer_opt` is `Some` here because RT forces the
                    // pre-pass on.
                    let gb = gbuffer_opt
                        .as_ref()
                        .expect("RT forces the unified G-buffer pre-pass to exist");
                    let nd_views = gb.normal_depth_views();
                    let rough_views = gb.roughness_views();
                    let (geom_buffer, geom_size) = accel.geom_table();
                    match super::post::rt_reflections::RtReflectionsResources::new(
                        &instance,
                        &device,
                        physical_device,
                        render_extent.width,
                        render_extent.height,
                        frames,
                        rt_settings.expect("rt_wanted implies rt_settings is Some"),
                        vertex_buffer,
                        index_buffer,
                        accel.tlas(),
                        geom_buffer,
                        geom_size,
                        accel.deformed_verts(),
                        accel.skinned_indices(),
                        &hdr_views,
                        &nd_views,
                        &rough_views,
                        env_map.prefilter.view,
                        cube_sampler,
                        bindless_set_layout,
                        global_set_layout,
                        bindless_pool_size,
                        hot_reload,
                    ) {
                        Ok(rt) => (Some(accel), Some(rt)),
                        Err(e) => {
                            tracing::warn!(
                                "RT reflections pass build failed (falling back to SSR): {e}"
                            );
                            let mut accel = accel;
                            accel.destroy(&device);
                            (None, None)
                        }
                    }
                }
                Ok(None) => {
                    tracing::info!(
                        "RT reflections requested but no resident triangle geometry to trace; \
                         using SSR"
                    );
                    (None, None)
                }
                Err(e) => {
                    tracing::warn!(
                        "RT acceleration-structure build failed (falling back to SSR): {e}"
                    );
                    (None, None)
                }
            }
        } else {
            (None, None)
        };
        let rt_active = rt_opt.is_some();
        // The SSR *resolve* owns the post-stack scene image only when SSR is
        // authored and RT did not take the slot.
        let ssr_resolve_on = ssr_authored && !rt_active;
        // How the TLAS tracks moving props (`CN_RT_DYNAMIC`); inert when RT off.
        let rt_dynamic_mode = crate::vulkan::raytrace::RtDynamicMode::from_env();

        // Reflection composite: built whenever a reflection path owns the post-stack
        // scene image (the SSR resolve is active OR RT reflections are active, which
        // are mutually exclusive). Both resolves write radiance+weight into their
        // output target; this blurs by roughness and composites over the scene into
        // its own output, which then replaces the raw resolve output as the scene
        // image every downstream pass samples.
        let composite_opt = if rt_active || ssr_resolve_on {
            let gb = gbuffer_opt
                .as_ref()
                .expect("a reflection path implies the unified G-buffer pre-pass");
            let hdr_views: Vec<vk::ImageView> =
                hdr_resolve_images.iter().map(|img| img.view).collect();
            Some(
                super::post::reflection_composite::ReflectionCompositeResources::new(
                    &instance,
                    &device,
                    physical_device,
                    command_pool,
                    graphics_queue,
                    render_extent.width,
                    render_extent.height,
                    frames,
                    reflection_blur_scale,
                    &hdr_views,
                    &gb.normal_depth_views(),
                    &gb.roughness_views(),
                    hot_reload,
                )?,
            )
        } else {
            None
        };

        // Per-object cull-status buffers (one u32 each), built unconditionally
        // on the bindless cull path: phase-1 cull writes them (binding 3 of the
        // cull set), and phase-2 cull (two-pass occlusion) reads them. Always
        // present so the phase-1 kernel always has a valid binding; under
        // single-pass occlusion the values are simply never read. Device-local.
        // Mirrors `directx/cull.rs`.
        let (cull_status_buffers, cull_status_buffer_memories) = if bindless_active {
            let status_size = n_cull as u64 * std::mem::size_of::<u32>() as u64;
            let mut bufs = Vec::with_capacity(frames);
            let mut mems = Vec::with_capacity(frames);
            for _ in 0..frames {
                let (buf, mem) = create_buffer(
                    &instance,
                    &device,
                    physical_device,
                    status_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?;
                bufs.push(buf);
                mems.push(mem);
            }
            (bufs, mems)
        } else {
            (Vec::new(), Vec::new())
        };

        // Compute cull: the cull compute pipeline + per-frame draw-args /
        // indirect-command buffers + descriptor sets. Built under the same
        // condition as the bindless pass: the compute kernel writes one
        // indirect draw command per build-time object, which the bindless main
        // pass issues with a single multiDrawIndexedIndirect.
        #[allow(clippy::type_complexity)]
        let (
            cull_pipeline,
            cull_pipeline_layout,
            cull_set_layout,
            cull_sets,
            draw_args_buffers,
            draw_args_buffer_memories,
            draw_args_buffer_ptrs,
            indirect_buffers,
            indirect_buffer_memories,
            hiz,
        ): (
            Option<vk::Pipeline>,
            Option<vk::PipelineLayout>,
            Option<vk::DescriptorSetLayout>,
            Vec<vk::DescriptorSet>,
            Vec<vk::Buffer>,
            Vec<vk::DeviceMemory>,
            Vec<*mut u8>,
            Vec<vk::Buffer>,
            Vec<vk::DeviceMemory>,
            Option<crate::vulkan::hiz::HiZResources>,
        ) = if bindless_active {
            // Set 0: object SSBO + draw-args SSBO + indirect-command SSBO +
            // cull-status SSBO (binding 3: phase-1 writes the per-object cull
            // outcome for two-pass occlusion; the phase-2 kernel reads it).
            let set_bindings: Vec<_> = (0..4u32)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = unsafe {
                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&set_bindings),
                    None,
                )
            }
            .map_err(|e| format!("cull set layout: {e}"))?;

            // Hi-Z occlusion resources. Built under the same gating as the cull
            // pipeline; its `read_set_layout` becomes set 1 of the cull
            // pipeline (sampler2D Hi-Z + per-frame CullHizParams UBO).
            let depth_views: Vec<vk::ImageView> = depth_images.iter().map(|img| img.view).collect();
            let hiz = crate::vulkan::hiz::HiZResources::new(
                &instance,
                &device,
                physical_device,
                command_pool,
                graphics_queue,
                render_extent.width,
                render_extent.height,
                msaa_samples.as_raw(),
                frames,
                &depth_views,
                occlusion_two_pass,
                hot_reload,
            )?;

            let push_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(CULL_PUSH_CONSTANT_BYTES);
            let layouts = [set_layout, hiz.read_set_layout];
            let pipeline_layout = unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(std::slice::from_ref(&push_range)),
                    None,
                )
            }
            .map_err(|e| format!("cull pipeline layout: {e}"))?;

            let cs = compile_cull_shader(hot_reload)?;
            let pipeline = create_cull_pipeline(&device, pipeline_layout, &cs)?;

            // Per-frame GpuDrawArgs (host-visible, rebuilt each frame) and
            // indirect-command buffers (device-local, GPU-written). `n_cull`
            // covers the static objects plus the merged instances.
            let n = n_cull as u64;
            let object_buffer_size =
                n * std::mem::size_of::<crate::gfx::render_types::GpuObjectData>() as u64;
            let draw_args_size =
                n * std::mem::size_of::<crate::gfx::render_types::GpuDrawArgs>() as u64;
            let indirect_size = n * std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u64;
            let mut da_buffers = Vec::with_capacity(frames);
            let mut da_memories = Vec::with_capacity(frames);
            let mut da_ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
            let mut ind_buffers = Vec::with_capacity(frames);
            let mut ind_memories = Vec::with_capacity(frames);
            for _ in 0..frames {
                let (da_buf, da_mem) = create_buffer(
                    &instance,
                    &device,
                    physical_device,
                    draw_args_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?;
                let da_ptr = unsafe {
                    device
                        .map_memory(da_mem, 0, draw_args_size, vk::MemoryMapFlags::empty())
                        .map_err(|e| format!("map draw args buffer: {e}"))?
                        as *mut u8
                };
                da_buffers.push(da_buf);
                da_memories.push(da_mem);
                da_ptrs.push(da_ptr);

                let (ind_buf, ind_mem) = create_buffer(
                    &instance,
                    &device,
                    physical_device,
                    indirect_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?;
                ind_buffers.push(ind_buf);
                ind_memories.push(ind_mem);
            }

            // One cull set per frame: that frame's object / draw-args /
            // indirect-command buffers at bindings 0 / 1 / 2.
            let set_layouts: Vec<_> = (0..frames).map(|_| set_layout).collect();
            let sets = alloc_descriptor_sets(&device, descriptor_pool, &set_layouts)?;
            for (i, &set) in sets.iter().enumerate() {
                let obj_info = vk::DescriptorBufferInfo::default()
                    .buffer(object_buffers[i])
                    .offset(0)
                    .range(object_buffer_size);
                let arg_info = vk::DescriptorBufferInfo::default()
                    .buffer(da_buffers[i])
                    .offset(0)
                    .range(draw_args_size);
                let cmd_info = vk::DescriptorBufferInfo::default()
                    .buffer(ind_buffers[i])
                    .offset(0)
                    .range(indirect_size);
                let status_info = vk::DescriptorBufferInfo::default()
                    .buffer(cull_status_buffers[i])
                    .offset(0)
                    .range(n * std::mem::size_of::<u32>() as u64);
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&obj_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&arg_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&cmd_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(3)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&status_info)),
                ];
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }

            (
                Some(pipeline),
                Some(pipeline_layout),
                Some(set_layout),
                sets,
                da_buffers,
                da_memories,
                da_ptrs,
                ind_buffers,
                ind_memories,
                Some(hiz),
            )
        } else {
            (
                None,
                None,
                None,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                None,
            )
        };

        // GPU-driven instanced merge: write each instance's `GpuObjectData`
        // record (+ `GpuDrawArgs`) once into every frame buffer, after the
        // `n_objects` static records. Instances are placed at world load and
        // never move, so these records are static -- the per-frame static fill
        // (`build_object_buffer` / `build_draw_args_buffer`) writes only
        // `[0, n_objects)`, leaving the instance tail intact. Only runs when the
        // bindless cull buffers exist (the bindless pass is active with build-time
        // geometry) and the world declares instanced props. Mirrors
        // `directx/init/mod.rs`.
        if n_instances > 0 && !object_buffer_ptrs.is_empty() {
            use crate::gfx::render_types::{
                GpuDrawArgs, GpuObjectData, draw_args_flags, instance_object_records,
            };
            let records = instance_object_records(
                &instanced_clusters,
                gpu_textures.len() as u32,
                gpu_normal_maps.len() as u32,
            );
            // Cluster base LOD slice (absolute indices, so `base_vertex = 0`);
            // per-instance LOD is a follow-up. Every instance is visible +
            // resident + cullable, so its finite per-instance world AABB is
            // frustum / distance / Hi-Z tested independently by the cull kernel.
            let mut draw_args: Vec<GpuDrawArgs> = Vec::with_capacity(records.len());
            for cluster in &instanced_clusters {
                for _ in &cluster.instances {
                    draw_args.push(GpuDrawArgs {
                        index_count: cluster.index_count as u32,
                        index_offset: cluster.index_offset as u32,
                        base_vertex: 0,
                        flags: draw_args_flags(true, true, true),
                    });
                }
            }
            let n_objects = draw_objects.len();
            let obj_stride = std::mem::size_of::<GpuObjectData>();
            let da_stride = std::mem::size_of::<GpuDrawArgs>();
            for (obj_ptr, da_ptr) in object_buffer_ptrs.iter().zip(draw_args_buffer_ptrs.iter()) {
                // SAFETY: the buffers were sized for `n_objects + n_instances`
                // records, so writing `records.len()` past the `n_objects` offset
                // stays in bounds.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        records.as_ptr() as *const u8,
                        obj_ptr.add(n_objects * obj_stride),
                        records.len() * obj_stride,
                    );
                    std::ptr::copy_nonoverlapping(
                        draw_args.as_ptr() as *const u8,
                        da_ptr.add(n_objects * da_stride),
                        draw_args.len() * da_stride,
                    );
                }
            }
        }

        // GPU-driven shadow pass resources. Built when the bindless cull path is
        // active AND shadows are enabled: a frustum + distance only cull pipeline
        // (`SHADOW_CULL`, lean 3-SSBO set: objects + draw-args + this cascade's
        // indirect buffer), one indirect buffer + cull set per (frame, cascade),
        // and a depth-only bindless graphics pipeline (shadow-global set 0 + the
        // bindless GpuObjectData set 1 + a cascade-index push constant). Each
        // re-rendered cascade then runs one cull dispatch + one
        // `cmd_draw_indexed_indirect` (static + instance prefix) + one for the
        // skinned tail, replacing the CPU per-object shadow loop.
        #[allow(clippy::type_complexity)]
        let (
            shadow_cull_pipeline,
            shadow_cull_pipeline_layout,
            shadow_cull_set_layout,
            shadow_cull_sets,
            shadow_bindless_pipeline,
            shadow_bindless_pipeline_layout,
            shadow_indirect_buffers,
            shadow_indirect_buffer_memories,
        ): (
            Option<vk::Pipeline>,
            Option<vk::PipelineLayout>,
            Option<vk::DescriptorSetLayout>,
            Vec<Vec<vk::DescriptorSet>>,
            Option<vk::Pipeline>,
            Option<vk::PipelineLayout>,
            Vec<Vec<vk::Buffer>>,
            Vec<Vec<vk::DeviceMemory>>,
        ) = if bindless_active
            && shadow_pipeline_opt.is_some()
            && let Some(bl_set_layout) = bindless_set_layout
        {
            let cascades = crate::gfx::render_types::NUM_SHADOW_CASCADES;
            // Lean shadow cull set layout: objects(0) + draw-args(1) + commands(2).
            let sc_bindings: Vec<_> = (0..3u32)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let sc_set_layout = unsafe {
                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&sc_bindings),
                    None,
                )
            }
            .map_err(|e| format!("shadow cull set layout: {e}"))?;

            let sc_push = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(CULL_PUSH_CONSTANT_BYTES);
            let sc_layouts = [sc_set_layout];
            let sc_pl = unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&sc_layouts)
                        .push_constant_ranges(std::slice::from_ref(&sc_push)),
                    None,
                )
            }
            .map_err(|e| format!("shadow cull pipeline layout: {e}"))?;
            let sc_spv = compile_shadow_cull_shader(hot_reload)?;
            let sc_pipeline = create_cull_pipeline(&device, sc_pl, &sc_spv)?;

            // Depth-only bindless shadow graphics pipeline: shadow-global set 0 +
            // the bindless GpuObjectData set 1 + a cascade-index push constant.
            let sb_push = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .offset(0)
                .size(4);
            let sb_layouts = [shadow_global_set_layout, bl_set_layout];
            let sb_pl = unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&sb_layouts)
                        .push_constant_ranges(std::slice::from_ref(&sb_push)),
                    None,
                )
            }
            .map_err(|e| format!("shadow bindless pipeline layout: {e}"))?;
            let sb_spv = compile_shadow_bindless_vs(hot_reload)?;
            let sb_pipeline = create_shadow_pipeline(&device, shadow_render_pass, sb_pl, &sb_spv)?;

            // Per-(frame, cascade) indirect buffers + cull sets. Each cull set
            // binds this frame's object + draw-args SSBOs and this cascade's
            // indirect buffer; the cull dispatch for cascade `c` binds set
            // `[frame][c]`, and the cascade's draws read buffer `[frame][c]`.
            let n = n_cull as u64;
            let object_buffer_size =
                n * std::mem::size_of::<crate::gfx::render_types::GpuObjectData>() as u64;
            let draw_args_size =
                n * std::mem::size_of::<crate::gfx::render_types::GpuDrawArgs>() as u64;
            let indirect_size = n * std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u64;
            let mut sc_indirect_bufs: Vec<Vec<vk::Buffer>> = Vec::with_capacity(frames);
            let mut sc_indirect_mems: Vec<Vec<vk::DeviceMemory>> = Vec::with_capacity(frames);
            let mut sc_sets: Vec<Vec<vk::DescriptorSet>> = Vec::with_capacity(frames);
            for f in 0..frames {
                let mut bufs = Vec::with_capacity(cascades);
                let mut mems = Vec::with_capacity(cascades);
                for _ in 0..cascades {
                    let (buf, mem) = create_buffer(
                        &instance,
                        &device,
                        physical_device,
                        indirect_size,
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::INDIRECT_BUFFER,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )?;
                    bufs.push(buf);
                    mems.push(mem);
                }
                let set_layouts: Vec<_> = (0..cascades).map(|_| sc_set_layout).collect();
                let sets = alloc_descriptor_sets(&device, descriptor_pool, &set_layouts)?;
                for (c, &set) in sets.iter().enumerate() {
                    let obj_info = vk::DescriptorBufferInfo::default()
                        .buffer(object_buffers[f])
                        .offset(0)
                        .range(object_buffer_size);
                    let arg_info = vk::DescriptorBufferInfo::default()
                        .buffer(draw_args_buffers[f])
                        .offset(0)
                        .range(draw_args_size);
                    let cmd_info = vk::DescriptorBufferInfo::default()
                        .buffer(bufs[c])
                        .offset(0)
                        .range(indirect_size);
                    let writes = [
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(&obj_info)),
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(&arg_info)),
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(2)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(&cmd_info)),
                    ];
                    unsafe { device.update_descriptor_sets(&writes, &[]) };
                }
                sc_indirect_bufs.push(bufs);
                sc_indirect_mems.push(mems);
                sc_sets.push(sets);
            }

            (
                Some(sc_pipeline),
                Some(sc_pl),
                Some(sc_set_layout),
                sc_sets,
                Some(sb_pipeline),
                Some(sb_pl),
                sc_indirect_bufs,
                sc_indirect_mems,
            )
        } else {
            (
                None,
                None,
                None,
                Vec::new(),
                None,
                None,
                Vec::new(),
                Vec::new(),
            )
        };

        // GPU-driven G-buffer pre-pass resources. Built when the bindless cull
        // path is active AND the G-buffer is enabled: a 3-MRT bindless pipeline +
        // per-frame previous-frame model SSBOs, drawn by reusing the main pass's
        // per-frame indirect buffer (camera frustum, NO extra cull dispatch). The
        // prev_model buffers' instance region is init-written inside the helper;
        // the static + skinned regions are rewritten each frame.
        #[allow(clippy::type_complexity)]
        let (
            gbuffer_bindless_pipeline,
            gbuffer_bindless_pipeline_layout,
            gbuffer_set_layout,
            gbuffer_sets,
            prev_model_buffers,
            prev_model_memories,
            prev_model_ptrs,
        ): (
            Option<vk::Pipeline>,
            Option<vk::PipelineLayout>,
            Option<vk::DescriptorSetLayout>,
            Vec<vk::DescriptorSet>,
            Vec<vk::Buffer>,
            Vec<vk::DeviceMemory>,
            Vec<*mut u8>,
        ) = if let (true, Some(gb), Some(bl_set_layout)) =
            (gbuffer_active, gbuffer_opt.as_ref(), bindless_set_layout)
        {
            // Per-instance models in cluster-then-instance order (matches the
            // GpuObjectData instance records); the helper init-writes them into the
            // prev_model buffers' instance region for camera-only velocity.
            let inst_models: Vec<[[f32; 4]; 4]> = instanced_clusters
                .iter()
                .flat_map(|c| c.instances.iter().copied())
                .collect();
            let gbb = super::post::gbuffer::build_gbuffer_bindless(
                &instance,
                &device,
                physical_device,
                descriptor_pool,
                bl_set_layout,
                gb,
                &inst_models,
                draw_objects.len(),
                n_cull,
                frames,
                hot_reload,
            )?;
            (
                Some(gbb.pipeline),
                Some(gbb.pipeline_layout),
                Some(gbb.set_layout),
                gbb.sets,
                gbb.prev_model_buffers,
                gbb.prev_model_memories,
                gbb.prev_model_ptrs,
            )
        } else {
            (
                None,
                None,
                None,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        };

        // Two-pass Hi-Z occlusion resources. Built only when the world
        // requested `occlusion_two_pass` AND the bindless cull path is active:
        // the phase-2 cull pipeline (`main_phase2`, same layout as phase 1), a
        // second set of per-frame indirect buffers `Cull2` writes / `Main2`
        // reads, a dedicated descriptor pool + per-frame phase-2 cull sets
        // (bindings 0/1/2/3 = object / draw-args / second-indirect /
        // cull-status), and the phase-1/phase-2 main render passes. The Hi-Z
        // phase-2 cull-read sets live inside `HiZResources` (built above when
        // `occlusion_two_pass`). Mirrors `directx/init/pipelines.rs`.
        #[allow(clippy::type_complexity)]
        let (
            cull_pipeline_phase2,
            cull_sets2,
            two_pass_pool,
            indirect_buffers2,
            indirect_buffer2_memories,
            main_render_pass_phase1,
            main_render_pass_phase2,
        ): (
            Option<vk::Pipeline>,
            Vec<vk::DescriptorSet>,
            Option<vk::DescriptorPool>,
            Vec<vk::Buffer>,
            Vec<vk::DeviceMemory>,
            Option<vk::RenderPass>,
            Option<vk::RenderPass>,
        ) = if let (Some(set_layout), Some(pipeline_layout)) =
            (cull_set_layout, cull_pipeline_layout)
            && occlusion_two_pass
        {
            let n = n_cull as u64;
            let object_buffer_size =
                n * std::mem::size_of::<crate::gfx::render_types::GpuObjectData>() as u64;
            let draw_args_size =
                n * std::mem::size_of::<crate::gfx::render_types::GpuDrawArgs>() as u64;
            let indirect_size = n * std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u64;
            let status_size = n * std::mem::size_of::<u32>() as u64;

            // Phase-2 cull pipeline (`main_phase2` entry, shared layout).
            let cs2 = compile_cull_shader_phase2(hot_reload)?;
            let pipeline2 = create_cull_pipeline(&device, pipeline_layout, &cs2)?;

            // Second indirect-command buffers (device-local, GPU-written).
            let mut ind2_buffers = Vec::with_capacity(frames);
            let mut ind2_memories = Vec::with_capacity(frames);
            for _ in 0..frames {
                let (buf, mem) = create_buffer(
                    &instance,
                    &device,
                    physical_device,
                    indirect_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?;
                ind2_buffers.push(buf);
                ind2_memories.push(mem);
            }

            // Dedicated descriptor pool for the per-frame phase-2 cull sets
            // (4 storage buffers each), kept off the shared pool's exact sizing.
            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(4 * n_frames);
            let pool = unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(std::slice::from_ref(&pool_size))
                        .max_sets(n_frames),
                    None,
                )
            }
            .map_err(|e| format!("two-pass cull descriptor pool: {e}"))?;
            let set_layouts2: Vec<_> = (0..frames).map(|_| set_layout).collect();
            let sets2 = alloc_descriptor_sets(&device, pool, &set_layouts2)?;
            for (i, &set) in sets2.iter().enumerate() {
                let obj_info = vk::DescriptorBufferInfo::default()
                    .buffer(object_buffers[i])
                    .offset(0)
                    .range(object_buffer_size);
                let arg_info = vk::DescriptorBufferInfo::default()
                    .buffer(draw_args_buffers[i])
                    .offset(0)
                    .range(draw_args_size);
                // Binding 2: the *second* indirect buffer (Cull2 writes it).
                let cmd_info = vk::DescriptorBufferInfo::default()
                    .buffer(ind2_buffers[i])
                    .offset(0)
                    .range(indirect_size);
                // Binding 3: the cull-status buffer (phase 1 wrote it; read here).
                let status_info = vk::DescriptorBufferInfo::default()
                    .buffer(cull_status_buffers[i])
                    .offset(0)
                    .range(status_size);
                let infos = [obj_info, arg_info, cmd_info, status_info];
                let writes: Vec<_> = infos
                    .iter()
                    .enumerate()
                    .map(|(b, info)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(b as u32)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(info))
                    })
                    .collect();
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }

            // Phase-1 (STORE MSAA colour) + phase-2 (LOAD colour + depth) main
            // render passes, both compatible with the existing framebuffers.
            let rp1 = create_main_render_pass_two_pass(&device, HDR_FORMAT, msaa_samples, false)?;
            let rp2 = create_main_render_pass_two_pass(&device, HDR_FORMAT, msaa_samples, true)?;

            (
                Some(pipeline2),
                sets2,
                Some(pool),
                ind2_buffers,
                ind2_memories,
                Some(rp1),
                Some(rp2),
            )
        } else {
            (None, Vec::new(), None, Vec::new(), Vec::new(), None, None)
        };

        // Per-cluster (albedo, normal) sets share the per-object layout.
        let cluster_object_sets: Vec<vk::DescriptorSet> = if instanced_clusters.is_empty() {
            Vec::new()
        } else {
            let cluster_layouts: Vec<_> = instanced_clusters
                .iter()
                .map(|_| object_set_layout)
                .collect();
            let sets = alloc_descriptor_sets(&device, descriptor_pool, &cluster_layouts)?;
            let last_tex = gpu_textures.len().saturating_sub(1);
            let last_nm = gpu_normal_maps.len().saturating_sub(1);
            for (cluster, &set) in instanced_clusters.iter().zip(sets.iter()) {
                let tex_slot = cluster.texture_slot.min(last_tex);
                let nm_slot = cluster.normal_map_slot.min(last_nm);
                let albedo_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(gpu_textures[tex_slot].view)
                    .sampler(linear_sampler);
                let nm_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(gpu_normal_maps[nm_slot].view)
                    .sampler(linear_sampler);
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(&albedo_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(&nm_info)),
                ];
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }
            sets
        };

        // Per-frame, per-cluster instance storage buffers (host-mapped).
        let mut instance_buffers: Vec<Vec<vk::Buffer>> = Vec::with_capacity(frames);
        let mut instance_memories: Vec<Vec<vk::DeviceMemory>> = Vec::with_capacity(frames);
        let mut instance_ptrs: Vec<Vec<*mut u8>> = Vec::with_capacity(frames);
        let mut instance_sets: Vec<Vec<vk::DescriptorSet>> = Vec::with_capacity(frames);
        if !instanced_clusters.is_empty() {
            let instance_set_layout = instance_set_layout_opt.unwrap();
            for _ in 0..frames {
                let mut bufs: Vec<vk::Buffer> = Vec::with_capacity(instanced_clusters.len());
                let mut mems: Vec<vk::DeviceMemory> = Vec::with_capacity(instanced_clusters.len());
                let mut ptrs: Vec<*mut u8> = Vec::with_capacity(instanced_clusters.len());
                for cluster in &instanced_clusters {
                    let size_bytes = (cluster.instances.len().max(1)
                        * std::mem::size_of::<[[f32; 4]; 4]>())
                        as vk::DeviceSize;
                    let (buf, mem) = create_buffer(
                        &instance,
                        &device,
                        physical_device,
                        size_bytes,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )?;
                    let ptr = unsafe {
                        device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    }
                    .map_err(|e| format!("map instance buffer: {e}"))?
                        as *mut u8;
                    bufs.push(buf);
                    mems.push(mem);
                    ptrs.push(ptr);
                }
                // Allocate one descriptor set per cluster for this frame.
                let layouts: Vec<_> = instanced_clusters
                    .iter()
                    .map(|_| instance_set_layout)
                    .collect();
                let sets = alloc_descriptor_sets(&device, descriptor_pool, &layouts)?;
                // Wire each set to its buffer.
                for (i, &set) in sets.iter().enumerate() {
                    let info = vk::DescriptorBufferInfo::default()
                        .buffer(bufs[i])
                        .offset(0)
                        .range(vk::WHOLE_SIZE);
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&info));
                    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
                }
                instance_buffers.push(bufs);
                instance_memories.push(mems);
                instance_ptrs.push(ptrs);
                instance_sets.push(sets);
            }
        }

        // Text atlas sets.
        let text_atlas_layouts: Vec<_> = gpu_text_atlases.iter().map(|_| text_set_layout).collect();
        let text_atlas_sets = if text_atlas_layouts.is_empty() {
            vec![]
        } else {
            let sets = alloc_descriptor_sets(&device, descriptor_pool, &text_atlas_layouts)?;
            for (&set, atlas) in sets.iter().zip(gpu_text_atlases.iter()) {
                let img_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(atlas.view)
                    .sampler(text_sampler);
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&img_info));
                unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
            }
            sets
        };

        // Composite sets (one per frame-in-flight slot): binding 0 = the
        // scene image (SSR output when SSR is on, else this slot's HDR
        // resolve), binding 1 = that slot's bloom mip 0, binding 2 = the
        // shared 3D colour LUT. TAA's branch below overrides binding 0 to
        // the TAA output when TAA is on.
        let composite_layouts: Vec<_> = (0..frames).map(|_| composite_set_layout).collect();
        let composite_sets = alloc_descriptor_sets(&device, descriptor_pool, &composite_layouts)?;
        for (i, &set) in composite_sets.iter().enumerate() {
            // Scene image: the reflection composite output (the SSR / RT reflection
            // blended over the scene) when a reflection path is active, else the raw
            // HDR resolve (a SSGI-only build composited its bounce into the latter
            // upstream). TAA / upscale override this below.
            let scene_view = composite_opt
                .as_ref()
                .map(|c| c.output.view)
                .unwrap_or(hdr_resolve_images[i].view);
            write_composite_set(
                &device,
                set,
                scene_view,
                bloom_mips[i][0].view,
                color_lut.view,
                composite_sampler,
            );
        }

        //  Bloom descriptor pool + input sets
        // A dedicated, resettable pool isolates bloom's variable set count
        // (the octave count can shift on resize) from the main pool. Sized for
        // the worst case (`MAX_BLOOM_MIPS + 1` sets per frame).
        let bloom_pool_capacity = n_frames * (MAX_BLOOM_MIPS + 1);
        let bloom_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(bloom_pool_capacity);
        let bloom_descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(std::slice::from_ref(&bloom_pool_size))
                    .max_sets(bloom_pool_capacity),
                None,
            )
        }
        .map_err(|e| format!("bloom descriptor pool: {e}"))?;
        let bloom_input_sets = alloc_bloom_input_sets(
            &device,
            bloom_descriptor_pool,
            bloom_set_layout,
            composite_sampler,
            &hdr_resolve_images,
            &bloom_mips,
        )?;
        // The reflection composite replaces the bloom prefilter's scene input
        // (input 0) with its output, the same scene image the composite pass
        // samples when a reflection path is active and TAA is off (a SSGI-only
        // build leaves the prefilter on the raw HDR resolve). One shared image, so
        // every frame's prefilter input 0 points at it.
        if let Some(view) = composite_opt.as_ref().map(|c| c.output.view) {
            for frame_sets in &bloom_input_sets {
                rebind_bloom_input0(&device, frame_sets[0], view, composite_sampler);
            }
        }

        //  Temporal anti-aliasing
        // When TAA is on the history resolve produces a post-TAA scene image;
        // the bloom prefilter and composite pass must sample that instead of the
        // raw HDR resolve, so their binding-0 descriptor is re-pointed at the
        // per-frame TAA output image.
        let taa = if taa_enabled {
            let taa = TaaResources::new(
                &instance,
                &device,
                physical_device,
                command_pool,
                graphics_queue,
                frames,
                render_extent,
                &hdr_resolve_images,
                composite_sampler,
                hot_reload,
            )?;
            // When a reflection path owns the scene image, TAA samples the
            // reflection composite output (the HDR scene with reflections composited
            // in) instead of the raw HDR resolve. A SSGI-only build leaves TAA on the
            // raw HDR resolve.
            if let Some(view) = composite_opt.as_ref().map(|c| c.output.view) {
                taa.rewire_scene(&device, view, composite_sampler);
            }
            for (i, &set) in composite_sets.iter().enumerate() {
                write_composite_set(
                    &device,
                    set,
                    taa.output_view(i),
                    bloom_mips[i][0].view,
                    color_lut.view,
                    composite_sampler,
                );
            }
            for (i, frame_sets) in bloom_input_sets.iter().enumerate() {
                rebind_bloom_input0(
                    &device,
                    frame_sets[0],
                    taa.output_view(i),
                    composite_sampler,
                );
            }
            Some(taa)
        } else {
            None
        };

        // Temporal upscaling overrides the scene input: when FSR is active the
        // bloom prefilter + composite sample its reconstructed swapchain-res
        // output (a single shared image), not the per-frame TAA output. TAA
        // resources are forced built under upscaling (for the velocity pre-pass)
        // and the TAA block above pointed the sets at the TAA output, so this
        // override is the final word; the TAA *resolve* is dropped from the
        // graph and never runs.
        if let Some(up) = &upscale {
            let up_output_view = up.output_image().view;
            for (i, &set) in composite_sets.iter().enumerate() {
                write_composite_set(
                    &device,
                    set,
                    up_output_view,
                    bloom_mips[i][0].view,
                    color_lut.view,
                    composite_sampler,
                );
            }
            for frame_sets in &bloom_input_sets {
                rebind_bloom_input0(&device, frame_sets[0], up_output_view, composite_sampler);
            }
        }

        //  Unified G-buffer pre-pass reader re-wire
        // Re-point every reader's G-buffer / roughness / velocity descriptor at
        // the merged pre-pass's per-frame views now that the merged buffer + all
        // readers exist. RT was already wired to the unified views at its
        // construction; here we move the SSR resolve, SSGI, SSAO kernel/blur, and
        // the TAA resolve's velocity input. The merged pre-pass produces the
        // byte-identical normal+depth / roughness the separate pre-passes did, so
        // the resolve / kernel maths is unchanged. Mirrors DirectX re-pointing
        // every reader at `self.gbuffer` in init.
        if let Some(gb) = gbuffer_opt.as_ref() {
            let nd_views = gb.normal_depth_views();
            let rough_views = gb.roughness_views();
            let vel_views = gb.velocity_views();
            let hdr_views: Vec<vk::ImageView> =
                hdr_resolve_images.iter().map(|img| img.view).collect();
            if let Some(ssr) = ssr_opt.as_ref() {
                ssr.wire_resolve_sets(
                    &device,
                    &hdr_views,
                    &nd_views,
                    &rough_views,
                    env_map.prefilter.view,
                    cube_sampler,
                );
            }
            if let Some(ssgi) = ssgi_opt.as_ref() {
                ssgi.wire_sets_gbuffer(&device, &hdr_views, &nd_views);
            }
            if let Some(ssao) = ssao_opt.as_ref() {
                ssao.wire_kernel_and_blur_sets_gbuffer(&device, &nd_views);
            }
            if let Some(taa) = taa.as_ref() {
                taa.rewire_velocity(&device, &vel_views, composite_sampler);
            }
        }

        //  Projected decals
        // Pipeline + per-frame uniforms + per-decal albedo sets are always
        // built so runtime `add_decal` works from a world that started
        // with none. The encoder simply skips when every slot is `None`
        // or every live decal culls.
        let depth_views: Vec<vk::ImageView> = depth_images.iter().map(|img| img.view).collect();
        let hdr_resolve_views: Vec<vk::ImageView> =
            hdr_resolve_images.iter().map(|img| img.view).collect();
        let decals_state = Some(crate::vulkan::decal::DecalResources::new(
            &instance,
            &device,
            physical_device,
            command_pool,
            graphics_queue,
            frames,
            msaa_samples != vk::SampleCountFlags::TYPE_1,
            HDR_FORMAT,
            &hdr_resolve_views,
            &depth_views,
            linear_sampler,
            render_extent,
            hot_reload,
        )?);

        // Volumetric fog: pipeline + per-frame uniform ring. Built only
        // when the world declared a `VolumetricFog`; the encoder skips the
        // pass when `fog_settings` is `None`.
        let fog_resources = if fog_settings.is_some() {
            Some(crate::vulkan::fog::FogResources::new(
                &instance,
                &device,
                physical_device,
                command_pool,
                graphics_queue,
                frames,
                msaa_samples != vk::SampleCountFlags::TYPE_1,
                HDR_FORMAT,
                &hdr_resolve_views,
                &depth_views,
                linear_sampler,
                shadow_ubo,
                shadow_map.view,
                shadow_sampler,
                render_extent,
                hot_reload,
            )?)
        } else {
            None
        };

        // Raymarched SDF volumes: per-volume pipelines + the shared view ring,
        // descriptor pool, render passes, and scene snapshot. `None` when no
        // `.glsl` `SdfVolume` survived the backend filter, so the Raymarch pass
        // is omitted from the frame graph.
        let raymarch = crate::vulkan::raymarch::RaymarchResources::try_new(
            &instance,
            &device,
            physical_device,
            command_pool,
            graphics_queue,
            frames,
            msaa_samples,
            render_extent.width,
            render_extent.height,
            shadow_map.view,
            shadow_sampler,
            env_map.irradiance.view,
            env_map.prefilter.view,
            cube_sampler,
            linear_sampler,
            light_ubo,
            shadow_ubo,
            shadow_render_pass,
            &sdf_volumes,
            hot_reload,
        )?;

        // Planar reflections: group each glass pane's world-space plane into a
        // bounded set of distinct reflector planes (near-coplanar panes share one
        // mirror render; panes past the budget fall back to the probe cube), then
        // build one render-resolution mirror target per distinct plane. Built
        // before glass so each pane's planar binding can point at its plane's
        // target. `slots[i]` is pane `i`'s target slot (or `None`).
        let planar_planes: Vec<[f32; 4]> = glass_panels
            .iter()
            .map(|p| crate::vulkan::planar::pane_plane(p.normal, p.centre))
            .collect();
        let planar_assignment = crate::gfx::planar_reflection::assign_planar_slots(
            &planar_planes,
            crate::vulkan::planar::MAX_PLANAR_PLANES,
        );
        // The reflected-frustum mirror cull is bindless-only (it needs the GPU cull
        // set layout + the per-frame object/draw-args SSBOs); a non-bindless world
        // has no `cull_set_layout`, so planar is skipped and its panes keep the
        // probe / sky reflection. Mirrors `metal::planar`'s bindless gate.
        let planar_reflection = if planar_assignment.representatives.is_empty() {
            None
        } else if let Some(csl) = cull_set_layout {
            let cull_sources = crate::vulkan::planar::PlanarCullSources {
                frame_object_buffers: &object_buffers,
                frame_draw_args_buffers: &draw_args_buffers,
                cull_set_layout: csl,
                cull_count: n_cull,
                hiz: hiz.as_ref().map(|h| {
                    let (view, sampler) = h.read_set_sources();
                    (h.read_set_layout, view, sampler)
                }),
            };
            Some(crate::vulkan::planar::PlanarReflectionSet::new(
                &instance,
                &device,
                physical_device,
                frames,
                msaa_samples,
                render_extent.width,
                render_extent.height,
                &planar_assignment.representatives,
                main_render_pass,
                global_set_layout,
                light_ubo,
                light_ubo_size,
                shadow_ubo,
                shadow_ubo_size,
                shadow_map.view,
                shadow_sampler,
                env_map.irradiance.view,
                env_map.prefilter.view,
                cube_sampler,
                ssao_white.view,
                linear_sampler,
                cull_sources,
            )?)
        } else {
            None
        };
        let planar_target_views: Vec<vk::ImageView> = planar_reflection
            .as_ref()
            .map(|s| (0..s.plane_count()).map(|i| s.target_view(i)).collect())
            .unwrap_or_default();

        // Translucent glass panels: the generic producer for the shared
        // transparent pass. `Some` only when the world declared any
        // `GlassPanel`. The pass blends into the post-SSR scene image (SSR
        // output when SSR is on, else this slot's HDR resolve), so the scene
        // target per frame slot is resolved here from `ssr_opt`; the main-depth
        // views feed the fragment shader's manual occlusion test.
        let glass = if glass_panels.is_empty() {
            None
        } else {
            let (glass_scene_views, glass_scene_images): (Vec<vk::ImageView>, Vec<vk::Image>) = (0
                ..frames)
                .map(|i| {
                    // Glass blends into the post-reflection scene: the reflection
                    // composite output when a reflection path is active, else the raw
                    // HDR resolve.
                    if let Some(c) = composite_opt.as_ref() {
                        (c.output.view, c.output.image)
                    } else {
                        (hdr_resolve_images[i].view, hdr_resolve_images[i].image)
                    }
                })
                .unzip();
            let glass_depth_views: Vec<vk::ImageView> =
                depth_images.iter().map(|img| img.view).collect();
            // The initial acceleration-structure handles for the glass RT path
            // (`None` when RT is off at launch; the per-frame `rt_dynamic_update`
            // fills the ring before the RT path is taken). The glass RT pipelines
            // themselves are built whenever the device is RT-capable.
            let glass_rt_inputs = rt_accel_opt.as_ref().map(|a| {
                let (geom_buffer, geom_size) = a.geom_table();
                crate::vulkan::glass::GlassRtInputs {
                    tlas: a.tlas(),
                    geom_buffer,
                    geom_size,
                    deformed_verts: a.deformed_verts(),
                    skinned_indices: a.skinned_indices(),
                }
            });
            Some(crate::vulkan::glass::GlassResources::new(
                &instance,
                &device,
                physical_device,
                command_pool,
                graphics_queue,
                frames,
                msaa_samples,
                render_extent.width,
                render_extent.height,
                &glass_scene_views,
                &glass_scene_images,
                &glass_depth_views,
                linear_sampler,
                global_set_layout,
                &planar_assignment.slots,
                &planar_target_views,
                rt_capable,
                vertex_buffer,
                index_buffer,
                glass_rt_inputs,
                bindless_set_layout,
                bindless_pool_size,
                &glass_panels,
                hot_reload,
            )?)
        };

        // Auto-exposure (EV adaptation): histogram + average compute
        // pipelines, the device-local histogram + output buffers, and the
        // per-frame readback ring. Built only when the world's
        // `PostProcessConfig` opted in. With auto-exposure off every
        // field below is None and the static authored EV continues to
        // drive `post_process.exposure` unchanged.
        let (auto_exposure, auto_exposure_state) =
            if let Some(settings) = auto_exposure_settings.as_ref() {
                let resources = crate::vulkan::auto_exposure::AutoExposureResources::new(
                    &instance,
                    &device,
                    physical_device,
                    frames,
                    &hdr_resolve_views,
                    linear_sampler,
                    hot_reload,
                )?;
                let state = crate::gfx::auto_exposure::AutoExposureState::new(settings);
                (Some(resources), Some(state))
            } else {
                (None, None)
            };

        //  Command buffers
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(frames as u32);
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }
            .map_err(|e| format!("allocate command buffers: {e}"))?;

        //  Parallel command-buffer recording: a `start` outer buffer per frame
        //  (leading timestamp) plus one command pool + primary buffer per
        //  (frame, pass) slot. Vulkan command pools are externally
        //  synchronized, so every slot gets its own pool - the rayon workers in
        //  `execute_graph` never share a pool. `RESET_COMMAND_BUFFER` so each
        //  buffer can be reset + re-recorded per frame (the per-frame
        //  `in_flight` fence gates reuse). Indexed `frame * PASS_COUNT + pass`.
        let pass_pool_flags = vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
            | vk::CommandPoolCreateFlags::TRANSIENT;
        let make_pool_with_buffer =
            |device: &ash::Device| -> Result<(vk::CommandPool, vk::CommandBuffer), String> {
                let pool = unsafe {
                    device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .flags(pass_pool_flags)
                            .queue_family_index(graphics_family),
                        None,
                    )
                }
                .map_err(|e| format!("per-pass command pool: {e}"))?;
                let buf = unsafe {
                    device.allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                }
                .map_err(|e| format!("per-pass command buffer: {e}"))?[0];
                Ok((pool, buf))
            };
        let mut start_command_pools = Vec::with_capacity(frames);
        let mut start_command_buffers = Vec::with_capacity(frames);
        for _ in 0..frames {
            let (pool, buf) = make_pool_with_buffer(&device)?;
            start_command_pools.push(pool);
            start_command_buffers.push(buf);
        }
        let pass_pool_count = frames * crate::gfx::render_graph::PASS_COUNT;
        let mut pass_command_pools = Vec::with_capacity(pass_pool_count);
        let mut pass_command_buffers = Vec::with_capacity(pass_pool_count);
        for _ in 0..pass_pool_count {
            let (pool, buf) = make_pool_with_buffer(&device)?;
            pass_command_pools.push(pool);
            pass_command_buffers.push(buf);
        }

        //  Sync objects
        // `image_available` + `in_flight` are per-frame-in-flight. The
        // render-finished semaphore is signalled by submit and waited on by
        // present, so it must be one-per-swapchain-image (indexed by the
        // acquired image index): a per-frame semaphore can still be queued
        // for presentation when its frame slot comes round again.
        let sem_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mut image_available = Vec::with_capacity(frames);
        let mut in_flight = Vec::with_capacity(frames);
        let mut render_finished = Vec::with_capacity(swapchain_images.len());
        for _ in 0..swapchain_images.len() {
            render_finished.push(
                unsafe { device.create_semaphore(&sem_info, None) }
                    .map_err(|e| format!("semaphore: {e}"))?,
            );
        }
        for _ in 0..frames {
            image_available.push(
                unsafe { device.create_semaphore(&sem_info, None) }
                    .map_err(|e| format!("semaphore: {e}"))?,
            );
            in_flight.push(
                unsafe { device.create_fence(&fence_info, None) }
                    .map_err(|e| format!("fence: {e}"))?,
            );
        }

        let (cull_bvh, always_draw) = crate::gfx::bvh::partition_draw_objects(&draw_objects);

        // Membership flags parallel to `draw_objects` so a recycled draw slot is
        // added to `always_draw` at most once. The free-list allocator starts
        // with every build-time slot already in use; runtime spawns and streamed
        // chunks pop a vacated slot before appending past this count.
        let always_draw_member = {
            let mut member = vec![false; draw_objects.len()];
            for &i in &always_draw {
                member[i as usize] = true;
            }
            member
        };
        let draw_slots = crate::gfx::draw_slot::DrawSlotAllocator::with_len(draw_objects.len());

        let shadow_pipeline_layout_field = if shadow_pipeline_opt.is_some() {
            Some(shadow_pipeline_layout)
        } else {
            unsafe { device.destroy_pipeline_layout(shadow_pipeline_layout, None) };
            None
        };

        // Shader hot-reload: spawn a filesystem watcher over
        // `vulkan/shaders/` only under `cn debug`. The shared atomic flag
        // is also handed to the debug WebSocket server elsewhere so the
        // `reload-shaders` command converges on the same trigger path.
        let (shader_reload_pending, shader_watcher) = if hot_reload {
            let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let watcher = crate::vulkan::hot_reload::spawn(std::sync::Arc::clone(&flag));
            (Some(flag), watcher)
        } else {
            (None, None)
        };

        let mut me = Self {
            instance,
            device,
            physical_device,
            surface,
            surface_loader,
            graphics_queue,
            present_queue,
            graphics_family,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
            render_extent,
            last_present_index: None,
            main_render_pass,
            composite_render_pass,
            msaa_samples,
            color_images,
            depth_images,
            hdr_resolve_images,
            framebuffers,
            composite_framebuffers,
            shadow: VkShadow {
                render_pass: shadow_render_pass,
                map: shadow_map,
                map_size: effective_shadow_size,
                framebuffers: shadow_framebuffers_vec,
                pipeline: shadow_pipeline_opt,
                pipeline_layout: shadow_pipeline_layout_field,
                global_set_layout: Some(shadow_global_set_layout),
                global_sets: shadow_global_sets,
                sampler: shadow_sampler,
                skinned_pipeline: None,
                skinned_pipeline_layout: None,
                ubo: shadow_ubo,
                ubo_memory: shadow_ubo_memory,
                uniforms: shadow_uniforms,
                light_dir: shadow_light_dir,
                update: shadow_update,
                distance: shadow_distance,
                cascades: shadow_cascades,
                scheduler: Default::default(),
                render_mask: 0,
            },
            textures: gpu_textures,
            normal_map_textures: gpu_normal_maps,
            text_atlas_textures: gpu_text_atlases,
            linear_sampler,
            text_sampler,
            main_pipeline,
            main_pipeline_layout,
            cull: VkCull {
                bindless_pipeline,
                bindless_pipeline_layout,
                bindless_set_layout,
                bindless_sets,
                object_buffers,
                object_buffer_memories,
                object_buffer_ptrs,
                cull_pipeline,
                cull_pipeline_layout,
                cull_set_layout,
                cull_sets,
                draw_args_buffers,
                draw_args_buffer_memories,
                draw_args_buffer_ptrs,
                indirect_buffers,
                indirect_buffer_memories,
                cull_status_buffers,
                cull_status_buffer_memories,
                occlusion_two_pass,
                cull_pipeline_phase2,
                cull_sets2,
                two_pass_pool,
                indirect_buffers2,
                indirect_buffer2_memories,
                main_render_pass_phase1,
                main_render_pass_phase2,
                hiz,
                hiz_valid: false,
                hiz_prev_view_proj: IDENTITY4,
                shadow_cull_pipeline,
                shadow_cull_pipeline_layout,
                shadow_cull_set_layout,
                shadow_cull_sets,
                shadow_bindless_pipeline,
                shadow_bindless_pipeline_layout,
                shadow_indirect_buffers,
                shadow_indirect_buffer_memories,
                gbuffer_bindless_pipeline,
                gbuffer_bindless_pipeline_layout,
                gbuffer_set_layout,
                gbuffer_sets,
                prev_model_buffers,
                prev_model_memories,
                prev_model_ptrs,
            },
            text_pipeline: text_pipeline_opt,
            text_pipeline_layout,
            instanced: VkInstanced {
                pipeline: instanced_pipeline_opt,
                pipeline_layout: instanced_pipeline_layout_opt,
                set_layout: instance_set_layout_opt,
                object_sets: cluster_object_sets,
                sets: instance_sets,
                buffers: instance_buffers,
                memories: instance_memories,
                ptrs: instance_ptrs,
                lod_buckets: vec![Vec::new(); instanced_clusters.len()],
                clusters: instanced_clusters,
            },
            composite_pipeline,
            composite_pipeline_layout,
            composite_set_layout,
            composite_sets,
            composite_sampler,
            color_lut,
            bloom_write_pass,
            bloom_blend_pass,
            bloom_pipeline_prefilter,
            bloom_pipeline_downsample,
            bloom_pipeline_upsample,
            bloom_pipeline_layout,
            bloom_set_layout,
            bloom_descriptor_pool,
            bloom_mips,
            bloom_mip_extents,
            bloom_write_framebuffers,
            bloom_blend_framebuffers,
            bloom_input_sets,
            post_process,
            taa,
            upscale,
            upscale_requested: upscale_backend,
            ssao: ssao_opt,
            ssao_white,
            transient_pool,
            ssr: ssr_opt,
            ssr_resolve_active: ssr_resolve_on,
            reflection_composite: composite_opt,
            ssgi: ssgi_opt,
            gbuffer: gbuffer_opt,
            rt_reflections: rt_opt,
            rt_accel: rt_accel_opt,
            rt_dynamic_mode,
            rt_topology_dirty: false,
            rt_capable,
            rt_static_vertex_count: vertices.len(),
            decals_state,
            decals: Vec::new(),
            decal_free_slots: Vec::new(),
            hdr_mode,
            vsync,
            particle_resources: None,
            particles: Vec::new(),
            particle_emitter_state: Vec::new(),
            particle_free_slots: Vec::new(),
            particle_last_elapsed: std::cell::Cell::new(0.0),
            particle_frame_index: std::cell::Cell::new(0),
            fog_resources,
            fog_settings,
            fog_sun_dir,
            fog_sun_color,
            raymarch,
            glass,
            planar_reflection,
            auto_exposure,
            auto_exposure_settings,
            auto_exposure_state,
            auto_exposure_bias_ev,
            auto_exposure_last_elapsed: 0.0,
            hot_reload,
            shader_reload_pending,
            shader_watcher,
            frame_stats: std::cell::Cell::new(crate::gfx::profile::RenderStats::default()),
            draw_calls_accum: std::sync::atomic::AtomicU32::new(0),
            timestamp_query_pool,
            timestamp_period_ns: timestamp_period,
            device_local_heaps,
            memory_budget_supported,
            descriptors: VkDescriptors {
                global_set_layout,
                object_set_layout,
                text_set_layout,
                descriptor_pool,
                global_sets,
                object_sets,
                text_atlas_sets,
            },
            geometry: VkGeometry {
                vertex_buffer,
                vertex_buffer_memory,
                index_buffer,
                index_buffer_memory,
                mesh_vtx_alloc: crate::gfx::range_alloc::RangeAllocator::new(),
                mesh_idx_alloc: crate::gfx::range_alloc::RangeAllocator::new(),
                vertex_buffer_bytes,
                index_buffer_bytes,
            },
            chunk_stream: VkChunkStream {
                vtx_alloc: crate::gfx::range_alloc::RangeAllocator::new(),
                idx_alloc: crate::gfx::range_alloc::RangeAllocator::new(),
                descriptor_pool: None,
                object_set: None,
                texture_slot: None,
                normal_map_slot: None,
            },
            n_objects: draw_objects.len(),
            n_instances,
            // Streamed-chunk record reserve (fixed at init = the worst-case resident
            // chunk window). The cull buffers reserve `[n_objects + n_instances,
            // +n_chunk)`; resident chunks fold in per frame, the unused tail is
            // disabled. 0 for a non-voxel world.
            n_chunk: n_chunk_max,
            // Set in `upload_skinned` once the skin fold is built; the cull buffers
            // reserve the tail at init via the threaded `n_skinned` capacity, but
            // `cull_count()` reads this runtime count.
            n_skinned: 0,
            clone_descriptor_pool: None,
            clone_object_sets: Vec::new(),
            clone_free_offsets: Vec::new(),
            clone_slot_by_draw_idx: std::collections::HashMap::new(),
            clone_texture_slots: Vec::new(),
            clone_normal_map_slots: Vec::new(),
            skinned: VkSkinned {
                pipeline: None,
                pipeline_layout: None,
                joint_set_layout: None,
                descriptor_pool: None,
                vertex_buffer: vk::Buffer::null(),
                vertex_buffer_memory: vk::DeviceMemory::null(),
                vertex_buffer_bytes: 0,
                index_buffer: vk::Buffer::null(),
                index_buffer_memory: vk::DeviceMemory::null(),
                index_buffer_bytes: 0,
                draw_objects: Vec::new(),
                object_sets: Vec::new(),
                joint_buffers: Vec::new(),
                joint_memories: Vec::new(),
                joint_ptrs: Vec::new(),
                joint_sets: Vec::new(),
                joint_matrices: Vec::new(),
                skin: None,
                deformed: Vec::new(),
                deformed_primed: std::sync::atomic::AtomicBool::new(false),
            },
            skinned_pool: crate::gfx::skinned_pool::SkinnedInstancePool::new(),
            uniforms: VkUniforms {
                view_ubo_buffers,
                view_ubo_memories,
                view_ubo_ptrs,
                probe_set_ubo_buffers,
                probe_set_ubo_memories,
                probe_set_ubo_ptrs,
                light_ubo,
                light_ubo_memory,
                light_uniforms,
            },
            frame_sync: VkFrameSync {
                image_available,
                render_finished,
                in_flight,
            },
            current_frame: 0,
            frames_in_flight: frames,
            commands: VkCommands {
                command_pool,
                command_buffers,
                start_command_pools,
                start_command_buffers,
                pass_command_pools,
                pass_command_buffers,
            },
            cull_bvh,
            always_draw,
            always_draw_member,
            draw_slots,
            visible_scratch: Vec::new(),
            draw_objects,
            clear_color,
            view_matrix: IDENTITY4,
            prefilter_mip_count: env_map.prefilter_mip_count,
            cube_sampler,
            env_map,
            probe_placements: Vec::new(),
            probe_set: super::probe_uniforms::ProbeSet::EMPTY,
            probe_maps: Vec::new(),
            probe_bake_queue: crate::gfx::reflection_probe::ProbeBakeQueue::new(0),
            probe_rendering: None,
            probe_converting: None,
            deferred_destroy: RefCell::new(Vec::new()),
            window,
            debug_utils,
            debug_messenger,
            debug_filter,
            _entry: entry,
        };
        // Push every world-authored `DecalRecord` through `add_decal` so
        // its albedo descriptor lands in the reserved slot before the
        // first frame runs.
        me.upload_initial_decals(decals)?;
        // Same pattern for particle emitters: each world-authored record
        // routes through `add_particle_emitter` so its pool, counter, and
        // descriptor sets land before the first frame.
        me.upload_initial_particles(particles)?;
        Ok(me)
    }
}

//  Shadow uniforms
//
// Per-frame cascade computation lives in `gfx::csm::compute_shadow_uniforms`
// and is invoked from `draw.rs` each frame using the current view matrix +
// camera position. The init path no longer computes a shadow VP; it just
// stores `empty_shadow_uniforms()` so the descriptor write at startup has a
// valid (fully-lit) buffer.

// Validation layer debug callback: logs validation errors and warnings.
// DLSS's first EvaluateFeature samples two NGX-internal resources it leaves in
// UNDEFINED, tripping VUID-vkCmdDraw-None-09600 exactly twice per feature
// creation. They are internal to nvngx_dlss.dll (not bindable through the NGX
// parameter API, confirmed by supplying our own exposure input, which did not
// displace them) and benign (the upscale output is correct). The debug messenger
// drops this many such messages while DLSS is the active upscaler. D3D12 never
// surfaces them (it has no image-layout validation model).
pub(super) const DLSS_FIRST_FRAME_LAYOUT_SUPPRESS: u32 = 2;

// Decide whether to drop a validation message rather than log it: true only for
// the benign DLSS first-frame layout VUID while `budget` is positive (consuming
// one unit of it). Every other VUID, and an exhausted budget, returns false so
// the message still surfaces. Split out from `debug_callback` so the suppression
// logic is unit testable without a live Vulkan instance.
fn drop_benign_dlss_layout_error(message_id: &[u8], budget: &std::sync::atomic::AtomicU32) -> bool {
    if message_id != b"VUID-vkCmdDraw-None-09600" {
        return false;
    }
    budget
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |n| (n > 0).then(|| n - 1),
        )
        .is_ok()
}

// Validation messages route through here (installed only when validation is on).
// `user` is a `*const AtomicU32`: a budget of benign DLSS first-frame layout
// errors to drop, set after `build_upscaler` resolves to DLSS (and reset on
// resize, which re-creates the feature). Null when no budget is wired.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if data.is_null() {
        return vk::FALSE;
    }
    let data = unsafe { &*data };

    // Drop the benign DLSS first-frame layout errors (see the helper); any other
    // VUID, or an exhausted budget, still logs.
    if !user.is_null() && !data.p_message_id_name.is_null() {
        let vuid = unsafe { CStr::from_ptr(data.p_message_id_name) };
        let budget = unsafe { &*(user as *const std::sync::atomic::AtomicU32) };
        if drop_benign_dlss_layout_error(vuid.to_bytes(), budget) {
            return vk::FALSE;
        }
    }

    let msg = unsafe { CStr::from_ptr(data.p_message) }.to_string_lossy();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        tracing::error!("[Vulkan] {}", msg);
    } else {
        tracing::warn!("[Vulkan] {}", msg);
    }
    vk::FALSE
}

#[cfg(test)]
mod tests {
    use super::{DLSS_FIRST_FRAME_LAYOUT_SUPPRESS, drop_benign_dlss_layout_error};
    use std::sync::atomic::{AtomicU32, Ordering};

    const LAYOUT_VUID: &[u8] = b"VUID-vkCmdDraw-None-09600";

    #[test]
    fn drops_exactly_the_budgeted_layout_errors_then_logs() {
        let budget = AtomicU32::new(DLSS_FIRST_FRAME_LAYOUT_SUPPRESS);
        for _ in 0..DLSS_FIRST_FRAME_LAYOUT_SUPPRESS {
            assert!(drop_benign_dlss_layout_error(LAYOUT_VUID, &budget));
        }
        // Budget spent: a further occurrence logs, so a real bug would surface.
        assert!(!drop_benign_dlss_layout_error(LAYOUT_VUID, &budget));
        assert_eq!(budget.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn never_drops_other_vuids_or_touches_budget() {
        let budget = AtomicU32::new(DLSS_FIRST_FRAME_LAYOUT_SUPPRESS);
        assert!(!drop_benign_dlss_layout_error(
            b"VUID-vkCmdDraw-None-02699",
            &budget
        ));
        assert!(!drop_benign_dlss_layout_error(b"", &budget));
        assert_eq!(
            budget.load(Ordering::Relaxed),
            DLSS_FIRST_FRAME_LAYOUT_SUPPRESS
        );
    }

    #[test]
    fn drops_nothing_when_budget_is_zero() {
        let budget = AtomicU32::new(0);
        assert!(!drop_benign_dlss_layout_error(LAYOUT_VUID, &budget));
    }
}
