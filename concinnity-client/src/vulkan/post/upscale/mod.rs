// src/vulkan/post/upscale/mod.rs
//
// Temporal upscaling for the Vulkan backend. The engine renders the 3D scene at
// a fraction of the swapchain extent (`render_extent`) and the
// `PassId::Upscale` pass reconstructs a swapchain-resolution image the bloom +
// composite stack consumes.
//
// Three interchangeable backends sit behind the `VkUpscaleBackend` trait,
// mirroring the DirectX `directx/post/upscale/` split:
//   fsr   AMD FidelityFX FSR (cross-vendor; the default fallback; ffx_api VK)
//   dlss  NVIDIA DLSS via raw NGX (RTX only; cfg(ngx_sdk_bundled))
//   xess  Intel XeSS (cross-vendor; runtime libxess.dll)
// `build_upscaler` resolves the requested `UpscalerBackend` against runtime
// availability and constructs the first that initialises, falling back to
// native-resolution rendering when none is available. The shared per-frame
// `VkContext::encode_upscale` (below) drives whichever backend is active through
// the trait; only the inner vendor dispatch differs.
//
// DLSS and XeSS additionally need Vulkan instance / device extensions (and, for
// XeSS, device features) enabled at instance / device creation, before the
// upscaler context exists. `UpscaleSdk` is queried up front (in `init.rs`,
// before `create_instance`) and threaded into `device::create_logical_device`;
// see its docs.

use std::cell::Cell;
use std::ffi::{CStr, CString, c_char};

use ash::{Device, vk};

use crate::assets::UpscalerBackend;
use crate::vulkan::context::{HDR_FORMAT, VkContext};
use crate::vulkan::graph_exec::GraphFrameParams;
use crate::vulkan::texture::{GpuImage, create_image, create_image_view, one_shot_submit};

#[cfg(ngx_sdk_bundled)]
mod dlss;
mod fsr;
mod xess;

// One render-resolution input image handed to a backend's `dispatch`. FSR only
// needs the raw `image` + a backend-chosen format; DLSS / XeSS need the full
// view + format + dimensions (their resource descriptors carry a `VkImageView`
// and `VkFormat`). Carrying all of it keeps the trait uniform.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct UpscaleImage {
    pub(in crate::vulkan) image: vk::Image,
    pub(in crate::vulkan) view: vk::ImageView,
    pub(in crate::vulkan) format: vk::Format,
    pub(in crate::vulkan) width: u32,
    pub(in crate::vulkan) height: u32,
    pub(in crate::vulkan) aspect: vk::ImageAspectFlags,
}

// One temporal-upscaling backend. `encode_upscale` (below) transitions the
// scene / depth / motion inputs and the output image, then calls `dispatch`;
// each backend records its vendor upscale onto the supplied command buffer.
// Smaller than the DX trait: Vulkan has no descriptor-heap plumbing, and the
// output is a self-contained `GpuImage` whose state is a single `vk::ImageLayout`
// (GENERAL while written / SHADER_READ_ONLY while sampled).
pub(in crate::vulkan) trait VkUpscaleBackend: Send {
    // Off-screen scene render dimensions (the backend's input size).
    fn render_dims(&self) -> (u32, u32);
    // Swapchain (output) dimensions the backend reconstructs.
    fn output_dims(&self) -> (u32, u32);
    // Per-axis render-to-output ratio resolved from the quality preset.
    fn scale(&self) -> f32;
    // The output image the bloom + composite stack samples as the scene.
    fn output_image(&self) -> &GpuImage;
    // Whether the output currently rests in GENERAL (the write window) vs
    // SHADER_READ_ONLY (the post-dispatch sample window). Tracked across frames
    // by `encode_upscale`.
    fn output_layout(&self) -> vk::ImageLayout;
    fn set_output_layout(&self, layout: vk::ImageLayout);
    // Sub-pixel jitter for this frame's index, shared with the camera
    // projection so the jittered VP and the upscale agree (render-pixel units).
    fn jitter_offset(&self, frame_index: u32) -> [f32; 2];
    // Stash this frame's jitter (set from `draw.rs` on the main thread before
    // the parallel fan-out) and read it back on the worker in `encode_upscale`.
    fn set_jitter(&self, offset: [f32; 2]);
    fn jitter(&self) -> [f32; 2];
    // Record the upscale onto `cmd`. Inputs are claimed in the layouts
    // `encode_upscale` transitioned them into (color / motion / depth in
    // SHADER_READ_ONLY_OPTIMAL, output in GENERAL). `elapsed` is the frame's
    // elapsed-seconds stamp; each backend keeps its own frame-delta clock +
    // reset state.
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        color: &UpscaleImage,
        depth: &UpscaleImage,
        motion: &UpscaleImage,
        jitter_offset: [f32; 2],
        elapsed: f32,
        camera_near: f32,
        camera_far: f32,
        camera_fov_y_radians: f32,
    ) -> Result<(), String>;
    // Tear down owned GPU + SDK resources. Called from `VkContext::drop` after
    // `device_wait_idle`.
    fn destroy(&mut self, device: &Device);
}

// Per-axis render-to-output resolution split, shared by all three backends.
// The temporal kernels support up to a 3x per-axis upscale (ratio >= 1/3);
// clamp the requested scale into `[1/3, 1]` so an out-of-range quality preset
// can't ask for an unsupported ratio. A `1.0` scale makes `render == output`
// (TAA-replacement mode). Returns the render dims + the clamped scale used.
pub(super) fn resolve_render_dims(
    output_width: u32,
    output_height: u32,
    upscale_scale: f32,
) -> (u32, u32, f32) {
    let scale = if upscale_scale > 0.0 {
        upscale_scale.clamp(1.0 / 3.0, 1.0)
    } else {
        1.0
    };
    let render_width = (((output_width as f32) * scale).round() as u32).max(1);
    let render_height = (((output_height as f32) * scale).round() as u32).max(1);
    (render_width, render_height, scale)
}

// Frame-delta in milliseconds from a per-backend `prev_elapsed` clock. `prev`
// starts at 0.0, so the first frame's raw `(now - prev)` is whatever
// elapsed-since-startup happens to be; the temporal heuristics expect
// frame-time-ish numbers, so clamp to [1, 100] ms. Shared by the backends that
// consume it (FSR); DLSS / XeSS ignore it but still advance their clock.
pub(super) fn frame_delta_ms(prev: &Cell<f32>, now: f32) -> f32 {
    let last = prev.replace(now);
    ((now - last) * 1000.0).clamp(1.0, 100.0)
}

// Sub-pixel jitter shared by the DLSS + XeSS backends (FSR queries its own
// FFX-prescribed sequence instead). A 16-phase Halton-2/3 sequence in
// [-0.5, 0.5] render-pixel units; the same value jitters the camera projection
// (see `draw.rs`) so the rasterised scene and the upscale agree.
pub(super) fn halton_jitter_offset(frame_index: u32) -> [f32; 2] {
    let idx = (frame_index % 16) + 1;
    [radical_inverse(idx, 2) - 0.5, radical_inverse(idx, 3) - 0.5]
}

// Van der Corput radical inverse of `i` in the given base, in [0, 1).
fn radical_inverse(mut i: u32, base: u32) -> f32 {
    let inv_base = 1.0 / base as f32;
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    while i > 0 {
        f *= inv_base;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

// Create the display-res output image a backend writes (RGBA16F,
// STORAGE | SAMPLED), transitioned UNDEFINED -> GENERAL so the first frame's
// dispatch finds it in the UNORDERED_ACCESS (GENERAL) state. Shared by all
// three backends.
pub(super) fn create_output_image(
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
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::SampleCountFlags::TYPE_1,
    )?;
    let view = create_image_view(device, image, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
    one_shot_submit(device, command_pool, queue, |cmd| {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        }
    })?;
    Ok(GpuImage {
        image,
        memory,
        view,
        aux_views: Vec::new(),
    })
}

// One image barrier with explicit stages/access (the upscalers read their
// inputs in the COMPUTE stage; the generic `transition_image_layout` helper
// targets FRAGMENT, which would not synchronise the compute reads).
#[allow(clippy::too_many_arguments)]
pub(super) fn image_barrier(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_stage: vk::PipelineStageFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&barrier),
        );
    }
}

// Backend selection

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::vulkan) enum ResolvedBackend {
    Fsr,
    Dlss,
    Xess,
    Native,
}

// Ordered candidate list for a requested backend + availability: the
// explicitly requested one first (when available), then the Auto priority
// order (DLSS, XeSS, FSR), then Native (always last, always available).
// Mirrors `directx::post::upscale::backend_order`.
fn backend_order(
    requested: UpscalerBackend,
    dlss_avail: bool,
    xess_avail: bool,
    fsr_avail: bool,
) -> Vec<ResolvedBackend> {
    let mut order: Vec<ResolvedBackend> = Vec::new();
    match requested {
        UpscalerBackend::Dlss if dlss_avail => order.push(ResolvedBackend::Dlss),
        UpscalerBackend::Xess if xess_avail => order.push(ResolvedBackend::Xess),
        UpscalerBackend::Fsr3 if fsr_avail => order.push(ResolvedBackend::Fsr),
        _ => {}
    }
    for (cand, avail) in [
        (ResolvedBackend::Dlss, dlss_avail),
        (ResolvedBackend::Xess, xess_avail),
        (ResolvedBackend::Fsr, fsr_avail),
    ] {
        if avail && !order.contains(&cand) {
            order.push(cand);
        }
    }
    order.push(ResolvedBackend::Native);
    order
}

// Compile-time availability of each backend. DLSS is gated on the NGX static
// lib being linked; XeSS / FSR are runtime-loaded DLLs but the `*_sdk_bundled`
// cfg mirrors DX's gating (the DLL is on the candidate list only when build.rs
// bundled it).
fn dlss_available() -> bool {
    cfg!(ngx_sdk_bundled)
}
fn xess_available() -> bool {
    cfg!(xess_sdk_bundled)
}
fn fsr_available() -> bool {
    cfg!(ffx_sdk_bundled)
}

// Construct the upscaler for the requested backend, falling through the
// candidate order on any `try_new` that returns `None` (DLL miss, unsupported
// GPU, context-init failure). Returns the boxed backend (or `None` for native
// rendering) and the tag that actually built. The instance / device extensions
// for the *first* candidate were enabled at device creation (see `UpscaleSdk`);
// a fallback past that candidate can only land on FSR / Native (which need no
// extra extensions), so a DLSS / XeSS context-create failure degrades to FSR.
#[allow(clippy::too_many_arguments)]
pub(in crate::vulkan) fn build_upscaler(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    output_width: u32,
    output_height: u32,
    upscale_scale: f32,
    requested: UpscalerBackend,
) -> Result<(Option<Box<dyn VkUpscaleBackend>>, ResolvedBackend), String> {
    for cand in backend_order(
        requested,
        dlss_available(),
        xess_available(),
        fsr_available(),
    ) {
        let built: Option<Box<dyn VkUpscaleBackend>> = match cand {
            ResolvedBackend::Fsr => fsr::FsrUpscaler::try_new(
                instance,
                device,
                physical_device,
                command_pool,
                queue,
                output_width,
                output_height,
                upscale_scale,
            )?
            .map(|u| Box::new(u) as Box<dyn VkUpscaleBackend>),
            ResolvedBackend::Xess => xess::XessUpscaler::try_new(
                instance,
                device,
                physical_device,
                command_pool,
                queue,
                output_width,
                output_height,
                upscale_scale,
            )?
            .map(|u| Box::new(u) as Box<dyn VkUpscaleBackend>),
            ResolvedBackend::Dlss => {
                #[cfg(ngx_sdk_bundled)]
                {
                    dlss::DlssUpscaler::try_new(
                        instance,
                        device,
                        physical_device,
                        command_pool,
                        queue,
                        output_width,
                        output_height,
                        upscale_scale,
                    )?
                    .map(|u| Box::new(u) as Box<dyn VkUpscaleBackend>)
                }
                #[cfg(not(ngx_sdk_bundled))]
                {
                    None
                }
            }
            ResolvedBackend::Native => None,
        };
        if let Some(b) = built {
            tracing::info!(
                "temporal upscaling: using {cand:?} backend (output {output_width}x{output_height})"
            );
            return Ok((Some(b), cand));
        }
        if cand != ResolvedBackend::Native {
            tracing::warn!("temporal upscaling: {cand:?} unavailable, trying next backend");
        }
    }
    tracing::info!("temporal upscaling: no backend available, rendering at native resolution");
    Ok((None, ResolvedBackend::Native))
}

// Vulkan instance / device extension requirements for DLSS / XeSS, resolved
// before `create_instance`. DLSS and XeSS each need extensions (and XeSS device
// features) enabled at creation time, queried from the SDK before the device
// exists. `prepare` runs first (loading only the chosen SDK and calling its
// extension-enumeration entry points, which need at most the loaded DLL); the
// instance extensions feed `create_instance`, and the struct is then threaded
// into `device::create_logical_device` for the device extensions / features.
// Inert (`choice == Native`, empty lists) when upscaling is off or the chosen
// backend needs nothing.
pub(in crate::vulkan) struct UpscaleSdk {
    pub(in crate::vulkan) choice: ResolvedBackend,
    // Held so XeSS's SDK-owned device-feature chain stays mapped through
    // `vkCreateDevice` (the chain memory is owned by libxess.dll). `None` for
    // DLSS (static-linked) / FSR / Native.
    xess: Option<xess::XessExtQuery>,
    // Owned instance-extension names merged into the instance create info. Held
    // here so the raw pointers from `instance_extension_ptrs` stay valid until
    // `create_instance` consumes them.
    instance_exts: Vec<CString>,
    // DLSS device extensions captured up front (NGX's RequiredExtensions yields
    // both instance + device lists in one call). XeSS queries device extensions
    // later, in `create_logical_device` (they need the instance + physical
    // device).
    dlss_device_exts: Vec<CString>,
    // Minimum Vulkan instance `apiVersion` the chosen backend needs (XeSS 3.x
    // requires 1.3 for SPV_KHR_integer_dot_product). 0 = no requirement beyond
    // the engine default. The caller clamps to loader support.
    min_api_version: u32,
}

impl UpscaleSdk {
    // Resolve which backend's extensions to enable and query the instance
    // extensions for it. Never fails: any SDK miss / query error degrades the
    // choice to a backend that needs no extra extensions, so instance / device
    // creation proceeds exactly as without upscaling.
    pub(in crate::vulkan) fn prepare(temporal_upscaling: bool, requested: UpscalerBackend) -> Self {
        let mut sdk = UpscaleSdk {
            choice: ResolvedBackend::Native,
            xess: None,
            instance_exts: Vec::new(),
            dlss_device_exts: Vec::new(),
            min_api_version: 0,
        };
        if !temporal_upscaling {
            return sdk;
        }
        let first = backend_order(
            requested,
            dlss_available(),
            xess_available(),
            fsr_available(),
        )[0];
        sdk.choice = first;
        match first {
            ResolvedBackend::Dlss =>
            {
                #[cfg(ngx_sdk_bundled)]
                match dlss::required_extensions() {
                    Some((inst, dev)) => {
                        sdk.instance_exts = inst;
                        sdk.dlss_device_exts = dev;
                    }
                    None => {
                        tracing::warn!(
                            "temporal upscaling: DLSS required-extensions query failed; \
                             device creation will skip DLSS extensions (build_upscaler will \
                             fall back to FSR / native)"
                        );
                        sdk.choice = ResolvedBackend::Fsr;
                    }
                }
            }
            ResolvedBackend::Xess => match xess::XessExtQuery::load() {
                Some(q) => {
                    let (exts, min_api) = q.instance_extensions();
                    sdk.instance_exts = exts;
                    sdk.min_api_version = min_api;
                    sdk.xess = Some(q);
                }
                None => {
                    tracing::warn!(
                        "temporal upscaling: XeSS DLL / extension query unavailable; device \
                         creation will skip XeSS extensions (build_upscaler will fall back to \
                         FSR / native)"
                    );
                    sdk.choice = ResolvedBackend::Fsr;
                }
            },
            _ => {}
        }
        sdk
    }

    // Raw instance-extension name pointers for `create_instance`. Valid as long
    // as `self` lives (the `CString`s are owned by `self.instance_exts`).
    pub(in crate::vulkan) fn instance_extension_ptrs(&self) -> Vec<*const std::os::raw::c_char> {
        self.instance_exts.iter().map(|c| c.as_ptr()).collect()
    }

    // Minimum Vulkan instance `apiVersion` the chosen backend needs (0 = none).
    pub(in crate::vulkan) fn min_api_version(&self) -> u32 {
        self.min_api_version
    }

    // Device-extension names required by the chosen backend, filtered to those
    // the physical device actually exposes and not already requested in
    // `already`. DLSS returns its up-front list; XeSS queries now (needs the
    // instance + physical device).
    pub(in crate::vulkan) fn device_extensions(
        &self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        already: &[CString],
    ) -> Vec<CString> {
        let supported = supported_device_extensions(instance, physical_device);
        let raw: Vec<CString> = match self.choice {
            ResolvedBackend::Dlss => self.dlss_device_exts.clone(),
            ResolvedBackend::Xess => self
                .xess
                .as_ref()
                .map(|q| q.device_extensions(instance, physical_device))
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        raw.into_iter()
            .filter(|name| supported.iter().any(|s| s == name))
            .filter(|name| !already.iter().any(|a| a == name))
            .collect()
    }

    // The XeSS-required device-feature chain head (an SDK-owned `pNext` chain to
    // splice into `VkDeviceCreateInfo`), or null for every other backend. The
    // chain memory is owned by libxess.dll and stays valid while `self` lives
    // (it holds the loaded library), which spans `vkCreateDevice`. `head` is the
    // caller's existing `pNext` chain that the XeSS chain is appended in front
    // of, so the SDK can also patch fields on the caller's structs.
    pub(in crate::vulkan) fn xess_device_features(
        &self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        head: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void {
        match (self.choice, self.xess.as_ref()) {
            (ResolvedBackend::Xess, Some(q)) => q.device_features(instance, physical_device, head),
            _ => head,
        }
    }
}

// Copy a `const char* const*` array (`count` entries) returned by an SDK
// extension query into owned `CString`s, severing the dependence on the
// SDK-owned memory. Shared by the DLSS + XeSS extension queries.
//
// SAFETY: `exts` must be null or point to `count` valid, null-terminated C
// strings (the SDK contract).
pub(super) unsafe fn copy_ext_names(count: u32, exts: *const *const c_char) -> Vec<CString> {
    if exts.is_null() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        let p = unsafe { *exts.add(i) };
        if !p.is_null() {
            out.push(unsafe { CStr::from_ptr(p) }.to_owned());
        }
    }
    out
}

// Names of every device extension the physical device exposes, as owned
// `CString`s for equality checks against SDK-requested names.
fn supported_device_extensions(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Vec<CString> {
    let props = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .unwrap_or_default();
    props
        .iter()
        .map(|e| {
            let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
            CString::from(name)
        })
        .collect()
}

impl VkContext {
    // Encode the temporal upscale onto `cmd`. Runs after SSR resolve / fog /
    // particles / transparent (so the scene input is the fully decorated
    // post-SSR colour) and before Bloom + Composite (which sample the
    // upscaler's output, rewired at init / resize). Recorded onto the
    // `PassId::Upscale` per-pass command buffer by the executor. Backend-
    // agnostic: the barrier choreography (output GENERAL, color / motion / depth
    // SHADER_READ_ONLY) is identical for FSR / DLSS / XeSS; only the inner
    // `dispatch` differs.
    pub(in crate::vulkan) fn encode_upscale(
        &self,
        cmd: vk::CommandBuffer,
        params: &GraphFrameParams<'_>,
    ) -> Result<(), String> {
        let upscaler = match &self.upscale {
            Some(u) => u,
            None => return Ok(()),
        };
        let frame = params.frame_idx;

        // One-time info log confirming the worker arm fires.
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let (rw, rh) = upscaler.render_dims();
            let (ow, oh) = upscaler.output_dims();
            tracing::info!(
                "temporal upscaling: first encode_upscale firing (render {rw}x{rh} -> upscale {ow}x{oh})"
            );
        }

        // Render-res motion + depth the upscalers consume. The unified G-buffer
        // pre-pass owns these (the `velocity` MRT target + the pre-pass's private
        // depth, both STORE'd shader-readable / depth attachment). Init builds the
        // merged pre-pass whenever upscaling is on (it forces `taa_enabled`), so it
        // is always present here; velocity rests in SHADER_READ_ONLY and depth in
        // DEPTH_STENCIL_ATTACHMENT so the barriers below are unchanged.
        let gb = self.gbuffer.as_ref().ok_or(
            "Upscale enabled but the unified G-buffer pre-pass is absent; upscaling needs its motion + depth",
        )?;
        let velocity = gb
            .velocity_images
            .get(frame)
            .ok_or("upscale: gbuffer velocity slot out of range")?;
        let depth = gb
            .depth_images
            .get(frame)
            .ok_or("upscale: gbuffer depth slot out of range")?;

        // Scene colour: SSR resolve output when the SSR resolve is active (HDR +
        // reflections), else this slot's HDR resolve target (also the SSGI-only
        // case, where `ssr` exists for the G-buffer but the resolve is off).
        // Both rest in SHADER_READ_ONLY_OPTIMAL after their writer.
        let scene = match self.ssr.as_ref().filter(|_| self.ssr_resolve_active) {
            Some(s) => &s.output,
            None => self
                .hdr_resolve_images
                .get(frame)
                .ok_or("upscale: hdr resolve slot out of range")?,
        };

        let (rw, rh) = upscaler.render_dims();

        // Make the producer writes (colour / velocity = COLOR_ATTACHMENT_WRITE,
        // depth = DEPTH_STENCIL_ATTACHMENT_WRITE) visible to the upscaler's
        // COMPUTE reads, and transition the depth from its attachment layout to
        // SHADER_READ_ONLY. The colour + velocity already rest in
        // SHADER_READ_ONLY (their render-pass final layout), so those are
        // same-layout execution+memory barriers.
        image_barrier(
            &self.device,
            cmd,
            scene.image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ,
        );
        image_barrier(
            &self.device,
            cmd,
            velocity.image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ,
        );
        image_barrier(
            &self.device,
            cmd,
            depth.image,
            vk::ImageAspectFlags::DEPTH,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ,
        );
        // The output rests in SHADER_READ_ONLY after the previous frame's
        // bloom + composite sampled it; flip it back to GENERAL for the write
        // (skipped on the first frame, where it starts in GENERAL).
        if upscaler.output_layout() != vk::ImageLayout::GENERAL {
            image_barrier(
                &self.device,
                cmd,
                upscaler.output_image().image,
                vk::ImageAspectFlags::COLOR,
                upscaler.output_layout(),
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
            );
            upscaler.set_output_layout(vk::ImageLayout::GENERAL);
        }

        let color = UpscaleImage {
            image: scene.image,
            view: scene.view,
            format: HDR_FORMAT,
            width: rw,
            height: rh,
            aspect: vk::ImageAspectFlags::COLOR,
        };
        let motion = UpscaleImage {
            image: velocity.image,
            view: velocity.view,
            format: vk::Format::R16G16_SFLOAT,
            width: rw,
            height: rh,
            aspect: vk::ImageAspectFlags::COLOR,
        };
        let depth_in = UpscaleImage {
            image: depth.image,
            view: depth.view,
            format: vk::Format::D32_SFLOAT,
            width: rw,
            height: rh,
            aspect: vk::ImageAspectFlags::DEPTH,
        };

        let near = params.near.max(1e-3);
        let far = params.far.max(near + 1.0);
        upscaler.dispatch(
            cmd,
            &color,
            &depth_in,
            &motion,
            upscaler.jitter(),
            params.elapsed,
            near,
            far,
            params.fov_y_radians,
        )?;

        // Flip the output GENERAL -> SHADER_READ_ONLY so bloom + composite can
        // sample it. (Inputs are left where the upscaler leaves them; the next
        // frame's render passes reset them.)
        image_barrier(
            &self.device,
            cmd,
            upscaler.output_image().image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        );
        upscaler.set_output_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::UpscalerBackend as B;

    fn resolved(req: B, dlss: bool, xess: bool, fsr: bool) -> ResolvedBackend {
        backend_order(req, dlss, xess, fsr)[0]
    }

    #[test]
    fn auto_prefers_dlss_then_xess_then_fsr_then_native() {
        assert_eq!(resolved(B::Auto, true, true, true), ResolvedBackend::Dlss);
        assert_eq!(resolved(B::Auto, false, true, true), ResolvedBackend::Xess);
        assert_eq!(resolved(B::Auto, false, false, true), ResolvedBackend::Fsr);
        assert_eq!(
            resolved(B::Auto, false, false, false),
            ResolvedBackend::Native
        );
    }

    #[test]
    fn explicit_choice_used_when_available() {
        assert_eq!(resolved(B::Dlss, true, true, true), ResolvedBackend::Dlss);
        assert_eq!(resolved(B::Xess, true, true, true), ResolvedBackend::Xess);
        assert_eq!(resolved(B::Fsr3, true, true, true), ResolvedBackend::Fsr);
    }

    #[test]
    fn explicit_choice_falls_through_when_unavailable() {
        // Requested DLSS unavailable falls to the next available (XeSS).
        assert_eq!(resolved(B::Dlss, false, true, true), ResolvedBackend::Xess);
        // Requested XeSS unavailable, only FSR left.
        assert_eq!(resolved(B::Xess, false, false, true), ResolvedBackend::Fsr);
        // Requested FSR unavailable, nothing left.
        assert_eq!(
            resolved(B::Fsr3, false, false, false),
            ResolvedBackend::Native
        );
    }

    #[test]
    fn halton_jitter_is_centered_and_bounded() {
        for f in 0..64u32 {
            let [x, y] = halton_jitter_offset(f);
            assert!((-0.5..0.5).contains(&x), "x={x} out of range");
            assert!((-0.5..0.5).contains(&y), "y={y} out of range");
        }
        // radical_inverse(1, 2) = 0.5 -> offset 0.0; (1,3) = 1/3 -> -1/6.
        let [x, y] = halton_jitter_offset(0);
        assert!((x - 0.0).abs() < 1e-6);
        assert!((y - (1.0 / 3.0 - 0.5)).abs() < 1e-6);
    }

    #[test]
    fn render_dims_apply_quality_scale() {
        let (w, h, s) = resolve_render_dims(1920, 1080, 2.0 / 3.0);
        assert_eq!((w, h), (1280, 720));
        assert!((s - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(resolve_render_dims(1920, 1080, 0.5).0, 960);
        assert_eq!(resolve_render_dims(1920, 1080, 0.5).1, 540);
    }

    #[test]
    fn render_dims_clamp_out_of_range_scale() {
        let (w, h, s) = resolve_render_dims(800, 600, 2.0);
        assert_eq!((w, h), (800, 600));
        assert!((s - 1.0).abs() < 1e-6);
        assert_eq!(resolve_render_dims(800, 600, 0.0), (800, 600, 1.0));
        let (_, _, s2) = resolve_render_dims(900, 900, 0.1);
        assert!((s2 - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn frame_delta_is_clamped() {
        let prev = Cell::new(0.0);
        // First frame: now=10s, raw delta huge, clamped to 100 ms.
        assert!((frame_delta_ms(&prev, 10.0) - 100.0).abs() < 1e-3);
        // 16 ms later.
        assert!((frame_delta_ms(&prev, 10.016) - 16.0).abs() < 1e-2);
        // A zero/negative delta clamps up to 1 ms.
        assert!((frame_delta_ms(&prev, 10.016) - 1.0).abs() < 1e-3);
    }
}
