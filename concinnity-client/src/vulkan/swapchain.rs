// src/vulkan/swapchain.rs
//
// Vulkan swapchain, attachment, and framebuffer creation, plus the
// swapchain rebuild path.
use ash::{Device, vk};

use super::context::*;
use super::device::*;
use super::post::bloom::{
    alloc_bloom_input_sets, create_bloom_chain, create_bloom_framebuffers, rebind_bloom_input0,
};
use super::texture::*;

//  Swapchain rebuild

impl VkContext {
    pub(super) fn destroy_swapchain_resources(&mut self) {
        let device = &self.device;
        for fb in &self.framebuffers {
            unsafe { device.destroy_framebuffer(*fb, None) };
        }
        for fb in &self.composite_framebuffers {
            unsafe { device.destroy_framebuffer(*fb, None) };
        }
        for frame_fbs in self
            .bloom_write_framebuffers
            .iter()
            .chain(&self.bloom_blend_framebuffers)
        {
            for &fb in frame_fbs {
                unsafe { device.destroy_framebuffer(fb, None) };
            }
        }
        for frame_mips in &self.bloom_mips {
            for img in frame_mips {
                // A null-memory mip is a borrowed pooled `bloom_top` (mip 0); the
                // transient pool owns and frees it. Committed mips free here.
                if img.memory != vk::DeviceMemory::null() {
                    img.destroy(device);
                }
            }
        }
        for img in &self.color_images {
            img.destroy(device);
        }
        for img in &self.depth_images {
            img.destroy(device);
        }
        for img in &self.hdr_resolve_images {
            img.destroy(device);
        }
        for iv in &self.swapchain_image_views {
            unsafe { device.destroy_image_view(*iv, None) };
        }
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None)
        };
        self.framebuffers.clear();
        self.composite_framebuffers.clear();
        self.bloom_write_framebuffers.clear();
        self.bloom_blend_framebuffers.clear();
        self.bloom_mips.clear();
        self.bloom_mip_extents.clear();
        self.color_images.clear();
        self.depth_images.clear();
        self.hdr_resolve_images.clear();
        self.swapchain_image_views.clear();
    }

    pub(super) fn rebuild_swapchain(&mut self) -> Result<(), String> {
        self.wait_idle();
        // The previous swapchain's images are about to be destroyed; invalidate
        // the screenshot read-back index until the next present repopulates it.
        self.last_present_index = None;
        self.destroy_swapchain_resources();

        let (width, height) = self.window.window.get_framebuffer_size();
        let (sc, imgs, fmt, ext) = create_swapchain_inner(
            &self.instance,
            &self.device,
            self.physical_device,
            &self.surface_loader,
            self.surface,
            &self.swapchain_loader,
            width as u32,
            height as u32,
            self.graphics_family,
            // re-query present family
            {
                let (_, pf) = query_queue_families(
                    &self.instance,
                    self.physical_device,
                    &self.surface_loader,
                    self.surface,
                )?;
                pf
            },
            vk::SwapchainKHR::null(),
            self.hdr_mode,
            self.vsync,
        )?;
        self.swapchain = sc;
        self.swapchain_images = imgs;
        self.swapchain_format = fmt;
        self.swapchain_extent = ext;
        // Temporal upscaling: the FSR context bakes its max render / upscale
        // sizes at creation, so a resize must recreate it at the new output
        // size (same quality scale). `device_wait_idle` at the top of this
        // function guarantees the old context is idle before destroy. The new
        // render dims then drive `render_ext`; off-screen scene passes rebuild
        // to it while bloom / composite / swapchain stay at `ext`.
        if let Some(scale) = self.upscale.as_ref().map(|u| u.scale()) {
            if let Some(mut old) = self.upscale.take() {
                old.destroy(&self.device);
            }
            // Rebuild the backend the world requested (not a hardcoded FSR). The
            // DLSS / XeSS device extensions are fixed at device creation, and
            // `build_upscaler` re-resolves `upscale_requested` deterministically
            // to the same first choice, so the rebuilt backend matches the device.
            let (built, resolved) = super::post::build_upscaler(
                &self.instance,
                &self.device,
                self.physical_device,
                self.commands.command_pool,
                self.graphics_queue,
                ext.width,
                ext.height,
                scale,
                self.upscale_requested,
            )?;
            // The rebuilt feature re-emits the benign DLSS first-frame layout
            // errors; re-arm the messenger budget so they stay suppressed.
            if resolved == super::post::ResolvedBackend::Dlss
                && let Some(f) = &self.debug_filter
            {
                f.store(
                    super::init::DLSS_FIRST_FRAME_LAYOUT_SUPPRESS,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            self.upscale = built;
        }
        let render_ext = match &self.upscale {
            Some(u) => {
                let (w, h) = u.render_dims();
                vk::Extent2D {
                    width: w,
                    height: h,
                }
            }
            None => ext,
        };
        self.render_extent = render_ext;

        // Rebuild the transient image pool before the off-screen attachments /
        // bloom chain / SSAO targets that bind its images. `ao_output` is
        // render-res, `bloom_top` is half the output (swapchain) extent; both are
        // per frame in flight. `bloom_top_pairs` feeds the bloom chain's mip 0
        // below (empty when bloom is off, so mip 0 is committed instead).
        self.transient_pool.rebuild(
            &self.instance,
            &self.device,
            self.physical_device,
            self.frames_in_flight,
            &super::transient_pool::transient_slots(
                self.ssao.is_some(),
                self.post_process.bloom_intensity > 0.0,
                render_ext,
                ext,
            ),
        )?;
        let bloom_top_pairs = self
            .transient_pool
            .pairs_for_frames("bloom_top", self.frames_in_flight);

        self.swapchain_image_views =
            create_swapchain_image_views(&self.device, &self.swapchain_images, fmt)?;

        let (color_images, depth_images, hdr_resolve_images) = create_attachments(
            &self.instance,
            &self.device,
            self.physical_device,
            self.commands.command_pool,
            self.graphics_queue,
            render_ext.width,
            render_ext.height,
            self.msaa_samples,
            self.frames_in_flight,
        )?;
        self.color_images = color_images;
        self.depth_images = depth_images;
        self.hdr_resolve_images = hdr_resolve_images;
        self.framebuffers = create_main_framebuffers(
            &self.device,
            self.main_render_pass,
            &self.color_images,
            &self.depth_images,
            &self.hdr_resolve_images,
            render_ext,
            self.msaa_samples,
        )?;
        self.composite_framebuffers = create_composite_framebuffers(
            &self.device,
            self.composite_render_pass,
            &self.swapchain_image_views,
            ext,
        )?;

        // Rebuild the bloom chain at the new resolution.
        let (bloom_mips, bloom_mip_extents) = create_bloom_chain(
            &self.instance,
            &self.device,
            self.physical_device,
            self.commands.command_pool,
            self.graphics_queue,
            ext,
            self.frames_in_flight,
            &bloom_top_pairs,
        )?;
        self.bloom_mips = bloom_mips;
        self.bloom_mip_extents = bloom_mip_extents;
        let (bloom_write_framebuffers, bloom_blend_framebuffers) = create_bloom_framebuffers(
            &self.device,
            self.bloom_write_pass,
            self.bloom_blend_pass,
            &self.bloom_mips,
            &self.bloom_mip_extents,
        )?;
        self.bloom_write_framebuffers = bloom_write_framebuffers;
        self.bloom_blend_framebuffers = bloom_blend_framebuffers;

        // The bloom input sets reference the destroyed mips; reset the pool
        // (the octave count may have changed) and re-allocate. wait_idle()
        // above guarantees none are still in flight.
        unsafe {
            self.device
                .reset_descriptor_pool(
                    self.bloom_descriptor_pool,
                    vk::DescriptorPoolResetFlags::empty(),
                )
                .map_err(|e| format!("reset bloom pool: {e}"))?;
        }
        self.bloom_input_sets = alloc_bloom_input_sets(
            &self.device,
            self.bloom_descriptor_pool,
            self.bloom_set_layout,
            self.composite_sampler,
            &self.hdr_resolve_images,
            &self.bloom_mips,
        )?;

        // Rebuild the unified G-buffer pre-pass targets at the new resolution
        // *first*: every reader (SSR resolve, SSAO, SSGI, RT, TAA velocity, FSR)
        // re-points its descriptors at the rebuilt per-frame normal+depth /
        // roughness / velocity views below, so the merged buffer must already be
        // current. The render pass, pipelines, UBOs, and descriptor sets survive.
        if let Some(mut gb) = self.gbuffer.take() {
            gb.rebuild(
                &self.instance,
                &self.device,
                self.physical_device,
                self.commands.command_pool,
                self.graphics_queue,
                render_ext.width,
                render_ext.height,
                self.frames_in_flight,
            )?;
            self.gbuffer = Some(gb);
        }

        // Rebuild the SSR targets at the new resolution. The G-buffer +
        // roughness + private depth + output are all resolution-dependent;
        // the resolve sets re-point automatically at the new HDR resolve +
        // SSR targets via wire_resolve_sets. With SSR on, the bloom prefilter
        // input 0 also moves to the new SSR output below; TAA (when on)
        // overrides that in turn to the new TAA output.
        if let Some(mut ssr) = self.ssr.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            // Per-frame unified G-buffer views (rebuilt above) when present, else
            // empty so the SSR resolve falls back to its own pre-pass targets.
            let (nd_views, rough_views) = match self.gbuffer.as_ref() {
                Some(gb) => (gb.normal_depth_views(), gb.roughness_views()),
                None => (Vec::new(), Vec::new()),
            };
            ssr.rebuild(
                &self.instance,
                &self.device,
                self.physical_device,
                self.commands.command_pool,
                self.graphics_queue,
                render_ext.width,
                render_ext.height,
                &hdr_views,
                &nd_views,
                &rough_views,
                self.env_map.prefilter.view,
                self.cube_sampler,
            )?;
            // Bloom prefilter samples SSR output only when the SSR resolve is
            // active (TAA overrides this below if TAA is also on). A SSGI-only
            // build rebuilt the pre-pass G-buffer above but leaves the bloom
            // prefilter on the raw HDR resolve.
            if self.ssr_resolve_active {
                for frame_sets in &self.bloom_input_sets {
                    rebind_bloom_input0(
                        &self.device,
                        frame_sets[0],
                        ssr.output.view,
                        self.composite_sampler,
                    );
                }
            }
            self.ssr = Some(ssr);
        }

        // Rebuild the SSGI gi target + composite framebuffers and re-wire its
        // descriptor sets to the rebuilt HDR resolves + SSR pre-pass G-buffer.
        // The SSR rebuild above already ran, so `ssr.gbuffer` is current. The
        // render passes, pipelines, sampler, and descriptor pool all survive.
        if let Some(mut ssgi) = self.ssgi.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            // SSGI samples the unified G-buffer's per-frame normal+depth views.
            // The merged pre-pass was rebuilt above, so they are current.
            let nd_views = self
                .gbuffer
                .as_ref()
                .expect("SSGI keeps the unified G-buffer pre-pass alive")
                .normal_depth_views();
            ssgi.rebuild(
                &self.instance,
                &self.device,
                self.physical_device,
                render_ext.width,
                render_ext.height,
                &hdr_views,
                &nd_views,
            )?;
            self.ssgi = Some(ssgi);
        }

        // Rebuild the RT-reflection output target + re-wire its static
        // descriptors (the SSR pre-pass G-buffer / roughness + the HDR resolves
        // all moved). The acceleration structure is resolution-independent, so it
        // survives; the per-frame TLAS + geometry-table descriptors are re-pointed
        // by `rt_dynamic_update` as usual. RT output is a single shared image, so
        // the bloom prefilter input 0 moves to it (TAA / upscale override below).
        if let Some(mut rt) = self.rt_reflections.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            // RT samples the unified G-buffer's per-frame normal+depth + roughness
            // views. The merged pre-pass was rebuilt above, so they are current.
            let gb = self
                .gbuffer
                .as_ref()
                .expect("RT keeps the unified G-buffer pre-pass alive");
            let nd_views = gb.normal_depth_views();
            let rough_views = gb.roughness_views();
            rt.rebuild(
                &self.instance,
                &self.device,
                self.physical_device,
                render_ext.width,
                render_ext.height,
                self.geometry.vertex_buffer,
                self.geometry.index_buffer,
                &hdr_views,
                &nd_views,
                &rough_views,
                self.env_map.prefilter.view,
                self.cube_sampler,
            )?;
            for frame_sets in &self.bloom_input_sets {
                rebind_bloom_input0(
                    &self.device,
                    frame_sets[0],
                    rt.output.view,
                    self.composite_sampler,
                );
            }
            self.rt_reflections = Some(rt);
        }

        // Rebuild the TAA velocity + history targets at the new resolution.
        // When TAA is on the bloom prefilter + composite sample its output
        // image; otherwise they sample the raw HDR resolve (or SSR output
        // when SSR is on but TAA is off). wait_idle() above guarantees none
        // of these are still in flight.
        if let Some(mut taa) = self.taa.take() {
            taa.rebuild(
                &self.instance,
                &self.device,
                self.physical_device,
                self.commands.command_pool,
                self.graphics_queue,
                render_ext,
                self.frames_in_flight,
                &self.hdr_resolve_images,
                self.composite_sampler,
            )?;
            // When RT reflections or the SSR resolve owns the scene image, TAA
            // samples that (HDR + reflections) instead of the raw HDR resolve. RT
            // takes precedence; a SSGI-only build leaves TAA on the raw HDR resolve.
            if let Some(rt) = self.rt_reflections.as_ref() {
                taa.rewire_scene(&self.device, rt.output.view, self.composite_sampler);
            } else if let Some(ssr) = self.ssr.as_ref().filter(|_| self.ssr_resolve_active) {
                taa.rewire_scene(&self.device, ssr.output.view, self.composite_sampler);
            }
            // The TAA resolve's velocity input is the unified G-buffer's per-frame
            // velocity channel (rebuilt above), replacing TAA's own velocity
            // pre-pass output. Mirrors the init-time `rewire_velocity`.
            if let Some(gb) = self.gbuffer.as_ref() {
                let vel_views = gb.velocity_views();
                taa.rewire_velocity(&self.device, &vel_views, self.composite_sampler);
            }
            for (i, frame_sets) in self.bloom_input_sets.iter().enumerate() {
                rebind_bloom_input0(
                    &self.device,
                    frame_sets[0],
                    taa.output_view(i),
                    self.composite_sampler,
                );
            }
            self.taa = Some(taa);
        }

        // Temporal upscaling: bloom prefilter samples the FSR output (the
        // reconstructed swapchain-res scene), overriding the SSR / TAA rebinds
        // above. A single shared image, so every frame's set points at it.
        if let Some(up) = &self.upscale {
            let up_output_view = up.output_image().view;
            for frame_sets in &self.bloom_input_sets {
                rebind_bloom_input0(
                    &self.device,
                    frame_sets[0],
                    up_output_view,
                    self.composite_sampler,
                );
            }
        }

        // Rebuild the decal framebuffers at the new resolution + re-point
        // the per-frame depth descriptor at the rebuilt depth view. The
        // pipeline, layouts, buffers, sampler, and per-decal albedo sets
        // all survive: only the targets the framebuffers + depth binding
        // reference moved.
        if let Some(mut decals) = self.decals_state.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            let depth_views: Vec<vk::ImageView> =
                self.depth_images.iter().map(|img| img.view).collect();
            decals.rebuild(&self.device, &hdr_views, &depth_views, render_ext)?;
            self.decals_state = Some(decals);
        }

        // Rebuild the fog framebuffers + re-point the per-frame depth
        // descriptor at the rebuilt depth view. Mirrors the decal rebuild;
        // the pipeline, layouts, UBOs, and sampler all survive.
        if let Some(mut fog) = self.fog_resources.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            let depth_views: Vec<vk::ImageView> =
                self.depth_images.iter().map(|img| img.view).collect();
            fog.rebuild(&self.device, &hdr_views, &depth_views, render_ext)?;
            self.fog_resources = Some(fog);
        }

        // Recreate the raymarch scene snapshot at the new resolution + re-point
        // the `scene_color` binding of every view set. The pipelines, layouts,
        // UBOs, cube buffers, and render passes survive; the pass reuses the
        // rebuilt main framebuffers, so only the snapshot moved.
        if let Some(mut rm) = self.raymarch.take() {
            rm.rebuild(
                &self.instance,
                &self.device,
                self.physical_device,
                self.commands.command_pool,
                self.graphics_queue,
                render_ext.width,
                render_ext.height,
            )?;
            self.raymarch = Some(rm);
        }

        // Rebuild the glass scene snapshot + per-frame framebuffers at the new
        // resolution + re-point the snapshot / depth bindings. The scene target
        // moved with the rebuilt SSR output / HDR resolve, so resolve it again
        // here (SSR output when SSR is on, else this slot's HDR resolve). The
        // SSR + HDR resolve rebuilds above already ran, so the handles are
        // current. The pipeline, layouts, panel buffers, view UBOs, and render
        // pass all survive.
        if let Some(mut glass) = self.glass.take() {
            let (scene_views, scene_images): (Vec<vk::ImageView>, Vec<vk::Image>) = (0..self
                .frames_in_flight)
                .map(
                    |i| match self.ssr.as_ref().filter(|_| self.ssr_resolve_active) {
                        Some(s) => (s.output.view, s.output.image),
                        None => (
                            self.hdr_resolve_images[i].view,
                            self.hdr_resolve_images[i].image,
                        ),
                    },
                )
                .unzip();
            let depth_views: Vec<vk::ImageView> =
                self.depth_images.iter().map(|img| img.view).collect();
            glass.rebuild(
                &self.instance,
                &self.device,
                self.physical_device,
                self.commands.command_pool,
                self.graphics_queue,
                render_ext.width,
                render_ext.height,
                &scene_views,
                &scene_images,
                &depth_views,
            )?;
            self.glass = Some(glass);
        }

        // Rebuild the Hi-Z pyramid at the new resolution + re-point its init
        // sets' depth bindings and the cull-read set's pyramid sampler. The
        // build pipelines, layouts, sampler, and per-frame cull UBOs survive.
        // Invalidate the pyramid so the next frame's cull falls back to frustum
        // + distance until a pyramid at the new resolution has been built.
        if let Some(mut hiz) = self.cull.hiz.take() {
            let depth_views: Vec<vk::ImageView> =
                self.depth_images.iter().map(|img| img.view).collect();
            hiz.resize_to(
                &self.instance,
                &self.device,
                self.physical_device,
                self.commands.command_pool,
                self.graphics_queue,
                render_ext.width,
                render_ext.height,
                &depth_views,
            )?;
            self.cull.hiz = Some(hiz);
            self.cull.hiz_valid = false;
        }

        // Rebuild the particle framebuffers at the new resolution. The
        // pipelines, layouts, view UBOs, per-emitter pools, and
        // descriptor sets all survive: only the framebuffers reference
        // the moved hdr_resolve targets.
        if let Some(mut p) = self.particle_resources.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            p.rebuild(&self.device, &hdr_views, render_ext)?;
            self.particle_resources = Some(p);
        }

        // Re-point the auto-exposure build sets at the rebuilt HDR resolve
        // views. The histogram / output / readback buffers are
        // resolution-independent and survive the rebuild untouched.
        if let Some(mut ae) = self.auto_exposure.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            ae.rebuild(&self.device, &hdr_views, self.linear_sampler);
            self.auto_exposure = Some(ae);
        }

        // Rebuild the SSAO targets + re-point the SSAO descriptor at set 0
        // binding 6 of every global set against the per-frame pooled `ao_output`
        // views (the transient pool was already rebuilt above). SSAO's stale
        // blur framebuffers are torn down inside `ssao.rebuild` (the device is
        // idle, so freeing the pool views ahead of those framebuffers is sound).
        // When SSAO is off the pool holds no `ao_output` and binding 6 stays on
        // the (resolution-independent) 1×1 white fallback, so no rebuild needed.
        let frames = self.frames_in_flight;
        if let Some(mut ssao) = self.ssao.take() {
            // SSAO kernel/blur sample the unified G-buffer's per-frame normal+depth
            // views (rebuilt above) when present, else SSAO's own pre-pass target.
            let nd_views = match self.gbuffer.as_ref() {
                Some(gb) => gb.normal_depth_views(),
                None => Vec::new(),
            };
            let ao_views = self.transient_pool.views_for_frames("ao_output", frames);
            ssao.rebuild(
                &self.instance,
                &self.device,
                self.physical_device,
                render_ext.width,
                render_ext.height,
                &nd_views,
                &ao_views,
            )?;
            for (i, &set) in self.descriptors.global_sets.iter().enumerate() {
                let ao_view = self
                    .transient_pool
                    .view_for("ao_output", i)
                    .unwrap_or(self.ssao_white.view);
                let info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(ao_view)
                    .sampler(self.linear_sampler);
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&info));
                unsafe {
                    self.device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
            }
            self.ssao = Some(ssao);
        }

        // Re-point the composite descriptor sets at the rebuilt scene-input
        // image (FSR upscale output > TAA output > RT reflection output > SSR
        // output > HDR resolve) + bloom mip 0. The 3D colour LUT is
        // resolution-independent, so it survives the resize untouched and is just
        // re-bound at binding 2.
        for (i, &set) in self.composite_sets.iter().enumerate() {
            let scene_view = if let Some(up) = &self.upscale {
                up.output_image().view
            } else if let Some(taa) = &self.taa {
                taa.output_view(i)
            } else if let Some(rt) = self.rt_reflections.as_ref() {
                rt.output.view
            } else if let Some(ssr) = self.ssr.as_ref().filter(|_| self.ssr_resolve_active) {
                ssr.output.view
            } else {
                self.hdr_resolve_images[i].view
            };
            write_composite_set(
                &self.device,
                set,
                scene_view,
                self.bloom_mips[i][0].view,
                self.color_lut.view,
                self.composite_sampler,
            );
        }

        // The render-finished semaphores are one-per-swapchain-image; a
        // resize can change the image count, so resize the pool to match.
        // wait_idle() above guarantees none are still in flight.
        if self.frame_sync.render_finished.len() != self.swapchain_images.len() {
            for &s in &self.frame_sync.render_finished {
                unsafe { self.device.destroy_semaphore(s, None) };
            }
            let sem_info = vk::SemaphoreCreateInfo::default();
            self.frame_sync.render_finished = (0..self.swapchain_images.len())
                .map(|_| unsafe { self.device.create_semaphore(&sem_info, None) })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("semaphore: {e}"))?;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn create_swapchain_inner(
    _instance: &ash::Instance,
    _device: &Device,
    pd: vk::PhysicalDevice,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_loader: &ash::khr::swapchain::Device,
    width: u32,
    height: u32,
    graphics_family: u32,
    present_family: u32,
    old_swapchain: vk::SwapchainKHR,
    // Resolved output mode, picking the swapchain (format, colour space):
    //   - `Sdr`                        -> `B8G8R8A8_UNORM` + sRGB-nonlinear.
    //   - `Hdr{ ExtendedLinear }`      -> `R16G16B16A16_SFLOAT` +
    //     `EXTENDED_SRGB_LINEAR_EXT` (scRGB linear).
    //   - `Hdr{ Pq }`                  -> HDR10 PQ: `R16G16B16A16_SFLOAT` +
    //     `HDR10_ST2084_EXT` preferred (keeps the composite + screenshot paths
    //     identical to the scRGB float swapchain), else
    //     `A2B10G10R10_UNORM_PACK32` + `HDR10_ST2084_EXT`.
    // The caller has already enabled `VK_EXT_swapchain_colorspace` and gated the
    // resolved mode on the surface advertising the matching pair (see the HDR
    // resolve in init.rs), so the chosen encoding and colour space stay in
    // sync. Each arm falls back through scRGB to the SDR default if its
    // preferred pair is unexpectedly absent.
    hdr_mode: crate::gfx::hdr_output::HdrOutputMode,
    // Lock presentation to the display refresh. `true` forces FIFO (always
    // present, vsync); `false` prefers MAILBOX (uncapped render loop, no
    // tearing), then IMMEDIATE, falling back to FIFO when neither is offered.
    vsync: bool,
) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D), String> {
    use crate::gfx::hdr_output::{HdrEncoding, HdrOutputMode};
    let caps = unsafe { surface_loader.get_physical_device_surface_capabilities(pd, surface) }
        .map_err(|e| format!("surface caps: {e}"))?;
    let formats = unsafe { surface_loader.get_physical_device_surface_formats(pd, surface) }
        .map_err(|e| format!("surface formats: {e}"))?;
    let present_modes =
        unsafe { surface_loader.get_physical_device_surface_present_modes(pd, surface) }
            .map_err(|e| format!("present modes: {e}"))?;

    // Pick surface format. scRGB HDR: `R16G16B16A16_SFLOAT` + scRGB-linear
    // (Rec.709 primaries, gamma 1.0, extended range; `1.0` = SDR reference
    // white). HDR10 PQ: a `HDR10_ST2084_EXT` pair (float preferred). SDR:
    // `B8G8R8A8_UNORM` + sRGB-nonlinear. When the preferred pair is absent the
    // arm falls back through scRGB to the first reported format.
    let scrgb_pair = (
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
    );
    let sdr_pair = (
        vk::Format::B8G8R8A8_UNORM,
        vk::ColorSpaceKHR::SRGB_NONLINEAR,
    );
    // PQ candidates, float first so the composite render pass + screenshot
    // read-back stay on the same `R16G16B16A16_SFLOAT` swapchain the scRGB
    // path uses; the 10-bit packed format is the secondary option.
    let pq_pairs = [
        (
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ColorSpaceKHR::HDR10_ST2084_EXT,
        ),
        (
            vk::Format::A2B10G10R10_UNORM_PACK32,
            vk::ColorSpaceKHR::HDR10_ST2084_EXT,
        ),
    ];
    let pick = |target: (vk::Format, vk::ColorSpaceKHR)| {
        formats
            .iter()
            .find(|f| f.format == target.0 && f.color_space == target.1)
            .copied()
    };
    let surface_format = match hdr_mode {
        HdrOutputMode::Hdr {
            encoding: HdrEncoding::Pq,
            ..
        } => pq_pairs
            .iter()
            .find_map(|&p| pick(p))
            .or_else(|| pick(scrgb_pair))
            .or_else(|| pick(sdr_pair))
            .unwrap_or(formats[0]),
        HdrOutputMode::Hdr { .. } => pick(scrgb_pair)
            .or_else(|| pick(sdr_pair))
            .unwrap_or(formats[0]),
        HdrOutputMode::Sdr => pick(sdr_pair).unwrap_or(formats[0]),
    };

    // FIFO is always available and is the vsync mode. Uncapped prefers MAILBOX
    // (no tearing) then IMMEDIATE (tearing) before falling back to FIFO.
    let present_mode = if vsync {
        vk::PresentModeKHR::FIFO
    } else {
        let has = |m: vk::PresentModeKHR| present_modes.contains(&m);
        if has(vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if has(vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        }
    };

    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };

    let image_count = (caps.min_image_count + 1).min(if caps.max_image_count == 0 {
        u32::MAX
    } else {
        caps.max_image_count
    });

    let queue_families = [graphics_family, present_family];
    let (sharing, families) = if graphics_family == present_family {
        (vk::SharingMode::EXCLUSIVE, &queue_families[..0])
    } else {
        (vk::SharingMode::CONCURRENT, &queue_families[..])
    };

    let sc_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        // TRANSFER_SRC so the `screenshot` debug command can copy the presented
        // image back to a host buffer (see vulkan/screenshot.rs).
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
        .image_sharing_mode(sharing)
        .queue_family_indices(families)
        .pre_transform(caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(old_swapchain);

    let swapchain = unsafe { swapchain_loader.create_swapchain(&sc_info, None) }
        .map_err(|e| format!("create swapchain: {e}"))?;

    if old_swapchain != vk::SwapchainKHR::null() {
        unsafe { swapchain_loader.destroy_swapchain(old_swapchain, None) };
    }

    let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }
        .map_err(|e| format!("get swapchain images: {e}"))?;

    Ok((swapchain, images, surface_format.format, extent))
}

pub(super) fn create_swapchain_image_views(
    device: &Device,
    images: &[vk::Image],
    format: vk::Format,
) -> Result<Vec<vk::ImageView>, String> {
    images
        .iter()
        .map(|&img| create_image_view(device, img, format, vk::ImageAspectFlags::COLOR))
        .collect()
}

// Main scene render pass. Renders linear-light HDR into an off-screen
// `R16G16B16A16_SFLOAT` target (the MSAA colour image when multisampled, or
// the resolve image directly otherwise) and ends with the resolve image in
// `SHADER_READ_ONLY_OPTIMAL` so the composite pass can sample it.

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn create_attachments(
    instance: &ash::Instance,
    device: &Device,
    pd: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    width: u32,
    height: u32,
    msaa: vk::SampleCountFlags,
    count: usize,
) -> Result<(Vec<GpuImage>, Vec<GpuImage>, Vec<GpuImage>), String> {
    let mut color_images = Vec::new();
    let mut depth_images = Vec::new();
    let mut resolve_images = Vec::new();
    for _ in 0..count {
        let depth = create_depth_image(
            instance,
            device,
            pd,
            command_pool,
            queue,
            width,
            height,
            msaa,
        )?;
        depth_images.push(depth);
        resolve_images.push(create_hdr_resolve_image(
            instance, device, pd, width, height, HDR_FORMAT,
        )?);
        if msaa != vk::SampleCountFlags::TYPE_1 {
            let color = create_msaa_color_image(
                instance,
                device,
                pd,
                command_pool,
                queue,
                width,
                height,
                HDR_FORMAT,
                msaa,
            )?;
            color_images.push(color);
        }
    }
    Ok((color_images, depth_images, resolve_images))
}

// Main-pass framebuffers, one per frame-in-flight slot. Each attaches the HDR
// colour (MSAA colour + resolve, or just the resolve image) and depth.
pub(super) fn create_main_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    color_images: &[GpuImage],
    depth_images: &[GpuImage],
    resolve_images: &[GpuImage],
    extent: vk::Extent2D,
    msaa: vk::SampleCountFlags,
) -> Result<Vec<vk::Framebuffer>, String> {
    (0..resolve_images.len())
        .map(|i| {
            let attachments: Vec<vk::ImageView> = if msaa != vk::SampleCountFlags::TYPE_1 {
                vec![
                    color_images[i].view,
                    depth_images[i].view,
                    resolve_images[i].view,
                ]
            } else {
                vec![resolve_images[i].view, depth_images[i].view]
            };
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            unsafe { device.create_framebuffer(&fb_info, None) }
                .map_err(|e| format!("framebuffer[{i}]: {e}"))
        })
        .collect()
}

// Composite-pass framebuffers, one per swapchain image.
pub(super) fn create_composite_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    swapchain_views: &[vk::ImageView],
    extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>, String> {
    swapchain_views
        .iter()
        .enumerate()
        .map(|(i, &sc_view)| {
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(std::slice::from_ref(&sc_view))
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            unsafe { device.create_framebuffer(&fb_info, None) }
                .map_err(|e| format!("composite framebuffer[{i}]: {e}"))
        })
        .collect()
}

// Write a composite descriptor set: binding 0 = HDR resolve image,
// binding 1 = bloom mip 0, binding 2 = the 3D colour-grading LUT. All sampled
// through `sampler`.
pub(super) fn write_composite_set(
    device: &Device,
    set: vk::DescriptorSet,
    hdr_view: vk::ImageView,
    bloom_view: vk::ImageView,
    lut_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let hdr_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(hdr_view)
        .sampler(sampler);
    let bloom_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(bloom_view)
        .sampler(sampler);
    let lut_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(lut_view)
        .sampler(sampler);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&hdr_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&bloom_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&lut_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

// Create one framebuffer per cascade slice of the array shadow map. Each
// framebuffer attaches a single-layer depth view from
// `shadow_map.aux_views`. Returns one framebuffer per available slice.
pub(super) fn create_shadow_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    shadow_map: &GpuImage,
    size: u32,
) -> Result<Vec<vk::Framebuffer>, String> {
    let mut fbs = Vec::with_capacity(shadow_map.aux_views.len());
    for &view in &shadow_map.aux_views {
        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(std::slice::from_ref(&view))
            .width(size)
            .height(size)
            .layers(1);
        let fb = unsafe { device.create_framebuffer(&fb_info, None) }
            .map_err(|e| format!("shadow framebuffer: {e}"))?;
        fbs.push(fb);
    }
    Ok(fbs)
}
