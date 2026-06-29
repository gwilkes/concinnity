// src/vulkan/planar.rs
//
// Planar reflection for flat glass panes on the Vulkan backend. Each frame the
// scene is rendered a second time from the camera reflected across each pane's
// plane (mirror view + oblique near-plane clip so geometry behind the plane
// never leaks in) into a render-resolution target; the pane's fragment shader
// then samples that target projectively for a sharp, scene-correct reflection
// instead of the box-projected probe cube.
//
// GLSL/Vulkan port of src/directx/planar.rs (itself a port of src/metal/planar.rs),
// glass-only (water is a Metal-only producer). One mirror render per DISTINCT
// plane: near-coplanar panes (one wall of windows) share a render, and panes past
// the budget (MAX_PLANAR_PLANES) fall back to the probe cube. The plane -> slot
// grouping + the mirror matrices come from the pure, unit-tested
// gfx::planar_reflection.
//
// Each plane gets a DEDICATED reflected-frustum cull (the shared probe-bake
// encode_probe_cull): the GPU cull re-runs against the reflected-camera frustum
// into that plane's own indirect buffer, reading the FRAME's camera-independent
// object + draw-args SSBOs. So geometry visible only in the reflection (behind /
// beside the main camera) is captured, not just the main camera's visible set; the
// reflected view-proj's oblique near-plane clip also rejects geometry behind the
// reflector. The face render then draws that indirect. Like the probe capture, the
// skinned tail is not drawn into a mirror (static + instance + chunk only).

use ash::{Device, vk};

use super::context::{HDR_FORMAT, VkContext};
use super::draw::ViewUniforms;
use super::resources::alloc_descriptor_sets;
use super::texture::{GpuImage, create_buffer, create_image, create_image_view};

// Maximum number of distinct reflection planes that render a mirror pass each
// frame. Each plane is a full scene re-render, so this caps the per-frame cost;
// panes past the budget fall back to the box-projected probe cube. Matches
// metal::planar / directx::planar MAX_PLANAR_PLANES.
pub(in crate::vulkan) const MAX_PLANAR_PLANES: usize = 4;

// Clip the reflection a hair toward the kept (camera) side of the plane so
// geometry exactly on the surface is not lost to near-plane precision. Matches
// the other backends' PLANAR_CLIP_BIAS.
const PLANAR_CLIP_BIAS: f32 = 0.02;
const PLANAR_DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

// World-space plane [nx, ny, nz, d] (unit normal, n . p + d = 0 on the surface)
// for a glass pane with unit normal through centre. Pure; unit tested. The init
// path feeds these to assign_planar_slots.
pub(in crate::vulkan) fn pane_plane(normal: [f32; 3], centre: [f32; 3]) -> [f32; 4] {
    [
        normal[0],
        normal[1],
        normal[2],
        -(normal[0] * centre[0] + normal[1] * centre[1] + normal[2] * centre[2]),
    ]
}

// The set of distinct reflection planes for the world, each rendering its mirror
// into the shared colour + depth then resolving into its own shader-readable
// target. A pane samples the target of the slot it was assigned at init (see
// gfx::planar_reflection::assign_planar_slots). Recreated on resize alongside the
// HDR targets; the planes + slot assignment are fixed at init.
pub(in crate::vulkan) struct PlanarReflectionSet {
    planes: Vec<[f32; 4]>,
    frames: usize,
    sample_count: vk::SampleCountFlags,
    width: u32,
    height: u32,
    // Borrowed from VkContext (render-pass-compatible with the bindless main
    // pipeline). Not owned, never destroyed here.
    main_render_pass: vk::RenderPass,

    // Shared MSAA colour (Some only when MSAA) + shared depth, reused across
    // planes (rendered one plane at a time on the frame's cmd buffer) and across
    // frames (the single graphics queue executes submissions in order). Recreated
    // on resize.
    color: Option<GpuImage>,
    depth: GpuImage,
    // Per-plane shader-readable target: the MSAA resolve when MSAA, else the
    // single-sample colour attachment itself. The glass pass samples it. Recreated
    // on resize.
    targets: Vec<GpuImage>,
    framebuffers: Vec<vk::Framebuffer>,

    // Per-(plane, frame) reflected ViewUniforms UBO ring (HOST_VISIBLE, mapped),
    // indexed plane * frames + frame, so the CPU writes this frame's slot without
    // racing the GPU reading a prior frame's. Bound at binding 0 of the matching
    // planar global set.
    view_bufs: Vec<vk::Buffer>,
    view_mems: Vec<vk::DeviceMemory>,
    view_ptrs: Vec<*mut u8>,
    // Per-(plane, frame) global set (the bindless main set): binding 0 = that
    // (plane, frame) reflected view, 1/2 = the shared light/shadow UBOs, 3..6 =
    // the static shadow/env/ssao images, 7 = an EMPTY ProbeSet (the mirror render
    // reflects only sky -- no recursion), 8 = the sky-filled probe cube array.
    global_sets: Vec<vk::DescriptorSet>,
    // EMPTY ProbeSet UBO (count 0), shared by every planar global set.
    probeset_buf: vk::Buffer,
    probeset_mem: vk::DeviceMemory,

    // Per-(plane, frame) reflected-frustum mirror cull: a DEVICE_LOCAL indirect +
    // status SSBO each (indexed plane * frames + frame), and a cull set that reads
    // the FRAME's object + draw-args SSBOs (camera-independent, so the reflected
    // cull sees every object) and writes this plane's indirect + status. Sized by
    // the build-time object count, so resize never touches them.
    cull_indirect_bufs: Vec<vk::Buffer>,
    cull_indirect_mems: Vec<vk::DeviceMemory>,
    cull_status_bufs: Vec<vk::Buffer>,
    cull_status_mems: Vec<vk::DeviceMemory>,
    cull_sets: Vec<vk::DescriptorSet>,
    // A bake-style Hi-Z read set (cull set 1) with hiz_enabled = 0 so the mirror
    // cull is frustum-only -- the main camera's pyramid is meaningless for a
    // reflected frustum. `Some` only when the world runs Hi-Z. Shared across planes.
    hiz_set: Option<vk::DescriptorSet>,
    hiz_ubo: Option<(vk::Buffer, vk::DeviceMemory)>,
    pool: vk::DescriptorPool,
}

// The frame-side handles the planar reflected-frustum cull needs: the per-frame
// object + draw-args SSBOs it reads (camera-independent, so the reflected cull sees
// every object, not just the main camera's visible set), the cull descriptor-set
// layout, the build-time object count, and -- when the world runs Hi-Z -- the Hi-Z
// read-set layout + pyramid (view, sampler) so a hiz_enabled = 0 set can be bound
// (the cull pipeline layout statically references set 1).
pub(in crate::vulkan) struct PlanarCullSources<'a> {
    pub(in crate::vulkan) frame_object_buffers: &'a [vk::Buffer],
    pub(in crate::vulkan) frame_draw_args_buffers: &'a [vk::Buffer],
    pub(in crate::vulkan) cull_set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) cull_count: usize,
    pub(in crate::vulkan) hiz: Option<(vk::DescriptorSetLayout, vk::ImageView, vk::Sampler)>,
}

// The mapped view-ring pointers are POD raw pointers; the upload buffers stay
// alive through the struct fields and the pointers are written on the render
// thread only. Mirrors GlassResources.
unsafe impl Send for PlanarReflectionSet {}
unsafe impl Sync for PlanarReflectionSet {}

// Create the shared colour (MSAA only) + shared depth + per-plane targets at the
// given render dimensions.
#[allow(clippy::too_many_arguments)]
fn create_targets(
    instance: &ash::Instance,
    device: &Device,
    pd: vk::PhysicalDevice,
    sample_count: vk::SampleCountFlags,
    width: u32,
    height: u32,
    plane_count: usize,
) -> Result<(Option<GpuImage>, GpuImage, Vec<GpuImage>), String> {
    let msaa = sample_count != vk::SampleCountFlags::TYPE_1;
    let w = width.max(1);
    let h = height.max(1);

    let color = if msaa {
        let (img, mem) = create_image(
            instance,
            device,
            pd,
            w,
            h,
            HDR_FORMAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            sample_count,
        )?;
        let view = create_image_view(device, img, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
        Some(GpuImage {
            image: img,
            memory: mem,
            view,
            aux_views: Vec::new(),
        })
    } else {
        None
    };

    let (depth_img, depth_mem) = create_image(
        instance,
        device,
        pd,
        w,
        h,
        PLANAR_DEPTH_FORMAT,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        sample_count,
    )?;
    let depth_view = create_image_view(
        device,
        depth_img,
        PLANAR_DEPTH_FORMAT,
        vk::ImageAspectFlags::DEPTH,
    )?;
    let depth = GpuImage {
        image: depth_img,
        memory: depth_mem,
        view: depth_view,
        aux_views: Vec::new(),
    };

    let mut targets = Vec::with_capacity(plane_count);
    for _ in 0..plane_count {
        let (img, mem) = create_image(
            instance,
            device,
            pd,
            w,
            h,
            HDR_FORMAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::SampleCountFlags::TYPE_1,
        )?;
        let view = create_image_view(device, img, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
        targets.push(GpuImage {
            image: img,
            memory: mem,
            view,
            aux_views: Vec::new(),
        });
    }
    Ok((color, depth, targets))
}

// One framebuffer per plane, render-pass-compatible with the bindless main pass:
// MSAA -> [shared colour, shared depth, plane target (resolve)], single-sample ->
// [plane target (colour), shared depth].
#[allow(clippy::too_many_arguments)]
fn create_framebuffers(
    device: &Device,
    main_render_pass: vk::RenderPass,
    sample_count: vk::SampleCountFlags,
    color: Option<&GpuImage>,
    depth: &GpuImage,
    targets: &[GpuImage],
    width: u32,
    height: u32,
) -> Result<Vec<vk::Framebuffer>, String> {
    let msaa = sample_count != vk::SampleCountFlags::TYPE_1;
    let mut out = Vec::with_capacity(targets.len());
    for target in targets {
        let attachments: Vec<vk::ImageView> = if msaa {
            vec![color.unwrap().view, depth.view, target.view]
        } else {
            vec![target.view, depth.view]
        };
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(main_render_pass)
            .attachments(&attachments)
            .width(width.max(1))
            .height(height.max(1))
            .layers(1);
        let fb = unsafe { device.create_framebuffer(&info, None) }
            .map_err(|e| format!("planar framebuffer: {e}"))?;
        out.push(fb);
    }
    Ok(out)
}

impl PlanarReflectionSet {
    // Build the planar set: shared colour + depth + per-plane targets at render
    // dimensions, per-plane framebuffers, the per-(plane, frame) reflected-view
    // UBO ring, and the per-(plane, frame) global sets (each carrying its reflected
    // view + the shared lighting / env bindings + an EMPTY ProbeSet so the mirror
    // render samples only sky) + the per-(plane, frame) reflected-frustum cull
    // resources (indirect + status + cull set reading the frame's object/draw-args).
    // The bindless object SSBO + texture pool (the bindless set) is the FRAME's,
    // bound at encode time.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vulkan) fn new(
        instance: &ash::Instance,
        device: &Device,
        pd: vk::PhysicalDevice,
        frames: usize,
        sample_count: vk::SampleCountFlags,
        width: u32,
        height: u32,
        planes: &[[f32; 4]],
        main_render_pass: vk::RenderPass,
        global_set_layout: vk::DescriptorSetLayout,
        light_ubo: vk::Buffer,
        light_size: u64,
        shadow_ubo: vk::Buffer,
        shadow_size: u64,
        shadow_map_view: vk::ImageView,
        shadow_sampler: vk::Sampler,
        irradiance_view: vk::ImageView,
        prefilter_view: vk::ImageView,
        cube_sampler: vk::Sampler,
        ssao_white_view: vk::ImageView,
        linear_sampler: vk::Sampler,
        cull: PlanarCullSources<'_>,
    ) -> Result<Self, String> {
        use super::probe_uniforms::{MAX_PROBES, ProbeSet};

        let plane_count = planes.len();
        let (color, depth, targets) = create_targets(
            instance,
            device,
            pd,
            sample_count,
            width,
            height,
            plane_count,
        )?;
        let framebuffers = create_framebuffers(
            device,
            main_render_pass,
            sample_count,
            color.as_ref(),
            &depth,
            &targets,
            width,
            height,
        )?;

        // EMPTY ProbeSet UBO (count 0): every mirror face reflects only the sky, so
        // the planar render never recurses into the probe set it is feeding.
        let empty = ProbeSet::EMPTY;
        let probeset_size = std::mem::size_of::<ProbeSet>() as u64;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let (probeset_buf, probeset_mem) = create_buffer(
            instance,
            device,
            pd,
            probeset_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            host,
        )?;
        unsafe {
            let p = device
                .map_memory(probeset_mem, 0, probeset_size, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("planar map probeset: {e}"))?;
            std::ptr::copy_nonoverlapping(
                &empty as *const ProbeSet as *const u8,
                p as *mut u8,
                probeset_size as usize,
            );
            device.unmap_memory(probeset_mem);
        }

        // Per-(plane, frame) reflected-view UBO ring.
        let view_size = std::mem::size_of::<ViewUniforms>() as u64;
        let ring = plane_count * frames;
        let mut view_bufs = Vec::with_capacity(ring);
        let mut view_mems = Vec::with_capacity(ring);
        let mut view_ptrs: Vec<*mut u8> = Vec::with_capacity(ring);
        for _ in 0..ring {
            let (buf, mem) = create_buffer(
                instance,
                device,
                pd,
                view_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                host,
            )?;
            let ptr = unsafe { device.map_memory(mem, 0, view_size, vk::MemoryMapFlags::empty()) }
                .map_err(|e| format!("planar map view ubo: {e}"))? as *mut u8;
            view_bufs.push(buf);
            view_mems.push(mem);
            view_ptrs.push(ptr);
        }

        // Per-(plane, frame) reflected-frustum cull output: a DEVICE_LOCAL indirect +
        // status SSBO each, sized by the build-time object count (resize never
        // touches them).
        use crate::gfx::render_types::{GpuDrawArgs, GpuObjectData};
        let object_range = (cull.cull_count * std::mem::size_of::<GpuObjectData>()).max(4) as u64;
        let args_range = (cull.cull_count * std::mem::size_of::<GpuDrawArgs>()).max(4) as u64;
        let indirect_size =
            (cull.cull_count * std::mem::size_of::<vk::DrawIndexedIndirectCommand>()).max(4) as u64;
        let status_size = (cull.cull_count * std::mem::size_of::<u32>()).max(4) as u64;
        let mut cull_indirect_bufs = Vec::with_capacity(ring);
        let mut cull_indirect_mems = Vec::with_capacity(ring);
        let mut cull_status_bufs = Vec::with_capacity(ring);
        let mut cull_status_mems = Vec::with_capacity(ring);
        for _ in 0..ring {
            let (ib, im) = create_buffer(
                instance,
                device,
                pd,
                indirect_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let (sb, sm) = create_buffer(
                instance,
                device,
                pd,
                status_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            cull_indirect_bufs.push(ib);
            cull_indirect_mems.push(im);
            cull_status_bufs.push(sb);
            cull_status_mems.push(sm);
        }

        // One pool: the per-(plane, frame) global sets (4 UBO + 4 sampler + the cube
        // array each) + the per-(plane, frame) cull sets (4 storage each) + one Hi-Z
        // set (1 sampler + 1 UBO) when the world runs Hi-Z.
        let has_hiz = cull.hiz.is_some();
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count((ring * 4 + usize::from(has_hiz)).max(1) as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count((ring * (4 + MAX_PROBES) + usize::from(has_hiz)).max(1) as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count((ring * 4).max(1) as u32),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets((ring * 2 + usize::from(has_hiz)).max(1) as u32);
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| format!("planar descriptor pool: {e}"))?;

        let layouts: Vec<_> = (0..ring).map(|_| global_set_layout).collect();
        let global_sets = alloc_descriptor_sets(device, pool, &layouts)?;

        let probe_cube_sky: Vec<vk::DescriptorImageInfo> = (0..MAX_PROBES)
            .map(|_| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(prefilter_view)
                    .sampler(cube_sampler)
            })
            .collect();
        for (i, &set) in global_sets.iter().enumerate() {
            let view_info = buf_info(view_bufs[i], view_size);
            let light_info = buf_info(light_ubo, light_size);
            let shadow_info = buf_info(shadow_ubo, shadow_size);
            let probeset_info = buf_info(probeset_buf, probeset_size);
            let shadow_img = img_info(shadow_map_view, shadow_sampler);
            let irr_img = img_info(irradiance_view, cube_sampler);
            let pre_img = img_info(prefilter_view, cube_sampler);
            let ssao_img = img_info(ssao_white_view, linear_sampler);
            let writes = [
                ubo_write(set, 0, &view_info),
                ubo_write(set, 1, &light_info),
                ubo_write(set, 2, &shadow_info),
                sampler_write(set, 3, &shadow_img),
                sampler_write(set, 4, &irr_img),
                sampler_write(set, 5, &pre_img),
                sampler_write(set, 6, &ssao_img),
                ubo_write(set, 7, &probeset_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(super::descriptor_layout::PROBE_CUBE_ARRAY_BINDING)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&probe_cube_sky),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        // Per-(plane, frame) cull sets: read the frame's object + draw-args SSBOs
        // (b0 / b1), write this plane's indirect + status (b2 / b3). Ring index
        // slot * frames + frame, so `i % frames` selects the frame's buffers.
        let cull_layouts: Vec<_> = (0..ring).map(|_| cull.cull_set_layout).collect();
        let cull_sets = alloc_descriptor_sets(device, pool, &cull_layouts)?;
        for (i, &set) in cull_sets.iter().enumerate() {
            let frame = i % frames;
            write_storage(
                device,
                set,
                0,
                cull.frame_object_buffers[frame],
                object_range,
            );
            write_storage(
                device,
                set,
                1,
                cull.frame_draw_args_buffers[frame],
                args_range,
            );
            write_storage(device, set, 2, cull_indirect_bufs[i], indirect_size);
            write_storage(device, set, 3, cull_status_bufs[i], status_size);
        }

        // The Hi-Z set (cull set 1) with hiz_enabled = 0: a frustum-only reflected
        // cull never samples the main camera's pyramid. Only when Hi-Z runs (the
        // cull pipeline layout statically references set 1 then). Shared across planes.
        let (hiz_set, hiz_ubo) = if let Some((hiz_layout, hiz_view, hiz_sampler)) = cull.hiz {
            use super::hiz::CullHizParams;
            let params = CullHizParams {
                prev_view_proj: [[0.0; 4]; 4],
                hiz_size: [1.0, 1.0],
                hiz_mip_count: 1,
                hiz_enabled: 0,
            };
            let params_size = std::mem::size_of::<CullHizParams>() as u64;
            let (ubo, umem) = create_buffer(
                instance,
                device,
                pd,
                params_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                host,
            )?;
            unsafe {
                let p = device
                    .map_memory(umem, 0, params_size, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("planar map hiz ubo: {e}"))?;
                std::ptr::copy_nonoverlapping(
                    &params as *const CullHizParams as *const u8,
                    p as *mut u8,
                    params_size as usize,
                );
                device.unmap_memory(umem);
            }
            let set = alloc_descriptor_sets(device, pool, std::slice::from_ref(&hiz_layout))?[0];
            let img = img_info(hiz_view, hiz_sampler);
            let ubo_info = buf_info(ubo, params_size);
            let writes = [sampler_write(set, 0, &img), ubo_write(set, 1, &ubo_info)];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            (Some(set), Some((ubo, umem)))
        } else {
            (None, None)
        };

        Ok(Self {
            planes: planes.to_vec(),
            frames,
            sample_count,
            width,
            height,
            main_render_pass,
            color,
            depth,
            targets,
            framebuffers,
            view_bufs,
            view_mems,
            view_ptrs,
            global_sets,
            probeset_buf,
            probeset_mem,
            cull_indirect_bufs,
            cull_indirect_mems,
            cull_status_bufs,
            cull_status_mems,
            cull_sets,
            hiz_set,
            hiz_ubo,
            pool,
        })
    }

    // Number of distinct reflector planes (mirror renders per frame).
    pub(in crate::vulkan) fn plane_count(&self) -> usize {
        self.planes.len()
    }

    // The shader-readable target view for plane `slot` (what the glass pass binds
    // for a pane assigned to that slot).
    pub(in crate::vulkan) fn target_view(&self, slot: usize) -> vk::ImageView {
        self.targets[slot].view
    }

    // Re-point the reflected-frustum cull's Hi-Z set (binding 0) at a fresh pyramid
    // view + sampler after a resize. The Hi-Z resource recreates its pyramid image
    // on resize, destroying the view this set captured at `new`; the planar set
    // persists, so its set 1 would otherwise dangle a freed view (the cull binds set
    // 1 unconditionally, even though hiz_enabled = 0 keeps it unsampled). Called
    // after `hiz.resize_to`, with the device idle. A no-op when the world runs no
    // Hi-Z (`hiz_set` is None).
    pub(in crate::vulkan) fn rewrite_hiz_view(
        &self,
        device: &Device,
        view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
        let Some(set) = self.hiz_set else {
            return;
        };
        let img = img_info(view, sampler);
        let write = sampler_write(set, 0, &img);
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    }

    // Recreate the shared colour + depth + per-plane targets + framebuffers at new
    // render dimensions. The view UBO ring + global sets + pool survive (the global
    // sets reference only the unchanged shared lighting / env bindings + the
    // per-(plane, frame) view UBOs). The targets move, so the caller must re-point
    // the glass pass's per-pane planar binding afterward.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        pd: vk::PhysicalDevice,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        // Build the new targets + framebuffers first, then retire the old ones, so
        // a failure leaves the existing set intact.
        let (color, depth, targets) = create_targets(
            instance,
            device,
            pd,
            self.sample_count,
            width,
            height,
            self.planes.len(),
        )?;
        let framebuffers = create_framebuffers(
            device,
            self.main_render_pass,
            self.sample_count,
            color.as_ref(),
            &depth,
            &targets,
            width,
            height,
        )?;

        unsafe {
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
        }
        if let Some(c) = self.color.take() {
            c.destroy(device);
        }
        let old_depth = std::mem::replace(&mut self.depth, depth);
        old_depth.destroy(device);
        for t in std::mem::replace(&mut self.targets, targets) {
            t.destroy(device);
        }
        self.color = color;
        self.framebuffers = framebuffers;
        self.width = width;
        self.height = height;
        Ok(())
    }

    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        unsafe {
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            if let Some(c) = &self.color {
                c.destroy(device);
            }
            self.depth.destroy(device);
            for t in &self.targets {
                t.destroy(device);
            }
            for (&buf, &mem) in self.view_bufs.iter().zip(self.view_mems.iter()) {
                device.unmap_memory(mem);
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
            for (&buf, &mem) in self
                .cull_indirect_bufs
                .iter()
                .zip(self.cull_indirect_mems.iter())
                .chain(
                    self.cull_status_bufs
                        .iter()
                        .zip(self.cull_status_mems.iter()),
                )
            {
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
            if let Some((buf, mem)) = self.hiz_ubo {
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
            device.destroy_buffer(self.probeset_buf, None);
            device.free_memory(self.probeset_mem, None);
            // The pool frees every global / cull / Hi-Z set allocated from it.
            device.destroy_descriptor_pool(self.pool, None);
        }
        self.framebuffers.clear();
        self.targets.clear();
        self.view_bufs.clear();
        self.view_mems.clear();
        self.view_ptrs.clear();
        self.global_sets.clear();
        self.cull_indirect_bufs.clear();
        self.cull_indirect_mems.clear();
        self.cull_status_bufs.clear();
        self.cull_status_mems.clear();
        self.cull_sets.clear();
    }
}

fn buf_info(buffer: vk::Buffer, range: u64) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(0)
        .range(range)
}

fn write_storage(
    device: &Device,
    set: vk::DescriptorSet,
    binding: u32,
    buffer: vk::Buffer,
    range: u64,
) {
    let info = buf_info(buffer, range);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

fn img_info(view: vk::ImageView, sampler: vk::Sampler) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(view)
        .sampler(sampler)
}

fn ubo_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(info))
}

fn sampler_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorImageInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(info))
}

impl VkContext {
    // Render the scene reflected across each plane in the planar set into that
    // plane's target. A no-op when no set exists. For each plane: write the
    // reflected ViewUniforms into this (plane, frame) ring slot, run the dedicated
    // reflected-frustum cull into the plane's indirect, then render the culled set
    // from the reflected view through the shared bindless encode_main_into_face into
    // the plane's framebuffer. Encoded on `cmd` at the
    // head of the transparent pass, before the glass draws sample the targets;
    // same-cmd-buffer ordering retires each target before its glass sample. Each
    // plane is oriented toward the camera so the oblique near-plane clip keeps the
    // camera's side.
    pub(in crate::vulkan) fn encode_planar_reflections(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        vp_mat: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        elapsed: f32,
    ) -> Result<(), String> {
        let Some(set) = self.planar_reflection.as_ref() else {
            return Ok(());
        };
        let Some(&bindless_set) = self.cull.bindless_sets.get(frame_idx) else {
            return Ok(());
        };

        // Recover the (jittered) projection from this frame's view-projection so the
        // mirror render shares the main camera's projection + jitter, keeping the
        // reflection aligned with the reflective fragment's screen-space sample.
        let proj = super::math::mat4_mul(vp_mat, super::math::mat4_inverse(self.view_matrix));
        let prefilter_mip_count = self.prefilter_mip_count as f32;
        let extent = vk::Extent2D {
            width: set.width,
            height: set.height,
        };

        for slot in 0..set.plane_count() {
            let oriented =
                crate::gfx::planar_reflection::orient_plane_toward(set.planes[slot], cam_pos);
            let m = crate::gfx::planar_reflection::planar_matrices(
                self.view_matrix,
                proj,
                cam_pos,
                oriented,
                PLANAR_CLIP_BIAS,
            );
            let view = ViewUniforms {
                vp: m.view_proj,
                view_mat: m.view,
                elapsed,
                // No reflection composite runs over the mirror render, so the
                // forward probe specular is its only reflection source; the EMPTY
                // ProbeSet then leaves it on the sky path.
                reflections_enabled: 0.0,
                cam_x: m.eye[0],
                cam_y: m.eye[1],
                cam_z: m.eye[2],
                prefilter_mip_count,
                _ep0: 0.0,
                _ep1: 0.0,
            };
            let ring = slot * set.frames + frame_idx;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &view as *const ViewUniforms as *const u8,
                    set.view_ptrs[ring],
                    std::mem::size_of::<ViewUniforms>(),
                );
            }
            // Reflected-frustum cull (compute, outside any render pass) into this
            // plane's indirect, reading the frame's camera-independent object set so
            // geometry visible only in the reflection is captured. The oblique clip
            // already rides the view-proj, so the extracted frustum also rejects
            // geometry behind the reflector.
            let frustum = crate::gfx::frustum::Frustum::from_view_projection(m.view_proj);
            self.encode_probe_cull(cmd, set.cull_sets[ring], set.hiz_set, &frustum, m.eye);
            self.encode_main_into_face(
                cmd,
                set.framebuffers[slot],
                extent,
                set.global_sets[ring],
                bindless_set,
                set.cull_indirect_bufs[ring],
            );
        }

        // Make every freshly rendered target visible to the glass fragment read.
        // The main render pass leaves them in SHADER_READ_ONLY (final layout) but
        // adds no output-side dependency, so order the colour writes before the
        // sample explicitly. Layout is unchanged (SHADER_READ_ONLY -> same).
        let barriers: Vec<vk::ImageMemoryBarrier> = set
            .targets
            .iter()
            .map(|t| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(t.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
            })
            .collect();
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pane_plane_passes_through_centre_with_unit_normal() {
        // A pane facing +z through (1, 2, 3): the plane constant places the centre
        // on the surface (n . c + d == 0), and the normal is carried unchanged.
        let p = pane_plane([0.0, 0.0, 1.0], [1.0, 2.0, 3.0]);
        assert_eq!([p[0], p[1], p[2]], [0.0, 0.0, 1.0]);
        let signed = p[0] * 1.0 + p[1] * 2.0 + p[2] * 3.0 + p[3];
        assert!(signed.abs() < 1e-5, "centre lies on the plane");
    }

    #[test]
    fn pane_plane_offset_is_negative_normal_dot_centre() {
        // Tilted normal: d == -(n . c).
        let n = [0.6, 0.0, 0.8];
        let c = [2.0, 5.0, -1.0];
        let p = pane_plane(n, c);
        let expect_d = -(n[0] * c[0] + n[1] * c[1] + n[2] * c[2]);
        assert!((p[3] - expect_d).abs() < 1e-5);
    }

    #[test]
    fn planar_budget_matches_backends() {
        // The reserved planar targets + the per-frame mirror-render cost are sized
        // off this; keep it in lockstep with `metal::planar` / `directx::planar` so
        // the three backends pick the same reflectors.
        assert_eq!(MAX_PLANAR_PLANES, 4);
    }
}
