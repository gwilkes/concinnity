// src/directx/resize.rs
//
// D3D12 swapchain + render-target resize handler. Polls `win_state.width/height`
// at the top of each frame and, when the dimensions diverged from the live
// render-target sizing, rebuilds every render-resolution-sized GPU resource and
// rewrites the descriptors that point at them. The descriptor *slots* never move
// (only the resources they point at) so the live root signatures + pipelines +
// pre-bound GPU descriptor handles keep working without a re-bind.
//
// Mirrors the Vulkan `rebuild_swapchain` flow in src/vulkan/swapchain.rs: a
// `wait_idle` gate, a wholesale drop + recreate, then per-effect resource
// rebuilds (TAA / SSAO / SSR / bloom).
//
// Bloom mip count is held fixed at init's value rather than recomputed at the
// new resolution. `bloom_mip_count` only changes for very small windows (<128
// pixels in the smaller dimension), and keeping the count stable keeps the
// SRV/RTV heap layout stable so everything past the bloom block stays at its
// originally-allocated slot.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;

use crate::directx::context::{DxContext, FRAMES};
use crate::directx::post::bloom::{bloom_top_extent, create_bloom_mips_at, write_color_rtv};
use crate::directx::texture::{
    HDR_FORMAT, create_hdr_color_target, create_hdr_resolve_target, create_main_depth_texture,
    write_hdr_srv,
};

impl DxContext {
    // Poll the window state and, if the client area resized since the last
    // `draw_frame`, rebuild every render-target-sized GPU resource. Called at
    // the top of `draw_frame` before any rendering happens. Returns `Ok(())`
    // when no work was needed, when the work succeeded, or when the window is
    // minimised (one or both dimensions zero: we just skip the frame's
    // resize cycle and leave the targets at their previous size; the next
    // non-zero size restores them).
    pub(super) fn maybe_handle_resize(&mut self) -> Result<(), String> {
        let new_w = self.win_state.width.max(0) as u32;
        let new_h = self.win_state.height.max(0) as u32;
        if new_w == 0 || new_h == 0 {
            return Ok(());
        }
        // Compare against the *drawable* dims, not the render dims; with
        // temporal upscaling on, `render_width`/`render_height` are a
        // fraction of the window size and would never equal it, firing a
        // pointless rebuild every frame.
        if new_w == self.output_width && new_h == self.output_height {
            return Ok(());
        }
        self.handle_resize(new_w, new_h)
    }

    // (Re)acquire the swapchain back buffers and write their RTVs into the
    // pre-reserved RTV heap slots. Used by the resize path on both success (the
    // freshly-sized buffers) and failure (the unchanged old buffers), so a
    // failed `ResizeBuffers` never leaves `back_buffers` empty.
    fn populate_back_buffers(&mut self) -> Result<(), String> {
        self.back_buffers.clear();
        let rtv_base = unsafe { self.rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        for i in 0..FRAMES {
            let buf: ID3D12Resource = unsafe { self.swapchain.GetBuffer(i as u32) }
                .map_err(|e| format!("GetBuffer[{i}]: {e}"))?;
            let rtv_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv_base.ptr + i * self.rtv_descriptor_size,
            };
            unsafe { self.device.CreateRenderTargetView(&buf, None, rtv_handle) };
            self.back_buffers.push(buf);
        }
        Ok(())
    }

    // Wholesale resize. Caller has already validated `new_w` / `new_h` are
    // non-zero and differ from the live size. The flow mirrors the Vulkan
    // rebuild: `wait_idle`, drop the old resources, recreate at the new
    // resolution, rewrite every dependent SRV/RTV/DSV at its existing heap
    // slot, and refresh `render_width` / `render_height`.
    fn handle_resize(&mut self, new_w: u32, new_h: u32) -> Result<(), String> {
        self.wait_idle();

        // 0) Temporal upscaler. `new_w`/`new_h` are the new drawable dims;
        //    rebuild the FFX context for them (its `max_render`/`max_upscale`
        //    sizes are baked at creation, so a resize needs a fresh context)
        //    and recompute the off-screen scene render dims from its quality
        //    scale. The output texture is recreated at the new drawable size
        //    and its UAV/SRV are rewritten into the same pre-reserved heap
        //    slots, so the live `scene_srv_for_post` binding stays valid.
        //    A failed rebuild degrades to native-resolution rendering (render
        //    == output). Everything downstream sizes off `render_w`/`render_h`.
        let (render_w, render_h) = if let Some(old) = self.upscale.backend.as_ref() {
            let scale = old.upscale_scale();
            let (uav, srv_cpu, srv_gpu) = old.output_descriptors();
            let backend = self.upscale.requested;
            // Drop the old context before building the replacement (its
            // max_render / max_upscale sizes are baked at creation).
            self.upscale.backend = None;
            let rebuilt = crate::directx::post::upscale::build_upscaler(
                &self.device,
                &self.command_queue,
                new_w,
                new_h,
                scale,
                uav,
                srv_cpu,
                srv_gpu,
                backend,
            )?
            .0;
            let dims = match &rebuilt {
                Some(u) => u.render_dims(),
                None => (new_w, new_h),
            };
            self.upscale.backend = rebuilt;
            dims
        } else {
            (new_w, new_h)
        };

        // 1) Swapchain back-buffers. `ResizeBuffers` requires every reference to
        //    the back-buffer resources to be released first. Besides the
        //    `back_buffers` Vec, the composite pass records each back buffer onto
        //    its slot's end command list, and those recorded references persist
        //    until the list is reset. `maybe_handle_resize` runs at the top of
        //    `draw_frame` before this frame's lists are reset, so every in-flight
        //    slot's end list still pins a back buffer; reset + close them here
        //    (the GPU is already drained by `wait_idle`) so no reference outlives
        //    the clear. Without this, `ResizeBuffers` fails with
        //    DXGI_ERROR_INVALID_CALL and the window can never be resized.
        for i in 0..FRAMES {
            unsafe {
                let _ = self.commands.end_command_allocators[i].Reset();
                if self.commands.end_command_lists[i]
                    .Reset(&self.commands.end_command_allocators[i], None)
                    .is_ok()
                {
                    let _ = self.commands.end_command_lists[i].Close();
                }
            }
        }
        self.back_buffers.clear();
        // ResizeBuffers must be passed the same flags the swapchain was created
        // with, so an ALLOW_TEARING (uncapped) swapchain keeps the flag here.
        let resize_flags = if self.allow_tearing {
            DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        } else {
            DXGI_SWAP_CHAIN_FLAG(0)
        };
        if let Err(e) = unsafe {
            self.swapchain.ResizeBuffers(
                FRAMES as u32,
                new_w,
                new_h,
                self.swap_format,
                resize_flags,
            )
        } {
            // The resize failed; `ResizeBuffers` leaves the swapchain at its
            // current size, so re-acquire the existing back buffers + RTVs. This
            // keeps the renderer presenting at the old size instead of indexing
            // an empty `back_buffers` (a panic) on the next frame; the resize
            // poll retries on the following frame.
            self.populate_back_buffers()?;
            return Err(format!("ResizeBuffers: {e}"));
        }
        self.populate_back_buffers()?;
        // The recreated back buffers invalidate any previously captured present
        // index; a `screenshot` before the next present then returns a clean
        // error instead of reading a stale buffer.
        self.last_present_index = None;

        // 2) Main HDR colour + (optional) HDR resolve + depth. The HDR scene
        //    SRV (`hdr_srv_gpu`) and the decal/fog main-depth SRV
        //    (`decal_depth_srv_gpu` on `DecalResources` / `FogResources`) are
        //    rewritten into their existing heap slots, so the consumers don't
        //    need a re-bind.
        self.hdr.color = create_hdr_color_target(
            &self.device,
            render_w,
            render_h,
            self.hdr.msaa_samples,
            self.hdr.color_rtv,
            self.clear_color,
        )?;
        if self.hdr.msaa_samples > 1 {
            let resolve = create_hdr_resolve_target(&self.device, render_w, render_h)?;
            // Rewrite the resolve target's RTV at the slot the decal pass
            // already binds. The CPU handle was captured at init when MSAA
            // turned the slot on; it stays valid across resize.
            if let Some(rtv) = self.hdr.resolve_rtv {
                let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
                    Format: HDR_FORMAT,
                    ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                    ..Default::default()
                };
                unsafe {
                    self.device
                        .CreateRenderTargetView(&resolve, Some(&rtv_desc), rtv)
                };
            }
            self.hdr.resolve = Some(resolve);
        }
        // Recreate main depth (shader-readable so the decal/fog/auto-exposure
        // paths can sample it). The DSV is rewritten at the same slot.
        self.depth_resource = create_main_depth_texture(
            &self.device,
            render_w,
            render_h,
            self.depth_dsv,
            self.hdr.msaa_samples,
            true,
        )?;

        // Cache the SRV-heap CPU/GPU bases so the per-resource SRV rewrites
        // below can derive the CPU handle from each stored GPU handle without
        // borrowing `self` again (the per-effect rebuilds need `&mut self`).
        let srv_cpu_base = unsafe {
            self.descriptors
                .srv_heap
                .GetCPUDescriptorHandleForHeapStart()
        };
        let srv_gpu_base = unsafe {
            self.descriptors
                .srv_heap
                .GetGPUDescriptorHandleForHeapStart()
        };
        let srv_cpu_of = |gpu: D3D12_GPU_DESCRIPTOR_HANDLE| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: srv_cpu_base.ptr + (gpu.ptr - srv_gpu_base.ptr) as usize,
        };

        // 3) Refresh SRVs that point at the recreated resources. The GPU
        //    handles stored on the various Resources structs already match
        //    these heap slots; we just rewrite the descriptors in place.
        write_hdr_srv(
            &self.device,
            self.hdr.resolve.as_ref().unwrap_or(&self.hdr.color),
            srv_cpu_of(self.hdr.srv_gpu),
        );

        // The main-depth SRV is shared by the decal pass and the fog pass.
        // Both store the same `depth_srv_gpu`; rewrite the descriptor once
        // and both consumers pick it up.
        if let Some(decals) = self.decal.state.as_ref() {
            crate::directx::decal::write_main_depth_srv(
                &self.device,
                &self.depth_resource,
                srv_cpu_of(decals.depth_srv_gpu),
                self.hdr.msaa_samples,
            );
        }

        // Rebuild the transient pool (`bloom_top` + `ao_output`) at the new
        // resolution up front, before the consumers below read it back: the
        // bloom chain takes its pooled `mips[0]` and SSAO re-points its
        // `ao_output` RTV/SRV from it. The device is idle at the top of resize,
        // so dropping the old placed resources + heaps is sound.
        let ssao_on = self.ssao.resources.is_some();
        self.transient_pool.rebuild(
            &self.device,
            &self.command_queue,
            &super::transient_pool::transient_slots(
                ssao_on,
                (render_w, render_h),
                bloom_top_extent(new_w, new_h),
            ),
        )?;

        // 4) Bloom mip chain. Keep the count fixed at the init-time value
        //    (`self.bloom.mips.len()`) so the heap layout past the bloom
        //    block (LUT, TAA SRVs, SSAO SRVs, ...) stays anchored. `mips[0]`
        //    (`bloom_top`) is the pooled placed resource; the finer mips are
        //    committed.
        let bloom_count = self.bloom.mips.len();
        if bloom_count > 0 {
            let bloom_top = self
                .transient_pool
                .resource_for("bloom_top")
                .ok_or("transient pool missing bloom_top on resize")?
                .clone();
            let new_mips =
                create_bloom_mips_at(&self.device, new_w, new_h, bloom_count, bloom_top)?;
            self.bloom.mips = new_mips.0;
            self.bloom.mip_extents = new_mips.1;
            // Rewrite each mip's RTV + SRV into the existing slots.
            for i in 0..bloom_count {
                write_color_rtv(&self.device, &self.bloom.mips[i], self.bloom.mip_rtvs[i]);
                write_hdr_srv(
                    &self.device,
                    &self.bloom.mips[i],
                    srv_cpu_of(self.bloom.mip_srv_gpus[i]),
                );
            }
        }

        // 5) TAA: velocity + private depth + ping-pong history. Rebuild
        //    resources at the new size; the `frame` counter resets so the
        //    resolve pass treats the next frame as the first-after-resize
        //    (history is unreliable across a resize: the reprojection
        //    coordinates were generated at the old resolution).
        if let Some(taa) = self.taa.as_mut() {
            taa.resize_to(&self.device, render_w, render_h, srv_cpu_base, srv_gpu_base)?;
        }

        // 6) SSAO: pre-pass G-buffer + private depth + raw/blurred AO. The
        // blurred `ao_output` is pooled and was rebuilt above; SSAO rewrites its
        // RTV + SRV from the new pooled resource.
        if let Some(ao_resource) = self.transient_pool.resource_for("ao_output").cloned()
            && let Some(ssao) = self.ssao.resources.as_mut()
        {
            ssao.resize_to(
                &self.device,
                render_w,
                render_h,
                srv_cpu_base,
                srv_gpu_base,
                &ao_resource,
            )?;
        }

        // 7) SSR: pre-pass G-buffer + roughness + private depth + resolve output.
        if let Some(ssr) = self.ssr.as_mut() {
            ssr.resize_to(&self.device, render_w, render_h, srv_cpu_base, srv_gpu_base)?;
        }

        // 7-gbuffer) Unified G-buffer pre-pass: normal+depth + roughness +
        // velocity + private depth, all at render resolution.
        if let Some(gbuffer) = self.gbuffer.as_mut() {
            gbuffer.resize_to(&self.device, render_w, render_h, srv_cpu_base, srv_gpu_base)?;
        }

        // 7-ssgi) SSGI gather target. Re-uses its pre-reserved RTV/SRV slots;
        // the live pass binding (which points at the SRV slot's GPU handle)
        // stays valid after the in-place descriptor rewrite.
        if let Some(ssgi) = self.ssgi.as_mut() {
            ssgi.resize_to(&self.device, render_w, render_h, srv_cpu_base, srv_gpu_base)?;
        }

        // 7-rt) RT reflections output target. Re-uses its pre-reserved RTV/SRV
        //     slots; the acceleration structure is resolution-independent and is
        //     left untouched.
        if let Some(rt) = self.rt_reflections.as_mut() {
            rt.resize_to(&self.device, render_w, render_h, srv_cpu_base, srv_gpu_base)?;
        }

        // 7-refl) Reflection composite: the full-res composited output + the
        //     reduced-res roughness blur. Re-uses its pre-reserved RTV/SRV slots,
        //     so the live scene binding (which points at the output SRV slot) stays
        //     valid after the in-place descriptor rewrite.
        if let Some(rc) = self.reflection_composite.as_mut() {
            rc.resize_to(&self.device, render_w, render_h, srv_cpu_base, srv_gpu_base)?;
        }

        // 7a) Raymarch: recreate the `hdr_resolve_copy` scene snapshot at
        //     the new dims and rewrite its SRV descriptor in place. The
        //     descriptor slot itself doesn't move, so the live raymarch
        //     root-table binding stays valid without a re-bind.
        if let Some(rm) = self.raymarch.as_mut() {
            rm.resize_to(&self.device, render_w, render_h)?;
        }

        // 7b) Hi-Z (depth-mip pyramid). Resource sized to the depth buffer; the
        //     mip count adapts to the new dimensions. Re-uses the pre-reserved
        //     SRV / UAV heap slots; the live cull binding (which points at
        //     the SRV slot's GPU handle) stays valid. The pyramid is invalid
        //     until the next frame rebuilds it, so flip `hiz_valid` back to
        //     false so the next cull dispatch skips the occlusion test
        //     (NDC coords from the old resolution would mis-sample the new
        //     mip dimensions otherwise).
        if let Some(hiz) = self.cull.hiz.as_mut() {
            hiz.resize_to(&self.device, render_w, render_h)?;
        }
        self.cull.hiz_valid.set(false);

        // 7c) Transparent glass: recreate the scene snapshot at the new dims
        //     and rewrite its SRV in place. The depth SRV the glass pass also
        //     binds is the main-depth slot, rewritten by the decal path.
        if let Some(glass) = self.glass.as_mut() {
            glass.resize_to(&self.device, render_w, render_h)?;
        }

        // 7d) Planar reflections: recreate the shared mirror colour + depth + the
        //     per-plane resolves at the new render dims and rewrite their RTV / DSV /
        //     SRVs in place, so the glass pass's per-pane resolve bindings stay valid.
        if let Some(planar) = self.planar_reflection.as_mut() {
            planar.resize_to(&self.device, render_w, render_h)?;
        }

        // 8) Commit the new dimensions. `render_*` drives the scene-pass
        //    viewports + the sub-pixel jitter; `output_*` drives the
        //    composite viewport and is what the next resize poll compares
        //    against. They differ only while temporal upscaling is active.
        self.render_width = render_w;
        self.render_height = render_h;
        self.output_width = new_w;
        self.output_height = new_h;

        // 9) Reset the swapchain back-buffer index. After `ResizeBuffers` the
        //    swapchain's notion of "current back buffer" is the next one to be
        //    presented; we don't need to mirror that here because
        //    `current_frame` indexes our internal frames-in-flight ring (not
        //    the swapchain), and `GetCurrentBackBufferIndex` is queried fresh
        //    each frame inside `draw_frame`.

        Ok(())
    }
}
