// src/directx/planar.rs
//
// Planar reflection for flat glass panes on the D3D12 backend. Each frame the
// scene is rendered a second time from the camera reflected across each pane's
// plane (mirror view + oblique near-plane clip so geometry behind the plane
// never leaks in) into a dedicated render-resolution target; the pane's
// fragment shader then samples that target projectively for a sharp,
// scene-correct reflection instead of the blurry box-projected probe cube.
//
// Mirrors src/metal/planar.rs, glass-only (water is a Metal-only producer). One
// mirror render per DISTINCT plane: near-coplanar panes (one wall of windows)
// share a render, and panes past the budget (`MAX_PLANAR_PLANES`) fall back to
// the probe cube. The plane -> slot grouping + the mirror matrices come from the
// pure, unit-tested `gfx::planar_reflection`.
//
// Each plane gets a DEDICATED reflected-frustum mirror cull (`encode_planar_culls`,
// mirroring `metal::cull::encode_mirror_cull`): the GPU cull re-runs against the
// reflected-camera frustum into that plane's region of a per-frame indirect buffer,
// reading the frame's camera-independent object + draw-args buffers. So geometry
// visible only in the reflection (behind / beside the main camera, outside its
// frustum) is captured, not just the main camera's visible set; the reflected
// view-proj's oblique near-plane clip also rejects geometry behind the reflector.
// The face render then executes that region.
//
// V1 scope (documented, matches the probe capture's own simplification): static +
// instanced + chunk geometry only -- skinned meshes are not drawn into the mirror
// (the bindless face render omits the skinned tail), exactly like the probe capture.

#![allow(clippy::incompatible_msrv)]

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::context::{DxContext, FRAMES, align256};
use super::cull::INDIRECT_COMMAND_STRIDE;
use super::draw::ViewUniforms;
use super::graph_exec::GraphFrameParams;
use super::texture::{
    HDR_FORMAT, create_buffer, create_hdr_color_target, create_hdr_resolve_target,
    create_uav_buffer, transition_barrier, write_hdr_srv,
};

// Maximum number of distinct reflection planes that render a mirror pass each
// frame. Each plane is a full scene re-render, so this caps the per-frame cost;
// panes past the budget fall back to the box-projected probe cube. Matches
// `metal::planar::MAX_PLANAR_PLANES`.
pub(in crate::directx) const MAX_PLANAR_PLANES: usize = 4;

// Clip the reflection a hair toward the kept (camera) side of the plane so
// geometry exactly on the surface is not lost to near-plane precision. Matches
// `metal::planar::PLANAR_CLIP_BIAS`.
const PLANAR_CLIP_BIAS: f32 = 0.02;

// World-space plane `[nx, ny, nz, d]` (unit normal, `n . p + d = 0` on the
// surface) for a glass pane with unit `normal` through `centre`. Pure; unit
// tested. The init path feeds these to `assign_planar_slots`.
pub(in crate::directx) fn pane_plane(normal: [f32; 3], centre: [f32; 3]) -> [f32; 4] {
    [
        normal[0],
        normal[1],
        normal[2],
        -(normal[0] * centre[0] + normal[1] * centre[1] + normal[2] * centre[2]),
    ]
}

// The set of distinct reflection planes for the world, each rendering its mirror
// into the shared MSAA colour + depth then resolving into its own shader-readable
// resolve. A pane samples the resolve of the slot it was assigned at init (see
// `gfx::planar_reflection::assign_planar_slots`). Rebuilt on resize alongside the
// HDR targets; the planes + slot assignment are fixed at init.
pub(in crate::directx) struct PlanarReflectionSet {
    // Distinct reflector planes (the `assign_planar_slots` representatives), one
    // per resolve slot. Re-oriented toward the camera per frame.
    planes: Vec<[f32; 4]>,
    sample_count: u32,
    clear_color: [f32; 4],

    // Shared colour + depth, reused across planes (rendered then resolved one
    // plane at a time, exactly like the probe shares one face target across its
    // six faces). Own non-shader-visible RTV / DSV heaps.
    color: ID3D12Resource,
    _depth: ID3D12Resource,
    _rtv_heap: ID3D12DescriptorHeap,
    _dsv_heap: ID3D12DescriptorHeap,
    color_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    depth_dsv: D3D12_CPU_DESCRIPTOR_HANDLE,

    // Per-plane shader-readable resolve + its SRV (CPU handle for the resize
    // rewrite, GPU handle for the glass pass to bind). The SRVs live in reserved
    // slots of the main shader-visible heap.
    resolves: Vec<ID3D12Resource>,
    resolve_srv_cpu: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    resolve_srv_gpu: Vec<D3D12_GPU_DESCRIPTOR_HANDLE>,

    // Per-(plane, frame) reflected `ViewUniforms` CBV ring, persistently mapped.
    // Indexed `plane * FRAMES + frame_idx`, so each frame writes its own slot and
    // never races the GPU reading a prior frame's reflected view.
    _view_cbvs: Vec<ID3D12Resource>,
    view_ptrs: Vec<*mut u8>,
    view_gvas: Vec<u64>,

    // Per-frame reflected-frustum cull output: one indirect buffer per frame
    // holding `plane_count` regions of `n_cull` commands each, plus a per-frame
    // never-read cull-status scratch. Frame-indexed so frame N writes its own and
    // never races the GPU reading frame N-1's. The per-plane face render issues its
    // region (`region_offset`) of `planar_indirect[frame]`. Sized by the object
    // count (`n_cull`), which is fixed at init, so resize never touches them.
    planar_indirect: Vec<ID3D12Resource>,
    planar_status: Vec<ID3D12Resource>,
    n_cull: usize,
}

// The mapped view-ring pointers are POD raw pointers; the upload buffers stay
// alive through the `Vec<ID3D12Resource>` field and the pointers are written on
// the render thread only. Mirrors `GlassResources`.
unsafe impl Send for PlanarReflectionSet {}
unsafe impl Sync for PlanarReflectionSet {}

impl PlanarReflectionSet {
    // Build the planar set: shared colour + depth at `width`x`height` (matching
    // the main pass's formats + sample count so the bindless face render binds the
    // standard pipeline), one resolve per plane with its SRV written into the
    // reserved heap slot, and the per-(plane, frame) reflected-view CBV ring.
    // `resolve_srv_cpu` / `resolve_srv_gpu` are the reserved heap descriptors, one
    // per plane in `planes`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        sample_count: u32,
        width: u32,
        height: u32,
        planes: &[[f32; 4]],
        resolve_srv_cpu: &[D3D12_CPU_DESCRIPTOR_HANDLE],
        resolve_srv_gpu: &[D3D12_GPU_DESCRIPTOR_HANDLE],
        clear_color: [f32; 4],
        // Build-time draw-record count (`DxContext::cull_count`): sizes each plane's
        // region of the per-frame mirror-cull indirect buffer.
        n_cull: usize,
    ) -> Result<Self, String> {
        let rtv_heap = create_rtv_heap(device)?;
        let dsv_heap = create_dsv_heap(device)?;
        let color_rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        let depth_dsv = unsafe { dsv_heap.GetCPUDescriptorHandleForHeapStart() };

        let color = create_hdr_color_target(
            device,
            width.max(1),
            height.max(1),
            sample_count,
            color_rtv,
            clear_color,
        )?;
        let depth =
            create_planar_depth(device, width.max(1), height.max(1), sample_count, depth_dsv)?;

        let mut resolves = Vec::with_capacity(planes.len());
        for (i, _) in planes.iter().enumerate() {
            let resolve = create_hdr_resolve_target(device, width.max(1), height.max(1))?;
            write_hdr_srv(device, &resolve, resolve_srv_cpu[i]);
            resolves.push(resolve);
        }

        let mut view_cbvs = Vec::with_capacity(planes.len() * FRAMES);
        let mut view_ptrs = Vec::with_capacity(planes.len() * FRAMES);
        let mut view_gvas = Vec::with_capacity(planes.len() * FRAMES);
        for _ in 0..planes.len() * FRAMES {
            let cbv = create_buffer(
                device,
                256,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            unsafe { cbv.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("planar: map view cbv: {e}"))?;
            view_gvas.push(unsafe { cbv.GetGPUVirtualAddress() });
            view_ptrs.push(ptr as *mut u8);
            view_cbvs.push(cbv);
        }

        // Per-frame mirror-cull output: `plane_count` regions of `n_cull` commands,
        // plus a never-read status scratch. Created in COMMON (D3D12 promotes buffers
        // from COMMON on first use), matching the shadow indirect buffers.
        let indirect_size =
            align256((planes.len() * n_cull * INDIRECT_COMMAND_STRIDE as usize) as u64);
        let status_size = align256((n_cull * std::mem::size_of::<u32>()) as u64).max(256);
        let mut planar_indirect = Vec::with_capacity(FRAMES);
        let mut planar_status = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            planar_indirect.push(create_uav_buffer(
                device,
                indirect_size.max(256),
                D3D12_RESOURCE_STATE_COMMON,
            )?);
            planar_status.push(create_uav_buffer(
                device,
                status_size,
                D3D12_RESOURCE_STATE_COMMON,
            )?);
        }

        Ok(Self {
            planes: planes.to_vec(),
            sample_count,
            clear_color,
            color,
            _depth: depth,
            _rtv_heap: rtv_heap,
            _dsv_heap: dsv_heap,
            color_rtv,
            depth_dsv,
            resolves,
            resolve_srv_cpu: resolve_srv_cpu.to_vec(),
            resolve_srv_gpu: resolve_srv_gpu.to_vec(),
            _view_cbvs: view_cbvs,
            view_ptrs,
            view_gvas,
            planar_indirect,
            planar_status,
            n_cull,
        })
    }

    // Recreate the shared colour + depth + per-plane resolves at new render-target
    // dimensions and rewrite the RTV / DSV / resolve SRVs in place. The descriptor
    // slots do not move, so the glass pass's GPU handles stay valid. Mirrors the
    // other `resize_to` resources.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let (w, h) = (width.max(1), height.max(1));
        self.color = create_hdr_color_target(
            device,
            w,
            h,
            self.sample_count,
            self.color_rtv,
            self.clear_color,
        )?;
        self._depth = create_planar_depth(device, w, h, self.sample_count, self.depth_dsv)?;
        for i in 0..self.resolves.len() {
            let resolve = create_hdr_resolve_target(device, w, h)?;
            write_hdr_srv(device, &resolve, self.resolve_srv_cpu[i]);
            self.resolves[i] = resolve;
        }
        Ok(())
    }

    // GPU descriptor handle of plane `slot`'s resolve SRV (what the glass pass
    // binds for a pane assigned to this slot).
    pub(in crate::directx) fn resolve_srv_gpu(&self, slot: usize) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        self.resolve_srv_gpu[slot]
    }

    // Number of distinct reflector planes (mirror renders per frame).
    pub(in crate::directx) fn plane_count(&self) -> usize {
        self.planes.len()
    }

    // This frame's mirror-cull indirect buffer (the per-plane regions the face
    // render executes).
    fn indirect(&self, frame: usize) -> &ID3D12Resource {
        &self.planar_indirect[frame]
    }

    // GPU address of this frame's never-read mirror-cull status scratch.
    fn status_gva(&self, frame: usize) -> u64 {
        unsafe { self.planar_status[frame].GetGPUVirtualAddress() }
    }

    // Byte offset to plane `slot`'s region (of `n_cull` commands) in the indirect
    // buffer, for the face render's `ExecuteIndirect`.
    fn region_offset(&self, slot: usize) -> u32 {
        (slot * self.n_cull * INDIRECT_COMMAND_STRIDE as usize) as u32
    }

    // Resolve the shared colour (just rendered for plane `slot`) into that plane's
    // shader-readable resolve, leaving the resolve in PIXEL_SHADER_RESOURCE and the
    // colour back in RENDER_TARGET for the next plane. MSAA resolves; a
    // single-sample target copies.
    fn resolve_into(&self, cmd: &ID3D12GraphicsCommandList, slot: usize) {
        let resolve = &self.resolves[slot];
        if self.sample_count > 1 {
            unsafe {
                cmd.ResourceBarrier(&[
                    transition_barrier(
                        &self.color,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                    ),
                    transition_barrier(
                        resolve,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_RESOLVE_DEST,
                    ),
                ]);
                cmd.ResolveSubresource(resolve, 0, &self.color, 0, HDR_FORMAT);
                cmd.ResourceBarrier(&[
                    transition_barrier(
                        resolve,
                        D3D12_RESOURCE_STATE_RESOLVE_DEST,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    ),
                    transition_barrier(
                        &self.color,
                        D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                    ),
                ]);
            }
        } else {
            unsafe {
                cmd.ResourceBarrier(&[
                    transition_barrier(
                        &self.color,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ),
                    transition_barrier(
                        resolve,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    ),
                ]);
                cmd.CopyResource(resolve, &self.color);
                cmd.ResourceBarrier(&[
                    transition_barrier(
                        resolve,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    ),
                    transition_barrier(
                        &self.color,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                    ),
                ]);
            }
        }
    }
}

impl DxContext {
    // Render the scene reflected across each plane in the planar set into that
    // plane's resolve. A no-op (returns Ok) when no set exists. For each plane: a
    // dedicated reflected-frustum mirror cull fills the plane's region of this
    // frame's indirect buffer (reading the frame's camera-independent object +
    // draw-args), then the bindless face render draws that region from the reflected
    // view into the shared colour + depth, and resolves into the plane's resolve.
    // Encoded on `cmd` before the transparent pass samples the resolves; same-cmd
    // -list ordering retires each resolve before its glass sample. Each plane is
    // oriented toward the camera so the oblique near-plane clip keeps the camera's
    // side.
    pub(in crate::directx) fn encode_planar_reflections(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        params: &GraphFrameParams<'_>,
    ) -> Result<(), String> {
        let Some(set) = self.planar_reflection.as_ref() else {
            return Ok(());
        };

        // Recover the (jittered) projection from this frame's view-projection so
        // the mirror render shares the main camera's projection + jitter, keeping
        // the reflection aligned with the reflective fragment's screen-space sample.
        let proj =
            super::math::mat4_mul(params.vp_mat, super::math::mat4_inverse(self.view_matrix));
        let prefilter_mip_count = self.env_map.prefilter_mip_count as f32;
        let (w, h) = (self.render_width, self.render_height);

        // Per plane: compute the reflected matrices, write the reflected view CBV,
        // and collect the reflected frustum + eye for the mirror cull.
        let mut cull_planes: Vec<(crate::gfx::frustum::Frustum, [f32; 3])> =
            Vec::with_capacity(set.plane_count());
        for slot in 0..set.plane_count() {
            let oriented = crate::gfx::planar_reflection::orient_plane_toward(
                set.planes[slot],
                params.cam_pos,
            );
            let m = crate::gfx::planar_reflection::planar_matrices(
                self.view_matrix,
                proj,
                params.cam_pos,
                oriented,
                PLANAR_CLIP_BIAS,
            );
            let view = ViewUniforms {
                vp: m.view_proj,
                view_mat: m.view,
                elapsed: params.elapsed,
                // No reflection resolve runs over the planar mirror render, so
                // the forward probe specular is its only reflection source.
                reflections_enabled: 0.0,
                cam_x: m.eye[0],
                cam_y: m.eye[1],
                cam_z: m.eye[2],
                prefilter_mip_count,
                _ep0: 0.0,
                _ep1: 0.0,
            };
            let ring = slot * FRAMES + params.frame_idx;
            // SAFETY: `ring < planes.len() * FRAMES`; the CBV is 256 bytes and
            // `ViewUniforms` is 160. The slot is this frame's own, written before
            // the GPU reads it later on this cmd list.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &view as *const ViewUniforms as *const u8,
                    set.view_ptrs[ring],
                    std::mem::size_of::<ViewUniforms>(),
                );
            }
            cull_planes.push((
                crate::gfx::frustum::Frustum::from_view_projection(m.view_proj),
                m.eye,
            ));
        }

        // Reflected-frustum mirror cull into the per-plane regions of this frame's
        // indirect buffer (one barrier flip around all planes).
        self.encode_planar_culls(
            cmd,
            params.frame_idx,
            &cull_planes,
            set.indirect(params.frame_idx),
            set.status_gva(params.frame_idx),
            // Stride regions by the SAME fixed capacity `region_offset` reads with.
            set.n_cull,
        );

        // Per plane: render the culled region from the reflected view into the
        // shared colour + depth (against the frame's object buffer), then resolve.
        let frame_object_gva =
            unsafe { self.cull.object_buffer_resources[params.frame_idx].GetGPUVirtualAddress() };
        let indirect = set.indirect(params.frame_idx);
        for slot in 0..set.plane_count() {
            let ring = slot * FRAMES + params.frame_idx;
            self.encode_main_into_face(
                cmd,
                set.color_rtv,
                set.depth_dsv,
                set.view_gvas[ring],
                params.light_gva,
                params.shadow_ubo_gva,
                indirect,
                set.region_offset(slot),
                frame_object_gva,
                w,
                h,
            );
            set.resolve_into(cmd, slot);
        }
        Ok(())
    }
}

// A one-entry non-shader-visible RTV heap for the shared planar colour target.
fn create_rtv_heap(device: &ID3D12Device) -> Result<ID3D12DescriptorHeap, String> {
    let desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    unsafe { device.CreateDescriptorHeap(&desc) }.map_err(|e| format!("planar: rtv heap: {e}"))
}

// A one-entry non-shader-visible DSV heap for the shared planar depth target.
fn create_dsv_heap(device: &ID3D12Device) -> Result<ID3D12DescriptorHeap, String> {
    let desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    unsafe { device.CreateDescriptorHeap(&desc) }.map_err(|e| format!("planar: dsv heap: {e}"))
}

// Create the shared planar depth target (D32_FLOAT, matching the main pass's DSV
// format + the colour's sample count) and write its DSV. Created in DEPTH_WRITE
// and left there (the face render clears it every plane). Mirrors
// `probe::create_bake_depth` at a rectangular render resolution.
fn create_planar_depth(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    sample_count: u32,
    dsv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = D3D12_CLEAR_VALUE {
        Format: DXGI_FORMAT_D32_FLOAT,
        Anonymous: D3D12_CLEAR_VALUE_0 {
            DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                Depth: 1.0,
                Stencil: 0,
            },
        },
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            Some(&clear_value),
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("planar: create depth: {e}"))?;
    let texture = tex_opt.ok_or_else(|| "planar: create depth returned None".to_string())?;
    let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
        Format: DXGI_FORMAT_D32_FLOAT,
        ViewDimension: if sample_count > 1 {
            D3D12_DSV_DIMENSION_TEXTURE2DMS
        } else {
            D3D12_DSV_DIMENSION_TEXTURE2D
        },
        Flags: D3D12_DSV_FLAG_NONE,
        ..Default::default()
    };
    unsafe { device.CreateDepthStencilView(&texture, Some(&dsv_desc), dsv_cpu) };
    Ok(texture)
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
        // The reserved planar-resolve heap block + the per-frame mirror-render cost
        // are sized off this; keep it in lockstep with `metal::planar` /
        // `vulkan::planar` so the three backends pick the same reflectors.
        assert_eq!(MAX_PLANAR_PLANES, 4);
    }
}
