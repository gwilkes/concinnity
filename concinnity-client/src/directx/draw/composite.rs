// src/directx/draw/composite.rs
//
// Composite + text overlay: tonemap (and optionally LUT-grade) the HDR scene
// target onto the swapchain backbuffer, then layer the text vertices on top.
// The composite pass samples `scene_srv` (the post-TAA image when TAA is on,
// the HDR scene SRV otherwise) plus bloom mip 0; the text pass appends each
// label's vertex / index geometry into this frame slot's persistent upload
// buffer (see [`TextUploadRing`]) and binds sub-views into it, so no per-frame
// GPU buffers are allocated.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16_UINT;

use crate::gfx::render_types::{PostProcessParams, TextDrawCall, TextVertex};

use crate::directx::context::DxContext;
use crate::directx::texture::transition_barrier;

// Root constants for the text pass (16 bytes = 4 DWORDs): window dimensions.
#[derive(Copy, Clone)]
#[repr(C)]
struct TextPush {
    win_width: f32,
    win_height: f32,
    _pad0: f32,
    _pad1: f32,
}

// Per-invocation binding context for the composite pass. The back-buffer is a
// cheap COM-refcount clone so `Args` carries no borrow (the trait's associated
// type can't name a lifetime).
pub(crate) struct DxCompositeArgs {
    back_buffer: ID3D12Resource,
    back_buffer_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    scene_srv: D3D12_GPU_DESCRIPTOR_HANDLE,
    width: u32,
    height: u32,
    frame_idx: usize,
}

// The composite + text orchestration lives once in `gfx::fullscreen`; this impl
// drives each step in D3D12. The back buffer enters in `PRESENT`, is transitioned
// to `RENDER_TARGET` for the draws, and is returned to `PRESENT` on exit; the HDR
// target is expected to already be in `PIXEL_SHADER_RESOURCE` (the main pass
// leaves it that way).
impl crate::gfx::fullscreen::CompositeEncoder for DxContext {
    type Rec = ID3D12GraphicsCommandList;
    type Args = DxCompositeArgs;

    fn begin_composite(&self, cmd: &Self::Rec, args: &Self::Args) {
        let to_rt = transition_barrier(
            &args.back_buffer,
            D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        unsafe { cmd.ResourceBarrier(&[to_rt]) };
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&args.back_buffer_rtv), false, None);
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: args.width as f32,
                Height: args.height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: args.width as i32,
                bottom: args.height as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);
        }
    }

    fn composite_draw(&self, cmd: &Self::Rec, args: &Self::Args) {
        unsafe {
            cmd.SetPipelineState(&self.composite_pso);
            cmd.SetGraphicsRootSignature(&self.composite_root_sig);
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);
            // Root param [0]: scene SRV (t0): the TAA output when TAA is on,
            // the HDR scene target otherwise.
            cmd.SetGraphicsRootDescriptorTable(0, args.scene_srv);
            // Root param [1]: bloom mip 0 SRV (t1).
            cmd.SetGraphicsRootDescriptorTable(1, self.bloom.mip_srv_gpus[0]);
            // Root param [2]: PostProcessParams (exposure / vignette / bloom +
            // the `hdr_output` + `pq_output` HDR-branch toggles, 8 DWORDs
            // total, matching the root-sig declaration). Pushed verbatim so
            // the HLSL cbuffer reads the same byte order as the Rust struct.
            cmd.SetGraphicsRoot32BitConstants(
                2,
                8,
                &self.post_process as *const PostProcessParams as *const std::ffi::c_void,
                0,
            );
            // Root param [3]: 3D colour-grading LUT SRV (t2).
            cmd.SetGraphicsRootDescriptorTable(3, self.color_lut.srv_gpu);
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            // The composite VS builds the fullscreen triangle from SV_VertexID.
            cmd.IASetVertexBuffers(0, None);
            cmd.IASetIndexBuffer(None);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
        self.inc_draw_calls(1);
    }

    fn begin_text(&self, cmd: &Self::Rec, args: &Self::Args) -> bool {
        let Some(text_pso) = &self.text_pso else {
            return false;
        };
        if self.descriptors.text_atlas_srv_gpus.is_empty() {
            return false;
        }
        let text_push = TextPush {
            win_width: args.width as f32,
            win_height: args.height as f32,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        unsafe {
            cmd.SetPipelineState(text_pso);
            cmd.SetGraphicsRootSignature(&self.text_root_sig);
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);
            cmd.SetGraphicsRoot32BitConstants(
                0,
                4,
                &text_push as *const TextPush as *const std::ffi::c_void,
                0,
            );
            cmd.SetGraphicsRootDescriptorTable(2, self.descriptors.text_sampler_gpu);
        }
        true
    }

    fn text_draw(
        &self,
        cmd: &Self::Rec,
        args: &Self::Args,
        call: &TextDrawCall,
    ) -> Result<(), String> {
        if call.vertices.is_empty() || self.descriptors.text_atlas_srv_gpus.is_empty() {
            return Ok(());
        }

        // Scissor a clipped (scrollable-panel) call to its band, restoring the
        // full-window scissor for an unclipped call so chrome is never cropped.
        // The clip rect is already in attachment pixels (see `clip_rect_to_scissor`).
        let scissor = match call.clip_rect {
            Some(clip) => {
                match crate::gfx::fullscreen::clip_rect_to_scissor(clip, args.width, args.height) {
                    // Row scrolled fully out of its band: nothing to draw.
                    None => return Ok(()),
                    Some((x, y, w, h)) => RECT {
                        left: x,
                        top: y,
                        right: x + w as i32,
                        bottom: y + h as i32,
                    },
                }
            }
            None => RECT {
                left: 0,
                top: 0,
                right: args.width as i32,
                bottom: args.height as i32,
            },
        };
        unsafe { cmd.RSSetScissorRects(&[scissor]) };

        let atlas_idx = call
            .atlas_slot
            .min(self.descriptors.text_atlas_srv_gpus.len() - 1);

        // Append this label's vertex + index geometry into the frame slot's
        // persistent upload buffer (sized up front by `reserve` in
        // `encode_composite_and_text`) and bind sub-views into it.
        let vert_bytes = unsafe {
            std::slice::from_raw_parts(
                call.vertices.as_ptr() as *const u8,
                call.vertices.len() * std::mem::size_of::<TextVertex>(),
            )
        };
        let idx_bytes = unsafe {
            std::slice::from_raw_parts(
                call.indices.as_ptr() as *const u8,
                call.indices.len() * std::mem::size_of::<u16>(),
            )
        };

        let vert_va = self.text_upload.push(args.frame_idx, vert_bytes)?;
        let idx_va = self.text_upload.push(args.frame_idx, idx_bytes)?;

        let vbv = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: vert_va,
            SizeInBytes: vert_bytes.len() as u32,
            StrideInBytes: std::mem::size_of::<TextVertex>() as u32,
        };
        let ibv = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: idx_va,
            SizeInBytes: idx_bytes.len() as u32,
            Format: DXGI_FORMAT_R16_UINT,
        };

        unsafe {
            cmd.SetGraphicsRootDescriptorTable(1, self.descriptors.text_atlas_srv_gpus[atlas_idx]);
            cmd.IASetVertexBuffers(0, Some(&[vbv]));
            cmd.IASetIndexBuffer(Some(&ibv));
            cmd.DrawIndexedInstanced(call.indices.len() as u32, 1, 0, 0, 0);
        }
        self.inc_draw_calls(1);
        Ok(())
    }

    fn end_composite(&self, cmd: &Self::Rec, args: &Self::Args) {
        let to_present = transition_barrier(
            &args.back_buffer,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PRESENT,
        );
        unsafe { cmd.ResourceBarrier(&[to_present]) };
    }
}

impl DxContext {
    // Encode the composite + text passes into `cmd` via the shared
    // `gfx::fullscreen` driver. Transitions the back buffer to `RENDER_TARGET`
    // for the draws and back to `PRESENT` on exit; the HDR target is expected to
    // already be in `PIXEL_SHADER_RESOURCE` (the main pass leaves it that way).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn encode_composite_and_text(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        back_buffer: &ID3D12Resource,
        back_buffer_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        text_calls: &[TextDrawCall],
        scene_srv: D3D12_GPU_DESCRIPTOR_HANDLE,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        // Reset this slot's text-upload cursor and ensure its buffer holds the
        // whole frame's text up front, so each `text_draw` only appends (and
        // never reallocates out from under an already-bound sub-view). The frame
        // fence in `draw_frame` has already confirmed the GPU is done with this
        // slot, so resetting / growing it now is race-free.
        let text_bytes = super::text_upload::text_calls_byte_size(text_calls);
        self.text_upload
            .reserve(&self.device, frame_idx, text_bytes)?;

        let args = DxCompositeArgs {
            back_buffer: back_buffer.clone(),
            back_buffer_rtv,
            scene_srv,
            width,
            height,
            frame_idx,
        };
        crate::gfx::fullscreen::encode_composite_chain(self, cmd, &args, text_calls)
    }
}
