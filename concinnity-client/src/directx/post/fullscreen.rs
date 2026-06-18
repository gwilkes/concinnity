// src/directx/post/fullscreen.rs
//
// Shared lifecycle for single-draw fullscreen post passes (SSR resolve, TAA
// resolve, ...): the PIXEL_SHADER_RESOURCE <-> RENDER_TARGET barrier bracket +
// the render-target bind + full-resolution viewport / scissor that every such
// pass repeats. The per-pass encoders (ssr.rs / taa.rs) implement
// `gfx::fullscreen::FullscreenPass` and call these from their begin/end so the
// bracket lives once. See gfx/fullscreen.rs for the cross-backend driver.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::context::DxContext;
use crate::directx::texture::transition_barrier;

impl DxContext {
    // Begin a fullscreen render-target pass: transition `output` from its sampled
    // resting state into RENDER_TARGET, bind it as the sole RTV, set the
    // viewport / scissor to `output`'s own dimensions, and bind the SRV heap the
    // pass's root tables index. Paired with `end_fullscreen_rt`.
    //
    // The viewport tracks the target size (not a fixed render resolution) so a
    // pass writing a reduced-resolution target -- the SSGI gather's `gi_scale`
    // gather, which the composite then bilateral-upsamples -- rasterizes the full
    // fullscreen triangle across its smaller target. Every other caller (SSR /
    // TAA resolve, the SSGI composite) writes a full-resolution target, so their
    // viewport is unchanged.
    pub(in crate::directx) fn begin_fullscreen_rt(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        output: &ID3D12Resource,
        output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        let to_rt = transition_barrier(
            output,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        unsafe { cmd.ResourceBarrier(&[to_rt]) };

        let desc = unsafe { output.GetDesc() };
        let w = desc.Width as u32;
        let h = desc.Height;
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&output_rtv), false, None);
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: w as f32,
                Height: h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: w as i32,
                bottom: h as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
        }
    }

    // End a fullscreen render-target pass: transition `output` back to its sampled
    // resting state for the downstream consumer. Paired with `begin_fullscreen_rt`.
    pub(in crate::directx) fn end_fullscreen_rt(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        output: &ID3D12Resource,
    ) {
        let to_psr = transition_barrier(
            output,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[to_psr]) };
    }
}
