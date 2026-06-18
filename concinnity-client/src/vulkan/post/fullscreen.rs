// src/vulkan/post/fullscreen.rs
//
// Shared lifecycle for single-draw fullscreen post passes (SSR resolve, TAA
// resolve, ...): the render-pass bracket + full-resolution viewport / scissor
// every such pass repeats. The per-pass encoders (ssr.rs / taa.rs) implement
// `gfx::fullscreen::FullscreenPass` and call these from their begin/end so the
// bracket lives once. See gfx/fullscreen.rs for the cross-backend driver.

use ash::vk;

use crate::vulkan::context::VkContext;

impl VkContext {
    // Begin a fullscreen render pass: begin `render_pass` over `framebuffer` at the
    // full render extent and set the matching viewport / scissor. Paired with
    // `end_fullscreen_pass`.
    pub(in crate::vulkan) fn begin_fullscreen_pass(
        &self,
        cmd: vk::CommandBuffer,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
    ) {
        self.begin_fullscreen_pass_sized(cmd, render_pass, framebuffer, self.render_extent);
    }

    // As `begin_fullscreen_pass`, but with an explicit target extent for a pass
    // whose framebuffer is not the full render resolution -- the SSGI gather
    // writes a `gi_scale`-reduced gi target, so its render area + viewport must
    // match that smaller framebuffer (the composite then bilateral-upsamples it).
    // The full-res callers go through `begin_fullscreen_pass` above.
    pub(in crate::vulkan) fn begin_fullscreen_pass_sized(
        &self,
        cmd: vk::CommandBuffer,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        extent: vk::Extent2D,
    ) {
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D::default().extent(extent));
        let vp = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        unsafe {
            self.device
                .cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            self.device
                .cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            self.device
                .cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
        }
    }

    // End a fullscreen render pass. Paired with `begin_fullscreen_pass`.
    pub(in crate::vulkan) fn end_fullscreen_pass(&self, cmd: vk::CommandBuffer) {
        unsafe { self.device.cmd_end_render_pass(cmd) };
    }
}
