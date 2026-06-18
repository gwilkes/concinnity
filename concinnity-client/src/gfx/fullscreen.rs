// src/gfx/fullscreen.rs
//
// Backend-agnostic fullscreen-pass encoder seam, the first pilot of a hardware
// abstraction layer over the three render backends. The bloom
// prefilter -> downsample -> upsample chain is structurally identical on every
// backend, so its orchestration lives here once and each backend implements
// `BloomEncoder` to bind + draw one sub-pass in its own command stream.
//
// Two associated types absorb the only real divergence, so the trait names no
// backend types: `Rec` hides the per-backend command recorder, and `Args`
// carries the per-invocation binding context (DirectX passes the scene-colour
// SRV its prefilter samples; Vulkan threads the frame-in-flight index that
// selects its per-frame framebuffers + descriptor sets). Everything else each
// impl reads from `&self`, consistent with the read-only parallel-encode
// contract.
//
// Implemented by DirectX + Vulkan. Metal keeps its hand-rolled `encode_bloom`,
// already factored through its own `fullscreen_pass`, so this seam is unused
// (dead code) on a Metal build.

use crate::gfx::render_types::TextDrawCall;

// Convert a `TextDrawCall.clip_rect` (a window-space rectangle `[x, y, w, h]`,
// already mapped through the overlay transform by `gfx::text::band_to_window`)
// into an integer scissor rect `(x, y, w, h)` clamped to the attachment's pixel
// bounds. Returns `None` when the clamped rectangle is empty (a row scrolled
// fully out of its band), so the caller skips the draw entirely.
//
// On DirectX / Vulkan the overlay's logical size is the swapchain pixel size
// (see each backend's `logical_size`), so a clip band's window-space rect is
// already in attachment pixels and needs no DPI scaling -- only the clamp, which
// keeps a partially-scrolled row's rect inside the target. (Metal scales
// logical points to its larger drawable in its own composite encoder.)
pub(crate) fn clip_rect_to_scissor(
    clip: [f32; 4],
    attach_w: u32,
    attach_h: u32,
) -> Option<(i32, i32, u32, u32)> {
    let aw = attach_w as f32;
    let ah = attach_h as f32;
    let x0 = clip[0].floor().clamp(0.0, aw);
    let y0 = clip[1].floor().clamp(0.0, ah);
    let x1 = (clip[0] + clip[2]).ceil().clamp(0.0, aw);
    let y1 = (clip[1] + clip[3]).ceil().clamp(0.0, ah);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some((x0 as i32, y0 as i32, (x1 - x0) as u32, (y1 - y0) as u32))
}

pub(crate) trait BloomEncoder {
    // Per-backend command recorder (DX `ID3D12GraphicsCommandList`, VK `vk::CommandBuffer`).
    type Rec;
    // Per-invocation binding context (DX scene-colour SRV handle, VK frame index).
    type Args;

    // Number of bloom mips; zero means bloom is off and the driver no-ops.
    fn bloom_mip_count(&self) -> usize;
    // One-time per-encode preamble (DX root signature / heap / IA state; VK no-op).
    fn begin_bloom(&self, rec: &Self::Rec, args: &Self::Args);
    // Prefilter: scene colour -> mip 0 (soft-knee threshold + Karis average).
    fn bloom_prefilter(&self, rec: &Self::Rec, args: &Self::Args);
    // Downsample: mip `dst - 1` -> mip `dst`.
    fn bloom_downsample(&self, rec: &Self::Rec, args: &Self::Args, dst: usize);
    // Upsample: mip `dst + 1` -> mip `dst`, additively blended.
    fn bloom_upsample(&self, rec: &Self::Rec, args: &Self::Args, dst: usize);
}

// The bloom chain orchestration, previously hand-duplicated in each backend's
// `encode_bloom`. On return, mip 0 holds the accumulated glow the composite pass
// samples.
pub(crate) fn encode_bloom_chain<E: BloomEncoder>(enc: &E, rec: &E::Rec, args: E::Args) {
    let n = enc.bloom_mip_count();
    if n == 0 {
        return;
    }
    enc.begin_bloom(rec, &args);
    // Prefilter: scene -> mip 0.
    enc.bloom_prefilter(rec, &args);
    // Downsample chain: mip i-1 -> mip i.
    for dst in 1..n {
        enc.bloom_downsample(rec, &args, dst);
    }
    // Upsample chain: mip i+1 -> mip i, walking back down to mip 0.
    for dst in (0..n - 1).rev() {
        enc.bloom_upsample(rec, &args, dst);
    }
}

// The composite pass: tonemap (+ optional LUT grade) the post-stack scene onto
// the swapchain image, then layer the text overlay on top in the same pass. Its
// begin -> composite-draw -> text-loop -> end shape is identical on every
// backend; the swapchain target lifecycle, the descriptor binding, and the
// transient text-buffer uploads stay backend-specific behind the trait. `Args`
// carries the per-frame binding context each backend needs (DX: the swapchain
// back-buffer + its RTV, the scene SRV, the window size, the frame slot; VK: the
// acquired image index + the frame slot).
pub(crate) trait CompositeEncoder {
    // Per-backend command recorder (DX `ID3D12GraphicsCommandList`, VK `vk::CommandBuffer`).
    type Rec;
    // Per-invocation binding context (see the trait doc).
    type Args;

    // Begin the pass: target the swapchain image (DX transitions it to
    // RENDER_TARGET + binds the RTV; VK begins the composite render pass) and set
    // the full-window viewport / scissor.
    fn begin_composite(&self, rec: &Self::Rec, args: &Self::Args);
    // The fullscreen tonemap draw: bind the composite pipeline + its inputs
    // (scene, bloom, LUT) + push constants, draw the fullscreen triangle.
    fn composite_draw(&self, rec: &Self::Rec, args: &Self::Args);
    // Bind the text pipeline + any one-time text state. Returns false when text
    // is inert (no pipeline or no atlases), so the driver skips the call loop.
    fn begin_text(&self, rec: &Self::Rec, args: &Self::Args) -> bool;
    // Encode one text draw call: upload its vertex/index geometry, bind the
    // atlas, and draw. How the geometry is uploaded is backend-specific (DX
    // appends into a persistent per-frame upload buffer; VK allocates transient
    // buffers and stashes them for deferred destruction).
    fn text_draw(
        &self,
        rec: &Self::Rec,
        args: &Self::Args,
        call: &TextDrawCall,
    ) -> Result<(), String>;
    // End the pass: DX transitions the back-buffer back to PRESENT; VK ends the
    // render pass.
    fn end_composite(&self, rec: &Self::Rec, args: &Self::Args);
}

// The composite + text orchestration, previously hand-duplicated in each
// backend's `encode_composite_and_text`. An error mid-text propagates without
// closing the pass, matching the prior DX/VK behaviour (the frame fails either
// way: the target is just left mis-stated). This is unused on Metal, where a
// render encoder must be `endEncoding`-ed before the command buffer commits:
// skipping `end_composite` on a text error would crash at commit, so Metal's
// `encode_composite_and_text` ends the encoder on any `?` with a `ScopedEncoder`
// RAII guard instead.
pub(crate) fn encode_composite_chain<E: CompositeEncoder>(
    enc: &E,
    rec: &E::Rec,
    args: &E::Args,
    text_calls: &[TextDrawCall],
) -> Result<(), String> {
    enc.begin_composite(rec, args);
    enc.composite_draw(rec, args);
    if !text_calls.is_empty() && enc.begin_text(rec, args) {
        for call in text_calls {
            enc.text_draw(rec, args, call)?;
        }
    }
    enc.end_composite(rec, args);
    Ok(())
}

// A single-draw fullscreen post pass (SSR resolve, TAA resolve, ...): target a
// render target, bind a pipeline + inputs, draw one fullscreen triangle, restore.
// Unlike the bloom + composite chains (whose drivers hold a mip / text loop), a
// fullscreen pass has no loop, so the driver is a fixed begin -> draw -> end. The
// value is the shared per-backend lifecycle factored behind begin/end (DX: the
// PSR<->RENDER_TARGET barrier bracket + render-target bind; VK: the render-pass
// bracket), reused across every such pass instead of re-pasted per pass.
//
// The inert-pass guard lives at each backend's call site: it resolves the pass's
// resources (returning early if a required one is absent) BEFORE constructing the
// encoder, so the driver always runs all three steps over a fully-resolved pass
// and can never leave a render pass / barrier half-open. There is no `Args`: each
// backend's encoder is a small struct holding the already-resolved references +
// per-call scalars, so the trait names no backend types (like `BloomEncoder`).
//
// Implemented by DirectX + Vulkan. Metal keeps its own `fullscreen_pass` helper,
// which already factors this begin/draw/end skeleton, so this seam is unused
// (dead code) on a Metal build.
pub(crate) trait FullscreenPass {
    // Per-backend command recorder (DX `ID3D12GraphicsCommandList`, VK `vk::CommandBuffer`).
    type Rec;

    // Begin: bind the target render target + set the full-resolution viewport /
    // scissor. DX transitions the target PIXEL_SHADER_RESOURCE -> RENDER_TARGET,
    // binds its RTV + the SRV heap; VK begins the pass's render pass.
    fn begin(&self, rec: &Self::Rec);
    // Bind the pipeline + inputs + per-frame params and draw the fullscreen
    // triangle (3 vertices; the vertex shader builds the triangle from the id).
    fn draw(&self, rec: &Self::Rec);
    // End: DX transitions the target back to PIXEL_SHADER_RESOURCE; VK ends the
    // render pass.
    fn end(&self, rec: &Self::Rec);
}

// The fullscreen-pass orchestration. Trivial by design (a single draw), but kept
// as a driver so every fullscreen post pass shares one begin -> draw -> end
// contract across backends, matching `encode_bloom_chain` / `encode_composite_chain`.
pub(crate) fn encode_fullscreen<E: FullscreenPass>(enc: &E, rec: &E::Rec) {
    enc.begin(rec);
    enc.draw(rec);
    enc.end(rec);
}

#[cfg(test)]
mod tests {
    use super::clip_rect_to_scissor;

    #[test]
    fn clip_inside_attachment_passes_through() {
        // A band fully inside the attachment maps 1:1 (no scaling on DX/VK).
        assert_eq!(
            clip_rect_to_scissor([100.0, 50.0, 300.0, 200.0], 1280, 720),
            Some((100, 50, 300, 200))
        );
    }

    #[test]
    fn clip_is_clamped_to_attachment_bounds() {
        // A band hanging off the right / bottom edge is clamped to the target.
        assert_eq!(
            clip_rect_to_scissor([1200.0, 700.0, 400.0, 400.0], 1280, 720),
            Some((1200, 700, 80, 20))
        );
        // A negative origin is clamped to zero, shrinking the width/height.
        assert_eq!(
            clip_rect_to_scissor([-40.0, -10.0, 100.0, 100.0], 1280, 720),
            Some((0, 0, 60, 90))
        );
    }

    #[test]
    fn fully_offscreen_clip_is_skipped() {
        // A band entirely past the attachment yields no scissor (skip the draw).
        assert_eq!(
            clip_rect_to_scissor([2000.0, 50.0, 100.0, 100.0], 1280, 720),
            None
        );
        // A zero-area band is also skipped.
        assert_eq!(
            clip_rect_to_scissor([10.0, 10.0, 0.0, 50.0], 1280, 720),
            None
        );
    }
}
