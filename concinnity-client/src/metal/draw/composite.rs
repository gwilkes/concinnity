// src/metal/draw/composite.rs
//
// Composite (post-process) pass + text overlay. The post-process pipeline
// reads `scene_color`, the bloom mip-0 target, and the 3D colour-grading LUT,
// then writes ACES tonemap + gamma + FXAA into the drawable. Text is drawn
// after in the same render pass so it sits on top of the tonemapped image in
// display-referred LDR space.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice as _;
use objc2_metal::{
    MTLCommandBuffer as _, MTLIndexType, MTLLoadAction, MTLPrimitiveType,
    MTLRenderCommandEncoder as _, MTLResourceOptions, MTLScissorRect, MTLStoreAction, MTLTexture,
};

use crate::gfx::render_types::{TextDrawCall, TextVertex};
use crate::metal::context::MtlContext;
use crate::metal::scoped_encoder::ScopedEncoder;

impl MtlContext {
    // pub(in crate::metal) so the render-graph executor in
    // `metal/graph_exec.rs` can dispatch to this from outside `metal/draw/`.
    pub(in crate::metal) fn encode_composite_and_text(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        scene_color: &Retained<ProtocolObject<dyn MTLTexture>>,
        text_calls: &[TextDrawCall],
    ) -> Result<u32, String> {
        let composite_pass_desc = self
            .mtk_view
            .currentRenderPassDescriptor()
            .ok_or("no current render pass descriptor")?;
        unsafe {
            let ca = composite_pass_desc
                .colorAttachments()
                .objectAtIndexedSubscript(0);
            ca.setLoadAction(MTLLoadAction::DontCare);
            ca.setStoreAction(MTLStoreAction::Store);
        }

        if let Some(t) = &self.pass_timing {
            t.attach_render(
                &composite_pass_desc,
                super::super::pass_timing::PassId::Composite,
            );
        }
        // ScopedEncoder guarantees the pass ends even if the text-overlay block
        // below hits an `?` (empty glyph slice / buffer-alloc failure) mid-encode.
        let post_encoder = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&composite_pass_desc)
                .ok_or("failed to get post-process render encoder")?,
            "composite",
        );

        post_encoder.setRenderPipelineState(&self.post_pipeline_state);
        unsafe {
            post_encoder.setFragmentTexture_atIndex(Some(scene_color.as_ref()), 0);
            // Bloom mip 0 at texture(1). Always bound so the binding resolves;
            // the shader skips the sample when bloom_intensity == 0.
            post_encoder.setFragmentTexture_atIndex(Some(self.bloom_targets.mips[0].as_ref()), 1);
            // 3D colour-grading LUT at texture(2). Always bound -- an identity
            // LUT stands in when the world declares no ColorLut.
            post_encoder.setFragmentTexture_atIndex(Some(self.color_lut.as_ref()), 2);
            post_encoder.setFragmentSamplerState_atIndex(Some(&self.post_sampler), 0);
            // Post-process tunables (bloom intensity) at buffer(0).
            post_encoder.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&self.post_process).cast(),
                std::mem::size_of::<crate::gfx::render_types::PostProcessParams>(),
                0,
            );
            // Fullscreen triangle: 3 vertices, no vertex buffer (post_vertex_main
            // synthesises position + UV from [[vertex_id]]).
            post_encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        let mut draw_calls: u32 = 1;

        // Text overlay: rendered in the same composite pass so it sits on
        // top of the tonemapped image in display-referred LDR space.
        if let Some(text_ps) = self.text_pipeline_state.clone()
            && !text_calls.is_empty()
            && !self.text_atlas_textures.is_empty()
        {
            let logical = self.mtk_view.bounds().size;
            let win_w = logical.width as f32;
            let win_h = logical.height as f32;
            let text_uniforms = crate::gfx::render_types::TextUniforms {
                win_width: win_w,
                win_height: win_h,
                _pad: [0.0; 2],
            };

            // The text vertices are in logical points (mapped to NDC by the
            // shader's divide by win_width/height); the scissor is in framebuffer
            // pixels. Recover the drawable's pixel size from the composite color
            // attachment so a per-call clip rect scales from points to pixels.
            let (fb_w, fb_h) = unsafe {
                match composite_pass_desc
                    .colorAttachments()
                    .objectAtIndexedSubscript(0)
                    .texture()
                {
                    Some(t) => (t.width(), t.height()),
                    None => (win_w.max(0.0) as usize, win_h.max(0.0) as usize),
                }
            };
            let scale_x = fb_w as f32 / win_w.max(1.0);
            let scale_y = fb_h as f32 / win_h.max(1.0);

            post_encoder.setRenderPipelineState(&text_ps);

            for call in text_calls {
                if call.vertices.is_empty() {
                    continue;
                }
                // Clip this call to its band (scrollable panel content) or reset
                // to the full drawable (chrome / HUD). A clip rect that scales to
                // an empty rectangle means the element scrolled fully out of its
                // band: skip the draw entirely.
                match call.clip_rect {
                    Some([cx, cy, cw, ch]) => {
                        let x0 = (cx * scale_x).floor().clamp(0.0, fb_w as f32) as usize;
                        let y0 = (cy * scale_y).floor().clamp(0.0, fb_h as f32) as usize;
                        let x1 = ((cx + cw) * scale_x).ceil().clamp(0.0, fb_w as f32) as usize;
                        let y1 = ((cy + ch) * scale_y).ceil().clamp(0.0, fb_h as f32) as usize;
                        if x1 <= x0 || y1 <= y0 {
                            continue;
                        }
                        post_encoder.setScissorRect(MTLScissorRect {
                            x: x0,
                            y: y0,
                            width: x1 - x0,
                            height: y1 - y0,
                        });
                    }
                    None => {
                        post_encoder.setScissorRect(MTLScissorRect {
                            x: 0,
                            y: 0,
                            width: fb_w,
                            height: fb_h,
                        });
                    }
                }
                let atlas_idx = call.atlas_slot.min(self.text_atlas_textures.len() - 1);
                unsafe {
                    post_encoder.setFragmentTexture_atIndex(
                        Some(self.text_atlas_textures[atlas_idx].as_ref()),
                        0,
                    );
                    post_encoder.setFragmentSamplerState_atIndex(Some(&self.text_sampler), 0);
                }

                unsafe {
                    post_encoder.setVertexBytes_length_atIndex(
                        std::ptr::NonNull::from(&text_uniforms).cast(),
                        std::mem::size_of::<crate::gfx::render_types::TextUniforms>(),
                        0,
                    );
                }

                let vert_bytes_slice = unsafe {
                    std::slice::from_raw_parts(
                        call.vertices.as_ptr() as *const u8,
                        call.vertices.len() * std::mem::size_of::<TextVertex>(),
                    )
                };
                let idx_bytes_slice = unsafe {
                    std::slice::from_raw_parts(
                        call.indices.as_ptr() as *const u8,
                        call.indices.len() * std::mem::size_of::<u16>(),
                    )
                };
                let text_vbuf = unsafe {
                    let ptr = std::ptr::NonNull::new(vert_bytes_slice.as_ptr() as *mut _)
                        .ok_or("text vertex slice is empty")?;
                    self.device
                        .newBufferWithBytes_length_options(
                            ptr,
                            vert_bytes_slice.len(),
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or("failed to create text vertex buffer")?
                };
                let text_ibuf = unsafe {
                    let ptr = std::ptr::NonNull::new(idx_bytes_slice.as_ptr() as *mut _)
                        .ok_or("text index slice is empty")?;
                    self.device
                        .newBufferWithBytes_length_options(
                            ptr,
                            idx_bytes_slice.len(),
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or("failed to create text index buffer")?
                };

                unsafe {
                    post_encoder.setVertexBuffer_offset_atIndex(Some(&text_vbuf), 0, 1);
                    post_encoder
                        .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                            MTLPrimitiveType::Triangle,
                            call.indices.len(),
                            MTLIndexType::UInt16,
                            &text_ibuf,
                            0,
                        );
                }
                draw_calls += 1;
            }
        }

        Ok(draw_calls)
    }
}
