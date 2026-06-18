// src/vulkan/render_pass.rs
//
// Vulkan render-pass construction for the main, shadow, composite, and
// bloom passes.
use ash::{Device, vk};

pub(super) fn create_main_render_pass(
    device: &Device,
    format: vk::Format,
    msaa: vk::SampleCountFlags,
) -> Result<vk::RenderPass, String> {
    let multisampled = msaa != vk::SampleCountFlags::TYPE_1;

    // When multisampled the resolve attachment ends shader-readable; the MSAA
    // colour image is transient. When single-sampled, attachment [0] is itself
    // the resolve image, so it ends shader-readable.
    let color_final_layout = if multisampled {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    } else {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    };

    let mut attachments = vec![
        // [0] color (MSAA HDR, or the single-sample HDR resolve image)
        vk::AttachmentDescription::default()
            .format(format)
            .samples(msaa)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(if multisampled {
                vk::AttachmentStoreOp::DONT_CARE
            } else {
                vk::AttachmentStoreOp::STORE
            })
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(color_final_layout),
        // [1] depth: STORE because the projected-decal and volumetric-fog
        // passes sample this attachment after the main pass ends. Without
        // STORE the contents are spec-undefined post-pass: tilers may
        // discard them and the depth-read in fog returns garbage values
        // that bias toward the `depth >= 1.0` "no scene hit" branch,
        // producing per-pixel bright sparkles after bloom.
        vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(msaa)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];
    if multisampled {
        // [2] resolve target (single-sample HDR resolve image)
        attachments.push(
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        );
    }

    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let resolve_ref = vk::AttachmentReference::default()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let mut subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref))
        .depth_stencil_attachment(&depth_ref);
    if multisampled {
        subpass = subpass.resolve_attachments(std::slice::from_ref(&resolve_ref));
    }

    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );

    let rp_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dependency));

    unsafe { device.create_render_pass(&rp_info, None) }
        .map_err(|e| format!("main render pass: {e}"))
}

// Main-pass render pass for two-pass occlusion culling. Two variants share
// this builder, selected by `load`:
//
//   * `load = false` (phase 1, `Main`): same as `create_main_render_pass` but
//     the MSAA colour is STORE'd (not DONT_CARE) so the phase-2 pass can load
//     the samples back and composite onto them. Colour ends
//     COLOR_ATTACHMENT_OPTIMAL so `Main2` loads it directly.
//   * `load = true` (phase 2, `Main2`): loads (does not clear) the phase-1
//     colour + depth, redraws the disoccluded statics, and resolves the
//     combined scene into `hdr_resolve` for the post stack.
//
// Both are render-pass-compatible with the existing main framebuffers (same
// attachment count / formats / sample counts), so no extra framebuffers are
// needed. `Main` (phase 1) still resolves into `hdr_resolve`; that resolve is
// overwritten by `Main2`'s and nothing reads it in between (HizBuild / Cull2
// are compute), so the combined result is what the post stack sees.
pub(super) fn create_main_render_pass_two_pass(
    device: &Device,
    format: vk::Format,
    msaa: vk::SampleCountFlags,
    load: bool,
) -> Result<vk::RenderPass, String> {
    let multisampled = msaa != vk::SampleCountFlags::TYPE_1;

    // Phase 1 leaves the colour in COLOR_ATTACHMENT_OPTIMAL for phase 2 to
    // load. Phase 2 ends shader-readable when single-sampled (the colour image
    // is itself the resolve target the post stack samples); when multisampled
    // the resolve attachment carries that role and the MSAA colour stays
    // transient in COLOR_ATTACHMENT_OPTIMAL.
    let color_final_layout = if multisampled || !load {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    } else {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    };
    let color_load_op = if load {
        vk::AttachmentLoadOp::LOAD
    } else {
        vk::AttachmentLoadOp::CLEAR
    };
    let color_initial_layout = if load {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    } else {
        vk::ImageLayout::UNDEFINED
    };
    let depth_load_op = if load {
        vk::AttachmentLoadOp::LOAD
    } else {
        vk::AttachmentLoadOp::CLEAR
    };
    let depth_initial_layout = if load {
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    } else {
        vk::ImageLayout::UNDEFINED
    };

    let mut attachments = vec![
        // [0] colour (MSAA HDR, or the single-sample HDR resolve image). STORE
        // unconditionally: phase 1 must keep the MSAA samples for phase 2's
        // load, and the single-sample image is the post-stack input.
        vk::AttachmentDescription::default()
            .format(format)
            .samples(msaa)
            .load_op(color_load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(color_initial_layout)
            .final_layout(color_final_layout),
        // [1] depth: STORE (HizBuild reduces it, the post decals/fog sample it,
        // and phase 2 depth-tests against it).
        vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(msaa)
            .load_op(depth_load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(depth_initial_layout)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];
    if multisampled {
        // [2] resolve target (single-sample HDR resolve image). Always
        // overwritten by the resolve, so its initial layout is irrelevant.
        attachments.push(
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        );
    }

    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let resolve_ref = vk::AttachmentReference::default()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let mut subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref))
        .depth_stencil_attachment(&depth_ref);
    if multisampled {
        subpass = subpass.resolve_attachments(std::slice::from_ref(&resolve_ref));
    }

    // Use the EXACT same subpass dependency as `create_main_render_pass`. Both
    // two-pass variants share the main framebuffers + the bindless pipeline,
    // which were created against `main_render_pass`; render-pass compatibility
    // (validated on `vkCmdBeginRenderPass` / `vkCmdDraw`) treats differing
    // dependencies as incompatible, so the dependency must match. Phase 2's
    // LOAD of the phase-1 colour + depth is instead ordered by an explicit
    // `vkCmdPipelineBarrier` in `encode_main_pass_phase2` (this backend owns
    // its cross-pass sync inline anyway).
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );

    let rp_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dependency));

    unsafe { device.create_render_pass(&rp_info, None) }
        .map_err(|e| format!("two-pass main render pass: {e}"))
}

pub(super) fn create_shadow_render_pass(device: &Device) -> Result<vk::RenderPass, String> {
    let attachment = vk::AttachmentDescription::default()
        .format(vk::Format::D32_SFLOAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let depth_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .depth_stencil_attachment(&depth_ref);

    let rp_info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass));

    unsafe { device.create_render_pass(&rp_info, None) }
        .map_err(|e| format!("shadow render pass: {e}"))
}

// Composite render pass. A single subpass renders the fullscreen tonemap +
// FXAA triangle (and the text overlay) into the swapchain backbuffer.
pub(super) fn create_composite_render_pass(
    device: &Device,
    swapchain_format: vk::Format,
) -> Result<vk::RenderPass, String> {
    // The fullscreen triangle overwrites every pixel, so the backbuffer is
    // not cleared or loaded.
    let attachment = vk::AttachmentDescription::default()
        .format(swapchain_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));

    // External dependency: the composite fragment shader must wait for the
    // main pass to finish writing (and resolving into) the HDR image, and the
    // backbuffer write must wait for the acquire semaphore's stage.
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ);

    let rp_info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dependency));

    unsafe { device.create_render_pass(&rp_info, None) }
        .map_err(|e| format!("composite render pass: {e}"))
}

// Create the off-screen HDR attachment set, `count` slots deep (one per
// frame-in-flight). Returns `(msaa_color, depth, hdr_resolve)`; `msaa_color`
// is empty when MSAA is disabled, in which case the main pass renders
// straight into the resolve image.
// A bloom-chain render pass: one single-sample HDR colour attachment, no
// depth. With `load` set the attachment is loaded (the additive upsample
// blends onto existing content) and its initial layout is the
// shader-readable layout the prior write pass left it in; otherwise the
// attachment is discarded on load. Either way it ends `SHADER_READ_ONLY`.
pub(super) fn create_bloom_render_pass(
    device: &Device,
    format: vk::Format,
    load: bool,
) -> Result<vk::RenderPass, String> {
    let attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(if load {
            vk::AttachmentLoadOp::LOAD
        } else {
            vk::AttachmentLoadOp::DONT_CARE
        })
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(if load {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::UNDEFINED
        })
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));

    // Order this pass after the prior bloom pass: its fragment shader samples
    // the mip the prior pass wrote, and (in the LOAD case) its own attachment
    // was also written by an earlier pass.
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::SHADER_READ,
        );

    let rp_info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dependency));

    unsafe { device.create_render_pass(&rp_info, None) }
        .map_err(|e| format!("bloom render pass: {e}"))
}
