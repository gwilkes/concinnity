// src/metal/planar.rs
//
// Planar reflection for flat reflectors (water surfaces + glass panes). Each
// frame, when ray tracing is off, the scene is rendered a second time from the
// camera reflected across each reflector plane (mirror view + oblique near-plane
// clip so geometry behind the plane never leaks in) into a dedicated target; the
// reflective surface then samples that target projectively for a sharp,
// scene-correct reflection instead of the blurry box-projected probe cube.
//
// One mirror render per DISTINCT plane. Water is a single horizontal plane; glass
// panes can be vertical or angled, and a world can hold several at different
// planes. Each is a full scene re-render, so the number of mirror renders is
// budgeted (`MAX_PLANAR_PLANES`): near-coplanar reflectors share one render (one
// wall of windows = one plane), and reflectors past the budget fall back to the
// probe cube (logged at init, see `metal/init`). The plane -> slot grouping is
// the pure, unit-tested `gfx::planar_reflection::assign_planar_slots`.
//
// Each plane gets a DEDICATED mirror cull against its reflected-camera frustum,
// so geometry visible only in the reflection (behind or beside the main camera,
// outside its frustum) is captured, not just the main camera's visible set. The
// reflected view-proj carries the oblique near-plane clip, so the extracted
// frustum also rejects geometry behind the reflector. On the bindless path the GPU
// cull kernel re-runs into that plane's own mirror ICB (`encode_mirror_cull`),
// which the face render executes; on the non-bindless path the CPU cull re-runs
// against the reflected frustum (`reflected_visible_set`) into a per-frame scratch
// list the legacy face render draws. The matrices + frustum come from the pure,
// unit-tested `gfx::planar_reflection` + `gfx::frustum`.

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice, MTLPixelFormat, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};

use super::context::MtlContext;

// Clip the reflection a hair toward the kept (camera) side of the plane so
// geometry exactly on the surface is not lost to near-plane precision.
const PLANAR_CLIP_BIAS: f32 = 0.02;

// The engine capacity ceiling for distinct reflection planes (water + glass): the
// count the mirror-target set + mirror ICB slots below are sized to. Single-sourced
// from `gfx::planar_reflection` so the three backends stay in lockstep by
// construction. The per-frame budget passed to `assign_planar_slots` at init can be
// lower under a quality preset / GPU tier (fewer full-res mirror passes on weaker
// hardware), never higher; reflectors past the active budget fall back to the
// box-projected probe cube.
pub(in crate::metal) const MAX_PLANAR_PLANES: usize =
    crate::gfx::planar_reflection::MAX_PLANAR_PLANES;

// Per-frame planar reflection render targets for one plane, sized to the render
// resolution. MSAA colour + depth (rendered into, then resolved) plus a
// single-sample resolve the reflective shader samples.
pub(in crate::metal) struct PlanarReflectionTargets {
    pub(in crate::metal) msaa_color: Retained<ProtocolObject<dyn MTLTexture>>,
    pub(in crate::metal) msaa_depth: Retained<ProtocolObject<dyn MTLTexture>>,
    pub(in crate::metal) resolve: Retained<ProtocolObject<dyn MTLTexture>>,
}

// The set of distinct reflection planes for the world, each with its own render
// targets. `planes[i]` is the world-space plane (`[nx, ny, nz, d]`, n unit) that
// renders its mirror into `targets[i]`; a water surface or glass pane samples the
// resolve of the slot it was assigned at init (see `assign_planar_slots`). The
// plane geometry is recomputed (oriented toward the camera) per frame, but the
// count + each reflector's slot are fixed at init. Rebuilt on resize alongside
// `hdr_targets` (the planes carry over).
pub(in crate::metal) struct PlanarReflectionSet {
    pub(in crate::metal) targets: Vec<PlanarReflectionTargets>,
    pub(in crate::metal) planes: Vec<[f32; 4]>,
}

// Build the planar reflection targets at `width`x`height`. Colour + depth match
// the main pipeline's attachment formats + sample count so `encode_main_into_face`
// binds the standard pipelines; the resolve is shader-readable.
pub(in crate::metal) fn create_planar_targets(
    device: &ProtocolObject<dyn MTLDevice>,
    width: u32,
    height: u32,
    sample_count: u32,
) -> Result<PlanarReflectionTargets, String> {
    let color = {
        let desc = MTLTextureDescriptor::new();
        unsafe {
            desc.setTextureType(MTLTextureType::Type2DMultisample);
            desc.setPixelFormat(MTLPixelFormat::RGBA16Float);
            desc.setWidth(width as usize);
            desc.setHeight(height as usize);
            desc.setSampleCount(sample_count as usize);
            desc.setUsage(MTLTextureUsage::RenderTarget);
            desc.setStorageMode(MTLStorageMode::Private);
        }
        device
            .newTextureWithDescriptor(&desc)
            .ok_or("planar: failed to create MSAA colour target")?
    };
    let depth = {
        let desc = MTLTextureDescriptor::new();
        unsafe {
            desc.setTextureType(MTLTextureType::Type2DMultisample);
            desc.setPixelFormat(MTLPixelFormat::Depth32Float);
            desc.setWidth(width as usize);
            desc.setHeight(height as usize);
            desc.setSampleCount(sample_count as usize);
            desc.setUsage(MTLTextureUsage::RenderTarget);
            desc.setStorageMode(MTLStorageMode::Private);
        }
        device
            .newTextureWithDescriptor(&desc)
            .ok_or("planar: failed to create MSAA depth target")?
    };
    let resolve = {
        let desc = MTLTextureDescriptor::new();
        unsafe {
            desc.setTextureType(MTLTextureType::Type2D);
            desc.setPixelFormat(MTLPixelFormat::RGBA16Float);
            desc.setWidth(width as usize);
            desc.setHeight(height as usize);
            desc.setUsage(MTLTextureUsage(
                MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0,
            ));
            desc.setStorageMode(MTLStorageMode::Private);
        }
        device
            .newTextureWithDescriptor(&desc)
            .ok_or("planar: failed to create resolve target")?
    };
    Ok(PlanarReflectionTargets {
        msaa_color: color,
        msaa_depth: depth,
        resolve,
    })
}

// Build a `PlanarReflectionSet` with one set of targets per plane in `planes`,
// each at `width`x`height`. `planes` is the deduplicated representative list from
// `assign_planar_slots`; an empty slice yields no set (the caller stores `None`).
pub(in crate::metal) fn create_planar_set(
    device: &ProtocolObject<dyn MTLDevice>,
    width: u32,
    height: u32,
    sample_count: u32,
    planes: &[[f32; 4]],
) -> Result<PlanarReflectionSet, String> {
    let mut targets = Vec::with_capacity(planes.len());
    for _ in planes {
        targets.push(create_planar_targets(device, width, height, sample_count)?);
    }
    Ok(PlanarReflectionSet {
        targets,
        planes: planes.to_vec(),
    })
}

impl MtlContext {
    // Render the scene reflected across each plane in the planar set into that
    // plane's target, reusing this frame's cull ICB + bindless buffers. A no-op
    // (returns Ok) when no set exists. Each plane is oriented toward the camera so
    // the oblique near-plane clip keeps the camera's side (a no-op for water above
    // the surface; flips a glass pane's normal when viewed from its back). Encoded
    // on `cmd_buf` before the transparent pass that samples the resolves;
    // command-buffer order + Metal's texture hazard tracking order each resolve
    // before its sample.
    pub(in crate::metal) fn encode_planar_reflections(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        params: &super::graph_exec::GraphFrameParams,
    ) -> Result<(), String> {
        let Some(set) = self.planar_reflection.as_ref() else {
            return Ok(());
        };

        // Recover the (jittered) projection from this frame's view-projection so
        // the mirror render shares the main camera's projection + jitter, keeping
        // the reflection aligned with the reflective fragment's screen-space sample.
        let proj = super::math::mat4_mul(params.vp, super::math::mat4_inverse(self.view_matrix));
        // Reused across planes for the non-bindless CPU reflected-frustum cull:
        // `reflected_visible_set` clears it each plane, so one allocation serves
        // the whole frame. Unused on the bindless path (the GPU mirror cull drives
        // the face render through the mirror ICB instead).
        let mut reflected_visible: Vec<u32> = Vec::new();
        for (slot, (plane, targets)) in set.planes.iter().zip(set.targets.iter()).enumerate() {
            let oriented =
                crate::gfx::planar_reflection::orient_plane_toward(*plane, params.cam_pos);
            let m = crate::gfx::planar_reflection::planar_matrices(
                self.view_matrix,
                proj,
                params.cam_pos,
                oriented,
                PLANAR_CLIP_BIAS,
            );

            // Re-cull against this plane's reflected-camera frustum so geometry
            // visible only in the reflection (behind or beside the main camera,
            // outside its frustum) is captured. The reflected view-proj already
            // carries the oblique near-plane clip, so its extracted frustum also
            // rejects geometry behind the reflector.
            let mirror_frustum = crate::gfx::frustum::Frustum::from_view_projection(m.view_proj);
            let (icb_override, visible): (
                Option<&ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>>,
                &[u32],
            ) = if let (Some(object_buffer), Some(draw_args)) =
                (params.object_buffer, params.draw_args_buffer)
            {
                // Bindless path: the GPU cull kernel re-runs into this plane's
                // mirror ICB, which the face render executes (`icb_override`); the
                // CPU `visible` slice is unused on the bindless static path.
                self.encode_mirror_cull(
                    cmd_buf,
                    object_buffer,
                    draw_args,
                    &mirror_frustum,
                    m.eye,
                    slot,
                )?;
                (
                    self.cull.mirror_slots.get(slot).map(|s| s.icb.as_ref()),
                    params.visible,
                )
            } else {
                // Non-bindless path: no GPU cull to mirror. Re-cull the CPU visible
                // set against the reflected frustum so the legacy face render draws
                // the reflection's geometry instead of the main camera's set.
                crate::gfx::planar_reflection::reflected_visible_set(
                    &self.cull_bvh,
                    &mirror_frustum,
                    m.eye,
                    &self.always_draw,
                    &mut reflected_visible,
                );
                (None, reflected_visible.as_slice())
            };

            self.encode_main_into_face(
                cmd_buf,
                &targets.msaa_color,
                &targets.msaa_depth,
                &targets.resolve,
                m.view,
                m.view_proj,
                m.eye,
                params.elapsed,
                visible,
                params.prepared_instances,
                params.skinned_joint_bufs,
                params.object_buffer,
                params.bindless_tex_args,
                params.deformed_skinned,
                icb_override,
            )?;
        }
        Ok(())
    }
}
