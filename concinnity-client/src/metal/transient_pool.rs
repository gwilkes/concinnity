// src/metal/transient_pool.rs
//
// Backing store for the render graph's transient textures on Metal. The shared
// `gfx::render_graph::alias` planner decides which transient resources may share
// physical memory; this pool is where the Metal backend realises that plan.
// Features stop owning these textures and read them back by label, so the pool
// can later repoint several labels at one aliased allocation without touching
// the features. This mirrors how the graph plans barriers while each backend
// emits them, and the Vulkan `transient_pool.rs`.
//
// Stage 1 (current): each managed transient owns its own committed texture (no
// `MTLHeap`, no aliasing). The pool's only job today is to relocate ownership of
// `ao_output` off SSAO so a later stage can place it and `bloom_top` on one heap
// slot and fence the reuse boundary. Single-buffered is correct here because
// Metal still auto-tracks a single texture across frames; the per-frame
// buffering an aliased slot needs arrives with the heap.
//
// A texture is "managed" iff its owning feature is enabled at build time;
// `texture_for` returns `None` otherwise and the consumer falls back exactly as
// before (the main pass samples a 1x1 white texture when SSAO is off).
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice as _, MTLPixelFormat, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};

use crate::metal::context::MtlContext;

// One managed transient texture: the graph label plus the parameters the pool
// needs to allocate it. The label is the same string the shared
// `build_frame_graph` declares, so every feature consumer agrees on one
// identifier.
pub(super) struct TextureSpec {
    pub label: &'static str,
    pub width: u32,
    pub height: u32,
    pub format: MTLPixelFormat,
}

struct PooledTexture {
    label: &'static str,
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
}

// The transient texture pool owned by `MtlContext`. Resolution-dependent, so it
// is rebuilt on resize alongside the other render-resolution targets.
pub(super) struct TransientTexturePool {
    textures: Vec<PooledTexture>,
}

impl TransientTexturePool {
    // Allocate every managed transient. Stage 1: one committed `Private` texture
    // each (sampled render target), no sharing. Each starts undefined; its
    // first-use contents come from its graph producer pass exactly as when the
    // feature owned it.
    pub(super) fn build(
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        specs: &[TextureSpec],
    ) -> Result<Self, String> {
        let sampled =
            MTLTextureUsage(MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0);
        let mut textures = Vec::with_capacity(specs.len());
        for spec in specs {
            let desc = MTLTextureDescriptor::new();
            unsafe {
                desc.setTextureType(MTLTextureType::Type2D);
                desc.setPixelFormat(spec.format);
                desc.setWidth(spec.width.max(1) as usize);
                desc.setHeight(spec.height.max(1) as usize);
                desc.setUsage(sampled);
                desc.setStorageMode(MTLStorageMode::Private);
            }
            let texture = device
                .newTextureWithDescriptor(&desc)
                .ok_or_else(|| format!("failed to create transient texture {}", spec.label))?;
            textures.push(PooledTexture {
                label: spec.label,
                texture,
            });
        }
        Ok(Self { textures })
    }

    // The managed texture for `label`, or `None` when its owning feature was
    // disabled at build time (so the pool holds no entry for it).
    pub(super) fn texture_for(&self, label: &str) -> Option<&ProtocolObject<dyn MTLTexture>> {
        self.textures
            .iter()
            .find(|t| t.label == label)
            .map(|t| t.texture.as_ref())
    }

    // Rebuild every managed texture at a new extent. Metal reference-counts, so
    // the old textures stay alive until the last command buffer referencing them
    // retires; the caller rebinds the new ones into the affected passes (the
    // per-frame bindless argument buffer re-encodes `ao_output` itself).
    pub(super) fn rebuild(
        &mut self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        specs: &[TextureSpec],
    ) -> Result<(), String> {
        *self = Self::build(device, specs)?;
        Ok(())
    }
}

// Build the managed-transient list for this build. Centralises the label ->
// (format, extent) mapping so init and resize stay in lockstep. Stage 1 manages
// only `ao_output` (SSAO's blurred occlusion, render-resolution R8); a later
// stage adds `bloom_top` and groups the two into one aliased heap slot.
pub(super) fn transient_specs(ssao_enabled: bool, ao_w: u32, ao_h: u32) -> Vec<TextureSpec> {
    let mut specs = Vec::new();
    if ssao_enabled {
        specs.push(TextureSpec {
            label: "ao_output",
            width: ao_w,
            height: ao_h,
            format: MTLPixelFormat::R8Unorm,
        });
    }
    specs
}

impl MtlContext {
    // The texture the main pass and the bindless argument buffer sample for
    // ambient occlusion: the pooled `ao_output` (SSAO's blurred occlusion) when
    // SSAO is on, else the SSAO state's 1x1 white fallback so `shade_surface`
    // reads a constant 1.0 (fully unoccluded). Replaces `SsaoState::ao_texture`
    // now that the occlusion target lives in the transient pool.
    pub(in crate::metal) fn ao_output_texture(&self) -> &ProtocolObject<dyn MTLTexture> {
        self.transient_pool
            .texture_for("ao_output")
            .unwrap_or_else(|| self.ssao.white.as_ref())
    }
}
