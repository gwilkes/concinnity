// src/gfx/render_graph/passes.rs
//
// Stable identity for every render-graph pass. Used by:
//
//   - The graph itself, as the dispatch key the executor matches on.
//   - The per-pass GPU timer ([`crate::metal::pass_timing`]), which keys
//     its sample-buffer slots off the same integer. Adding a new pass = a
//     new variant here + a new entry in [`PASS_NAMES`] + a bumped
//     `PASS_COUNT`; nothing else needs to change for timing to flow. The
//     `every_pass_id_round_trips_to_its_name` test forces all three edits
//     at compile time (a missed registration would otherwise report zero
//     GPU time for the pass), and `pass_timing::slot_pair` debug_asserts
//     the index at runtime.
//
// Variants are intentionally `#[repr(u32)]` so a `PassId` round-trips
// through `as usize` into the [`PASS_NAMES`] / counter-sample-buffer
// slot index. Adding a variant in the middle of the list will renumber
// later variants and silently shift every timing slot; append-only.

// Stable display name for each pass. Index = `PassId as usize`. Used by
// the WS `profile.passes` reply and the per-pass timing readback.
pub const PASS_NAMES: [&str; PASS_COUNT] = [
    "cull",
    "shadow",
    "ssr_prepass",
    "ssao_prepass",
    "ssao_kernel",
    "ssao_blur",
    "main",
    "auto_exposure",
    "decals",
    "fog",
    "particles_sim",
    "particles_draw",
    "ssr_resolve",
    "velocity",
    "taa_resolve",
    "bloom",
    "composite",
    "fog_froxel",
    "upscale",
    "transparent",
    "raymarch",
    "hiz_build",
    "cull2",
    "main2",
    "ssgi",
    "rt_reflections",
    "gbuffer_prepass",
];

// Number of distinct passes the engine times. Sized to match
// [`PASS_NAMES`]; the per-pass timing array in
// [`crate::gfx::profile::RenderStats`] is sized to at least this many
// slots.
pub const PASS_COUNT: usize = 27;

// One per-pass identity. Cast to `usize` to index [`PASS_NAMES`] or any
// `[T; PASS_COUNT]` companion array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum PassId {
    Cull = 0,
    Shadow = 1,
    SsrPrepass = 2,
    SsaoPrepass = 3,
    SsaoKernel = 4,
    SsaoBlur = 5,
    Main = 6,
    AutoExposure = 7,
    Decals = 8,
    Fog = 9,
    ParticlesSim = 10,
    ParticlesDraw = 11,
    SsrResolve = 12,
    Velocity = 13,
    TaaResolve = 14,
    Bloom = 15,
    Composite = 16,
    // Volumetric-fog froxel-volume compute pass. Populates a 3D
    // `(scattered, transmittance)` texture once per frame, sampled by the
    // fullscreen `Fog` render pass instead of an inline ray-march. Metal
    // only; DX/Vulkan keep the ray-march and never insert this pass.
    FogFroxel = 17,
    // Temporal upscaling pass. When the world's `PostProcessConfig`
    // enables `temporal_upscaling`, the renderer draws the 3D scene at a
    // fraction of drawable size and inserts this pass between the post-SSR
    // scene and the Bloom + Composite stack. The backend runs its
    // platform-native temporal upscaler (MetalFX on macOS; FSR / DLSS /
    // XeSS slots on the Windows backends are placeholders today) to
    // reconstruct a drawable-resolution image. Replaces `TaaResolve`:
    // the upscaler does temporal accumulation itself, so adding both
    // would double-temporal the scene.
    Upscale = 18,
    // Transparent / translucent geometry pass. Runs after `SsrResolve`
    // (so water + glass see opaque reflections) and before
    // `TaaResolve` / `Upscale` (so translucents pick up temporal
    // accumulation). Reads the latest scene-pre-taa colour + main
    // depth as sampled textures; writes scene-pre-taa blended
    // (SRC_ALPHA / ONE_MINUS_SRC_ALPHA). Each transparent draw owns
    // its own pipeline + descriptor set; the pass aggregates them as
    // a back-to-front sorted list at encode time. Gated on
    // `FrameGraphInputs::transparent_enabled`; when no consumer is
    // in the world, the slot is omitted entirely.
    Transparent = 19,
    // Raymarched SDF volume pass. Rasterises the back faces of each
    // `SdfVolume`'s world-space bounding box and runs the user-authored
    // fragment shader, which sphere-traces a signed distance field
    // inside the box. Hit fragments write opaque colour into
    // `hdr_resolve` (RMW between `AutoExposure` and `Decals`) and
    // update the main depth attachment so the raymarched surface
    // composites with rasterised geometry naturally: decals, fog,
    // SSR-resolve, and TAA all consume the post-Raymarch depth and
    // colour. Gated on `FrameGraphInputs::raymarch_enabled`; when no
    // `SdfVolume` is in the world the slot is omitted entirely.
    Raymarch = 20,
    // Mid-frame Hi-Z (depth-mip pyramid) rebuild for two-pass occlusion
    // culling. Inserted only when `FrameGraphInputs::two_pass_occlusion_enabled`
    // is on: after `Main` (phase 1) has written this frame's depth, this
    // compute pass reduces it into the Hi-Z pyramid so `Cull2` can re-test
    // the objects phase 1 occluded against up-to-date depth. Distinct from
    // the end-of-frame Hi-Z build (which feeds the *next* frame's phase-1
    // cull and stays an inline action, not a graph node). Metal only; the
    // other backends keep `two_pass_occlusion_enabled` false so this node
    // never appears in their graphs.
    HizBuild = 21,
    // Phase-2 GPU cull for two-pass occlusion. Re-tests the objects `Cull`
    // (phase 1) marked Hi-Z-occluded against the freshly rebuilt pyramid
    // (`HizBuild`) and encodes a draw for any that turn out visible into a
    // second indirect command buffer `Main2` consumes. Reads the per-object
    // status buffer phase-1 cull wrote + the `draw_args2` buffer it writes.
    // Gated on `FrameGraphInputs::two_pass_occlusion_enabled`; Metal only.
    Cull2 = 22,
    // Phase-2 main pass for two-pass occlusion. Loads (does not clear) the
    // HDR colour + depth `Main` wrote and re-runs only the bindless-static
    // indirect draw through `Cull2`'s command buffer, depth-compositing the
    // disoccluded geometry with phase 1. Instanced + skinned geometry is not
    // Hi-Z-culled, so it is fully drawn in phase 1 and not repeated here.
    // Becomes the new head of the hdr_resolve post-decoration chain (so
    // AutoExposure / Decals / Fog / SSR see the combined result). Gated on
    // `FrameGraphInputs::two_pass_occlusion_enabled`; Metal only.
    Main2 = 23,
    // Screen-space global illumination. A refinement of SSR: it reuses the
    // SSR depth + normal pre-pass G-buffer and screen-space ray-march, but
    // integrates bounced radiance over a cosine-weighted hemisphere instead
    // of along one reflection vector. Sits on the hdr_resolve RMW chain (after
    // `Raymarch`, before `Decals`): it reads the lit scene as the bounce
    // radiance source and additively composites the gathered + denoised
    // indirect term back into it, so the near-field colour bleed layers on top
    // of the IBL ambient. Gated on `FrameGraphInputs::ssgi_enabled`; when
    // `indirect_lighting` is IBL-only the slot is omitted entirely. Metal only
    // today; the other backends keep the flag false so this node never appears
    // in their graphs.
    Ssgi = 24,
    // Hardware ray-traced reflections. Occupies the same scene-pre-taa slot as
    // `SsrResolve` (reads the post-decoration `hdr_resolve`, writes
    // `scene_pre_taa`) and takes precedence over it: when this pass is live the
    // builder inserts it and omits `SsrResolve` (a world may author both; RT
    // runs where available, SSR is the fallback). It still relies on the SSR
    // depth + normal + roughness
    // pre-pass (so `SsrPrepass` is forced on), but instead of a screen-space
    // march it traces a world-space reflection ray against an acceleration
    // structure built over the static scene geometry, so off-screen reflected
    // geometry appears. Gated on `FrameGraphInputs::rt_reflections_enabled`;
    // Metal only, and only on GPUs that report ray-tracing support.
    RtReflections = 25,
    // Unified geometry G-buffer pre-pass. One jittered traversal of the visible
    // set writes view-space normal + linear depth, perceptual roughness, and
    // screen-space motion into a single MRT (plus a sampleable depth), replacing
    // the separate `SsrPrepass` + `Velocity` (and the SSAO-owned prepass): every
    // consumer (SSR, SSAO, SSGI, RT, TAA, upscaler) reads this one output. Gated
    // on `FrameGraphInputs::unified_gbuffer_prepass`; Metal only, so the other
    // backends keep the flag false and emit their separate prepasses instead.
    GBufferPrepass = 26,
}

impl PassId {
    // Stable display name, looked up in [`PASS_NAMES`]. `'static` since
    // the table is `const`.
    #[allow(dead_code)] // Used by per-backend executor + debug formatting.
    pub fn name(self) -> &'static str {
        PASS_NAMES[self as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every `PassId` variant, in declaration (index) order. This is a sized
    // `[PassId; PASS_COUNT]` array, so adding a variant forces three edits to
    // keep this compiling: bump `PASS_COUNT`, add the `PASS_NAMES` entry, and
    // list the variant here. Combined with `expected_name`'s wildcard-free
    // match below, a new graph pass cannot ship without a timing name (which
    // would otherwise read as zero GPU time).
    const ALL: [PassId; PASS_COUNT] = [
        PassId::Cull,
        PassId::Shadow,
        PassId::SsrPrepass,
        PassId::SsaoPrepass,
        PassId::SsaoKernel,
        PassId::SsaoBlur,
        PassId::Main,
        PassId::AutoExposure,
        PassId::Decals,
        PassId::Fog,
        PassId::ParticlesSim,
        PassId::ParticlesDraw,
        PassId::SsrResolve,
        PassId::Velocity,
        PassId::TaaResolve,
        PassId::Bloom,
        PassId::Composite,
        PassId::FogFroxel,
        PassId::Upscale,
        PassId::Transparent,
        PassId::Raymarch,
        PassId::HizBuild,
        PassId::Cull2,
        PassId::Main2,
        PassId::Ssgi,
        PassId::RtReflections,
        PassId::GBufferPrepass,
    ];

    // Expected timing name per variant. The match has no wildcard arm, so
    // adding a `PassId` variant fails to compile here until it is named. This
    // is the forcing function for the timing-name registration gotcha.
    fn expected_name(pass: PassId) -> &'static str {
        match pass {
            PassId::Cull => "cull",
            PassId::Shadow => "shadow",
            PassId::SsrPrepass => "ssr_prepass",
            PassId::SsaoPrepass => "ssao_prepass",
            PassId::SsaoKernel => "ssao_kernel",
            PassId::SsaoBlur => "ssao_blur",
            PassId::Main => "main",
            PassId::AutoExposure => "auto_exposure",
            PassId::Decals => "decals",
            PassId::Fog => "fog",
            PassId::ParticlesSim => "particles_sim",
            PassId::ParticlesDraw => "particles_draw",
            PassId::SsrResolve => "ssr_resolve",
            PassId::Velocity => "velocity",
            PassId::TaaResolve => "taa_resolve",
            PassId::Bloom => "bloom",
            PassId::Composite => "composite",
            PassId::FogFroxel => "fog_froxel",
            PassId::Upscale => "upscale",
            PassId::Transparent => "transparent",
            PassId::Raymarch => "raymarch",
            PassId::HizBuild => "hiz_build",
            PassId::Cull2 => "cull2",
            PassId::Main2 => "main2",
            PassId::Ssgi => "ssgi",
            PassId::RtReflections => "rt_reflections",
            PassId::GBufferPrepass => "gbuffer_prepass",
        }
    }

    #[test]
    fn pass_names_match_pass_count() {
        assert_eq!(PASS_NAMES.len(), PASS_COUNT);
    }

    #[test]
    fn every_pass_id_round_trips_to_its_name() {
        // Each variant's integer value must equal its position in both `ALL`
        // and `PASS_NAMES` so the per-pass timing arrays stay aligned, and
        // every pass must carry a non-empty, expected name.
        for (i, &pass) in ALL.iter().enumerate() {
            assert_eq!(pass as usize, i, "{pass:?} index out of order");
            assert_eq!(pass.name(), PASS_NAMES[i], "{pass:?} name table mismatch");
            assert_eq!(pass.name(), expected_name(pass), "{pass:?} name drifted");
            assert!(!pass.name().is_empty(), "{pass:?} has an empty name");
        }
        // The last listed variant pins the high end of the index range.
        assert_eq!(ALL[PASS_COUNT - 1] as usize, PASS_COUNT - 1);
    }
}
