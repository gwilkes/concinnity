// src/directx/init/mod.rs
//
// DxContext construction. The constructor is intentionally a flat top-to-
// bottom sequence so the order of dependencies stays obvious; helpers for
// self-contained sub-phases live in sibling modules:
//
//   window.rs    Win32 window + raw input + DXGI factory + adapter +
//                D3D12 device + info-queue + command queue + swapchain
//                + MSAA support query.
//   pipelines.rs Shader compile + main / shadow / instanced / text /
//                composite PSOs + bindless main pass + GPU-cull compute
//                pipeline and its per-frame UAV / upload buffers.
//   effects.rs   Bloom mip targets + pipelines, TAA velocity + history,
//                SSAO pre-pass + kernel + blur, SSAO white fallback.
//                Each gated on per-world settings.
//
// What still lives inline here:
//   * Descriptor heap creation (RTV / DSV / CBV+SRV+UAV / sampler) with
//     the cross-cutting slot layout.
//   * Sampler creation.
//   * Texture pool uploads, per-object + per-cluster SRV pair writes,
//     text atlas uploads, shadow map array, IBL cubes, colour LUT,
//     main-depth + HDR scene targets.
//   * Geometry + per-frame view / light / shadow constant buffers.
//   * Per-frame command infrastructure (allocator/list/fence), per-cluster
//     instance upload buffers, and the final `Self { ... }` literal.

use std::cell::RefCell;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::CreateEventW;

use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::*;

use super::context::*;
use super::draw::*;
use super::math::*;
use super::post::bloom::bloom_mip_count;
use super::texture::*;

mod effects;
mod heap_layout;
pub(in crate::directx) mod pipelines;
mod window;

// Maximum Hi-Z mip count we reserve descriptor slots for. 15 mips covers
// every render target up to 16384 pixels in the larger dimension; an
// 8K display sits at 13. The Hi-Z resource clamps `mip_count` against this
// so the heap layout stays anchored even when the window resizes.
pub(in crate::directx) const HIZ_MAX_MIPS: usize = 15;

#[allow(clippy::too_many_arguments)]
impl DxContext {
    pub fn new(
        title: &str,
        width: u32,
        height: u32,
        validation: bool,
        _frames_in_flight: usize, // we always use FRAMES=3 for D3D12
        vsync: bool,
        clear_color: [f32; 4],
        vertices: &[Vertex],
        indices: &[u32],
        draw_objects: Vec<DrawObject>,
        instanced_clusters: Vec<InstancedCluster>,
        // Skinned draw-object count, threaded purely to size the shared cull /
        // object / draw-args / indirect buffers for the merged total at init
        // (`n_objects + n_instances + n_skinned`); the skinned geometry itself is
        // uploaded later by `upload_skinned`, which sets the live `self.n_skinned`.
        // 0 when the world has no skinned meshes.
        n_skinned: usize,
        // Worst-case resident chunk count for a streaming VoxelWorld (0 otherwise).
        // Reserves a chunk record region in the shared cull buffers at init
        // (`[n_objects + n_instances, +n_chunk_max)`); resident chunks fold into
        // the indirect path each frame. Sets the live `self.n_chunk`.
        n_chunk_max: usize,
        vert_bytes: &[u8],
        frag_bytes: &[u8],
        shadow_bytes: &[u8],
        vert_instanced_bytes: &[u8],
        textures: &[(u32, u32, Vec<u8>)],
        normal_maps: &[(u32, u32, Vec<u8>)],
        light_uniforms: LightUniforms,
        shadow_map_size: u32,
        // Shadow-cascade re-render policy: hybrid amortizes the far cascades
        // across frames, every_frame refreshes all cascades every frame.
        shadow_update: crate::assets::ShadowUpdate,
        text_atlases: Vec<(u32, u32, Vec<u8>)>,
        // Serialised EnvironmentMap payload (irradiance + prefilter cubemaps).
        // None disables IBL; the runtime binds 1×1 grey fallback cubes and
        // sets ViewUniforms::prefilter_mip_count to 0 so the shader falls
        // back to the legacy ambient/skybox path.
        env_map_bytes: Option<&[u8]>,
        // Post-process tunables. `bloom_intensity` / `bloom_threshold` /
        // `bloom_knee` drive the bloom chain; `exposure` / `vignette` /
        // `lut_strength` feed the composite pass.
        post_process: crate::gfx::render_types::PostProcessParams,
        // Serialised ColorLut payload (3D grading LUT) baked into the composite
        // pass. `None` binds a 2×2×2 identity LUT, so the grade is a no-op at
        // any `lut_strength`.
        color_lut_bytes: Option<&[u8]>,
        // Temporal anti-aliasing toggle (resolved from `PostProcessConfig.taa`).
        // When set, the renderer jitters the projection, runs a velocity
        // pre-pass + a history-resolve pass, and feeds the resolved image to
        // the bloom + composite passes.
        taa_enabled: bool,
        // SSAO (GTAO) settings. When `Some`, the renderer runs a depth +
        // normal pre-pass, the GTAO horizon-search kernel, and a depth-aware
        // blur; the main pass then samples the blurred occlusion to modulate
        // its ambient term. `None` skips every SSAO pass and binds a 1x1
        // white fallback so the ambient multiplier is a constant 1.0.
        ssao_settings: Option<crate::gfx::ssao::SsaoSettings>,
        // SSR settings. When `Some`, the renderer runs a depth + normal +
        // roughness pre-pass and a fullscreen ray-march resolve; the resolve
        // output then replaces `hdr_resolve` as the scene colour the TAA /
        // bloom / composite passes consume. `None` skips every SSR pass and
        // the post stack samples `hdr_resolve` directly.
        ssr_settings: Option<crate::gfx::ssr::SsrSettings>,
        // SSGI settings. When `Some`, the renderer runs the SSR depth + normal
        // pre-pass (forced on so the gather has a G-buffer, even when SSR
        // resolve is off) plus a hemisphere-gather + depth-aware-blur composite
        // that additively bleeds nearby lit surfaces' colour onto one another
        // on top of the IBL ambient. `None` skips the SSGI pass entirely.
        ssgi_settings: Option<crate::gfx::ssgi::SsgiSettings>,
        // Hardware ray-traced-reflection settings. `Some` only when the world
        // authored `ray_traced_reflections`. When present AND the GPU supports
        // the DXR tier, the renderer builds the scene acceleration structure +
        // the inline-`RayQuery` reflection pass (forcing the SSR pre-pass
        // G-buffer on, like SSGI) and the frame graph runs `RtReflections` in
        // the `SsrResolve` slot. Any failure (no DXR, DXC absent, build error)
        // falls back to SSR. Reuses the SSR `intensity` / `max_distance` knobs.
        rt_reflection_settings: Option<crate::gfx::rt_reflections::RtReflectionSettings>,
        // Authored projected decals resolved from the world's `Decal`
        // components. The decal pipeline + unit-cube buffers are always built
        // (so runtime `add_decal` works from an empty world); the encoder
        // simply skips when every slot is `None`.
        decals: Vec<crate::gfx::decal::DecalRecord>,
        // Particle-emitter records resolved from the world's
        // `ParticleEmitter` components. The compute + render pipelines and the
        // per-emitter GPU pool buffers are built only when at least one
        // emitter is declared (or when runtime `add_emitter` fires); the
        // encoder skips the passes when `particle_resources` is `None`.
        particles: Vec<crate::gfx::particles::ParticleEmitterRecord>,
        // Volumetric-fog settings resolved from the world's `VolumetricFog`.
        // `Some` builds the fog pipeline + per-frame uniform ring; `None`
        // skips the fog pass entirely.
        fog_settings: Option<crate::gfx::volumetric_fog::FogSettings>,
        // Auto-exposure settings resolved from `PostProcessConfig`. `Some`
        // builds the histogram + average compute pipelines and drives the EMA;
        // `None` disables auto-exposure entirely (the authored `exposure_ev`
        // remains the only input to `post_process.exposure`).
        auto_exposure_settings: Option<crate::gfx::auto_exposure::AutoExposureSettings>,
        // Authored `exposure_ev` carried through as a bias on the adapted EV
        // when auto-exposure is on; ignored when it is off.
        auto_exposure_bias_ev: f32,
        // World-side HDR display request from `PostProcessConfig.hdr_display`.
        // When `true` and the active adapter reports an HDR-capable output, the
        // swapchain is created in `RGBA16Float` + scRGB-linear colour space and
        // the composite shader's `hdr_output` branch skips ACES + gamma + FXAA
        // + LUT, emitting linear extended-range values directly. When `false`
        // or the panel reports no EDR headroom, the renderer stays on the SDR
        // path (BGRA8Unorm + ACES tonemap chain) and the request is logged.
        hdr_display: bool,
        // PQ-encoded HDR output request from `PostProcessConfig.hdr_pq`.
        // When `true` and `hdr_display` resolves on AND the swapchain
        // advertises `DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020`, the
        // swapchain is created in that colour space and the composite
        // shader emits SMPTE ST 2084 PQ-encoded values directly (SDR
        // reference white = 203 nits per BT.2408). When the swapchain
        // does not advertise HDR10 PQ but does advertise scRGB linear,
        // the renderer falls back to extended-linear output with a
        // warning. Mirrors the Metal `kCGColorSpaceDisplayP3_PQ` path.
        hdr_pq: bool,
        // Temporal upscaling toggle from `PostProcessConfig.temporal_upscaling`.
        // When on AND the FFX SDK (`amd_fidelityfx_dx12.dll`) loads
        // successfully, the engine builds an FSR3 upscaler that runs
        // between the post-SSR scene and the bloom + composite stack. The
        // scene renders at `output * upscale_scale` and FSR reconstructs the
        // drawable resolution.
        temporal_upscaling: bool,
        // Per-axis input-to-output ratio from `PostProcessConfig.upscale_quality`.
        // Threaded through to the FSR3 upscaler when `temporal_upscaling`
        // is on; it sizes the off-screen scene targets at
        // `output * upscale_scale`. Ignored when upscaling is off.
        upscale_scale: f32,
        // `PostProcessConfig.upscale_backend`. Selects the temporal-upscaling
        // backend (auto / fsr3 / dlss / xess); resolved against runtime
        // availability by `build_upscaler`, falling back when the requested one
        // is unavailable. Only consumed when `temporal_upscaling` is on.
        upscale_backend: crate::assets::UpscalerBackend,
        // `PostProcessConfig.occlusion_two_pass`. When set (and the bindless
        // GPU-cull path is active), the renderer builds the phase-2 cull
        // pipeline + second indirect buffers and the frame graph inserts the
        // HizBuild / Cull2 / Main2 chain that re-tests phase-1-occluded objects
        // against a mid-frame-rebuilt Hi-Z pyramid.
        occlusion_two_pass: bool,
        // Raymarched SDF volumes drained from the world's `SdfVolume`
        // components, paired with their compiled-payload fragment shader
        // source bytes + asset label. Each entry is filtered at init:
        // `.hlsl` payloads compile + render, `.metal` (Metal-first) ones
        // are skipped with a logged warning and the rest of the world
        // renders unchanged.
        sdf_volumes: Vec<(crate::assets::SdfVolume, Vec<u8>, String)>,
        // Translucent glass panels drained from the world's `GlassPanel`
        // components. Each becomes one back-to-front-sorted draw in the shared
        // transparent pass. Empty leaves `glass` None and the pass skipped.
        glass_panels: Vec<crate::assets::GlassPanel>,
        // True only under `cn debug` (set via `crate::app::dev_flags`). Routes
        // every built-in shader compile through the disk-first
        // `shader_source` helper and spawns the `directx/shaders/` filesystem
        // watcher. `cn run` leaves it false; the baked-in include_str! sources
        // continue to drive every pipeline.
        hot_reload: bool,
    ) -> Result<Self, String> {
        // Record this (main) thread so the `RenderBackend` mutation entry
        // points can `debug_assert_main_thread` against it; the Send invariant
        // rests on the context being touched from this thread alone.
        super::context::record_main_thread();

        // FSR3 needs the velocity buffer + the TAA-velocity pre-pass
        // PSOs, both of which live inside `TaaResources`. When upscale
        // is on we force the TAA resources to be built even if the
        // world's `PostProcessConfig.taa` is off; the TAA *resolve*
        // pass is still skipped (see `record_frame::seed_inputs`),
        // because FSR owns the temporal accumulation.
        let taa_enabled = taa_enabled || temporal_upscaling;
        // Win32 window + DXGI factory + device + info-queue + command queue +
        // MSAA support + swapchain. See init/window.rs. The HDR-display
        // negotiation also happens in there: a capable adapter + a `true`
        // toggle yields a `RGBA16Float` scRGB swapchain; otherwise the
        // returned `hdr_mode` is `Sdr` and the swapchain stays at BGRA8Unorm.
        let window::DeviceAndWindow {
            win_state,
            device,
            info_queue,
            command_queue,
            swapchain,
            swapchain_format,
            allow_tearing,
            msaa_samples,
            adapter,
            hdr_mode,
        } = window::setup(title, width, height, validation, vsync, hdr_display, hdr_pq)?;
        // Presentation pacing derived from the vsync request + tearing support.
        // vsync on -> sync interval 1 (lock to refresh). vsync off + tearing ->
        // sync interval 0 with the tearing present flag (true uncapped). vsync
        // off without tearing -> sync interval 0, no flag (flip-model refresh
        // pacing, the best available fallback).
        let present_sync_interval: u32 = if vsync { 1 } else { 0 };
        // Surface the resolved mode to the composite shader via the post
        // uniform. On the SDR path both flags stay 0.0 and the shader runs
        // the full ACES + gamma + FXAA + LUT chain unchanged. Inside the
        // HDR branch, `pq_output` picks scRGB-linear passthrough (0.0) vs
        // SMPTE ST 2084 in-shader encode (1.0). Mirrors the Metal hop in
        // `metal/init/mod.rs`. `setup` may have already downgraded the
        // encoding when `CheckColorSpaceSupport(HDR10 PQ)` came back
        // negative, so reading `hdr_mode.pq_flag()` after `setup` returns
        // is the source of truth.
        let mut post_process = post_process;
        post_process.hdr_output = hdr_mode.shader_flag();
        post_process.pq_output = hdr_mode.pq_flag();

        // Hardware ray-tracing capability + update mode. RT reflection resources
        // + the acceleration structure are built only when the world authored
        // `ray_traced_reflections` AND the GPU reports the DXR 1.1 tier inline
        // `RayQuery` needs; otherwise the renderer falls back to SSR. The dynamic
        // mode (how the BVH tracks moving props) is read once from `CN_RT_DYNAMIC`.
        let raytracing_supported = super::raytrace::raytracing_supported(&device);
        let rt_dynamic_mode = super::raytrace::RtDynamicMode::from_env();
        let rt_enabled = rt_reflection_settings.is_some() && raytracing_supported;
        if rt_reflection_settings.is_some() && !raytracing_supported {
            tracing::warn!(
                "ray_traced_reflections requested but the GPU does not report DXR \
                 tier 1.1; falling back to screen-space reflections"
            );
        }

        // RTV heap
        // Slots: [0..FRAMES) = back-buffer RTVs, [FRAMES] = HDR scene RTV,
        // [FRAMES+1 .. FRAMES+1+bloom_count] = bloom mip RTVs, then (TAA only)
        // the velocity RTV + two ping-pong history RTVs.
        let bloom_count = bloom_mip_count(width, height) as usize;
        // The five live-toggleable Quality features (TAA, SSAO, SSR, SSGI, and
        // the unified G-buffer pre-pass they share) reserve their RTV / DSV / SRV
        // slots UNCONDITIONALLY, independent of the world's init-time gates. The
        // slots are fixed positions the passes bind by absolute index, so a live
        // toggle (`apply_quality_settings`) can build a feature that launched off
        // and write into its pre-reserved slot without shifting any other
        // feature's slots. A reserved-but-unbuilt feature leaves its slots
        // unwritten; that is safe because no always-running pass binds them (each
        // feature's own pass runs only when the feature is on, and the main pass's
        // SSAO occlusion binding falls back to the 1x1 white slot below), matching
        // the existing reserved-but-unwritten SSR slot in a SSGI-only build. The
        // `*_enabled` / `*_present` gates below still drive whether the resources
        // are BUILT at init, just not whether the slots exist.
        //
        // TAA: 2 ping-pong history RTVs after the bloom mip RTVs + 2 history SRVs
        // after the colour LUT SRV. Its motion comes from the G-buffer pre-pass,
        // so TAA reserves no DSV of its own.
        let taa_rtv_extra = 2;
        let taa_srv_extra = 2;
        // SSAO: 2 RTVs (ao_raw + ao) + 2 SRVs (ao_raw + ao); view normal + depth
        // come from the G-buffer pre-pass, so no DSV. A 1x1 white fallback always
        // sits one slot further so the main pass binds a constant 1.0 occlusion
        // when SSAO is off (this is the one feature SRV an always-running pass
        // binds, hence the always-present fallback).
        let ssao_enabled = ssao_settings.is_some();
        let ssao_rtv_extra = 2;
        let ssao_srv_extra = 2;
        // SSR: 1 RTV + 1 SRV (resolve output); view normal + depth + roughness
        // come from the G-buffer pre-pass, so no DSV. SSR / SSGI / RT all reuse
        // the pre-pass; `ssr_prepass_present` still gates whether `SsrResources`
        // is built at init.
        let ssr_prepass_present = ssr_settings.is_some() || ssgi_settings.is_some() || rt_enabled;
        let ssr_rtv_extra = 1;
        let ssr_srv_extra = 1;
        // SSGI gather target: 1 RTV (the gather writes it) + 1 SRV (the composite
        // reads it).
        let ssgi_rtv_extra = 1;
        let ssgi_srv_extra = 1;
        // RT-reflection output: 1 RTV (the trace writes it) at the RTV-heap tail
        // + 1 SRV (the post stack samples it) at the SRV-heap tail. Reserved
        // UNCONDITIONALLY like the other live-toggleable features, so a live
        // `apply_quality_settings` RT enable (on a DXR-capable GPU) builds the
        // output into its fixed slot without shifting any other feature's slots.
        // `rt_enabled` still gates whether RT is BUILT at init, just not the slot.
        let rt_rtv_extra = 1;
        let rt_srv_extra = 1;
        // Unified G-buffer pre-pass: 3 RTVs (normal+depth, roughness, velocity),
        // 1 DSV (private depth), 3 SRVs. Slots always reserved; `gbuffer_enabled`
        // still gates whether the pre-pass resources are built at init (any
        // screen-space consumer: SSR / SSGI, SSAO, or TAA / FSR velocity).
        // `taa_enabled` already folds in temporal upscaling, covering velocity.
        let gbuffer_enabled = taa_enabled || ssao_enabled || ssr_prepass_present;
        let gbuffer_rtv_extra = 3;
        let gbuffer_dsv_extra = 1;
        let gbuffer_srv_extra = 3;
        // Projected decals: always-on infrastructure so runtime `add_decal`
        // works from a world that started empty. One extra RTV for
        // `hdr_resolve` (only when MSAA is on; the MSAA-off path writes
        // through the existing `hdr_color` RTV), one SRV for the main depth,
        // and `MAX_DECALS` per-decal albedo SRV slots.
        let decal_rtv_extra = if msaa_samples > 1 { 1 } else { 0 };
        let decal_srv_extra = crate::directx::decal::MAX_DECALS + 1;
        let _ = &decals; // referenced below where the pipeline is built.
        let rtv_heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: FRAMES as u32
                    + 1
                    + bloom_count as u32
                    + taa_rtv_extra as u32
                    + ssao_rtv_extra as u32
                    + ssr_rtv_extra as u32
                    + ssgi_rtv_extra as u32
                    + decal_rtv_extra as u32
                    + gbuffer_rtv_extra as u32
                    + rt_rtv_extra as u32,
                ..Default::default()
            })
        }
        .map_err(|e| format!("RTV heap: {e}"))?;
        let rtv_descriptor_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) }
                as usize;

        // Back-buffer RTVs
        let mut back_buffers = Vec::with_capacity(FRAMES);
        let rtv_base = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        for i in 0..FRAMES {
            let buf: ID3D12Resource = unsafe { swapchain.GetBuffer(i as u32) }
                .map_err(|e| format!("GetBuffer[{i}]: {e}"))?;
            let rtv_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv_base.ptr + i * rtv_descriptor_size,
            };
            unsafe {
                device.CreateRenderTargetView(&buf, None, rtv_handle);
            }
            back_buffers.push(buf);
        }

        // DSV heap
        // Slots: [0] = main depth, [1..1+NUM_SHADOW_CASCADES] = per-cascade
        // shadow DSVs (one slice each into the shadow map array), then the
        // unified G-buffer pre-pass's private depth buffer.
        let dsv_heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: 1 + NUM_SHADOW_CASCADES as u32 + gbuffer_dsv_extra as u32,
                ..Default::default()
            })
        }
        .map_err(|e| format!("DSV heap: {e}"))?;
        let dsv_descriptor_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) }
                as usize;
        let dsv_base = unsafe { dsv_heap.GetCPUDescriptorHandleForHeapStart() };
        let main_dsv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: dsv_base.ptr };
        let shadow_dsv_base_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: dsv_base.ptr + dsv_descriptor_size,
        };

        // CBV/SRV/UAV heap slot layout. The full per-block map + the
        // positional cascade live in `heap_layout.rs`, which a unit test
        // anchors so a stray offset edit fails a test instead of silently
        // misbinding a descriptor at shader time.
        let n_objects = draw_objects.len();
        let n_clusters = instanced_clusters.len();
        let n_atlases = text_atlases.len();
        // Flat bindless pool sizes, derived from the resource pools built below:
        // the albedo region has one SRV per `gpu_textures` entry (a 1x1 white
        // fallback stands in when no textures exist), the normal region one per
        // `gpu_normal_maps` entry (slot 0 is the flat-normal fallback).
        let flat_albedo_count = textures.len().max(1);
        let flat_normal_count = normal_maps.len() + 1;
        let _ = decal_srv_extra; // folded into the heap_layout decal block.
        let heap_layout::SrvHeapLayout {
            object_base_slot,
            hdr_srv_slot,
            bloom_srv_base_slot,
            lut_srv_slot,
            taa_srv_base_slot,
            ssao_srv_base_slot,
            ssao_white_srv_slot,
            ssr_srv_base_slot,
            decal_depth_srv_slot,
            decal_srv_base_slot,
            chunk_srv_base_slot,
            skinned_srv_base_slot,
            particle_srv_base_slot,
            clone_srv_base_slot,
            fog_froxel_uav_slot,
            fog_froxel_srv_slot,
            upscale_uav_slot,
            upscale_srv_slot,
            raymarch_srv_base_slot,
            hiz_srv_slot,
            hiz_uav_base_slot,
            transparent_scene_copy_srv_slot,
            ssgi_gi_srv_slot,
            gbuffer_srv_base_slot,
            rt_output_srv_slot,
            flat_pool_base_slot,
            srv_slots,
        } = heap_layout::SrvHeapLayout::compute(&heap_layout::SrvHeapParams {
            n_objects,
            n_clusters,
            n_atlases,
            bloom_count,
            taa_srv_extra,
            ssao_srv_extra,
            ssr_srv_extra,
            ssgi_srv_extra,
            gbuffer_srv_extra,
            rt_output_srv_extra: rt_srv_extra,
            albedo_count: flat_albedo_count,
            normal_count: flat_normal_count,
        });
        let srv_heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                // `srv_slots` is the running total of every block in the
                // heap_layout cascade, so it sizes the heap to exactly cover
                // the highest slot any descriptor write addresses.
                NumDescriptors: srv_slots as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("SRV heap: {e}"))?;
        let srv_descriptor_size = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        } as usize;
        let srv_cpu_base = unsafe { srv_heap.GetCPUDescriptorHandleForHeapStart() };
        let srv_gpu_base = unsafe { srv_heap.GetGPUDescriptorHandleForHeapStart() };

        let slot_cpu = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: srv_cpu_base.ptr + i * srv_descriptor_size,
        };
        let slot_gpu = |i: usize| D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: srv_gpu_base.ptr + (i * srv_descriptor_size) as u64,
        };

        // Temporal upscaler (FSR3 via the FidelityFX SDK). Built here,
        // ahead of the HDR / depth / post-effect targets, because its
        // resolved render dimensions decide the size of every scene target.
        // When the world's `PostProcessConfig.temporal_upscaling` is on AND
        // the FFX DLL loads + the context creates successfully, this returns
        // `Some` and the scene renders at `output * upscale_scale`; FSR
        // reconstructs the drawable resolution into the upscaler's output
        // texture, which bloom + composite sample. Falls back silently (logs
        // a warning, leaves render == output) when the SDK isn't on `PATH`
        // or the GPU rejects the context build, so a missing SDK degrades
        // to native-resolution TAA rather than a low-res bilinear stretch.
        let upscaler = if temporal_upscaling {
            crate::directx::post::upscale::build_upscaler(
                &device,
                &command_queue,
                width,
                height,
                upscale_scale,
                slot_cpu(upscale_uav_slot),
                slot_cpu(upscale_srv_slot),
                slot_gpu(upscale_srv_slot),
                upscale_backend,
            )?
            .0
        } else {
            None
        };
        // Off-screen scene render resolution. The active backend reports the
        // resolved render dims (clamped to the backend's supported ratio
        // range); a missing / failed upscaler leaves the scene at full output.
        let (render_w, render_h) = match &upscaler {
            Some(u) => u.render_dims(),
            None => (width, height),
        };
        if upscaler.is_some() {
            tracing::info!(
                "DirectX: temporal upscaling active: scene render {}x{}, drawable {}x{}",
                render_w,
                render_h,
                width,
                height
            );
        }

        // Sampler heap
        // Slots: [0]=shadow comparison, [1]=linear repeat, [2]=cube linear-clamp+mip,
        //        [3]=linear clamp (text). linear+cube are placed contiguously so the
        //        main pass binds them via a single 2-descriptor table range.
        // Slots [4..7] are the raymarch pass's contiguous descriptor table:
        // shadow comparison, cube linear-clamp, and a linear-clamp scene
        // sampler. These duplicate samplers at slots 0 / 2 so the raymarch
        // root sig can bind a single 3-slot range; the cost is three extra
        // descriptors (a few bytes). Reserved unconditionally so the heap
        // layout stays anchored.
        let raymarch_sampler_base_slot = 4usize;
        let sampler_heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                NumDescriptors: 7,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("sampler heap: {e}"))?;
        let sampler_descriptor_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) }
                as usize;
        let samp_cpu_base = unsafe { sampler_heap.GetCPUDescriptorHandleForHeapStart() };
        let samp_gpu_base = unsafe { sampler_heap.GetGPUDescriptorHandleForHeapStart() };

        create_samplers(&device, samp_cpu_base, sampler_descriptor_size);

        let shadow_sampler_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: samp_gpu_base.ptr,
        };
        let linear_sampler_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: samp_gpu_base.ptr + sampler_descriptor_size as u64,
        };
        let text_sampler_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: samp_gpu_base.ptr + (3 * sampler_descriptor_size) as u64,
        };

        // Shadow map array
        // Real path: NUM_SHADOW_CASCADES-slice Texture2DArray with per-slice DSVs.
        // Fallback: 1×1 single-slice R32_FLOAT array with value 0.0 (LESS_EQUAL
        // always passes → fully lit), declared as Texture2DArray so the shader's
        // binding type stays identical between disabled and enabled cases.
        // CSM is gated on `shadow_map_size` (from GraphicsConfig; 0 disables
        // shadows). The shadow vertex shader is engine-internal (the baked
        // SHADOW_VERT_HLSL), so an empty `shadow_bytes` override no longer means
        // "no shadows": it just selects the built-in shader. Mirrors the Metal
        // internal-shadow path.
        let effective_shadow_size = shadow_map_size;
        let (shadow_resource_opt, shadow_dsvs, shadow_srv_gpu) = if effective_shadow_size > 0 {
            let (sm, dsvs) = create_shadow_map_array(
                &device,
                effective_shadow_size,
                NUM_SHADOW_CASCADES as u32,
                shadow_dsv_base_cpu,
                dsv_descriptor_size,
                slot_cpu(0),
                slot_gpu(0),
            )?;
            (Some(sm), dsvs, slot_gpu(0))
        } else {
            let fb =
                create_fallback_shadow_array(&device, &command_queue, slot_cpu(0), slot_gpu(0))?;
            (Some(fb), Vec::new(), slot_gpu(0))
        };

        // IBL cubemaps (irradiance + prefilter)
        // When env_map_bytes is Some, deserialise the EnvironmentMap payload and
        // upload both cubes. Otherwise bind a 1×1 grey fallback for each; the
        // shader keys off prefilter_mip_count == 0 to skip IBL math.
        let env_map = if let Some(bytes) = env_map_bytes {
            let view = crate::build::environment_map::deserialise(bytes)
                .map_err(|e| format!("EnvironmentMap payload malformed: {e}"))?;
            upload_environment_map(
                &device,
                &command_queue,
                view.irradiance_face,
                view.irradiance_bytes,
                view.prefilter_face,
                &view.prefilter_mip_bytes,
                slot_cpu(1),
                slot_gpu(1),
                slot_cpu(2),
                slot_gpu(2),
            )?
        } else {
            let irradiance = create_fallback_cubemap(
                &device,
                &command_queue,
                [0.05, 0.05, 0.05, 1.0],
                slot_cpu(1),
                slot_gpu(1),
            )?;
            let prefilter = create_fallback_cubemap(
                &device,
                &command_queue,
                [0.05, 0.05, 0.05, 1.0],
                slot_cpu(2),
                slot_gpu(2),
            )?;
            EnvironmentMapTextures {
                irradiance,
                prefilter,
                prefilter_mip_count: 0,
            }
        };

        // Cache the first directional light's direction for per-frame CSM updates.
        let shadow_light_dir = if light_uniforms.num_directional > 0 {
            light_uniforms.directional[0].direction
        } else {
            // Match LightUniforms::DEFAULT.
            [-0.3, 0.85, 0.4]
        };

        // Cache the first directional light's colour * intensity for the
        // volumetric-fog encoder. The DirectX backend uploads LightUniforms
        // once at init (no runtime light mutation), so the sun colour fed
        // into the fog ray-march is fixed.
        let fog_sun_dir = shadow_light_dir;
        let fog_sun_color = if light_uniforms.num_directional > 0 {
            let l = &light_uniforms.directional[0];
            [
                l.color[0] * l.intensity,
                l.color[1] * l.intensity,
                l.color[2] * l.intensity,
            ]
        } else {
            // Match LightUniforms::DEFAULT (colour [1, 1, 1] at intensity 1.0).
            [1.0, 1.0, 1.0]
        };

        // Albedo texture pool
        // One ID3D12Resource per input texture; SRVs are written below at
        // per-object pair slots so a single texture can be referenced by many
        // objects. When no textures were declared, a single 1x1 white fallback
        // stands in so every object's albedo slot resolves to opaque white.
        let gpu_textures: Vec<ID3D12Resource> = if textures.is_empty() {
            vec![create_fallback_white_resource(&device, &command_queue)?]
        } else {
            textures
                .iter()
                .enumerate()
                .map(|(i, (w, h, px))| {
                    upload_texture_resource(&device, &command_queue, *w, *h, px)
                        .map_err(|e| format!("texture[{i}]: {e}"))
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        // Normal map pool
        // Slot 0 is the flat-normal fallback (matching Metal/Vulkan), so the
        // material map's normal_map_slot of 0 means "no normal map".
        let mut gpu_normal_maps: Vec<ID3D12Resource> = vec![create_fallback_flat_normal_resource(
            &device,
            &command_queue,
        )?];
        for (i, (w, h, px)) in normal_maps.iter().enumerate() {
            gpu_normal_maps.push(
                upload_texture_resource(&device, &command_queue, *w, *h, px)
                    .map_err(|e| format!("normal_map[{i}]: {e}"))?,
            );
        }

        // Flat deduplicated bindless pool: one SRV per distinct albedo resource
        // followed by one per distinct normal resource. The bindless main pass
        // and the RT hit shader bind this region's base and index it by a flat
        // slot (`albedo = texture_slot`, `normal = albedo_count + normal_slot`),
        // mirroring Vulkan/Metal. A shared texture resolves to ONE descriptor
        // here, unlike the per-object pairs above which bake a copy per draw.
        debug_assert_eq!(gpu_textures.len(), flat_albedo_count);
        debug_assert_eq!(gpu_normal_maps.len(), flat_normal_count);
        for (k, tex) in gpu_textures.iter().enumerate() {
            write_rgba8_srv(&device, tex, slot_cpu(flat_pool_base_slot + k));
        }
        for (k, nm) in gpu_normal_maps.iter().enumerate() {
            write_rgba8_srv(
                &device,
                nm,
                slot_cpu(flat_pool_base_slot + flat_albedo_count + k),
            );
        }

        // Per-object albedo + normal SRV pairs
        // Layout: slot object_base_slot+obj_idx*2 = albedo, +1 = normal.
        // Each object's SRVs are CreateShaderResourceView'd from the pool
        // resource selected by texture_slot / normal_map_slot, clamped to the
        // pool length so out-of-range slots fall back to the last valid entry.
        if n_objects > 0 {
            let last_tex = gpu_textures.len() - 1;
            let last_nm = gpu_normal_maps.len() - 1;
            for (obj_idx, obj) in draw_objects.iter().enumerate() {
                let albedo_idx = obj.texture_slot.min(last_tex);
                let nm_idx = obj.normal_map_slot.min(last_nm);
                let albedo_slot_idx = object_base_slot + obj_idx * 2;
                let normal_slot_idx = albedo_slot_idx + 1;
                write_rgba8_srv(
                    &device,
                    &gpu_textures[albedo_idx],
                    slot_cpu(albedo_slot_idx),
                );
                write_rgba8_srv(&device, &gpu_normal_maps[nm_idx], slot_cpu(normal_slot_idx));
            }
        }

        // Per-cluster albedo + normal SRV pairs
        // Layout: slot (object_base_slot + n_objects*2 + cluster_idx*2) = albedo, +1 = normal.
        if n_clusters > 0 {
            let last_tex = gpu_textures.len() - 1;
            let last_nm = gpu_normal_maps.len() - 1;
            let cluster_base_slot = object_base_slot + n_objects * 2;
            for (cluster_idx, cluster) in instanced_clusters.iter().enumerate() {
                let albedo_idx = cluster.texture_slot.min(last_tex);
                let nm_idx = cluster.normal_map_slot.min(last_nm);
                let albedo_slot_idx = cluster_base_slot + cluster_idx * 2;
                let normal_slot_idx = albedo_slot_idx + 1;
                write_rgba8_srv(
                    &device,
                    &gpu_textures[albedo_idx],
                    slot_cpu(albedo_slot_idx),
                );
                write_rgba8_srv(&device, &gpu_normal_maps[nm_idx], slot_cpu(normal_slot_idx));
            }
        }

        // Text atlas textures
        let atlas_base_slot = object_base_slot + n_objects * 2 + n_clusters * 2;
        let mut gpu_text_atlases: Vec<GpuResource> = Vec::new();
        let mut text_atlas_srv_gpus: Vec<D3D12_GPU_DESCRIPTOR_HANDLE> = Vec::new();
        for (i, (w, h, px)) in text_atlases.iter().enumerate() {
            let s = atlas_base_slot + i;
            let res = upload_texture(
                &device,
                &command_queue,
                *w,
                *h,
                px,
                slot_cpu(s),
                slot_gpu(s),
            )
            .map_err(|e| format!("text_atlas[{i}]: {e}"))?;
            text_atlas_srv_gpus.push(slot_gpu(s));
            gpu_text_atlases.push(res);
        }

        // Main depth buffer. Allowed as SRV so the projected-decal pass can
        // sample it to reconstruct world positions; runtime `add_decal`
        // needs this even when no decals were declared at init.
        let depth_resource = create_main_depth_texture(
            &device,
            render_w,
            render_h,
            main_dsv_cpu,
            msaa_samples,
            true,
        )?;

        // Off-screen HDR scene target
        // The main + instanced passes render linear-light HDR into this; the
        // composite pass tonemaps it onto the swapchain. RTV heap slot [FRAMES]
        // (after the back-buffer RTVs) holds its render-target view.
        let hdr_color_rtv = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: rtv_base.ptr + FRAMES * rtv_descriptor_size,
        };
        let hdr_color = create_hdr_color_target(
            &device,
            render_w,
            render_h,
            msaa_samples,
            hdr_color_rtv,
            clear_color,
        )?;
        let hdr_resolve = if msaa_samples > 1 {
            Some(create_hdr_resolve_target(&device, render_w, render_h)?)
        } else {
            None
        };
        // RTV for `hdr_resolve`: the projected-decal pass renders into the
        // resolved scene target, so it needs a render-target view. Sits in
        // the RTV heap right after the SSR RTVs. Only created when MSAA is
        // on (MSAA off uses the existing `hdr_color_rtv`).
        let hdr_resolve_rtv = if let Some(resolve) = &hdr_resolve {
            let rtv_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv_base.ptr
                    + (FRAMES
                        + 1
                        + bloom_count
                        + taa_rtv_extra
                        + ssao_rtv_extra
                        + ssr_rtv_extra
                        + ssgi_rtv_extra)
                        * rtv_descriptor_size,
            };
            unsafe {
                let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
                    Format: crate::directx::texture::HDR_FORMAT,
                    ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                    ..Default::default()
                };
                device.CreateRenderTargetView(resolve, Some(&rtv_desc), rtv_handle);
            }
            Some(rtv_handle)
        } else {
            None
        };
        // The composite pass samples the resolved target (MSAA on) or the
        // directly-rendered HDR target (MSAA off).
        write_hdr_srv(
            &device,
            hdr_resolve.as_ref().unwrap_or(&hdr_color),
            slot_cpu(hdr_srv_slot),
        );
        let hdr_srv_gpu = slot_gpu(hdr_srv_slot);

        // Projected-decal main-depth SRV: bind point t0 in the decal pass.
        // The DSV-only flag was dropped above so this is valid.
        crate::directx::decal::write_main_depth_srv(
            &device,
            &depth_resource,
            slot_cpu(decal_depth_srv_slot),
            msaa_samples,
        );
        let decal_depth_srv_gpu = slot_gpu(decal_depth_srv_slot);

        // Colour-grading LUT
        // Upload the declared `ColorLut` payload, or build a 2×2×2 identity LUT
        // so the composite pass always binds a valid Texture3D. With the
        // identity LUT the grade is a no-op at any `lut_strength`.
        let color_lut = if let Some(bytes) = color_lut_bytes {
            let (size, data) = crate::build::color_lut::deserialise(bytes)
                .map_err(|e| format!("ColorLut payload malformed: {e}"))?;
            upload_color_lut(
                &device,
                &command_queue,
                size,
                data,
                slot_cpu(lut_srv_slot),
                slot_gpu(lut_srv_slot),
            )?
        } else {
            create_fallback_color_lut(
                &device,
                &command_queue,
                slot_cpu(lut_srv_slot),
                slot_gpu(lut_srv_slot),
            )?
        };

        // Geometry buffers
        let vert_bytes_raw = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                std::mem::size_of_val(vertices),
            )
        };
        let idx_bytes_raw = unsafe {
            std::slice::from_raw_parts(
                indices.as_ptr() as *const u8,
                std::mem::size_of_val(indices),
            )
        };
        let vertex_buffer = upload_buffer(
            &device,
            &command_queue,
            vert_bytes_raw,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
        )?;
        let index_buffer = upload_buffer(
            &device,
            &command_queue,
            idx_bytes_raw,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
        )?;

        let vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: unsafe { vertex_buffer.GetGPUVirtualAddress() },
            SizeInBytes: vert_bytes_raw.len().max(4) as u32,
            StrideInBytes: std::mem::size_of::<Vertex>() as u32,
        };
        let index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: unsafe { index_buffer.GetGPUVirtualAddress() },
            SizeInBytes: idx_bytes_raw.len().max(4) as u32,
            // Static IB is u32: the `indices: &[u32]` signature is honoured
            // end-to-end. A previous half-completed migration left this as
            // R16_UINT while the byte count was already widened: the GPU then
            // read each u32 index as a pair of u16s, indexing into garbage
            // vertices and shearing every static prop's geometry.
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
        };

        // Constant buffers
        let view_ubo_size = align256(std::mem::size_of::<ViewUniforms>() as u64);
        let light_ubo_size = align256(std::mem::size_of::<LightUniforms>() as u64);
        let shadow_ubo_size = align256(std::mem::size_of::<ShadowUniforms>() as u64);

        let mut view_ubo_resources = Vec::with_capacity(FRAMES);
        let mut view_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = create_buffer(
                &device,
                view_ubo_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map view ubo: {e}"))?;
            view_ubo_ptrs.push(ptr as *mut u8);
            view_ubo_resources.push(buf);
        }

        let light_ubo = create_buffer(
            &device,
            light_ubo_size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;

        // Triple-buffer the shadow UBO since cascade VPs are recomputed each
        // frame from the camera. Persistently mapped.
        let mut shadow_ubo_resources = Vec::with_capacity(FRAMES);
        let mut shadow_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = create_buffer(
                &device,
                shadow_ubo_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map shadow ubo: {e}"))?;
            shadow_ubo_ptrs.push(ptr as *mut u8);
            shadow_ubo_resources.push(buf);
        }

        let shadow_uniforms = crate::gfx::csm::empty_shadow_uniforms();
        // Seed every frame's shadow UBO with the empty uniforms; per-frame
        // compute_shadow_uniforms in record_frame overwrites them.
        for ptr in &shadow_ubo_ptrs {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &shadow_uniforms as *const ShadowUniforms as *const u8,
                    *ptr,
                    std::mem::size_of::<ShadowUniforms>(),
                );
            }
        }
        upload_light_uniforms(&light_ubo, &light_uniforms)?;

        // Shaders + root sigs + PSOs (main / shadow / instanced / text /
        // composite + bindless static main + GPU-cull compute). See
        // init/pipelines.rs.
        let need_instanced = !instanced_clusters.is_empty();
        // Total instances across all clusters, folded into the GPU-driven bindless
        // pass as `GpuObjectData` records after the `n_objects` static objects.
        let n_instances: usize = instanced_clusters.iter().map(|c| c.instances.len()).sum();
        let shaders = pipelines::compile_all_shaders(
            vert_bytes,
            frag_bytes,
            shadow_bytes,
            vert_instanced_bytes,
            need_instanced,
            hot_reload,
        )?;

        let main_pipelines = pipelines::build_main_pipelines(
            &device,
            info_queue.as_ref(),
            &shaders,
            vert_bytes,
            frag_bytes,
            msaa_samples,
            n_objects,
            n_instances,
            n_skinned,
            n_chunk_max,
            occlusion_two_pass,
            effective_shadow_size > 0,
            gbuffer_enabled,
            hot_reload,
        )?;
        let pipelines::MainPipelines {
            main_root_sig,
            main_pso,
            main_bindless_root_sig,
            main_bindless_pso,
            object_buffer_resources,
            object_buffer_ptrs,
            cull_root_sig,
            cull_pso,
            cull_pso_phase2,
            cull_command_signature,
            draw_args_buffer_resources,
            draw_args_buffer_ptrs,
            indirect_cmd_buffers,
            cull_status_buffers,
            indirect_cmd_buffers_2,
            shadow_bindless_root_sig,
            shadow_bindless_pso,
            shadow_bindless_cmd_sig,
            cull_pso_shadow,
            shadow_indirect_buffers,
            shadow_cull_status_buffers,
            gbuffer_bindless_root_sig,
            gbuffer_bindless_pso,
            gbuffer_bindless_cmd_sig,
            prev_model_buffer_resources,
            prev_model_buffer_ptrs,
        } = main_pipelines;

        // GPU-driven instanced merge: write each instance's `GpuObjectData` record
        // (+ `GpuDrawArgs`) once into every frame buffer, after the `n_objects`
        // static records. Instances are placed at world load and never move, so
        // these records are static -- the per-frame static fill (`build_object_buffer`
        // / `build_draw_args_buffer`) writes only `[0, n_objects)`, leaving the
        // instance tail intact. Only runs when the bindless cull buffers exist (the
        // bindless pass is active with build-time geometry) and the world declares
        // instanced props.
        if n_instances > 0 && !object_buffer_ptrs.is_empty() {
            use crate::gfx::render_types::{
                GpuDrawArgs, GpuObjectData, draw_args_flags, instance_object_records,
            };
            let records = instance_object_records(
                &instanced_clusters,
                flat_albedo_count as u32,
                flat_normal_count as u32,
            );
            // Cluster base index range (cluster indices are absolute, so
            // base_vertex = 0); per-instance LOD is a follow-up. Every instance is
            // visible + resident + cullable, so its finite per-instance world AABB
            // is frustum/distance/Hi-Z tested independently by the cull kernel.
            let mut draw_args: Vec<GpuDrawArgs> = Vec::with_capacity(records.len());
            for cluster in &instanced_clusters {
                for _ in &cluster.instances {
                    draw_args.push(GpuDrawArgs {
                        index_count: cluster.index_count as u32,
                        index_offset: cluster.index_offset as u32,
                        base_vertex: 0,
                        flags: draw_args_flags(true, true, true),
                    });
                }
            }
            let obj_stride = std::mem::size_of::<GpuObjectData>();
            let da_stride = std::mem::size_of::<GpuDrawArgs>();
            for (obj_ptr, da_ptr) in object_buffer_ptrs.iter().zip(draw_args_buffer_ptrs.iter()) {
                // SAFETY: the buffers were sized for `n_objects + n_instances`
                // records, so writing `records.len()` past the `n_objects` offset
                // stays in bounds.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        records.as_ptr() as *const u8,
                        obj_ptr.add(n_objects * obj_stride),
                        records.len() * obj_stride,
                    );
                    std::ptr::copy_nonoverlapping(
                        draw_args.as_ptr() as *const u8,
                        da_ptr.add(n_objects * da_stride),
                        draw_args.len() * da_stride,
                    );
                }
            }

            // GPU-driven G-buffer velocity: the instance region of the parallel
            // `prev_model` buffer is the instances' current models (immutable, so
            // motion is camera-only). Written once into every frame buffer after
            // the static prefix, exactly like the instance object records; the
            // per-frame `build_gbuffer_prev_models` fill writes only the static +
            // skinned regions, leaving this intact. A no-op when the G-buffer path
            // is inactive (the buffers were not allocated).
            if !prev_model_buffer_ptrs.is_empty() {
                let models: Vec<[[f32; 4]; 4]> = records.iter().map(|r| r.model).collect();
                let m_stride = std::mem::size_of::<[[f32; 4]; 4]>();
                for pm_ptr in prev_model_buffer_ptrs.iter() {
                    // SAFETY: the prev_model buffer was sized for
                    // `n_objects + n_instances + n_skinned` records, so writing
                    // `models.len()` past the `n_objects` offset stays in bounds.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            models.as_ptr() as *const u8,
                            pm_ptr.add(n_objects * m_stride),
                            models.len() * m_stride,
                        );
                    }
                }
            }
        }

        // The bindless texture pool's first slot is the flat deduplicated pool
        // base; pool index `texture_slot` lands on the albedo SRV and
        // `albedo_count + normal_slot` on the normal SRV. The bindless main pass
        // and the RT hit shader bind from here.
        let bindless_pool_gpu = slot_gpu(flat_pool_base_slot);

        // Only build the shadow PSO when shadows are enabled; the shadow pass
        // keys off `shadow_pso.is_some()`, so passing `None` when
        // `effective_shadow_size == 0` keeps a shadow-disabled world from
        // rendering into nonexistent cascade DSVs.
        let shadow_vs_for_pso = if effective_shadow_size > 0 {
            shaders.shadow_vs.as_deref()
        } else {
            None
        };
        let (shadow_root_sig, shadow_pso) =
            pipelines::build_shadow_pipeline(&device, info_queue.as_ref(), shadow_vs_for_pso)?;

        let (main_instanced_root_sig, main_instanced_pso) =
            pipelines::build_main_instanced_pipeline(
                &device,
                info_queue.as_ref(),
                shaders.main_vs_instanced.as_deref(),
                &shaders.main_ps,
                msaa_samples,
            )?;

        let (text_root_sig, text_pso) = pipelines::build_text_pipeline(
            &device,
            info_queue.as_ref(),
            &shaders.text_vs,
            &shaders.text_ps,
            swapchain_format,
            !text_atlases.is_empty(),
        )?;

        let (composite_root_sig, composite_pso) = pipelines::build_composite_pipeline(
            &device,
            info_queue.as_ref(),
            swapchain_format,
            hot_reload,
        )?;

        // Bloom mips + bloom PSOs + TAA + SSAO. See init/effects.rs.
        let bloom_rtv_for = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: rtv_base.ptr + (FRAMES + 1 + i) * rtv_descriptor_size,
        };
        let bloom_srv_cpu_for = |i: usize| slot_cpu(bloom_srv_base_slot + i);
        let bloom_srv_gpu_for = |i: usize| slot_gpu(bloom_srv_base_slot + i);

        let taa_rtv_for = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: rtv_base.ptr + (FRAMES + 1 + bloom_count + i) * rtv_descriptor_size,
        };
        let taa_slots = effects::TaaSlots {
            history_rtv: [taa_rtv_for(0), taa_rtv_for(1)],
            history_srv: [
                (slot_cpu(taa_srv_base_slot), slot_gpu(taa_srv_base_slot)),
                (
                    slot_cpu(taa_srv_base_slot + 1),
                    slot_gpu(taa_srv_base_slot + 1),
                ),
            ],
        };

        let ssr_rtv_for = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: rtv_base.ptr
                + (FRAMES + 1 + bloom_count + taa_rtv_extra + ssao_rtv_extra + i)
                    * rtv_descriptor_size,
        };
        let ssr_slots = effects::SsrSlots {
            output_rtv: ssr_rtv_for(0),
            output_srv: (slot_cpu(ssr_srv_base_slot), slot_gpu(ssr_srv_base_slot)),
        };

        // SSGI gather target: RTV right after the SSR RTVs, SRV at the heap tail.
        let ssgi_gi_rtv = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: rtv_base.ptr
                + (FRAMES + 1 + bloom_count + taa_rtv_extra + ssao_rtv_extra + ssr_rtv_extra)
                    * rtv_descriptor_size,
        };
        let ssgi_slots = effects::SsgiSlots {
            gi_rtv: ssgi_gi_rtv,
            gi_srv: (slot_cpu(ssgi_gi_srv_slot), slot_gpu(ssgi_gi_srv_slot)),
        };

        let ssao_rtv_for = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: rtv_base.ptr
                + (FRAMES + 1 + bloom_count + taa_rtv_extra + i) * rtv_descriptor_size,
        };
        let ssao_slots = effects::SsaoSlots {
            ao_raw_rtv: ssao_rtv_for(0),
            ao_raw_srv: (slot_cpu(ssao_srv_base_slot), slot_gpu(ssao_srv_base_slot)),
            ao_rtv: ssao_rtv_for(1),
            ao_srv: (
                slot_cpu(ssao_srv_base_slot + 1),
                slot_gpu(ssao_srv_base_slot + 1),
            ),
            white_srv: (slot_cpu(ssao_white_srv_slot), slot_gpu(ssao_white_srv_slot)),
        };

        // RT-reflection output target: RTV right after the gbuffer RTVs (the
        // last RTV block), SRV at the SRV-heap tail.
        let rt_output_rtv = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: rtv_base.ptr
                + (FRAMES
                    + 1
                    + bloom_count
                    + taa_rtv_extra
                    + ssao_rtv_extra
                    + ssr_rtv_extra
                    + ssgi_rtv_extra
                    + decal_rtv_extra
                    + gbuffer_rtv_extra)
                    * rtv_descriptor_size,
        };
        let rt_slots = effects::RtReflectionsSlots {
            output_rtv: rt_output_rtv,
            output_srv: (slot_cpu(rt_output_srv_slot), slot_gpu(rt_output_srv_slot)),
        };

        // Unified G-buffer pre-pass descriptor slots (always reserved). Minted
        // here so both the conditional init build below and the runtime
        // `apply_quality_settings` rebuild use the same fixed slots.
        let gb_rtv_base = FRAMES
            + 1
            + bloom_count
            + taa_rtv_extra
            + ssao_rtv_extra
            + ssr_rtv_extra
            + ssgi_rtv_extra
            + decal_rtv_extra;
        let gb_rtv = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: rtv_base.ptr + (gb_rtv_base + i) * rtv_descriptor_size,
        };
        let gbuffer_slots = crate::directx::post::gbuffer::GbufferSlots {
            normal_depth_rtv: gb_rtv(0),
            normal_depth_srv: (
                slot_cpu(gbuffer_srv_base_slot),
                slot_gpu(gbuffer_srv_base_slot),
            ),
            roughness_rtv: gb_rtv(1),
            roughness_srv: (
                slot_cpu(gbuffer_srv_base_slot + 1),
                slot_gpu(gbuffer_srv_base_slot + 1),
            ),
            velocity_rtv: gb_rtv(2),
            velocity_srv: (
                slot_cpu(gbuffer_srv_base_slot + 2),
                slot_gpu(gbuffer_srv_base_slot + 2),
            ),
            depth_dsv: D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: dsv_base.ptr + (1 + NUM_SHADOW_CASCADES) * dsv_descriptor_size,
            },
        };

        // Stash the live-toggleable effects' fixed slots so the runtime
        // `apply_quality_settings` can build a launched-off feature into its slot
        // without re-deriving the heap layout. Copied from the per-effect slot
        // structs before they move into `build_effects` below.
        let quality_slots = super::quality::QualitySlotHandles {
            taa_history_rtv: taa_slots.history_rtv,
            taa_history_srv: taa_slots.history_srv,
            ssao_ao_raw_rtv: ssao_slots.ao_raw_rtv,
            ssao_ao_raw_srv: ssao_slots.ao_raw_srv,
            ssao_ao_rtv: ssao_slots.ao_rtv,
            ssao_ao_srv: ssao_slots.ao_srv,
            ssr_output_rtv: ssr_slots.output_rtv,
            ssr_output_srv: ssr_slots.output_srv,
            ssgi_gi_rtv: ssgi_slots.gi_rtv,
            ssgi_gi_srv: ssgi_slots.gi_srv,
            rt_output_rtv: rt_slots.output_rtv,
            rt_output_srv: rt_slots.output_srv,
            gbuffer: gbuffer_slots,
        };

        let effects_bundle = effects::build_effects(
            &device,
            &command_queue,
            info_queue.as_ref(),
            width,
            height,
            render_w,
            render_h,
            taa_enabled,
            ssao_settings,
            ssr_settings,
            ssgi_settings,
            rt_reflection_settings,
            raytracing_supported,
            effects::BloomSlots {
                rtv_for: &bloom_rtv_for,
                srv_cpu_for: &bloom_srv_cpu_for,
                srv_gpu_for: &bloom_srv_gpu_for,
            },
            taa_slots,
            ssao_slots,
            ssr_slots,
            ssgi_slots,
            rt_slots,
            hot_reload,
        )?;
        let effects::EffectsBundle {
            transient_pool,
            bloom_mips,
            bloom_mip_rtvs,
            bloom_mip_srv_gpus,
            bloom_mip_extents,
            bloom_root_sig,
            bloom_pso_prefilter,
            bloom_pso_downsample,
            bloom_pso_upsample,
            taa,
            ssao,
            ssao_white,
            ssao_white_srv_gpu,
            ssr,
            ssgi,
            rt_reflections,
        } = effects_bundle;

        // Unified G-buffer pre-pass resources. Built whenever any screen-space
        // consumer drives it (see `gbuffer_enabled`). Its three MRT RTVs sit at
        // the tail of the RTV heap (after the decal RTV), its private depth DSV
        // right after the shadow DSVs, and its three SRVs in the reserved
        // `gbuffer_srv_base_slot` block. The skinned PSO builds lazily in
        // `upload_skinned` once the joint-bound vertex layout exists.
        let gbuffer = if gbuffer_enabled {
            Some(crate::directx::post::gbuffer::GbufferResources::new(
                &device,
                render_w,
                render_h,
                need_instanced,
                false,
                gbuffer_slots,
                info_queue.as_ref(),
                hot_reload,
            )?)
        } else {
            None
        };

        // Projected decals: pipeline + unit-cube buffers + per-frame
        // uniform rings. Always built so runtime `add_decal` works from a
        // world that started with none; pre-authored decals get their albedo
        // SRV written below.
        let decals_state = Some(crate::directx::decal::DecalResources::new(
            &device,
            &command_queue,
            msaa_samples,
            decal_srv_base_slot,
            decal_depth_srv_gpu,
            info_queue.as_ref(),
            hot_reload,
        )?);
        // Pre-authored decals: write each one's albedo SRV into its reserved
        // heap slot. Runtime adds via `DxContext::add_decal` follow the same
        // pattern.
        if decals.len() > crate::directx::decal::MAX_DECALS {
            return Err(format!(
                "decals: {} authored decals exceed MAX_DECALS ({})",
                decals.len(),
                crate::directx::decal::MAX_DECALS
            ));
        }
        let last_tex = gpu_textures.len().saturating_sub(1);
        for (i, rec) in decals.iter().enumerate() {
            let tex_idx = rec.texture_slot.min(last_tex);
            write_rgba8_srv(
                &device,
                &gpu_textures[tex_idx],
                slot_cpu(decal_srv_base_slot + i),
            );
        }
        let decals_init: Vec<Option<crate::gfx::decal::DecalRecord>> =
            decals.into_iter().map(Some).collect();

        // Volumetric fog: pipeline + per-frame uniform ring. Built only when
        // the world declared a `VolumetricFog`; the encoder simply skips the
        // pass when `fog_settings` is `None`. The fog pass shares the main-
        // depth SRV that the decal-init path already wrote into the heap.
        let fog_resources = if fog_settings.is_some() {
            Some(crate::directx::fog::FogResources::new(
                &device,
                msaa_samples,
                decal_depth_srv_gpu,
                shadow_srv_gpu,
                slot_cpu(fog_froxel_uav_slot),
                slot_gpu(fog_froxel_uav_slot),
                slot_cpu(fog_froxel_srv_slot),
                slot_gpu(fog_froxel_srv_slot),
                info_queue.as_ref(),
                hot_reload,
            )?)
        } else {
            None
        };

        // (The FSR3 temporal upscaler is built earlier; its resolved render
        // dimensions decide the scene-target sizes used above.)

        // Particles: compute + render pipelines + per-frame uniform rings,
        // plus one persistent GPU pool per emitter. Built only when the world
        // declared ≥1 emitter; the encoder skips the passes when
        // `particle_resources` is `None`, and runtime `add_emitter` builds the
        // pipelines lazily the same way. The emitter cap matches the SRV-heap
        // reservation made above.
        if particles.len() > crate::directx::particle::MAX_EMITTERS {
            return Err(format!(
                "particles: {} authored emitters exceed MAX_EMITTERS ({})",
                particles.len(),
                crate::directx::particle::MAX_EMITTERS
            ));
        }
        let (particle_resources, particle_records, particle_emitter_states) =
            if !particles.is_empty() {
                let resources = crate::directx::particle::ParticleResources::new(
                    &device,
                    particle_srv_base_slot,
                    info_queue.as_ref(),
                    hot_reload,
                )?;
                let mut states: Vec<Option<crate::directx::particle::ParticleEmitterGpuState>> =
                    Vec::with_capacity(particles.len());
                let last_tex = gpu_textures.len().saturating_sub(1);
                for (i, rec) in particles.iter().enumerate() {
                    let state = crate::directx::particle::build_emitter_gpu_state(
                        &device,
                        &command_queue,
                        rec,
                    )?;
                    states.push(Some(state));
                    // Write the per-emitter albedo SRV into its reserved heap slot.
                    let tex_idx = rec.texture_slot.min(last_tex);
                    write_rgba8_srv(
                        &device,
                        &gpu_textures[tex_idx],
                        slot_cpu(particle_srv_base_slot + i),
                    );
                }
                let recs: Vec<Option<crate::gfx::particles::ParticleEmitterRecord>> =
                    particles.into_iter().map(Some).collect();
                (Some(resources), recs, states)
            } else {
                (None, Vec::new(), Vec::new())
            };

        // Per-frame command infrastructure
        let mut command_allocators = Vec::with_capacity(FRAMES);
        let mut command_lists: Vec<ID3D12GraphicsCommandList> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let alloc: ID3D12CommandAllocator =
                unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                    .map_err(|e| format!("command allocator: {e}"))?;
            let list: ID3D12GraphicsCommandList = unsafe {
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            }
            .map_err(|e| format!("command list: {e}"))?;
            // Close immediately; we re-open each frame.
            unsafe { list.Close() }.map_err(|e| format!("close cmd list: {e}"))?;
            command_allocators.push(alloc);
            command_lists.push(list);
        }

        // Per-pass command allocator + cmd list pool for the parallel-
        // encoding path. Sized FRAMES * PASS_COUNT so each pass owns its
        // own allocator + cmd list per in-flight slot; workers reset
        // their own allocator + cmd list before recording, so multiple
        // workers can encode in parallel without contending. Allocators
        // are very lightweight (a few KB of CPU-side bookkeeping each);
        // a 21-pass × 3-frame pool is ~63 entries.
        let pass_pool_size = FRAMES * crate::gfx::render_graph::PASS_COUNT;
        let mut pass_allocators: Vec<ID3D12CommandAllocator> = Vec::with_capacity(pass_pool_size);
        let mut pass_cmd_lists: Vec<ID3D12GraphicsCommandList> = Vec::with_capacity(pass_pool_size);
        for _ in 0..pass_pool_size {
            let alloc: ID3D12CommandAllocator =
                unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                    .map_err(|e| format!("per-pass command allocator: {e}"))?;
            let list: ID3D12GraphicsCommandList = unsafe {
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            }
            .map_err(|e| format!("per-pass command list: {e}"))?;
            // Close immediately; we re-open per-pass each frame as needed.
            unsafe { list.Close() }.map_err(|e| format!("close per-pass cmd list: {e}"))?;
            pass_allocators.push(alloc);
            pass_cmd_lists.push(list);
        }

        // End-of-frame outer cmd list pair (composite + final timestamp +
        // resolve). Submitted last so its `ResolveQueryData` reads every
        // per-pass `EndQuery` write.
        let mut end_command_allocators: Vec<ID3D12CommandAllocator> = Vec::with_capacity(FRAMES);
        let mut end_command_lists: Vec<ID3D12GraphicsCommandList> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let alloc: ID3D12CommandAllocator =
                unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                    .map_err(|e| format!("end command allocator: {e}"))?;
            let list: ID3D12GraphicsCommandList = unsafe {
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            }
            .map_err(|e| format!("end command list: {e}"))?;
            unsafe { list.Close() }.map_err(|e| format!("close end cmd list: {e}"))?;
            end_command_allocators.push(alloc);
            end_command_lists.push(list);
        }

        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
            .map_err(|e| format!("create fence: {e}"))?;
        let fence_event = unsafe { CreateEventW(None, false, false, None) }
            .map_err(|e| format!("create fence event: {e}"))?;
        let fence_values = vec![0u64; FRAMES];

        // Timestamp infrastructure for the per-frame GPU time chip. Falls back
        // to `None`s with frequency 0 when the queue does not support
        // timestamps (every WDDM 2.0+ direct queue does, but the fallback keeps
        // the rest of the overlay working on adapters that don't).
        let (timestamp_query_heap, timestamp_readback, timestamp_readback_ptr, timestamp_frequency) =
            crate::directx::context::build_timestamp_resources(&device, &command_queue);

        // Shader hot-reload wiring. The atomic flag is shared between the
        // notify watcher thread and `draw_frame`, plus the debug WS
        // `reload-shaders` command path via `GraphicsSystem`. Watcher
        // creation is best-effort: a missing source dir or a notify error
        // logs a warning and disables only the watcher half -- the debug
        // command still works on the same flag.
        let (shader_reload_pending, shader_watcher) = if hot_reload {
            let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let watcher = crate::directx::hot_reload::spawn(std::sync::Arc::clone(&flag));
            (Some(flag), watcher)
        } else {
            (None, None)
        };

        let (cull_bvh, always_draw) = crate::gfx::bvh::partition_draw_objects(&draw_objects);

        // Per-frame instance upload buffers. One persistently-mapped buffer
        // per (frame, cluster). Sized to hold cluster.instances.len() float4x4
        // matrices, which is fixed at init time.
        let mut instance_upload_buffers: Vec<Vec<ID3D12Resource>> = Vec::with_capacity(FRAMES);
        let mut instance_upload_ptrs: Vec<Vec<*mut u8>> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let mut frame_bufs: Vec<ID3D12Resource> = Vec::with_capacity(instanced_clusters.len());
            let mut frame_ptrs: Vec<*mut u8> = Vec::with_capacity(instanced_clusters.len());
            for cluster in &instanced_clusters {
                let bytes =
                    (cluster.instances.len().max(1) * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
                let buf = create_buffer(
                    &device,
                    bytes,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )
                .map_err(|e| format!("instance upload buf: {e}"))?;
                let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
                unsafe {
                    buf.Map(0, None, Some(&mut ptr))
                        .map_err(|e| format!("map instance buf: {e}"))?;
                }
                frame_bufs.push(buf);
                frame_ptrs.push(ptr as *mut u8);
            }
            instance_upload_buffers.push(frame_bufs);
            instance_upload_ptrs.push(frame_ptrs);
        }

        // Auto-exposure: build the histogram + average compute pipelines plus
        // the GPU buffers (histogram UAV, output UAV, per-frame readback)
        // only when the world's PostProcessConfig opted in. With auto-exposure
        // off every path below is None and the static authored EV continues
        // to drive `post_process.exposure` unchanged.
        let (auto_exposure, auto_exposure_state) =
            if let Some(settings) = auto_exposure_settings.as_ref() {
                let resources = dump_on_err(
                    info_queue.as_ref(),
                    crate::directx::auto_exposure::AutoExposureResources::new(&device, hot_reload),
                )?;
                let state = crate::gfx::auto_exposure::AutoExposureState::new(settings);
                (Some(resources), Some(state))
            } else {
                (None, None)
            };

        // Raymarched SDF volumes. Builds per-volume PSOs from `.hlsl`
        // payloads and writes the raymarch SRV + sampler tables into
        // their reserved blocks. `.metal` payloads are filtered out
        // inside `try_new` with a logged warning; if every volume is
        // Metal-first (the current showcase shape), this returns `None`
        // and the render graph never adds `PassId::Raymarch`. The
        // shadow + IBL handles passed here mirror the matching slot-0/1/2
        // bindings the main pass uses, so raymarched surfaces sample the
        // same CSM cascades + IBL cubes as rasterised geometry.
        let raymarch = crate::directx::raymarch::RaymarchResources::try_new(
            &device,
            info_queue.as_ref(),
            &command_queue,
            &sdf_volumes,
            render_w,
            render_h,
            msaa_samples,
            shadow_resource_opt.as_ref().map(|r| &r.resource),
            NUM_SHADOW_CASCADES as u32,
            &env_map.irradiance.resource,
            &env_map.prefilter.resource,
            slot_cpu(raymarch_srv_base_slot),
            slot_gpu(raymarch_srv_base_slot),
            srv_descriptor_size,
            D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: samp_cpu_base.ptr + raymarch_sampler_base_slot * sampler_descriptor_size,
            },
            D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: samp_gpu_base.ptr
                    + (raymarch_sampler_base_slot * sampler_descriptor_size) as u64,
            },
            sampler_descriptor_size,
            hot_reload,
        )?;

        // Hi-Z pyramid. Built under the same condition as the cull pipeline
        // (bindless main pass active + build-time static geometry). The
        // resource owns its descriptors at the reserved Hi-Z heap slots;
        // when the gating condition fails the slots stay empty and the
        // cull kernel's `hiz_enabled` flag stays zero so it never samples
        // them. The init kernel reads the main-depth SRV that the decal +
        // fog passes already wrote; `decal_depth_srv_gpu` carries the
        // matching GPU handle.
        let hiz = if cull_pso.is_some() {
            let mut mip_uav_cpus: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> =
                Vec::with_capacity(HIZ_MAX_MIPS);
            let mut mip_uav_gpus: Vec<D3D12_GPU_DESCRIPTOR_HANDLE> =
                Vec::with_capacity(HIZ_MAX_MIPS);
            for i in 0..HIZ_MAX_MIPS {
                mip_uav_cpus.push(slot_cpu(hiz_uav_base_slot + i));
                mip_uav_gpus.push(slot_gpu(hiz_uav_base_slot + i));
            }
            Some(crate::directx::hiz::HiZResources::new(
                &device,
                info_queue.as_ref(),
                render_w,
                render_h,
                slot_cpu(hiz_srv_slot),
                slot_gpu(hiz_srv_slot),
                slot_cpu(decal_depth_srv_slot),
                decal_depth_srv_gpu,
                mip_uav_cpus,
                mip_uav_gpus,
                hot_reload,
            )?)
        } else {
            None
        };

        // Translucent glass panels: the generic producer for the shared
        // transparent pass. `Some` only when the world declared any
        // `GlassPanel`. Shares the main-depth SRV with the decal pass; the
        // scene-copy snapshot uses its own reserved heap slot.
        let glass = if glass_panels.is_empty() {
            None
        } else {
            Some(crate::directx::glass::GlassResources::new(
                &device,
                &command_queue,
                msaa_samples,
                &glass_panels,
                slot_cpu(transparent_scene_copy_srv_slot),
                slot_gpu(transparent_scene_copy_srv_slot),
                decal_depth_srv_gpu,
                render_w,
                render_h,
                info_queue.as_ref(),
                hot_reload,
            )?)
        };

        // Hardware-RT acceleration structure. Built once over the shared static
        // vertex/index buffers + the draw-object / cluster lists, only when the
        // RT reflection resources came up (DXR-capable GPU + DXC compile OK).
        // `Ok(None)` means an empty scene; an `Err` is non-fatal (logged, falls
        // back to SSR). `rt_reflections_active` gates the RT pass on both this
        // and the resources being `Some`. The init build is static-only; skinned
        // meshes are seeded into the BVH on the first dynamic frame
        // (`rebuild_skinned`), so the compute-skinning pipeline is built here and
        // attached. A skin-pipeline build failure (DXC absent) is non-fatal: the
        // RT pass still runs for static geometry, just without skinned hits.
        let rt_accel = if rt_reflections.is_some() {
            match super::raytrace::build_rt_accel(
                &device,
                &command_queue,
                &vertex_buffer,
                &index_buffer,
                &draw_objects,
                &instanced_clusters,
                vertices.len(),
                flat_albedo_count as u32,
                flat_normal_count as u32,
            ) {
                Ok(Some(mut accel)) => {
                    match super::raytrace::build_rt_skin_pipeline(&device, hot_reload) {
                        Ok(skin) => accel.set_skin_pipeline(skin),
                        Err(e) => tracing::warn!(
                            "RT skin pipeline build failed (skinned meshes absent from reflections): {e}"
                        ),
                    }
                    Some(accel)
                }
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!(
                        "RT acceleration-structure build failed, falling back to SSR: {e}"
                    );
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            win_state,
            device,
            command_queue,
            swapchain,
            back_buffers,
            rtv_heap,
            rtv_descriptor_size,
            hdr: super::context::HdrState {
                color: hdr_color,
                color_rtv: hdr_color_rtv,
                resolve: hdr_resolve,
                resolve_rtv: hdr_resolve_rtv,
                srv_gpu: hdr_srv_gpu,
                msaa_samples,
            },
            render_width: render_w,
            render_height: render_h,
            output_width: width,
            output_height: height,
            upscale: super::context::UpscaleState {
                backend: upscaler,
                requested: upscale_backend,
                jitter: std::cell::Cell::new([0.0, 0.0]),
                prev_elapsed: std::cell::Cell::new(0.0),
            },
            depth_dsv: main_dsv_cpu,
            depth_resource,
            dsv_heap,
            shadow: super::context::ShadowState {
                resource: shadow_resource_opt,
                dsvs: shadow_dsvs,
                map_size: effective_shadow_size,
                srv_gpu: shadow_srv_gpu,
                light_dir: shadow_light_dir,
                update: shadow_update,
                scheduler: Default::default(),
                render_mask: 0,
                uniforms: crate::gfx::csm::empty_shadow_uniforms(),
            },
            env_map,
            color_lut,
            descriptors: DxDescriptors {
                srv_heap,
                srv_descriptor_size,
                flat_pool_base_slot,
                sampler_heap,
                shadow_sampler_gpu,
                linear_sampler_gpu,
                text_sampler_gpu,
                textures: gpu_textures,
                normal_map_textures: gpu_normal_maps,
                text_atlas_textures: gpu_text_atlases,
                text_atlas_srv_gpus,
            },
            n_objects,
            n_instances,
            // Streamed-chunk record reserve (fixed at init = the worst-case
            // resident chunk window). The cull buffers reserve
            // `[n_objects + n_instances, +n_chunk)`; resident chunks are folded in
            // per frame and the unused tail is disabled. 0 for a non-voxel world.
            n_chunk: n_chunk_max,
            // Set in `upload_skinned` once skinned geometry is resident; the cull
            // buffers reserve the tail at init via the threaded `n_skinned`
            // capacity, but `cull_count()` reads this runtime count.
            n_skinned: 0,
            n_clusters,
            geometry: DxGeometry {
                vertex_buffer,
                index_buffer,
                vertex_buffer_view,
                index_buffer_view,
            },
            mesh_stream: super::context::MeshStreamState {
                vtx_alloc: crate::gfx::range_alloc::RangeAllocator::new(),
                idx_alloc: crate::gfx::range_alloc::RangeAllocator::new(),
            },
            chunk_stream: super::context::ChunkStreamState {
                vtx_alloc: crate::gfx::range_alloc::RangeAllocator::new(),
                idx_alloc: crate::gfx::range_alloc::RangeAllocator::new(),
                free_slots: Vec::new(),
                srv_base_slot: chunk_srv_base_slot,
            },
            skinned: SkinnedState {
                pso: None,
                root_sig: None,
                shadow_pso: None,
                shadow_root_sig: None,
                vertex_buffer: None,
                index_buffer: None,
                vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW::default(),
                index_buffer_view: D3D12_INDEX_BUFFER_VIEW::default(),
                draw_objects: Vec::new(),
                joint_buffers: Vec::new(),
                joint_ptrs: Vec::new(),
                joint_matrices: Vec::new(),
                srv_base_slot: skinned_srv_base_slot,
                skin_pipeline: None,
                deformed_primed: std::sync::atomic::AtomicBool::new(false),
                deformed_buffers: Vec::new(),
                deformed_vbvs: Vec::new(),
            },
            uniforms: DxUniforms {
                view_ubo_resources,
                view_ubo_ptrs,
                light_ubo,
                light_uniforms,
                shadow_ubo_resources,
                shadow_ubo_ptrs,
            },
            main_root_sig,
            main_pso,
            cull: CullState {
                main_bindless_root_sig,
                main_bindless_pso,
                object_buffer_resources,
                object_buffer_ptrs,
                bindless_pool_gpu,
                cull_root_sig,
                cull_pso,
                cull_pso_phase2,
                cull_command_signature,
                draw_args_buffer_resources,
                draw_args_buffer_ptrs,
                indirect_cmd_buffers,
                cull_status_buffers,
                indirect_cmd_buffers_2,
                shadow_bindless_root_sig,
                shadow_bindless_pso,
                shadow_bindless_cmd_sig,
                cull_pso_shadow,
                shadow_indirect_buffers,
                shadow_cull_status_buffers,
                gbuffer_bindless_root_sig,
                gbuffer_bindless_pso,
                gbuffer_bindless_cmd_sig,
                prev_model_buffers: prev_model_buffer_resources,
                prev_model_buffer_ptrs,
                occlusion_two_pass,
                hiz,
                prev_view_proj: std::cell::Cell::new(IDENTITY4),
                hiz_valid: std::cell::Cell::new(false),
            },
            shadow_root_sig,
            shadow_pso,
            text_root_sig,
            text_pso,
            composite_root_sig,
            composite_pso,
            bloom: BloomState {
                mips: bloom_mips,
                mip_rtvs: bloom_mip_rtvs,
                mip_srv_gpus: bloom_mip_srv_gpus,
                mip_extents: bloom_mip_extents,
                root_sig: bloom_root_sig,
                pso_prefilter: bloom_pso_prefilter,
                pso_downsample: bloom_pso_downsample,
                pso_upsample: bloom_pso_upsample,
            },
            post_process,
            gbuffer,
            taa,
            ssao: super::context::SsaoState {
                resources: ssao,
                white: ssao_white,
                white_srv_gpu: ssao_white_srv_gpu,
            },
            transient_pool,
            ssr,
            ssgi,
            rt_reflections,
            rt_accel,
            rt_dynamic_mode,
            decal: super::context::DecalState {
                state: decals_state,
                records: decals_init,
                free_slots: Vec::new(),
            },
            raymarch,
            glass,
            fog: super::context::FogState {
                resources: fog_resources,
                settings: fog_settings,
                sun_dir: fog_sun_dir,
                sun_color: fog_sun_color,
            },
            particle: super::context::ParticleState {
                resources: particle_resources,
                records: particle_records,
                emitter_state: particle_emitter_states,
                free_slots: Vec::new(),
                srv_base_slot: particle_srv_base_slot,
                last_elapsed: std::cell::Cell::new(0.0),
                frame_index: std::cell::Cell::new(0),
            },
            clone: super::context::CloneState {
                srv_base_slot: clone_srv_base_slot,
                count: 0,
                slot_by_draw_idx: std::collections::HashMap::new(),
            },
            commands: DxCommands {
                command_allocators,
                command_lists,
                pass_allocators,
                pass_cmd_lists,
                end_command_allocators,
                end_command_lists,
            },
            draw_calls_accum: std::sync::atomic::AtomicU32::new(0),
            frame_sync: DxFrameSync {
                fence,
                fence_values,
                next_fence_value: std::cell::Cell::new(1),
                fence_event,
            },
            current_frame: 0,
            cull_bvh,
            always_draw,
            visible_scratch: RefCell::new(Vec::new()),
            draw_objects,
            instanced: DxInstanced {
                root_sig: main_instanced_root_sig,
                pso: main_instanced_pso,
                clusters: instanced_clusters,
                upload_buffers: instance_upload_buffers,
                upload_ptrs: instance_upload_ptrs,
                // One outer Vec entry per cluster; populated each frame by
                // `build_instance_upload` from `lod_buckets(cam_pos)`. The
                // inner Vec is the bucket order (LOD0 → LODN) for that
                // cluster. Empty rows for clusters that never have visible
                // instances stay empty.
                bucket_layouts: std::sync::RwLock::new(vec![Vec::new(); n_clusters]),
            },
            clear_color,
            view_matrix: IDENTITY4,
            text_upload: super::draw::TextUploadRing::new(FRAMES),
            info_queue,
            adapter,
            frame_stats: std::cell::Cell::new(crate::gfx::profile::RenderStats::default()),
            timestamps: TimestampState {
                query_heap: timestamp_query_heap,
                readback: timestamp_readback,
                readback_ptr: timestamp_readback_ptr,
                frequency: timestamp_frequency,
            },
            auto_exposure: super::context::AutoExposureState {
                resources: auto_exposure,
                settings: auto_exposure_settings,
                state: auto_exposure_state,
                bias_ev: auto_exposure_bias_ev,
                last_elapsed: 0.0,
            },
            max_edr: match hdr_mode {
                crate::gfx::hdr_output::HdrOutputMode::Hdr { max_edr, .. } => Some(max_edr),
                crate::gfx::hdr_output::HdrOutputMode::Sdr => None,
            },
            swap_format: swapchain_format,
            present_sync_interval,
            allow_tearing,
            hdr_encoding: match hdr_mode {
                crate::gfx::hdr_output::HdrOutputMode::Hdr { encoding, .. } => Some(encoding),
                crate::gfx::hdr_output::HdrOutputMode::Sdr => None,
            },
            last_present_index: None,
            hot_reload: super::context::HotReloadState {
                enabled: hot_reload,
                reload_pending: shader_reload_pending,
                watcher: shader_watcher,
            },
            quality_slots,
            rt_capable: raytracing_supported,
            rt_static_vertex_count: vertices.len(),
        })
    }
}

fn create_samplers(device: &ID3D12Device, base_cpu: D3D12_CPU_DESCRIPTOR_HANDLE, stride: usize) {
    // [0] Shadow comparison sampler (LESS_EQUAL).
    let shadow_samp = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    unsafe {
        device.CreateSampler(
            &shadow_samp,
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base_cpu.ptr },
        )
    };

    // [1] Anisotropic repeat (albedo + normal map). Anisotropic filtering plus
    // the unclamped MaxLOD lets minified scene textures trilinear-select down
    // their mip chain instead of aliasing from mip 0. 8x is well within the
    // D3D12 feature-level-11 guaranteed maximum of 16.
    let linear_samp = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_ANISOTROPIC,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        MaxAnisotropy: 8,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    unsafe {
        device.CreateSampler(
            &linear_samp,
            D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base_cpu.ptr + stride,
            },
        )
    };

    // [2] Cube linear-clamp + mip linear (IBL irradiance / prefilter).
    let cube_samp = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    unsafe {
        device.CreateSampler(
            &cube_samp,
            D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base_cpu.ptr + stride * 2,
            },
        )
    };

    // [3] Linear clamp, mip 0 only (text atlas). The text atlas is a tightly
    // packed glyph SDF: its coarse mips bleed adjacent glyphs together, so
    // trilinear minification samples that garbage and the text reads choppy.
    // Clamp MaxLOD to 0 so only the full-resolution (supersampled) mip 0 is
    // sampled; the SDF stays crisp under bilinear minification on its own.
    // Mirrors the Vulkan text sampler (`create_sampler_linear_clamp`, whose
    // max_lod defaults to 0).
    let clamp_samp = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MinLOD: 0.0,
        MaxLOD: 0.0,
        ..Default::default()
    };
    unsafe {
        device.CreateSampler(
            &clamp_samp,
            D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base_cpu.ptr + stride * 3,
            },
        )
    };
}
