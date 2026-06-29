<!-- Auto-generated - do not edit. -->

# PostProcessConfig

Tunables for the post-process stack. One per world; the first declared
instance wins. With no `PostProcessConfig` present, the defaults below are
used (bloom on at a moderate intensity).

Colour-LUT grading is a separate [ColorLut](ColorLut.md) asset; `lut_strength`
here is the blend amount applied to whichever [ColorLut](ColorLut.md) the world
declares.

When `auto_exposure` is on, the scene's average brightness is measured each
frame and exposure adapts toward a balanced mid-tone. The authored
`exposure_ev` then acts as an additive bias (in stops) on top of the adapted
value.

```jsonl
{"name":"post","type":"PostProcessConfig","args":{"bloom_intensity":0.8}}
{"name":"post_dim","type":"PostProcessConfig","args":{"exposure_ev":-1.0,"vignette_strength":0.4}}
{"name":"post_taa","type":"PostProcessConfig","args":{"aa_mode":"taa"}}
{"name":"post_ssao","type":"PostProcessConfig","args":{"ssao":true,"ssao_radius":0.6}}
{"name":"post_ssr","type":"PostProcessConfig","args":{"ssr":true,"ssr_intensity":0.8}}
{"name":"post_rt","type":"PostProcessConfig","args":{"ray_traced_reflections":true,"ssr_intensity":0.8}}
{"name":"post_refl_blur","type":"PostProcessConfig","args":{"ssr":true,"reflection_blur_resolution":"quarter"}}
{"name":"post_ssgi","type":"PostProcessConfig","args":{"indirect_lighting":"ssgi","ssgi_intensity":0.6}}
{"name":"post_auto_ev","type":"PostProcessConfig","args":{"auto_exposure":true}}
{"name":"post_hdr","type":"PostProcessConfig","args":{"hdr_display":true}}
{"name":"post_upscale","type":"PostProcessConfig","args":{"temporal_upscaling":true,"upscale_quality":"balanced"}}
{"name":"post_dlss","type":"PostProcessConfig","args":{"temporal_upscaling":true,"upscale_backend":"dlss"}}
{"name":"post_occ2","type":"PostProcessConfig","args":{"occlusion_two_pass":true}}
{"name":"post_off","type":"PostProcessConfig","args":{"bloom_intensity":0.0}}
```

## Parameters

- `bloom_intensity`: A float. Additive bloom contribution. 0 skips bloom entirely. Defaults to `0.6`.
- `bloom_threshold`: A float. Brightness threshold for bloom. Pixels brighter than this contribute fully; pixels within `bloom_knee` below it ramp in softly. Defaults to `1.0`.
- `bloom_knee`: A float. Width of the soft knee just below `bloom_threshold`. Defaults to `0.5`.
- `exposure_ev`: A float. Exposure offset in photographic stops. Each +1 doubles scene brightness before bloom and tonemapping; 0 is neutral. Defaults to `0.0`.
- `vignette_strength`: A float. Vignette strength in `[0, 1]`. 0 disables the corner darkening. Defaults to `0.0`.
- `lut_strength`: A float. Colour-LUT blend in `[0, 1]`. Mixes the graded colour over the ungraded one by this amount. Only matters when the world declares a [ColorLut](ColorLut.md); with none, grading is a no-op at any strength. Defaults to `1.0`.
- `aa_mode`: A string (see [AaMode](AaMode.md)). Anti-aliasing mode. `fxaa` (default) applies a cheap composite-pass edge filter; `taa` adds a temporal pass that jitters the projection and accumulates detail across frames for the cleanest edges, at the cost of a velocity pre-pass and a history buffer; `off` disables edge smoothing.
- `ssao`: A boolean. Screen-space ambient occlusion toggle. Darkens creases and contact areas where ambient light is occluded. Defaults to `false`.
- `ssao_radius`: A float. How far the ambient-occlusion search reaches for occluders, in world units. Larger values pick up broader, softer occlusion. Defaults to `0.5`.
- `ssao_intensity`: A float. Ambient-occlusion strength, clamped to `[0, 4]`. 1.0 is the natural amount; higher values exaggerate the contact darkening. Defaults to `1.0`.
- `ssr`: A boolean. Screen-space reflection toggle. Mixes reflected scene colour over glossy surfaces (water, polished floors). Defaults to `false`.
- `ssr_intensity`: A float. Reflection blend strength, clamped to `[0, 1]`. Scales the Fresnel-weighted reflection mixed over the base shading. Defaults to `0.7`.
- `ssr_max_distance`: A float. How far a reflection reaches, in world units. Longer reaches catch more distant reflections, more coarsely. Defaults to `40.0`.
- `ray_traced_reflections`: A boolean. Hardware ray-traced reflection toggle. When the GPU supports ray tracing, traces real reflection rays so off-screen geometry still appears, instead of the screen-space method. Reuses the `ssr_intensity` / `ssr_max_distance` tunables and takes precedence over `ssr`, falling back to it where ray tracing isn't available. Defaults to `false`.
- `reflection_blur_resolution`: A string (see [ReflectionBlurResolution](ReflectionBlurResolution.md)). Internal resolution of the roughness-aware reflection blur the SSR / ray-traced reflection composite runs. `half` (default) blurs at a quarter of the pixels for a large saving and bilinearly upsamples; `full` blurs at native resolution; `quarter` is the cheapest. Smooth mirror surfaces stay sharp at any setting (the composite keeps the sharp reflection for low roughness). Only matters when `ssr` or `ray_traced_reflections` is on.
- `indirect_lighting`: A string (see [IndirectLighting](IndirectLighting.md)). Indirect-diffuse lighting source. `ibl` (default) uses the environment map's ambient alone. `ssgi` adds a screen-space global-illumination pass on top, so nearby lit surfaces bleed colour onto one another; the environment ambient still covers the off-screen / sky fallback.
- `ambient_intensity`: A float. Multiplier on the indirect (ambient / IBL) lighting term, clamped to `[0, 16]`. 1.0 (default) leaves the environment-derived ambient at its physical level. Raising it lifts fill light in areas the directional light cannot reach (shadowed facades, alleys) without brightening directly lit surfaces, which the sun already dominates. Scales the diffuse and specular IBL together, so reflections stay consistent with the brighter ambient. Useful for high-contrast exterior scenes where a strong sun would otherwise crush shadows to black.
- `ssgi_intensity`: A float. Indirect-bounce strength, clamped to `[0, 4]`. Scales the gathered indirect light added on top of the existing shading; 0 makes it a no-op. Only matters when `indirect_lighting` is `ssgi`. Defaults to `0.5`.
- `ssgi_max_distance`: A float. How far the indirect-light gather reaches, in world units. A near-field effect, so it defaults well below `ssr_max_distance`. Only matters when `indirect_lighting` is `ssgi`.
- `ssgi_resolution`: A string (see [SsgiResolution](SsgiResolution.md)). Internal resolution of the SSGI gather. `half` (default) trades a little sharpness for a large performance saving; `full` is native; `quarter` is the cheapest. Only matters when `indirect_lighting` is `ssgi`.
- `ssgi_rays`: An integer. Hemisphere rays cast per pixel by the SSGI gather, clamped to `[1, 32]`. More rays reduce noise at a higher cost. Only matters when `indirect_lighting` is `ssgi`.
- `ssgi_steps`: An integer. Ray-march samples per SSGI ray, clamped to `[1, 64]`. More samples catch finer occlusion at a higher cost. Only matters when `indirect_lighting` is `ssgi`.
- `auto_exposure`: A boolean. Auto-exposure toggle. Adapts exposure each frame toward a balanced mid-tone. The authored `exposure_ev` then acts as an additive bias in stops on top of the adapted value. Defaults to `false`.
- `auto_exposure_min_ev`: A float. Lower bound on the adapted exposure (EV). The `exposure_ev` bias is applied before this clamp. Defaults to `-8.0`.
- `auto_exposure_max_ev`: A float. Upper bound on the adapted exposure (EV). Defaults to `8.0`.
- `auto_exposure_speed`: A float. How quickly exposure chases a new target (per second). Higher converges faster but can pump under flickering content; 1-3 is comfortable. Defaults to `1.5`.
- `hdr_display`: A boolean. HDR display output toggle. On a capable display, emits extended-range HDR instead of the standard tonemapped output. Falls back to standard output when the display or platform doesn't support HDR. Defaults to `false`.
- `hdr_pq`: A boolean. PQ (HDR10) output mode. When true, and `hdr_display` is on, and the display has HDR headroom, output is PQ-encoded for HDR10 panels. No effect when `hdr_display` is off. Defaults to `false`.
- `temporal_upscaling`: A boolean. Temporal upscaling toggle. Renders the 3D scene at a lower resolution (set by `upscale_quality`) and reconstructs a full-resolution image, trading some sharpness for performance. Replaces TAA while on (the `taa` flag is ignored). Defaults to `false`.
- `upscale_quality`: A string (see [UpscaleQuality](UpscaleQuality.md)). Render-scale preset for `temporal_upscaling`; each step progressively lowers the internal resolution. No effect when `temporal_upscaling` is off.
- `upscale_backend`: A string (see [UpscalerBackend](UpscalerBackend.md)). Which upscaler backend `temporal_upscaling` uses. `auto` (default) picks the best available at runtime (DLSS on NVIDIA RTX, else XeSS, else FSR3); `fsr3` / `dlss` / `xess` request a specific one and fall back when it is unavailable on the current GPU or build. No effect when `temporal_upscaling` is off. DLSS and XeSS are DirectX-only.
- `occlusion_two_pass`: A boolean. Two-pass occlusion culling toggle. Reduces objects popping in a frame late when they're revealed by camera or occluder motion, at the cost of extra culling work each frame. Defaults to `false`.
