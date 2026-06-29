// Shared GPU data types used by all rendering backends. Defined here (no
// #[cfg] gate) so a future Vulkan backend can import them without pulling in
// Metal-specific code. metal.rs imports from this module rather than defining
// its own copies.

pub const MAX_DIRECTIONAL_LIGHTS: usize = 4;
pub const MAX_POINT_LIGHTS: usize = 8;

// Number of cascades the directional shadow pre-pass renders into the shadow
// map array. Hardcoded because changing N requires re-compiling the shaders
// (the array length appears in the MSL/HLSL/GLSL source).
pub const NUM_SHADOW_CASCADES: usize = 4;

// Maximum number of joints in a single skinned-mesh skeleton. Enforced
// CPU-side as a clamp on each `SkinnedDrawObject.joint_count` and on the
// matching `skinned_joint_matrices` Vec length. The skinned shaders read
// the joints buffer through a pointer (`constant float4x4 *joints`) using
// vertex-encoded joint indices, so the GPU buffer size and joint count are
// fully dynamic: this constant just caps how many matrices the per-frame
// upload may carry per object.
pub const MAX_JOINTS: usize = 64;

// Per-draw-call material parameters pushed to the fragment shader at buffer(3).
// Must stay in sync with the `MaterialUniforms` struct in every .metal shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct MaterialUniforms {
    // Perceptual roughness [0, 1]: 0 = mirror, 1 = fully diffuse.
    pub roughness: f32,
    // Metallic factor [0, 1]: 0 = dielectric, 1 = metal.
    pub metallic: f32,
    // Macro-variation strength [0, 1]; 0 disables the tiling-break noise.
    pub macro_variation: f32,
    // Terrain-shading blend [0, 1]; 0 disables (default PBR sampling),
    // non-zero switches the shader to a triplanar world-space projection
    // with slope-based rocky-tint blending. See `Material::terrain_blend`.
    pub terrain_blend: f32,
    // Linear-space RGB multiplier on the albedo sample.
    pub tint: [f32; 3],
    pub _pad2: f32,
    // Additive emission colour in linear space.
    pub emissive: [f32; 3],
    // Sharpness of the slope-based blend between the primary and the
    // `albedo_secondary` texture pair. 0 = wide soft gradient;
    // 1 = nearly hard cliff edge. Ignored unless both `terrain_blend > 0`
    // and the secondary texture pair is bound.
    pub secondary_blend_sharpness: f32,
    // Index into the per-draw albedo texture (legacy path) or the
    // bindless texture pool (bindless path) for the slope-shaded
    // secondary albedo. `0` is also the fallback when the material
    // doesn't declare a secondary texture; the shader's
    // `terrain_blend > 0 && secondary_blend_sharpness > 0` gate keeps
    // it from being sampled in that case.
    pub albedo_secondary_index: u32,
    // Companion to `albedo_secondary_index` for the slope-shaded
    // secondary normal map.
    pub normal_secondary_index: u32,
    // Bindless-pool index for the emissive map. `0` means no map: the shader
    // gates on a non-zero index, falling back to the scalar `emissive` factor.
    pub emissive_map_index: u32,
    // Bindless-pool index for the packed occlusion/roughness/metalness map
    // (R = occlusion, G = roughness, B = metalness). `0` means no map: the
    // shader keeps the scalar `roughness`/`metallic` and full occlusion.
    pub orm_map_index: u32,
    // Base surface opacity in [0, 1]; 1 = fully opaque (the default). Only
    // meaningful with `transparent`: it drives the glass alpha in the
    // transparent pass. Carried on the material so it rides Material ->
    // DrawObject.material to the backend; the opaque main-pass shader ignores it.
    pub opacity: f32,
    // 1 when this surface routes through the transparent pass instead of the
    // opaque one (a glass MESH on an RT-capable device); 0 for opaque. A CPU
    // routing flag: the backend reads it to skip the draw in the opaque pass +
    // the RT BLAS and to feed the per-pixel-RT transparent producer. No GPU
    // shader reads it (the opaque skip is decided CPU-side).
    pub transparent: u32,
    // 1 when a `transparent` glass MESH renders as genuinely see-through (Layer
    // 2: the scene behind shows through plus a sharp per-pixel reflection); 0
    // keeps it as opaque low-roughness reflective glass (Layer 1). Another CPU
    // routing flag (no GPU shader reads it): the see-through producer, the
    // opaque-pass skip, and the RT-BLAS exclude all key off it, so Layer 2 is
    // opt-in per material. Always 0 unless `transparent` is also 1.
    pub see_through: u32,
}

impl MaterialUniforms {
    // Neutral material: matte, non-metallic, white tint, no emission.
    pub const DEFAULT: Self = Self {
        roughness: 0.8,
        metallic: 0.0,
        macro_variation: 0.0,
        terrain_blend: 0.0,
        tint: [1.0, 1.0, 1.0],
        _pad2: 0.0,
        emissive: [0.0, 0.0, 0.0],
        secondary_blend_sharpness: 0.5,
        albedo_secondary_index: 0,
        normal_secondary_index: 0,
        emissive_map_index: 0,
        orm_map_index: 0,
        opacity: 1.0,
        transparent: 0,
        see_through: 0,
    };
}

// One directional light entry in LightUniforms.
// Layout (32 bytes) must match DirectionalLightData in every .metal shader.
// MSL shaders must declare float3 fields as packed_float3 in constant buffer
// structs; plain float3 has size=16 in MSL which shifts subsequent fields.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct DirectionalLightData {
    // Unit vector pointing TOWARD the light source (same as L in Blinn-Phong).
    pub direction: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
    pub _pad: f32,
}

// One point light entry in LightUniforms.
// Layout (32 bytes) must match PointLightData in every .metal shader.
// Same packed_float3 requirement as DirectionalLightData above.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct PointLightData {
    // World-space position of the light source.
    pub position: [f32; 3],
    // Maximum reach in metres; attenuation is zero at this distance.
    pub range: f32,
    pub color: [f32; 3],
    pub intensity: f32,
}

const ZERO_DIR_LIGHT: DirectionalLightData = DirectionalLightData {
    direction: [0.0; 3],
    intensity: 0.0,
    color: [0.0; 3],
    _pad: 0.0,
};

const ZERO_POINT_LIGHT: PointLightData = PointLightData {
    position: [0.0; 3],
    range: 0.0,
    color: [0.0; 3],
    intensity: 0.0,
};

// All scene lights packed into a single GPU buffer pushed at fragment buffer(4).
// Must stay in sync with LightUniforms in every .metal shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct LightUniforms {
    pub directional: [DirectionalLightData; MAX_DIRECTIONAL_LIGHTS],
    pub point: [PointLightData; MAX_POINT_LIGHTS],
    pub num_directional: i32,
    pub num_point: i32,
    // Multiplier on the indirect (IBL / flat-fallback) ambient term in the main
    // pass, resolved from `PostProcessConfig.ambient_intensity`. 1.0 leaves the
    // physically derived ambient untouched; higher values lift fill in shadowed
    // areas the directional light cannot reach. Occupies the first of the two
    // trailing pad words, so the 400-byte layout is unchanged.
    pub ambient_intensity: f32,
    pub _pad: f32,
}

impl LightUniforms {
    // Neutral directional sun; used when no Light components are declared.
    pub const DEFAULT: Self = Self {
        directional: [
            DirectionalLightData {
                direction: [-0.3, 0.85, 0.4],
                intensity: 1.0,
                color: [1.0, 1.0, 1.0],
                _pad: 0.0,
            },
            ZERO_DIR_LIGHT,
            ZERO_DIR_LIGHT,
            ZERO_DIR_LIGHT,
        ],
        point: [
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
        ],
        num_directional: 1,
        num_point: 0,
        ambient_intensity: 1.0,
        _pad: 0.0,
    };
}

// Cascaded shadow map view-projection matrices and split depths.
//
// Pushed to the shadow-pass vertex shader at buffer(0) (alongside a
// cascade_idx push constant that picks one matrix) and to the main-pass
// fragment shader at buffer(5). Layout must stay in sync with the
// `ShadowUniforms` struct in every .metal / .hlsl / .glsl shader.
//
// `cascade_splits` are the view-space depth thresholds (positive) marking the
// FAR end of each cascade. The fragment shader picks the first cascade whose
// split is greater than the fragment's view-space depth. Slot 3 (the last
// cascade's far end) doubles as the overall shadow distance.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ShadowUniforms {
    // One light-space VP matrix per cascade, column-major.
    pub light_vps: [[[f32; 4]; 4]; NUM_SHADOW_CASCADES],
    // View-space depth at the FAR end of each cascade. Stored as a vec4 to
    // keep MSL std140-like alignment trivial across backends. Slots
    // `[active_cascades..]` hold a negative sentinel so the fragment shader's
    // split comparison never selects an unrendered cascade.
    pub cascade_splits: [f32; NUM_SHADOW_CASCADES],
    // How many of the `NUM_SHADOW_CASCADES` slots are live this frame (1..=4,
    // from `GraphicsConfig.shadow_cascades`). The array capacity stays 4; only
    // the first `active_cascades` are split, rendered, and sampled. The fragment
    // shader bounds both its cascade fallback and its cross-cascade blend by this
    // so it never reads a slot the CPU did not render.
    pub active_cascades: u32,
    // Pads the struct to 288 bytes (a multiple of 16). The Rust `[[f32; 4]; 4]`
    // matrices are only 4-byte aligned, so Rust would otherwise leave this at 276;
    // the MSL `float4x4`-aligned struct and the GLSL std140 block both round up to
    // 288, so the upload size must match for the bound buffer range to cover them.
    pub _pad: [u32; 3],
}

// Per-shadow-pass push constant identifying which cascade is being rendered.
// Used so the shadow vertex shader can index `ShadowUniforms.light_vps[i]`
// from a single bound UBO instead of re-binding a different uniform per pass.
//
// Currently consumed only by the Metal shadow pass; Vulkan and DirectX each
// define their own private push-constant layouts in their respective `draw.rs`
// modules.
#[cfg(backend_metal)]
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ShadowPassPush {
    pub cascade_idx: u32,
    pub _pad: [u32; 3],
}

// Compact vertex type used exclusively by the text render pass.
// 32 bytes: screen-pixel position, atlas UV, text colour, and padding.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct TextVertex {
    // Screen-space position in pixels (x from left, y from top).
    pub pos: [f32; 2],
    // Normalised UV into the glyph atlas texture.
    pub uv: [f32; 2],
    // Linear-space RGB text colour.
    pub color: [f32; 3],
    pub _pad: f32,
}

// Uniforms pushed to the text vertex shader once per text draw call.
// Carries the framebuffer size so the shader can convert pixel coords to NDC.
#[derive(Copy, Clone)]
#[repr(C)]
#[allow(dead_code)]
pub struct TextUniforms {
    pub win_width: f32,
    pub win_height: f32,
    pub _pad: [f32; 2],
}

// Post-process tunables resolved from the `PostProcessConfig` asset (or its
// defaults) and threaded into each backend at init. Pushed verbatim to the
// bloom prefilter and composite fragment shaders, so the layout must stay in
// sync with the `PostUniforms` struct in those shaders. 36 bytes.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct PostProcessParams {
    // Additive bloom strength. 0 disables the bloom passes entirely.
    pub bloom_intensity: f32,
    // Luminance threshold for the bloom prefilter.
    pub bloom_threshold: f32,
    // Quadratic soft-knee width below the threshold.
    pub bloom_knee: f32,
    // Linear exposure multiplier applied to HDR radiance before the bloom
    // prefilter and the composite tonemap. Resolved from `exposure_ev` as
    // `2^ev`, so 1.0 is neutral.
    pub exposure: f32,
    // Vignette strength in `[0, 1]`. 0 disables the corner darkening.
    pub vignette: f32,
    // Colour-LUT blend in `[0, 1]`: `mix(scene, graded, lut_strength)` in the
    // composite pass. Has no effect when no `ColorLut` is declared: the
    // renderer then binds an identity LUT, so the grade is a no-op.
    pub lut_strength: f32,
    // HDR display output flag. `0.0` = SDR path (ACES tonemap + gamma 2.2 +
    // FXAA + ColorLut, output BGRA8Unorm). `1.0` = HDR EDR path (no tonemap
    // or gamma; the exposed HDR scene is emitted into a `RGBA16Float`
    // swapchain attached to a Display P3 EDR layer). FXAA + ColorLut are
    // skipped on the HDR path because both depend on display-referred
    // values. Stored as a float (not a uint) so the MSL shader can branch
    // on `> 0.5` without a cast.
    pub hdr_output: f32,
    // Output-encoding flag inside the HDR branch. `0.0` = scRGB-linear
    // passthrough (the OS handles the encode for the panel). `1.0` =
    // PQ-encode (SMPTE ST 2084) in-shader so the swapchain ships
    // PQ-encoded values directly to an HDR10 panel. Only read when
    // `hdr_output > 0.5`; always 0.0 on the SDR path.
    pub pq_output: f32,
    // FXAA edge-filter flag on the SDR path. `1.0` runs the composite's
    // FXAA pass; `0.0` skips it (the `Off` anti-aliasing mode). Resolved
    // from `PostProcessConfig.aa_mode`: on for `Fxaa` and `Taa`, off for
    // `Off`. Always ignored on the HDR path, which never runs FXAA. Stored
    // as a float so the shader branches on `> 0.5` without a cast.
    pub fxaa: f32,
}

impl PostProcessParams {
    // Matches `PostProcessConfig::default()`: used when no asset is declared.
    pub const DEFAULT: Self = Self {
        bloom_intensity: 0.6,
        bloom_threshold: 1.0,
        bloom_knee: 0.5,
        exposure: 1.0,
        vignette: 0.0,
        lut_strength: 1.0,
        hdr_output: 0.0,
        pq_output: 0.0,
        fxaa: 1.0,
    };
}

// Per-frame uniform for the SSAO (GTAO) horizon-search kernel. Carries the
// clamped authored tunables plus the view-ray scale the kernel needs to
// rebuild a view-space position from the linear depth the SSAO pre-pass
// writes. Pushed verbatim to the SSAO kernel fragment shader, so the layout
// must stay in sync with the `SsaoParams` struct there. 16 bytes.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct SsaoParams {
    // World-space hemisphere radius the horizon search covers.
    pub radius: f32,
    // Occlusion strength multiplier applied to the integrated visibility.
    pub intensity: f32,
    // `tan(fov_y / 2)`: the vertical view-ray half-extent at unit depth.
    pub tan_half_fov_y: f32,
    // Viewport aspect ratio (width / height).
    pub aspect: f32,
}

// Per-frame uniform for the screen-space reflection (SSR) ray-march. Carries
// the clamped authored tunables, the view-ray scale the resolve pass uses to
// project a view-space ray point back to a screen UV, and the data the resolve
// needs to sample the IBL prefilter cubemap as a fallback. Pushed verbatim to
// the SSR resolve fragment shader, so the layout must stay in sync with the
// `SsrParams` struct there. 96 bytes.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct SsrParams {
    // Reflection blend strength in `[0, 1]`; scales the Fresnel-weighted mix.
    pub intensity: f32,
    // Maximum world-space distance a reflection ray marches before giving up.
    pub max_distance: f32,
    // `tan(fov_y / 2)`: the vertical view-ray half-extent at unit depth.
    pub tan_half_fov_y: f32,
    // Viewport aspect ratio (width / height).
    pub aspect: f32,
    // World-space length of one ray-march step.
    pub stride: f32,
    // View-space depth tolerance for accepting a ray/scene-depth intersection.
    pub thickness: f32,
    // IBL prefilter cubemap mip count. 0 when no EnvironmentMap is bound: the
    // resolve then skips the cube fallback and keeps the base shading instead.
    pub prefilter_mip_count: f32,
    pub _pad: f32,
    // Camera-to-world transform (column-major, the rigid inverse of the view
    // matrix): the orthonormal 3x3 turns a view-space reflection ray into the
    // world-space direction the cubemap is sampled with, and the translation
    // column (the world camera position) lets the resolve rebuild the world-space
    // surface position a reflection probe box-projects against. Backends that only
    // sample the cube use the 3x3 (the `r_world` direction) and ignore translation.
    pub inv_view: [[f32; 4]; 4],
}

// Per-frame uniform for the screen-space global-illumination (SSGI) gather +
// composite. Carries the clamped authored tunables and the view-ray scale the
// gather pass uses to project a view-space ray point back to a screen UV.
// Pushed verbatim to the SSGI gather + composite fragment shaders, so the
// layout must stay in sync with the `SsgiParams` struct there. 32 bytes.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct SsgiParams {
    // Indirect-bounce blend strength; scales the gathered radiance the
    // composite pass adds on top of the existing shading.
    pub intensity: f32,
    // Maximum world-space distance a hemisphere ray marches before giving up.
    pub max_distance: f32,
    // `tan(fov_y / 2)`: the vertical view-ray half-extent at unit depth.
    pub tan_half_fov_y: f32,
    // Viewport aspect ratio (width / height).
    pub aspect: f32,
    // World-space length of one ray-march step.
    pub stride: f32,
    // View-space depth tolerance for accepting a ray/scene-depth intersection.
    pub thickness: f32,
    // Hemisphere rays cast per pixel (carried as f32; the shader reads it as an
    // int loop bound). Backends that still bake a compile-time ray count ignore
    // this field, so the 32-byte layout is unchanged.
    pub rays: f32,
    // Ray-march samples per ray (same f32-as-int-bound convention as `rays`).
    pub steps: f32,
}

// Per-frame uniform for the hardware ray-traced reflection pass. Like
// [`SsrParams`] it carries the clamped intensity / distance, the view-ray
// scale used to rebuild a view-space position from the SSR pre-pass G-buffer,
// and the IBL prefilter mip count for the miss fallback. Unlike SSR it traces
// a world-space ray against an acceleration structure, so it also carries the
// camera-to-world transform (to lift the view-space hit point + normal into
// world space), the world camera position (the ray origin), and the sun
// direction + colour the hit-shading uses. Pushed verbatim to the RT kernel,
// so the layout must stay in sync with the `RtParams` struct there. 144 bytes,
// 16-byte aligned (every `vec3` is padded to a `float4`).
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct RtParams {
    // Reflection blend strength in `[0, 1]`; scales the Fresnel-weighted mix.
    pub intensity: f32,
    // Maximum world-space distance a reflection ray travels before it misses.
    pub max_distance: f32,
    // `tan(fov_y / 2)`: the vertical view-ray half-extent at unit depth.
    pub tan_half_fov_y: f32,
    // Viewport aspect ratio (width / height).
    pub aspect: f32,
    // IBL prefilter cubemap mip count. 0 when no EnvironmentMap is bound: the
    // kernel then keeps the base shading for missed rays instead of a cube tap.
    pub prefilter_mip_count: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    // World-space camera position (`xyz`); the reflection ray origin. `w` unused.
    pub cam_pos: [f32; 4],
    // World-space unit direction *toward* the sun (`xyz`); the hit-shading N·L
    // term uses it. `w` unused.
    pub sun_dir: [f32; 4],
    // Sun radiance (`xyz`); the direct term at a reflected hit. `w` unused.
    pub sun_color: [f32; 4],
    // Camera-to-world transform (column-major, the rigid inverse of the view
    // matrix). Lifts the view-space reconstructed position + normal into the
    // world space the acceleration structure is built in.
    pub inv_view: [[f32; 4]; 4],
}

// One entry of the ray-tracing geometry table, indexed by the intersector's
// `instance_id` (the instance's position in the TLAS instance buffer, one
// entry per instance, in instance order). Lets the RT kernel find the hit
// triangle's indices in the shared index buffer, transform its local-space
// vertices into world space for the geometric normal, and pick a base albedo to
// shade the hit with. `#[repr(C)]`, 128 bytes: the layout must stay in sync
// with the `RtGeomEntry` struct in the RT kernel (`rt_reflections.metal`), where
// `tint` and `emissive` are `packed_float3` so the field offsets match; a plain
// `float3` there would stride the buffer differently and fault the trace. The
// `_pad` tail rounds the struct to 128 bytes so its array stride is a multiple
// of the 16-byte GPU alignment of the `float4x4` `model` (MSL/HLSL/GLSL round a
// matrix-bearing struct up to a 16-byte multiple, so a 116-byte Rust struct
// would stride 116 while the shader strides 128 and faults the trace). The
// `size_eq` / `offsets` unit tests below lock the Rust side. The model matrix is
// stored here (rather than read from the intersector's instance transform) so
// the kernel's normal math is self-contained.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct RtGeomEntry {
    // Element offset of this object's first index in the shared index buffer
    // (`DrawObject::index_offset`).
    pub index_offset: u32,
    // Value added to each fetched index before the vertex lookup
    // (`DrawObject::base_vertex`).
    pub base_vertex: u32,
    // Index into the bindless albedo texture pool (`DrawObject::texture_slot`,
    // clamped to the albedo count). The textured RT-hit shader samples
    // `tex_pool[albedo_index]` at the hit UV; the flat fallback ignores it.
    pub albedo_index: u32,
    // Index into the bindless texture pool for this object's normal map
    // (`albedo_count + normal_map_slot`). The textured RT-hit shader samples
    // `tex_pool[normal_index]` to perturb the hit normal via the interpolated
    // tangent frame; objects with no normal map resolve to the 1x1 flat-normal
    // fallback at normal slot 0, so the sample is always safe (no perturbation).
    pub normal_index: u32,
    // Base albedo for hit shading (the material tint), `[r, g, b]`. The
    // textured shader multiplies the sampled albedo by this; the flat fallback
    // uses it directly.
    pub tint: [f32; 3],
    // Surface roughness, used to pick the IBL prefilter mip at the hit.
    pub roughness: f32,
    // Metallic factor [0, 1] for the hit PBR response: metals drop the diffuse
    // term and tint the reflected environment specular by their albedo.
    pub metallic: f32,
    // Self-emission added to the hit colour, so glowing surfaces light up in
    // reflections regardless of incident lighting. Fills the three words that
    // pad `metallic` out to a 16-byte boundary (the MSL side reads it as a
    // `packed_float3` at the same offset; the layout test pins the exact size).
    pub emissive: [f32; 3],
    // Column-major object-to-world transform (`DrawObject::model`), used to
    // lift the hit triangle's local-space vertices into world space.
    pub model: [[f32; 4]; 4],
    // Bindless albedo-pool index for the emissive map (0 = none). The textured
    // RT-hit shader multiplies the self-emission by this map sample, so glowing
    // textured surfaces (e.g. the bistro string lights) reflect in colour rather
    // than the scalar `emissive`; the flat fallback ignores it. Mirrors
    // `MaterialUniforms::emissive_map_index` / `GpuObjectData::emissive_map_index`.
    pub emissive_map_index: u32,
    // Pads the struct to 128 bytes (a multiple of the 16-byte GPU alignment of
    // the `float4x4` `model`) so the Rust array stride matches the shader stride.
    pub _pad: [u32; 3],
}

// Per-frame uniform consumed by the volumetric-fog ray-march. Carries the
// clamped authored tunables plus the view inputs the shader needs to
// reconstruct world positions from depth and integrate sun-aligned
// scattering. Pushed verbatim to the fog fragment shader, so the layout must
// stay in sync with `FogParams` in `metal/shaders/fog.metal`. 176 bytes.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct FogParams {
    // Inverse view-projection: reconstructs world position from depth.
    pub inv_vp: [[f32; 4]; 4],
    // Linear-space fog tint (RGB); alpha unused.
    pub color: [f32; 4],
    // World-space camera position. The marcher starts here and walks the
    // view ray to the scene point. `_pad` keeps `cam_pos` 16-byte aligned
    // inside MSL.
    pub cam_pos: [f32; 3],
    pub _pad0: f32,
    // First directional light's world-space direction (toward the light),
    // used for the Henyey-Greenstein phase and the per-step sun radiance.
    pub sun_dir: [f32; 3],
    pub _pad1: f32,
    // First directional light's colour pre-multiplied with its intensity.
    // Drives the per-step sun in-scatter alongside the phase function.
    pub sun_color: [f32; 3],
    pub _pad2: f32,
    // Base density at `height_reference` (per world unit).
    pub density: f32,
    // Exponential height-falloff rate; 0 = homogeneous medium.
    pub height_falloff: f32,
    // World-space Y at which density equals `density`.
    pub height_reference: f32,
    // Ray-march cap in world units.
    pub max_distance: f32,
    // Henyey-Greenstein anisotropy in `(-0.95, 0.95)`.
    pub phase_g: f32,
    // Ambient (sky-side) scattering term added each step.
    pub ambient: f32,
    // Width / height of the HDR resolve target in pixels: drives the
    // screen→NDC conversion in the fragment shader.
    pub viewport: [f32; 2],
    // Pre-computed reciprocal of `max_distance`; the shader uses it to skip
    // a per-step `1 / max_distance` divide.
    pub inv_max_distance: f32,
    pub _pad3: [f32; 3],
}

// Per-frame uniform consumed by the volumetric-fog froxel compute kernel and
// the matching fragment-shader sample path. Carries the view matrix (so the
// compute kernel + the fragment sampler can map between world-space froxel
// positions and the volume's Z axis) and the discrete volume dimensions. Used
// by the froxel-volume fog path; only worlds that declare a `VolumetricFog`
// bind it.
//
// Bound at the fog fragment shader + the froxel kernel. Layout must stay in
// sync with `FogFroxelParams` in `metal/shaders/fog.metal`,
// `directx/shaders/fog_froxel.hlsl`, and `vulkan/shaders/fog_froxel.comp`.
// 96 bytes.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[allow(dead_code)] // Bound only by client backends; unused within the core crate.
pub struct FogFroxelParams {
    // View matrix (world → view), so the compute kernel can pick a CSM
    // cascade for each froxel and the fragment sampler can map a scene
    // depth into the volume's Z slice index.
    pub view: [[f32; 4]; 4],
    // Number of froxels along screen-x / screen-y / view-z.
    pub froxel_dims: [u32; 3],
    // Explicit 4-byte pad so the MSL `uint3 froxel_dims` (which the
    // MSL alignment rules promote to a 16-byte slot) and the Rust
    // `[u32; 3]` (which `#[repr(C)]` packs to 12 bytes) describe the
    // same on-wire layout. Without this padding `z_near` lands at
    // offset 76 in Rust but offset 80 in MSL, swapping `z_near` /
    // `z_far` and inverting the volume's Z mapping.
    pub _pad_align: u32,
    // Camera near-plane in view units. The fragment sampler maps a scene
    // view-z to a normalised volume w via `(z - z_near) / (z_far - z_near)`
    // for linear Z.
    pub z_near: f32,
    // Camera far-plane mirror: `FogSettings.max_distance`. The volume
    // covers `[z_near, max_distance]` along view-space depth. Beyond
    // `max_distance` the sampler clamps to the last slice (already
    // fully-integrated).
    pub z_far: f32,
    pub _pad: [f32; 2],
}

// Per-frame uniform consumed by the particle compute + render kernels. Carries
// the resolved emitter tunables (position, direction, gravity, ...) plus the
// dynamic per-frame inputs the compute kernel needs to age + integrate +
// respawn the pool. Pushed at compute buffer(2) and vertex buffer(1) of the
// Metal particle passes, so the layout must stay in sync with
// `ParticleParams` in `metal/shaders/particle.metal`. 144 bytes.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[allow(dead_code)] // Metal-only particle pipeline uniform; DirectX / Vulkan don't draw particles.
pub struct ParticleParams {
    // World-space spawn origin. `packed_float3` in MSL: alignment 4, with
    // `spread_cos` packed into the trailing float slot of the float4.
    pub position: [f32; 3],
    // Cosine of the emission cone's half-angle. `1.0` = straight jet,
    // `-1.0` = full sphere.
    pub spread_cos: f32,
    // Unit-length mean emission direction.
    pub direction: [f32; 3],
    pub speed_min: f32,
    // Constant per-frame acceleration applied to each particle.
    pub gravity: [f32; 3],
    pub speed_max: f32,
    // Linear-space RGBA at `age = 0`.
    pub color_start: [f32; 4],
    // Linear-space RGBA at `age = lifetime`.
    pub color_end: [f32; 4],
    pub lifetime_min: f32,
    pub lifetime_max: f32,
    pub size_start: f32,
    pub size_end: f32,
    // Frame delta time (seconds). Drives the age + integration step.
    pub dt: f32,
    // Integer count of new particles the compute kernel may emit this frame.
    // Carried atomically inside the kernel so threads racing for spawn slots
    // only succeed up to this many times.
    pub spawn_budget: u32,
    // Per-frame seed mixed with the thread id by the kernel's cheap RNG.
    pub random_seed: u32,
    // Pool size in slots; the kernel returns early past it.
    pub max_particles: u32,
}

// One text draw call: quads for all visible characters sharing one atlas texture.
pub struct TextDrawCall {
    // One quad (4 vertices, 6 indices) per character.
    pub vertices: Vec<TextVertex>,
    pub indices: Vec<u16>,
    // Index into the backend's text atlas texture array.
    pub atlas_slot: usize,
    // Optional clip rectangle in window pixels `[x, y, width, height]`. When
    // set, the backend scissors this call to the rect so a scrollable UI panel's
    // off-band rows do not bleed over its chrome. `None` draws unclipped.
    pub clip_rect: Option<[f32; 4]>,
}

// One LOD level past LOD0 for a `DrawObject`. Holds the rebased shared-index
// buffer slice and the camera-distance threshold above which the renderer
// picks this slice instead of the one stored on the parent `DrawObject`.
// `LodSlice`s are stored in ascending `switch_distance` order; the runtime
// picks the highest-indexed slice whose threshold is ≤ the current camera
// distance. The same `base_vertex` as the parent applies: LOD decimation
// reuses the LOD0 vertex range, only the index list changes.
#[derive(Copy, Clone, Debug)]
#[allow(dead_code)] // Vulkan still renders LOD0 only; the allow keeps that build warning-free.
pub struct LodSlice {
    pub index_offset: usize,
    pub index_count: usize,
    pub switch_distance: f32,
}

// One renderable object: vertex/index slice within the shared GPU buffers,
// a model matrix, albedo and normal-map texture slots, and material parameters.
pub struct DrawObject {
    // Byte offset into the shared vertex buffer.
    pub vertex_offset: usize,
    // Number of vertices this object occupies in the shared vertex buffer,
    // starting at `vertex_offset`. Used by mesh streaming to memcpy the
    // object's geometry region in and out of the GPU buffer.
    pub vertex_count: usize,
    // Element offset into the shared index buffer.
    pub index_offset: usize,
    // Number of indices to draw.
    pub index_count: usize,
    // Value added to every index before the vertex fetch (the
    // `drawIndexedPrimitives` `baseVertex` / D3D12 `BaseVertexLocation` /
    // Vulkan `vertex_offset`). 0 for static and streamed-mesh geometry, whose
    // indices are already absolute into the shared vertex buffer. Streamed
    // `VoxelWorld` chunks instead keep mesh-relative (0-based) indices and set
    // this to their vertex region's base, so a chunk placed past the
    // 65 535-vertex `u16` index range still renders. Read by all three
    // backends' shadow/main/velocity passes.
    pub base_vertex: i32,
    // Column-major model-to-world matrix.
    pub model: [[f32; 4]; 4],
    // Index into MtlContext::textures. Clamped to the last slot if out of range.
    #[allow(dead_code)]
    pub texture_slot: usize,
    // Index into MtlContext::normal_map_textures. 0 = flat-normal fallback.
    #[allow(dead_code)]
    pub normal_map_slot: usize,
    // Per-draw material scalars pushed to the fragment shader at buffer(3).
    pub material: MaterialUniforms,
    // When false the object is skipped in both the shadow and main passes.
    // Controlled by SceneReel to hide props belonging to inactive scenes.
    pub visible: bool,
    // When false the object's geometry is not yet uploaded to the GPU buffers
    // and the object is skipped in every pass. Always true unless the asset-
    // streaming subsystem is active; the mesh streamer flips it to true once
    // the geometry region is resident.
    // Read by the shadow/main/velocity passes.
    pub resident: bool,
    // World-space axis-aligned bounding box used for frustum culling.
    // Baked from the mesh's local bounds and the prop's initial model matrix
    // at GraphicsSystem init. Set to a degenerate (NaN) box to disable culling.
    pub bb_min: [f32; 3],
    pub bb_max: [f32; 3],
    // If > 0, the object is skipped when the camera is further than this from
    // the AABB centre. Lets a Prop opt in to distance-based unloading.
    pub cull_distance: f32,
    // LOD1..N slices for this draw, in ascending `switch_distance` order.
    // Empty when the mesh declared `lod_levels <= 1`; the renderer then
    // always uses the LOD0 `(index_offset, index_count)` carried on this
    // object. With non-empty alternates the renderer picks the highest
    // slice whose threshold is ≤ the camera→AABB-centre distance each frame.
    pub lod_alternates: Vec<LodSlice>,
}

impl DrawObject {
    // Pick the active `(index_offset, index_count)` for this object given
    // the current camera distance to its AABB centre. Returns the LOD0
    // pair when no alternates are present or when the camera is closer
    // than the first alternate's threshold; otherwise returns the
    // highest-indexed alternate whose `switch_distance` ≤ `distance`.
    pub fn active_lod(&self, distance: f32) -> (usize, usize) {
        let mut best = (self.index_offset, self.index_count);
        for slice in &self.lod_alternates {
            if distance >= slice.switch_distance {
                best = (slice.index_offset, slice.index_count);
            } else {
                break;
            }
        }
        best
    }

    // Returns true when the AABB encodes valid finite bounds suitable for
    // frustum/distance culling. A degenerate box (NaN min) disables culling.
    pub fn cullable(&self) -> bool {
        self.bb_min[0].is_finite()
            && self.bb_min[1].is_finite()
            && self.bb_min[2].is_finite()
            && self.bb_max[0].is_finite()
            && self.bb_max[1].is_finite()
            && self.bb_max[2].is_finite()
    }
}

// Per-object record consumed by the Metal "bindless" static main pass.
//
// The static pipeline no longer rebinds model/material/textures per draw
// call: every object's transform, material scalars, and texture-pool indices
// live in one GPU buffer, and the vertex/fragment shaders index it by the
// object id delivered through `[[base_instance]]`. That makes each draw call
// stateless, which is the prerequisite for the GPU-driven compute-cull +
// indirect-command-buffer pass.
//
// The renderer rebuilds the buffer each frame from its `DrawObject` list, so
// `update_model` / `update_visibility` changes are picked up automatically.
// `bb_min` / `bb_max` / `cull_distance` mirror the `DrawObject` AABB and are
// unused by the main-pass shaders; they are carried so a GPU-driven compute
// cull kernel can read object bounds straight from this buffer without a
// layout change.
//
// Layout (160 bytes) must stay in sync with the `GpuObjectData` struct in
// every backend's bindless static shader: `default.metal` (Metal), the inline
// `StructuredBuffer<GpuObjectData>` HLSL in `directx/pipeline.rs`, and the
// inline `std430` SSBO GLSL in `vulkan/pipeline.rs`. The `gpu_object_data_*`
// layout test below pins the offsets all three rely on.
//
// `albedo_index` / `normal_index` are *opaque* indices into each backend's
// texture pool: the pool's internal addressing differs per backend (Metal
// deduplicates into `[albedo..]++[normal..]`; DirectX and Vulkan address
// interleaved `(albedo, normal)` pairs), but the struct layout is identical.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GpuObjectData {
    // Column-major model-to-world matrix.
    pub model: [[f32; 4]; 4],
    // Linear-space RGB multiplier on the albedo sample.
    pub tint: [f32; 3],
    // Perceptual roughness [0, 1].
    pub roughness: f32,
    // Additive emission colour in linear space.
    pub emissive: [f32; 3],
    // Metallic factor [0, 1].
    pub metallic: f32,
    // Index into the bindless texture pool for this object's albedo map.
    pub albedo_index: u32,
    // Index into the bindless texture pool for this object's normal map.
    pub normal_index: u32,
    // Macro-variation strength [0, 1]; 0 disables the tiling-break noise.
    pub macro_variation: f32,
    // Terrain-shading blend [0, 1]; 0 disables (default PBR sampling),
    // non-zero switches the shader to a triplanar world-space projection
    // with slope-based rocky-tint blending. See `Material::terrain_blend`.
    pub terrain_blend: f32,
    // World-space AABB minimum (compute-cull bounds input).
    pub bb_min: [f32; 3],
    // View-distance cutoff; 0 = no cutoff (compute-cull bounds input).
    pub cull_distance: f32,
    // World-space AABB maximum (compute-cull bounds input).
    pub bb_max: [f32; 3],
    // Sharpness of the slope-based blend between primary and
    // `albedo_secondary_index` textures. 0 = wide soft gradient;
    // 1 = nearly hard cliff edge. Mirrors
    // `MaterialUniforms::secondary_blend_sharpness`.
    pub secondary_blend_sharpness: f32,
    // Bindless-pool index for the secondary albedo (slope-shaded).
    // 0 when the material doesn't declare a secondary texture; the
    // shader's `terrain_blend > 0` gate keeps it from being sampled
    // in that case.
    pub albedo_secondary_index: u32,
    // Bindless-pool index for the secondary normal map (slope-shaded).
    pub normal_secondary_index: u32,
    // Bindless-pool index for the emissive map (0 = none). Mirrors
    // `MaterialUniforms::emissive_map_index`.
    pub emissive_map_index: u32,
    // Bindless-pool index for the packed occlusion/roughness/metalness map
    // (0 = none). Mirrors `MaterialUniforms::orm_map_index`.
    pub orm_map_index: u32,
}

// Pack one `DrawObject` into its `GpuObjectData` record for the DirectX and
// Vulkan bindless static pass. The caller supplies the texture-pool indices
// because the pool addressing differs per backend: DirectX addresses an
// interleaved per-object SRV region (object `i` → albedo `2*i`, normal
// `2*i+1`); Vulkan addresses a deduplicated `[albedo..]++[normal..]` pool
// (albedo = `texture_slot`, normal = `albedo_count + normal_map_slot`),
// mirroring Metal. Only the 144-byte struct layout is shared across backends.
//
// `bb_min` / `bb_max` / `cull_distance` are copied even though the
// main-pass shaders ignore them, so a GPU-driven compute cull can read object
// bounds straight from this buffer without a layout change.
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub fn pack_object_record(obj: &DrawObject, albedo_index: u32, normal_index: u32) -> GpuObjectData {
    GpuObjectData {
        model: obj.model,
        tint: obj.material.tint,
        roughness: obj.material.roughness,
        emissive: obj.material.emissive,
        metallic: obj.material.metallic,
        albedo_index,
        normal_index,
        macro_variation: obj.material.macro_variation,
        terrain_blend: obj.material.terrain_blend,
        bb_min: obj.bb_min,
        cull_distance: obj.cull_distance,
        bb_max: obj.bb_max,
        secondary_blend_sharpness: obj.material.secondary_blend_sharpness,
        albedo_secondary_index: obj.material.albedo_secondary_index,
        normal_secondary_index: obj.material.normal_secondary_index,
        emissive_map_index: obj.material.emissive_map_index,
        orm_map_index: obj.material.orm_map_index,
    }
}

// Pack one instance of an `InstancedCluster` into a `GpuObjectData` for the
// GPU-driven bindless instanced path: the cluster's material + flat-pool texture
// indices, this instance's model matrix, and the instance's world AABB (the
// cluster's mesh-local AABB transformed by the model) so the compute cull tests
// each instance independently. Mirrors `pack_object_record`, sourcing the bounds
// from the cluster rather than a `DrawObject`. The instanced mesh's indices are
// already absolute (rebased at `append_mesh`), so the per-instance draw args use
// `base_vertex = 0`.
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub fn pack_instance_record(
    cluster: &InstancedCluster,
    model: [[f32; 4]; 4],
    albedo_index: u32,
    normal_index: u32,
) -> GpuObjectData {
    let (bb_min, bb_max) =
        crate::gfx::frustum::transform_aabb(cluster.local_bb_min, cluster.local_bb_max, model);
    GpuObjectData {
        model,
        tint: cluster.material.tint,
        roughness: cluster.material.roughness,
        emissive: cluster.material.emissive,
        metallic: cluster.material.metallic,
        albedo_index,
        normal_index,
        macro_variation: cluster.material.macro_variation,
        terrain_blend: cluster.material.terrain_blend,
        bb_min,
        cull_distance: cluster.cull_distance,
        bb_max,
        secondary_blend_sharpness: cluster.material.secondary_blend_sharpness,
        albedo_secondary_index: cluster.material.albedo_secondary_index,
        normal_secondary_index: cluster.material.normal_secondary_index,
        emissive_map_index: cluster.material.emissive_map_index,
        orm_map_index: cluster.material.orm_map_index,
    }
}

// Expand every instance of every cluster into a flat `GpuObjectData` list for the
// GPU-driven bindless instanced path, in cluster-then-instance order (so a
// parallel per-instance draw-args list can be walked in the same order). The flat
// texture-pool indices clamp to the pool (a stale slot reads the last valid entry
// rather than out of bounds), matching `build_object_buffer`'s static addressing.
// `albedo_count` is the albedo-pool size; `normal_count` the normal-pool size.
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub fn instance_object_records(
    clusters: &[InstancedCluster],
    albedo_count: u32,
    normal_count: u32,
) -> Vec<GpuObjectData> {
    let last_tex = albedo_count.saturating_sub(1);
    let last_nm = normal_count.saturating_sub(1);
    let total: usize = clusters.iter().map(|c| c.instances.len()).sum();
    let mut records = Vec::with_capacity(total);
    for cluster in clusters {
        let albedo = (cluster.texture_slot as u32).min(last_tex);
        let normal = albedo_count + (cluster.normal_map_slot as u32).min(last_nm);
        for &model in &cluster.instances {
            records.push(pack_instance_record(cluster, model, albedo, normal));
        }
    }
    records
}

// Factor the skinned mesh's bind-pose AABB half-extents are scaled by before the
// box is transformed into the per-frame world bound the GPU culler tests. Skinned
// geometry deforms every frame, so the bind-pose box does not cover poses that
// reach past it (raised arms, a wide stance); padding keeps those from being
// wrongly culled. Conservative -- loose, but never a false negative. A tight
// per-frame bound emitted by the skin compute kernel is a planned follow-up.
const SKINNED_BB_PAD_FACTOR: f32 = 2.0;

// Build the GPU-driven cull/draw record for one skinned object, folded into the
// bindless main pass as rigid geometry. The skin compute kernel deforms the
// bind-pose vertices into MODEL space, so `model` is applied after skinning
// (model -> world) exactly like a static object; the cull bound is the padded
// bind-pose AABB transformed by `model`. Mirrors `pack_instance_record`, sourcing
// material + flat texture indices from the skinned object.
// Consumed by the DirectX skinned fold; the allow keeps the Vulkan / Metal builds
// (whose folds are not yet wired) warning-free until they wire their callers.
#[allow(dead_code)]
pub fn pack_skinned_record(
    obj: &SkinnedDrawObject,
    albedo_index: u32,
    normal_index: u32,
) -> GpuObjectData {
    let mut lo = [0.0f32; 3];
    let mut hi = [0.0f32; 3];
    for a in 0..3 {
        let centre = 0.5 * (obj.local_bb_min[a] + obj.local_bb_max[a]);
        let half = 0.5 * (obj.local_bb_max[a] - obj.local_bb_min[a]) * SKINNED_BB_PAD_FACTOR;
        lo[a] = centre - half;
        hi[a] = centre + half;
    }
    let (bb_min, bb_max) = crate::gfx::frustum::transform_aabb(lo, hi, obj.model);
    GpuObjectData {
        model: obj.model,
        tint: obj.material.tint,
        roughness: obj.material.roughness,
        emissive: obj.material.emissive,
        metallic: obj.material.metallic,
        albedo_index,
        normal_index,
        macro_variation: obj.material.macro_variation,
        terrain_blend: obj.material.terrain_blend,
        bb_min,
        cull_distance: 0.0,
        bb_max,
        secondary_blend_sharpness: obj.material.secondary_blend_sharpness,
        albedo_secondary_index: obj.material.albedo_secondary_index,
        normal_secondary_index: obj.material.normal_secondary_index,
        emissive_map_index: obj.material.emissive_map_index,
        orm_map_index: obj.material.orm_map_index,
    }
}

// Per-object draw parameters consumed by the GPU-driven compute cull kernel.
// Parallel to `GpuObjectData` (one record per `DrawObject`, same index):
// `GpuObjectData` carries the cull bounds the kernel tests, this carries the
// indexed-draw arguments the kernel encodes into the indirect command buffer
// (Metal `MTLIndirectCommandBuffer` / DirectX `ExecuteIndirect` argument
// buffer / Vulkan `multiDrawIndexedIndirect` buffer) when an object survives
// the cull.
//
// The kernel reads `flags` to decide an object's fate without re-deriving it
// from the AABB: `ENABLED` gates the per-frame `visible` / `resident` state,
// `CULLABLE` selects whether the object is frustum/distance-tested at all
// (mirrors `DrawObject::cullable()`: non-cullable objects always draw).
//
// Layout (16 bytes) must stay in sync with the `GpuDrawArgs` struct in every
// backend's cull kernel: `default.metal` / the Metal `build_cull_pipeline`
// MSL, the inline cull HLSL in `directx/pipeline.rs`, and the inline cull
// GLSL in `vulkan/pipeline.rs`.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GpuDrawArgs {
    // Number of indices to draw (`DrawObject::index_count`).
    pub index_count: u32,
    // Element offset into the shared index buffer (`DrawObject::index_offset`).
    pub index_offset: u32,
    // Value added to every index before the vertex fetch
    // (`DrawObject::base_vertex`; always >= 0 in practice).
    pub base_vertex: u32,
    // Cull-decision bits; see `DrawArgsFlags`.
    pub flags: u32,
}

// Bit flags packed into `GpuDrawArgs::flags`.
pub struct DrawArgsFlags;

impl DrawArgsFlags {
    // The object is visible and its geometry is resident this frame. When
    // clear the kernel resets the object's indirect command (draws nothing).
    pub const ENABLED: u32 = 1;
    // The object has a finite AABB and should be frustum/distance-culled.
    // When clear the object always draws (subject to `ENABLED`).
    pub const CULLABLE: u32 = 2;
}

// Pack the per-frame cull-decision bits for one `DrawObject`. Mirrors the
// CPU `visible = BVH(cullable) + always_draw` partition: an object draws when
// it is visible and resident, and is frustum-tested only when it has finite
// bounds.
pub fn draw_args_flags(visible: bool, resident: bool, cullable: bool) -> u32 {
    let mut flags = 0;
    if visible && resident {
        flags |= DrawArgsFlags::ENABLED;
    }
    if cullable {
        flags |= DrawArgsFlags::CULLABLE;
    }
    flags
}

// One GPU-instanced draw: a shared mesh slice + material rendered at many
// world-space transforms in a single `drawIndexedInstanced` / `cmd_draw_indexed`
// (instance_count > 1). The cluster has a single union AABB across all
// instances; it is frustum-tested as a whole. Cluster culling trades per-
// instance precision for one draw call instead of N.
pub struct InstancedCluster {
    // Byte offset into the shared vertex buffer. Currently unused because
    // every backend keeps the shared vertex buffer bound at offset 0 and
    // `append_mesh` rebases indices accordingly; retained for symmetry with
    // `DrawObject` and future per-cluster vertex bindings.
    #[allow(dead_code)]
    pub vertex_offset: usize,
    // Number of vertices this cluster's mesh occupies in the shared vertex
    // buffer, starting at `vertex_offset`. Used by the shrinkable-seed
    // compaction to relocate the cluster's geometry region when streamed-mesh
    // gaps are removed from the shared buffer.
    #[allow(dead_code)]
    pub vertex_count: usize,
    // Element offset into the shared index buffer.
    pub index_offset: usize,
    // Number of indices per instance.
    pub index_count: usize,
    // Index into the backend's albedo texture pool.
    pub texture_slot: usize,
    // Index into the backend's normal-map pool. 0 = flat-normal fallback.
    pub normal_map_slot: usize,
    // Per-cluster material scalars, identical for every instance.
    pub material: MaterialUniforms,
    // Union AABB of all per-instance world AABBs; used for cluster-wide
    // frustum culling. A degenerate (NaN) box disables culling.
    pub cluster_bb_min: [f32; 3],
    pub cluster_bb_max: [f32; 3],
    // Mesh-local (object-space) AABB, identical for every instance. The
    // GPU-driven instanced path transforms this by each instance's model matrix
    // to get a per-instance world AABB for per-instance frustum/distance/Hi-Z
    // culling; the union AABB above is only the whole-cluster bound.
    pub local_bb_min: [f32; 3],
    pub local_bb_max: [f32; 3],
    // View-distance cutoff applied to the cluster centre. 0 = no cutoff.
    pub cull_distance: f32,
    // Per-instance column-major model matrices. Uploaded to a transient GPU
    // buffer each frame.
    pub instances: Vec<[[f32; 4]; 4]>,
    // LOD1..N slices for this cluster's mesh, in ascending `switch_distance`
    // order. Empty when the mesh declared `lod_levels <= 1`. Each per-instance
    // LOD is picked from camera distance to that instance's translation; the
    // per-pass draw loop partitions `instances` by their picked LOD and issues
    // one `drawIndexedInstanced` per non-empty bucket. The vertex set is
    // shared with LOD0 (QEM decimation preserves vertices), so only
    // `(index_offset, index_count)` varies per slice.
    pub lod_alternates: Vec<LodSlice>,
}

// One LOD bucket emitted by [`InstancedCluster::lod_buckets`]: the index
// range for that LOD plus the subset of instance matrices that picked it.
// Each bucket becomes one `drawIndexedInstanced` call.
#[derive(Clone, Debug)]
#[allow(dead_code)] // Consumed by the client backends' instanced draw paths.
pub struct InstancedLodBucket {
    pub index_offset: usize,
    pub index_count: usize,
    pub instances: Vec<[[f32; 4]; 4]>,
}

impl InstancedCluster {
    // True when the cluster AABB encodes valid finite bounds suitable for
    // frustum / distance culling.
    pub fn cullable(&self) -> bool {
        self.cluster_bb_min[0].is_finite()
            && self.cluster_bb_min[1].is_finite()
            && self.cluster_bb_min[2].is_finite()
            && self.cluster_bb_max[0].is_finite()
            && self.cluster_bb_max[1].is_finite()
            && self.cluster_bb_max[2].is_finite()
    }

    // Partition the cluster's instances into LOD buckets keyed by camera
    // distance to each instance's translation. Returns one entry per
    // non-empty bucket, in mesh-LOD order (LOD0 first). With no alternates
    // every instance lands in a single LOD0 bucket so the caller's loop
    // degenerates to the legacy one-`drawIndexedInstanced` path.
    //
    // Per-instance distance uses the model-matrix translation rather than
    // a transformed-AABB centre: close enough for distance-keyed swaps
    // without paying the per-instance AABB transform every pass.
    #[allow(dead_code)] // Consumed by every backend's instanced draw path (client crate).
    pub fn lod_buckets(&self, cam_pos: [f32; 3]) -> Vec<InstancedLodBucket> {
        // Fast path: no alternates → single LOD0 bucket containing every
        // instance. Cloning is unavoidable since the GPU buffer expects an
        // owned, contiguous slice.
        if self.lod_alternates.is_empty() || self.instances.is_empty() {
            if self.instances.is_empty() {
                return Vec::new();
            }
            return vec![InstancedLodBucket {
                index_offset: self.index_offset,
                index_count: self.index_count,
                instances: self.instances.clone(),
            }];
        }

        let n_levels = self.lod_alternates.len() + 1;
        let mut buckets: Vec<InstancedLodBucket> = Vec::with_capacity(n_levels);
        buckets.push(InstancedLodBucket {
            index_offset: self.index_offset,
            index_count: self.index_count,
            instances: Vec::new(),
        });
        for alt in &self.lod_alternates {
            buckets.push(InstancedLodBucket {
                index_offset: alt.index_offset,
                index_count: alt.index_count,
                instances: Vec::new(),
            });
        }

        for m in &self.instances {
            let dx = m[3][0] - cam_pos[0];
            let dy = m[3][1] - cam_pos[1];
            let dz = m[3][2] - cam_pos[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let mut pick = 0usize;
            for (i, alt) in self.lod_alternates.iter().enumerate() {
                if d >= alt.switch_distance {
                    pick = i + 1;
                } else {
                    break;
                }
            }
            buckets[pick].instances.push(*m);
        }

        buckets.retain(|b| !b.instances.is_empty());
        buckets
    }

    // Visit each non-empty LOD bucket for this cluster, calling
    // `visit(index_offset, index_count, instances)` once per bucket in
    // mesh-LOD order (LOD0 first). Stops and returns the first error a `visit`
    // produces.
    //
    // In the common no-alternates case `instances` borrows `self.instances`
    // directly (no allocation, no copy) so a caller that memcpy's the slice
    // into a GPU buffer skips the wholesale clone that the owned
    // [`lod_buckets`](Self::lod_buckets) makes. Clusters that declared LOD
    // alternates still regroup their instances per LOD (a copy that separate
    // per-LOD draw calls genuinely require); that path reuses `lod_buckets`.
    #[allow(dead_code)] // Metal consumes this; other backends use owned `lod_buckets`.
    pub fn try_for_each_lod_bucket<E>(
        &self,
        cam_pos: [f32; 3],
        mut visit: impl FnMut(usize, usize, &[[[f32; 4]; 4]]) -> Result<(), E>,
    ) -> Result<(), E> {
        if self.instances.is_empty() {
            return Ok(());
        }
        if self.lod_alternates.is_empty() {
            return visit(self.index_offset, self.index_count, &self.instances);
        }
        for b in self.lod_buckets(cam_pos) {
            visit(b.index_offset, b.index_count, &b.instances)?;
        }
        Ok(())
    }

    // Infallible [`try_for_each_lod_bucket`](Self::try_for_each_lod_bucket)
    // for callers whose per-bucket work cannot fail.
    #[allow(dead_code)] // Metal consumes this; other backends use owned `lod_buckets`.
    pub fn for_each_lod_bucket(
        &self,
        cam_pos: [f32; 3],
        mut visit: impl FnMut(usize, usize, &[[[f32; 4]; 4]]),
    ) {
        let _ = self.try_for_each_lod_bucket::<core::convert::Infallible>(
            cam_pos,
            |index_offset, index_count, instances| {
                visit(index_offset, index_count, instances);
                Ok(())
            },
        );
    }
}

// One skeletally animated draw: a slice of the shared skinned vertex/index
// buffers plus the per-joint matrix buffer the vertex shader blends.
//
// Skinned objects deform every frame, so they are excluded from the static
// BVH and drawn unconditionally (after the visibility flag). The joint
// matrices live in a separate per-object GPU buffer the renderer rewrites
// each frame from `AnimationSystem`'s output.
pub struct SkinnedDrawObject {
    // Vertex offset (in vertex units, not bytes) of this slot's first
    // vertex in the shared skinned vertex buffer. The shared index buffer
    // holds absolute vertex indices (each slot's mesh-relative indices are
    // re-based onto this `vertex_base` at upload time), so the value is
    // not consumed by the draw call itself, but the asset hot-reload's
    // `rebuild_skinned_geometry` path needs to know each slot's vertex
    // region to copy unchanged geometry across a size-changing rebuild.
    // Stored as u16 because the shared skinned index buffer is u16 and
    // every slot's `vertex_base` has to fit there.
    #[allow(dead_code)] // Read only by Metal's rebuild_skinned_geometry hot-reload.
    pub vertex_base: u16,
    // Number of vertices in this slot's region of the shared skinned
    // vertex buffer.
    #[allow(dead_code)] // Read only by Metal's rebuild_skinned_geometry hot-reload.
    pub vertex_count: usize,
    // Element offset into the shared skinned index buffer.
    pub index_offset: usize,
    // Number of indices to draw.
    pub index_count: usize,
    // Column-major model-to-world matrix. Applied after skinning.
    pub model: [[f32; 4]; 4],
    // Index into the backend's albedo texture pool.
    pub texture_slot: usize,
    // Index into the backend's normal-map pool. 0 = flat-normal fallback.
    pub normal_map_slot: usize,
    // Per-object material scalars.
    pub material: MaterialUniforms,
    // When false the object is skipped in both the shadow and main passes.
    pub visible: bool,
    // Number of joints in this object's skeleton, capped at `MAX_JOINTS`.
    // The vertex shader only blends matrices below this count.
    pub joint_count: usize,
    // Mesh-local (object-space) bind-pose AABB, computed from the bind-pose
    // vertices at load. The GPU-driven skinned fold pads this (animations reach
    // past the bind pose) and transforms it by `model` to get the per-frame world
    // bound the cull kernel frustum/Hi-Z tests. See `pack_skinned_record`.
    pub local_bb_min: [f32; 3],
    pub local_bb_max: [f32; 3],
    // LOD1..N slices, in ascending `switch_distance` order. Empty when
    // the SkinnedMesh declared `lod_levels <= 1`. Each slice points at a
    // rebased index range in the shared skinned IB; QEM half-edge
    // decimation keeps the vertex set unchanged so all LODs share this
    // slot's `vertex_base` / `vertex_count`.
    #[allow(dead_code)] // Consumed by every backend's skinned draw path (client crate).
    pub lod_alternates: Vec<LodSlice>,
}

#[allow(dead_code)] // Consumed by every backend's skinned draw path (client crate).
impl SkinnedDrawObject {
    // Pick the active `(index_offset, index_count)` for this object given
    // the camera distance to its model-matrix translation. Returns the
    // LOD0 pair when no alternates are present or the distance is below
    // the first threshold.
    pub fn active_lod(&self, distance: f32) -> (usize, usize) {
        let mut best = (self.index_offset, self.index_count);
        for slice in &self.lod_alternates {
            if distance >= slice.switch_distance {
                best = (slice.index_offset, slice.index_count);
            } else {
                break;
            }
        }
        best
    }

    // World-space translation of this object (column 3 of the model
    // matrix). Skinned objects have no static AABB (they deform every
    // frame), so per-frame LOD picks use the model translation as the
    // stand-in for an AABB centre.
    pub fn translation(&self) -> [f32; 3] {
        [self.model[3][0], self.model[3][1], self.model[3][2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn gpu_object_data_layout_matches_msl() {
        // The MSL `GpuObjectData` in default.metal lays the struct out with
        // float4x4's 16-byte alignment; the offsets below must match it
        // exactly or the bindless static pass reads garbage.
        assert_eq!(size_of::<GpuObjectData>(), 160);
        assert_eq!(offset_of!(GpuObjectData, model), 0);
        assert_eq!(offset_of!(GpuObjectData, tint), 64);
        assert_eq!(offset_of!(GpuObjectData, roughness), 76);
        assert_eq!(offset_of!(GpuObjectData, emissive), 80);
        assert_eq!(offset_of!(GpuObjectData, metallic), 92);
        assert_eq!(offset_of!(GpuObjectData, albedo_index), 96);
        assert_eq!(offset_of!(GpuObjectData, normal_index), 100);
        assert_eq!(offset_of!(GpuObjectData, macro_variation), 104);
        assert_eq!(offset_of!(GpuObjectData, terrain_blend), 108);
        assert_eq!(offset_of!(GpuObjectData, bb_min), 112);
        assert_eq!(offset_of!(GpuObjectData, cull_distance), 124);
        assert_eq!(offset_of!(GpuObjectData, bb_max), 128);
        assert_eq!(offset_of!(GpuObjectData, secondary_blend_sharpness), 140);
        assert_eq!(offset_of!(GpuObjectData, albedo_secondary_index), 144);
        assert_eq!(offset_of!(GpuObjectData, normal_secondary_index), 148);
        assert_eq!(offset_of!(GpuObjectData, emissive_map_index), 152);
        assert_eq!(offset_of!(GpuObjectData, orm_map_index), 156);
        // Size is a multiple of 16 so an array of records keeps every
        // float4x4 model matrix 16-byte aligned.
        assert_eq!(size_of::<GpuObjectData>() % 16, 0);
        assert_eq!(align_of::<f32>(), 4);
    }

    #[test]
    fn rt_geom_entry_layout_matches_msl() {
        // The MSL `RtGeomEntry` in rt_reflections.metal must match this exactly,
        // which requires `tint` and `emissive` to be `packed_float3` there (NOT
        // `float3`). A `float3` would 16-byte-align `tint`, pushing `roughness`
        // to 32 and `model` to 64. `model` at offset 48 is already 16-aligned, so
        // the float4x4 needs no padding; the `_pad` tail then rounds the struct
        // to 128 bytes so the shader's matrix-bearing struct (which the GPU rounds
        // up to a 16-byte multiple) strides identically to the Rust side.
        assert_eq!(size_of::<RtGeomEntry>(), 128);
        assert_eq!(offset_of!(RtGeomEntry, index_offset), 0);
        assert_eq!(offset_of!(RtGeomEntry, base_vertex), 4);
        assert_eq!(offset_of!(RtGeomEntry, albedo_index), 8);
        assert_eq!(offset_of!(RtGeomEntry, normal_index), 12);
        assert_eq!(offset_of!(RtGeomEntry, tint), 16);
        assert_eq!(offset_of!(RtGeomEntry, roughness), 28);
        assert_eq!(offset_of!(RtGeomEntry, metallic), 32);
        assert_eq!(offset_of!(RtGeomEntry, emissive), 36);
        assert_eq!(offset_of!(RtGeomEntry, model), 48);
        assert_eq!(offset_of!(RtGeomEntry, emissive_map_index), 112);
    }

    #[test]
    fn material_uniforms_layout_matches_msl() {
        // The MSL `MaterialUniforms` in default.metal declares
        // `tint` and `emissive` as packed_float3 (align 4), so the offsets line
        // up with this tightly-packed Rust struct. A plain float3 there would
        // 16-align `tint` and shift every following field.
        assert_eq!(size_of::<MaterialUniforms>(), 76);
        assert_eq!(offset_of!(MaterialUniforms, roughness), 0);
        assert_eq!(offset_of!(MaterialUniforms, metallic), 4);
        assert_eq!(offset_of!(MaterialUniforms, macro_variation), 8);
        assert_eq!(offset_of!(MaterialUniforms, terrain_blend), 12);
        assert_eq!(offset_of!(MaterialUniforms, tint), 16);
        assert_eq!(offset_of!(MaterialUniforms, _pad2), 28);
        assert_eq!(offset_of!(MaterialUniforms, emissive), 32);
        assert_eq!(offset_of!(MaterialUniforms, secondary_blend_sharpness), 44);
        assert_eq!(offset_of!(MaterialUniforms, albedo_secondary_index), 48);
        assert_eq!(offset_of!(MaterialUniforms, normal_secondary_index), 52);
        assert_eq!(offset_of!(MaterialUniforms, emissive_map_index), 56);
        assert_eq!(offset_of!(MaterialUniforms, orm_map_index), 60);
        assert_eq!(offset_of!(MaterialUniforms, opacity), 64);
        assert_eq!(offset_of!(MaterialUniforms, transparent), 68);
        assert_eq!(offset_of!(MaterialUniforms, see_through), 72);
    }

    #[test]
    fn directional_light_data_layout_matches_msl() {
        // MSL `DirectionalLightData` uses packed_float3 for `direction` and
        // `color` so the 32-byte stride matches; a plain float3 would read the
        // colour channel as zeros (see the comment in default.metal).
        assert_eq!(size_of::<DirectionalLightData>(), 32);
        assert_eq!(offset_of!(DirectionalLightData, direction), 0);
        assert_eq!(offset_of!(DirectionalLightData, intensity), 12);
        assert_eq!(offset_of!(DirectionalLightData, color), 16);
        assert_eq!(offset_of!(DirectionalLightData, _pad), 28);
    }

    #[test]
    fn point_light_data_layout_matches_msl() {
        // MSL `PointLightData` uses packed_float3 for `position` and `color`.
        assert_eq!(size_of::<PointLightData>(), 32);
        assert_eq!(offset_of!(PointLightData, position), 0);
        assert_eq!(offset_of!(PointLightData, range), 12);
        assert_eq!(offset_of!(PointLightData, color), 16);
        assert_eq!(offset_of!(PointLightData, intensity), 28);
    }

    #[test]
    fn light_uniforms_layout_matches_msl() {
        // MSL `LightUniforms` in default.metal (and `RaymarchLights` in
        // raymarch_helpers.metal, which is bound from this same Rust struct):
        // DirectionalLightData[4] then PointLightData[8] then two ints, then
        // ambient_intensity + one pad word in the trailing 16-byte block.
        assert_eq!(size_of::<LightUniforms>(), 400);
        assert_eq!(offset_of!(LightUniforms, directional), 0);
        assert_eq!(offset_of!(LightUniforms, point), 128);
        assert_eq!(offset_of!(LightUniforms, num_directional), 384);
        assert_eq!(offset_of!(LightUniforms, num_point), 388);
        assert_eq!(offset_of!(LightUniforms, ambient_intensity), 392);
        assert_eq!(offset_of!(LightUniforms, _pad), 396);
    }

    #[test]
    fn shadow_uniforms_layout_matches_msl() {
        // MSL `ShadowUniforms` in default.metal / shadow_map.metal (and
        // `RaymarchShadowUniforms` in raymarch_helpers.metal, bound from this
        // same struct): NUM_SHADOW_CASCADES float4x4s, then the splits, then the
        // active-cascade count. The MSL declares the splits as `float4`
        // (default/shadow) or `float[4]` (raymarch); both occupy the same 16
        // bytes as the Rust `[f32; 4]`. `active_cascades` is a `uint` at offset
        // 272; the struct rounds up to 288 for the 16-byte (float4x4) alignment.
        assert_eq!(size_of::<ShadowUniforms>(), 288);
        assert_eq!(offset_of!(ShadowUniforms, light_vps), 0);
        assert_eq!(offset_of!(ShadowUniforms, cascade_splits), 256);
        assert_eq!(offset_of!(ShadowUniforms, active_cascades), 272);
        // 16-aligned size keeps the float4x4 array head aligned in MSL.
        assert_eq!(size_of::<ShadowUniforms>() % 16, 0);
    }

    #[cfg(backend_metal)]
    #[test]
    fn shadow_pass_push_layout_matches_msl() {
        // MSL `ShadowPassPush` in shadow_map.metal: a uint + three pad uints.
        assert_eq!(size_of::<ShadowPassPush>(), 16);
        assert_eq!(offset_of!(ShadowPassPush, cascade_idx), 0);
        assert_eq!(offset_of!(ShadowPassPush, _pad), 4);
    }

    #[test]
    fn text_vertex_layout_matches_msl() {
        // The text vertex buffer is consumed through a vertex descriptor whose
        // attributes sit at offsets 0 (pos), 8 (uv), 16 (color) with a 32-byte
        // stride, matching the `TextVtxIn` attribute slots in text.metal.
        assert_eq!(size_of::<TextVertex>(), 32);
        assert_eq!(offset_of!(TextVertex, pos), 0);
        assert_eq!(offset_of!(TextVertex, uv), 8);
        assert_eq!(offset_of!(TextVertex, color), 16);
        assert_eq!(offset_of!(TextVertex, _pad), 28);
    }

    #[test]
    fn text_uniforms_layout_matches_msl() {
        // MSL `TextUniforms` in text.metal: four floats.
        assert_eq!(size_of::<TextUniforms>(), 16);
        assert_eq!(offset_of!(TextUniforms, win_width), 0);
        assert_eq!(offset_of!(TextUniforms, win_height), 4);
        assert_eq!(offset_of!(TextUniforms, _pad), 8);
    }

    #[test]
    fn post_process_params_layout_matches_msl() {
        // MSL `PostUniforms` in post.metal / bloom.metal: nine floats.
        assert_eq!(size_of::<PostProcessParams>(), 36);
        assert_eq!(offset_of!(PostProcessParams, bloom_intensity), 0);
        assert_eq!(offset_of!(PostProcessParams, bloom_threshold), 4);
        assert_eq!(offset_of!(PostProcessParams, bloom_knee), 8);
        assert_eq!(offset_of!(PostProcessParams, exposure), 12);
        assert_eq!(offset_of!(PostProcessParams, vignette), 16);
        assert_eq!(offset_of!(PostProcessParams, lut_strength), 20);
        assert_eq!(offset_of!(PostProcessParams, hdr_output), 24);
        assert_eq!(offset_of!(PostProcessParams, pq_output), 28);
        assert_eq!(offset_of!(PostProcessParams, fxaa), 32);
    }

    #[test]
    fn ssao_params_layout_matches_msl() {
        // MSL `SsaoParams` in ssao.metal: four floats.
        assert_eq!(size_of::<SsaoParams>(), 16);
        assert_eq!(offset_of!(SsaoParams, radius), 0);
        assert_eq!(offset_of!(SsaoParams, intensity), 4);
        assert_eq!(offset_of!(SsaoParams, tan_half_fov_y), 8);
        assert_eq!(offset_of!(SsaoParams, aspect), 12);
    }

    #[test]
    fn ssr_params_layout_matches_msl() {
        // MSL `SsrParams` in ssr.metal: eight scalars then a float4x4 at the
        // already-16-aligned offset 32.
        assert_eq!(size_of::<SsrParams>(), 96);
        assert_eq!(offset_of!(SsrParams, intensity), 0);
        assert_eq!(offset_of!(SsrParams, max_distance), 4);
        assert_eq!(offset_of!(SsrParams, tan_half_fov_y), 8);
        assert_eq!(offset_of!(SsrParams, aspect), 12);
        assert_eq!(offset_of!(SsrParams, stride), 16);
        assert_eq!(offset_of!(SsrParams, thickness), 20);
        assert_eq!(offset_of!(SsrParams, prefilter_mip_count), 24);
        assert_eq!(offset_of!(SsrParams, _pad), 28);
        assert_eq!(offset_of!(SsrParams, inv_view), 32);
    }

    #[test]
    fn ssgi_params_layout_matches_shaders() {
        // Eight floats, byte-identical across the MSL `SsgiParams` in
        // ssgi.metal and the HLSL `SsgiParams` cbuffer in ssgi.hlsl. The last
        // two floats (`rays`, `steps`) sit where `_pad0`/`_pad1` used to: the
        // Metal shader reads them as loop bounds; backends that still bake
        // compile-time counts leave those fields as inert padding, so the
        // layout is unchanged.
        assert_eq!(size_of::<SsgiParams>(), 32);
        assert_eq!(offset_of!(SsgiParams, intensity), 0);
        assert_eq!(offset_of!(SsgiParams, max_distance), 4);
        assert_eq!(offset_of!(SsgiParams, tan_half_fov_y), 8);
        assert_eq!(offset_of!(SsgiParams, aspect), 12);
        assert_eq!(offset_of!(SsgiParams, stride), 16);
        assert_eq!(offset_of!(SsgiParams, thickness), 20);
        assert_eq!(offset_of!(SsgiParams, rays), 24);
        assert_eq!(offset_of!(SsgiParams, steps), 28);
    }

    #[test]
    fn rt_params_layout_matches_msl() {
        // MSL `RtParams` in rt_reflections.metal: eight scalars, three float4s,
        // then a float4x4. Every float4/float4x4 lands at a 16-aligned offset.
        assert_eq!(size_of::<RtParams>(), 144);
        assert_eq!(offset_of!(RtParams, intensity), 0);
        assert_eq!(offset_of!(RtParams, max_distance), 4);
        assert_eq!(offset_of!(RtParams, tan_half_fov_y), 8);
        assert_eq!(offset_of!(RtParams, aspect), 12);
        assert_eq!(offset_of!(RtParams, prefilter_mip_count), 16);
        assert_eq!(offset_of!(RtParams, _pad0), 20);
        assert_eq!(offset_of!(RtParams, _pad1), 24);
        assert_eq!(offset_of!(RtParams, _pad2), 28);
        assert_eq!(offset_of!(RtParams, cam_pos), 32);
        assert_eq!(offset_of!(RtParams, sun_dir), 48);
        assert_eq!(offset_of!(RtParams, sun_color), 64);
        assert_eq!(offset_of!(RtParams, inv_view), 80);
    }

    fn sample_draw_object() -> DrawObject {
        DrawObject {
            vertex_offset: 0,
            vertex_count: 8,
            index_offset: 12,
            index_count: 36,
            base_vertex: 4,
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [5.0, 6.0, 7.0, 1.0],
            ],
            texture_slot: 9,
            normal_map_slot: 2,
            material: MaterialUniforms {
                roughness: 0.3,
                metallic: 0.7,
                macro_variation: 0.5,
                terrain_blend: 0.0,
                tint: [0.1, 0.2, 0.3],
                _pad2: 0.0,
                emissive: [0.4, 0.5, 0.6],
                secondary_blend_sharpness: 0.5,
                albedo_secondary_index: 0,
                normal_secondary_index: 0,
                emissive_map_index: 0,
                orm_map_index: 0,
                opacity: 1.0,
                transparent: 0,
                see_through: 0,
            },
            visible: true,
            resident: true,
            bb_min: [-1.0, -2.0, -3.0],
            bb_max: [1.0, 2.0, 3.0],
            cull_distance: 42.0,
            lod_alternates: Vec::new(),
        }
    }

    #[test]
    fn active_lod_returns_lod0_when_no_alternates() {
        let obj = sample_draw_object();
        assert_eq!(obj.active_lod(0.0), (obj.index_offset, obj.index_count));
        assert_eq!(obj.active_lod(1000.0), (obj.index_offset, obj.index_count));
    }

    #[test]
    fn instance_object_records_expand_clamp_and_transform_bounds() {
        use crate::gfx::frustum::transform_aabb;
        let translate = |x: f32| {
            let mut m = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            m[3][0] = x;
            m
        };
        // Cluster A: 2 instances, in-range texture/normal slots, unit-cube mesh.
        let mut a = sample_cluster(vec![translate(0.0), translate(10.0)], Vec::new());
        a.texture_slot = 3;
        a.normal_map_slot = 2;
        a.local_bb_min = [0.0, 0.0, 0.0];
        a.local_bb_max = [1.0, 1.0, 1.0];
        // Cluster B: 1 instance, out-of-range slots that must clamp to the pool.
        let mut b = sample_cluster(vec![translate(5.0)], Vec::new());
        b.texture_slot = 99;
        b.normal_map_slot = 99;
        b.local_bb_min = [0.0, 0.0, 0.0];
        b.local_bb_max = [2.0, 2.0, 2.0];

        let recs = instance_object_records(&[a, b], 8, 4);
        // Cluster-then-instance order: 2 from A, then 1 from B.
        assert_eq!(recs.len(), 3);
        // Flat pool addressing: albedo = texture_slot, normal = albedo_count + slot.
        assert_eq!(recs[0].albedo_index, 3);
        assert_eq!(recs[0].normal_index, 8 + 2);
        assert_eq!(recs[1].albedo_index, 3);
        // The second instance carries its own model + translated world AABB.
        assert_eq!(recs[1].model[3][0], 10.0);
        let (exp_min, exp_max) = transform_aabb([0.0; 3], [1.0, 1.0, 1.0], translate(10.0));
        assert_eq!(recs[1].bb_min, exp_min);
        assert_eq!(recs[1].bb_max, exp_max);
        // Out-of-range slots clamp to the last valid albedo (7) / normal (8+3) entry.
        assert_eq!(recs[2].albedo_index, 7);
        assert_eq!(recs[2].normal_index, 8 + 3);
    }

    #[test]
    fn pack_skinned_record_pads_and_transforms_bounds() {
        use crate::gfx::frustum::transform_aabb;
        let mut model = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        model[3][0] = 10.0;
        let obj = SkinnedDrawObject {
            vertex_base: 0,
            vertex_count: 10,
            index_offset: 0,
            index_count: 30,
            model,
            texture_slot: 5,
            normal_map_slot: 2,
            material: MaterialUniforms::DEFAULT,
            visible: true,
            joint_count: 4,
            local_bb_min: [-1.0, -1.0, -1.0],
            local_bb_max: [1.0, 1.0, 1.0],
            lod_alternates: Vec::new(),
        };
        let rec = pack_skinned_record(&obj, 5, 8 + 2);
        // Model is applied after skinning (the kernel deforms into model space).
        assert_eq!(rec.model, model);
        assert_eq!(rec.albedo_index, 5);
        assert_eq!(rec.normal_index, 8 + 2);
        // No distance cutoff for skinned (frustum + Hi-Z only).
        assert_eq!(rec.cull_distance, 0.0);
        // Bind-pose half-extent 1.0, padded x2 -> 2.0, then translated +10 on x.
        let (exp_min, exp_max) = transform_aabb([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0], model);
        assert_eq!(rec.bb_min, exp_min);
        assert_eq!(rec.bb_max, exp_max);
        assert_eq!(rec.bb_min[0], 8.0);
        assert_eq!(rec.bb_max[0], 12.0);
    }

    #[test]
    fn active_lod_picks_highest_passing_threshold() {
        let mut obj = sample_draw_object();
        obj.lod_alternates = vec![
            LodSlice {
                index_offset: 100,
                index_count: 18,
                switch_distance: 10.0,
            },
            LodSlice {
                index_offset: 200,
                index_count: 9,
                switch_distance: 25.0,
            },
            LodSlice {
                index_offset: 300,
                index_count: 3,
                switch_distance: 60.0,
            },
        ];
        // Below the first threshold → LOD0.
        assert_eq!(obj.active_lod(5.0), (obj.index_offset, obj.index_count));
        // Just past the first → LOD1.
        assert_eq!(obj.active_lod(15.0), (100, 18));
        // Between LOD2 and LOD3 thresholds → LOD2.
        assert_eq!(obj.active_lod(40.0), (200, 9));
        // Past everything → LOD3.
        assert_eq!(obj.active_lod(120.0), (300, 3));
    }

    fn sample_instance_translation(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [x, y, z, 1.0],
        ]
    }

    fn sample_cluster(
        instances: Vec<[[f32; 4]; 4]>,
        lod_alternates: Vec<LodSlice>,
    ) -> InstancedCluster {
        InstancedCluster {
            vertex_offset: 0,
            vertex_count: 0,
            index_offset: 0,
            index_count: 60,
            texture_slot: 0,
            normal_map_slot: 0,
            material: MaterialUniforms::DEFAULT,
            cluster_bb_min: [f32::NAN; 3],
            cluster_bb_max: [f32::NAN; 3],
            local_bb_min: [0.0; 3],
            local_bb_max: [0.0; 3],
            cull_distance: 0.0,
            instances,
            lod_alternates,
        }
    }

    #[test]
    fn lod_buckets_no_alternates_returns_single_lod0_bucket() {
        let c = sample_cluster(
            vec![
                sample_instance_translation(0.0, 0.0, 0.0),
                sample_instance_translation(50.0, 0.0, 0.0),
            ],
            Vec::new(),
        );
        let buckets = c.lod_buckets([0.0; 3]);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].index_offset, 0);
        assert_eq!(buckets[0].index_count, 60);
        assert_eq!(buckets[0].instances.len(), 2);
    }

    #[test]
    fn lod_buckets_partitions_instances_by_camera_distance() {
        let c = sample_cluster(
            vec![
                sample_instance_translation(0.0, 0.0, 0.0),   // d=0 → LOD0
                sample_instance_translation(20.0, 0.0, 0.0),  // d=20 → LOD1
                sample_instance_translation(50.0, 0.0, 0.0),  // d=50 → LOD2
                sample_instance_translation(120.0, 0.0, 0.0), // d=120 → LOD3
            ],
            vec![
                LodSlice {
                    index_offset: 100,
                    index_count: 30,
                    switch_distance: 10.0,
                },
                LodSlice {
                    index_offset: 200,
                    index_count: 15,
                    switch_distance: 25.0,
                },
                LodSlice {
                    index_offset: 300,
                    index_count: 5,
                    switch_distance: 60.0,
                },
            ],
        );
        let buckets = c.lod_buckets([0.0; 3]);
        // Each LOD got exactly one instance, so all four buckets survive.
        assert_eq!(buckets.len(), 4);
        assert_eq!(buckets[0].index_count, 60);
        assert_eq!(buckets[0].instances.len(), 1);
        assert_eq!(buckets[1].index_offset, 100);
        assert_eq!(buckets[1].instances.len(), 1);
        assert_eq!(buckets[2].index_offset, 200);
        assert_eq!(buckets[2].instances.len(), 1);
        assert_eq!(buckets[3].index_offset, 300);
        assert_eq!(buckets[3].instances.len(), 1);
    }

    #[test]
    fn lod_buckets_drops_empty_levels() {
        // Every instance lands in LOD0; the LOD1 bucket should not be emitted.
        let c = sample_cluster(
            vec![sample_instance_translation(0.0, 0.0, 0.0)],
            vec![LodSlice {
                index_offset: 100,
                index_count: 30,
                switch_distance: 50.0,
            }],
        );
        let buckets = c.lod_buckets([0.0; 3]);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].index_offset, 0);
    }

    #[test]
    fn lod_buckets_empty_instances_returns_no_buckets() {
        let c = sample_cluster(Vec::new(), Vec::new());
        assert!(c.lod_buckets([0.0; 3]).is_empty());
    }

    #[test]
    fn for_each_lod_bucket_no_alternates_borrows_full_instance_slice() {
        // The no-alternates fast path must hand the visitor the cluster's own
        // instance slice in one bucket at the LOD0 range: the zero-copy
        // contract the GPU upload relies on.
        let c = sample_cluster(
            vec![
                sample_instance_translation(0.0, 0.0, 0.0),
                sample_instance_translation(50.0, 0.0, 0.0),
            ],
            Vec::new(),
        );
        let mut seen: Vec<(usize, usize, usize)> = Vec::new();
        c.for_each_lod_bucket([0.0; 3], |index_offset, index_count, instances| {
            seen.push((index_offset, index_count, instances.len()));
            // Same length and contents as the cluster's own instances.
            assert_eq!(instances, c.instances.as_slice());
        });
        assert_eq!(seen, vec![(0, 60, 2)]);
    }

    #[test]
    fn for_each_lod_bucket_partitions_like_lod_buckets() {
        // With alternates, the closure path must visit the same buckets, in the
        // same order, as the owned `lod_buckets`.
        let c = sample_cluster(
            vec![
                sample_instance_translation(0.0, 0.0, 0.0),
                sample_instance_translation(20.0, 0.0, 0.0),
                sample_instance_translation(50.0, 0.0, 0.0),
            ],
            vec![
                LodSlice {
                    index_offset: 100,
                    index_count: 30,
                    switch_distance: 10.0,
                },
                LodSlice {
                    index_offset: 200,
                    index_count: 15,
                    switch_distance: 25.0,
                },
            ],
        );
        let mut seen: Vec<(usize, usize, usize)> = Vec::new();
        c.for_each_lod_bucket([0.0; 3], |index_offset, index_count, instances| {
            seen.push((index_offset, index_count, instances.len()));
        });
        let owned: Vec<(usize, usize, usize)> = c
            .lod_buckets([0.0; 3])
            .iter()
            .map(|b| (b.index_offset, b.index_count, b.instances.len()))
            .collect();
        assert_eq!(seen, owned);
    }

    #[test]
    fn for_each_lod_bucket_empty_instances_never_visits() {
        let c = sample_cluster(Vec::new(), Vec::new());
        let mut visits = 0;
        c.for_each_lod_bucket([0.0; 3], |_, _, _| visits += 1);
        assert_eq!(visits, 0);
    }

    #[test]
    fn try_for_each_lod_bucket_stops_on_first_error() {
        // Three buckets; the visitor fails on the second. Iteration must stop
        // there (two visits, error returned) rather than continue.
        let c = sample_cluster(
            vec![
                sample_instance_translation(0.0, 0.0, 0.0),
                sample_instance_translation(20.0, 0.0, 0.0),
                sample_instance_translation(50.0, 0.0, 0.0),
            ],
            vec![
                LodSlice {
                    index_offset: 100,
                    index_count: 30,
                    switch_distance: 10.0,
                },
                LodSlice {
                    index_offset: 200,
                    index_count: 15,
                    switch_distance: 25.0,
                },
            ],
        );
        let mut visits = 0;
        let result = c.try_for_each_lod_bucket::<&str>([0.0; 3], |_, _, _| {
            visits += 1;
            if visits == 2 { Err("boom") } else { Ok(()) }
        });
        assert_eq!(result, Err("boom"));
        assert_eq!(visits, 2);
    }

    #[test]
    fn pack_object_record_stores_caller_pool_indices() {
        // The texture-pool indices are caller-supplied (per-backend addressing)
        // and copied verbatim into the record the shader fetches.
        let obj = sample_draw_object();
        let rec = pack_object_record(&obj, 14, 15);
        assert_eq!(rec.albedo_index, 14);
        assert_eq!(rec.normal_index, 15);
    }

    #[test]
    fn pack_object_record_copies_transform_material_and_bounds() {
        let obj = sample_draw_object();
        let rec = pack_object_record(&obj, 6, 7);
        assert_eq!(rec.model, obj.model);
        assert_eq!(rec.tint, obj.material.tint);
        assert_eq!(rec.roughness, obj.material.roughness);
        assert_eq!(rec.emissive, obj.material.emissive);
        assert_eq!(rec.metallic, obj.material.metallic);
        assert_eq!(rec.macro_variation, obj.material.macro_variation);
        assert_eq!(rec.bb_min, obj.bb_min);
        assert_eq!(rec.bb_max, obj.bb_max);
        assert_eq!(rec.cull_distance, obj.cull_distance);
    }

    #[test]
    fn pack_object_record_zeroes_padding() {
        // The trailing index slots come straight from the material; the
        // sample material leaves the emissive + ORM map indices unset (0),
        // which the shader reads as "no map".
        let rec = pack_object_record(&sample_draw_object(), 0, 1);
        assert_eq!(rec.emissive_map_index, 0);
        assert_eq!(rec.orm_map_index, 0);
    }

    #[test]
    fn gpu_draw_args_layout_matches_shaders() {
        // The `GpuDrawArgs` struct in every backend's cull kernel (Metal MSL,
        // DirectX HLSL, Vulkan std430 GLSL) is four tightly packed uints; the
        // kernel reads garbage if the Rust record drifts from it.
        assert_eq!(size_of::<GpuDrawArgs>(), 16);
        assert_eq!(offset_of!(GpuDrawArgs, index_count), 0);
        assert_eq!(offset_of!(GpuDrawArgs, index_offset), 4);
        assert_eq!(offset_of!(GpuDrawArgs, base_vertex), 8);
        assert_eq!(offset_of!(GpuDrawArgs, flags), 12);
    }

    #[test]
    fn particle_params_layout_matches_msl() {
        // `ParticleParams` rides at compute buffer(2) and vertex buffer(1) of
        // the Metal particle passes; the MSL struct in
        // `metal/shaders/particle.metal` reads three packed_float3 + scalar
        // pairs, two float4 colour tints, then a scalar tail.
        assert_eq!(size_of::<ParticleParams>(), 112);
        assert_eq!(offset_of!(ParticleParams, position), 0);
        assert_eq!(offset_of!(ParticleParams, spread_cos), 12);
        assert_eq!(offset_of!(ParticleParams, direction), 16);
        assert_eq!(offset_of!(ParticleParams, speed_min), 28);
        assert_eq!(offset_of!(ParticleParams, gravity), 32);
        assert_eq!(offset_of!(ParticleParams, speed_max), 44);
        assert_eq!(offset_of!(ParticleParams, color_start), 48);
        assert_eq!(offset_of!(ParticleParams, color_end), 64);
        assert_eq!(offset_of!(ParticleParams, lifetime_min), 80);
        assert_eq!(offset_of!(ParticleParams, lifetime_max), 84);
        assert_eq!(offset_of!(ParticleParams, size_start), 88);
        assert_eq!(offset_of!(ParticleParams, size_end), 92);
        assert_eq!(offset_of!(ParticleParams, dt), 96);
        assert_eq!(offset_of!(ParticleParams, spawn_budget), 100);
        assert_eq!(offset_of!(ParticleParams, random_seed), 104);
        assert_eq!(offset_of!(ParticleParams, max_particles), 108);
        // Multiple of 16 so back-to-back records stay aligned.
        assert_eq!(size_of::<ParticleParams>() % 16, 0);
    }

    #[test]
    fn fog_params_layout_matches_msl() {
        // `FogParams` rides at fragment buffer(0) of the volumetric-fog
        // ray-march; the MSL struct in `metal/shaders/fog.metal` reads it as
        // float4x4 + float4 + 3×(packed_float3 + pad) + 6 scalars + a float2 +
        // a scalar + padding. The offsets below pin every field.
        assert_eq!(size_of::<FogParams>(), 176);
        assert_eq!(offset_of!(FogParams, inv_vp), 0);
        assert_eq!(offset_of!(FogParams, color), 64);
        assert_eq!(offset_of!(FogParams, cam_pos), 80);
        assert_eq!(offset_of!(FogParams, sun_dir), 96);
        assert_eq!(offset_of!(FogParams, sun_color), 112);
        assert_eq!(offset_of!(FogParams, density), 128);
        assert_eq!(offset_of!(FogParams, height_falloff), 132);
        assert_eq!(offset_of!(FogParams, height_reference), 136);
        assert_eq!(offset_of!(FogParams, max_distance), 140);
        assert_eq!(offset_of!(FogParams, phase_g), 144);
        assert_eq!(offset_of!(FogParams, ambient), 148);
        assert_eq!(offset_of!(FogParams, viewport), 152);
        assert_eq!(offset_of!(FogParams, inv_max_distance), 160);
        // Size is a multiple of 16 so the inv_vp at the head keeps a
        // following array of records 16-byte aligned. Only one is ever
        // pushed today, but the layout test pins the invariant.
        assert_eq!(size_of::<FogParams>() % 16, 0);
    }

    #[test]
    fn fog_froxel_params_layout_matches_msl() {
        // `FogFroxelParams` rides alongside `FogParams` for the Metal froxel
        // volume path. Pin every field offset so the MSL `FogFroxelParams`
        // struct in `metal/shaders/fog.metal` stays in sync.
        assert_eq!(size_of::<FogFroxelParams>(), 96);
        assert_eq!(offset_of!(FogFroxelParams, view), 0);
        assert_eq!(offset_of!(FogFroxelParams, froxel_dims), 64);
        // _pad_align at 76 so MSL `uint3 froxel_dims` (16-byte slot) lines
        // up with `z_near` at 80 on both sides.
        assert_eq!(offset_of!(FogFroxelParams, z_near), 80);
        assert_eq!(offset_of!(FogFroxelParams, z_far), 84);
        assert_eq!(size_of::<FogFroxelParams>() % 16, 0);
    }

    #[test]
    fn draw_args_flags_packs_cull_decision() {
        // A visible, resident, cullable object: both bits set.
        assert_eq!(
            draw_args_flags(true, true, true),
            DrawArgsFlags::ENABLED | DrawArgsFlags::CULLABLE
        );
        // Non-cullable (e.g. skybox): enabled but never frustum-tested.
        assert_eq!(draw_args_flags(true, true, false), DrawArgsFlags::ENABLED);
        // Hidden or not-yet-streamed objects clear ENABLED so the kernel
        // resets their indirect command.
        assert_eq!(draw_args_flags(false, true, true), DrawArgsFlags::CULLABLE);
        assert_eq!(draw_args_flags(true, false, true), DrawArgsFlags::CULLABLE);
        assert_eq!(draw_args_flags(false, false, false), 0);
    }
}
