#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice as _, MTLPixelFormat, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};

// Upload a 2-D RGBA texture from raw pixel bytes with a full mip chain.
// The chain is box-filtered on the CPU (`crate::gfx::mipmap`) and every level
// is written so the texture minifies through hardware trilinear / aniso
// selection instead of aliasing from a single mip-0 sample at a distance.
// The texture is created with ShaderRead usage so it can be sampled in
// fragment shaders. StorageModeShared is used so the CPU-side pixel data
// is accessible without an explicit blit encoder.
pub(super) fn upload_texture(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let base = (width as usize) * (height as usize) * 4;
    if pixels.len() < base {
        return Err(format!(
            "pixel data too short for {}x{} RGBA texture ({} bytes, need {})",
            width,
            height,
            pixels.len(),
            base
        ));
    }

    let chain = crate::gfx::mipmap::generate_mip_chain(width, height, pixels);

    let desc = MTLTextureDescriptor::new();
    unsafe {
        desc.setTextureType(MTLTextureType::Type2D);
        desc.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        desc.setWidth(width as usize);
        desc.setHeight(height as usize);
        desc.setMipmapLevelCount(chain.len());
        desc.setUsage(MTLTextureUsage::ShaderRead);
        desc.setStorageMode(objc2_metal::MTLStorageMode::Shared);
    }

    let texture = device
        .newTextureWithDescriptor(&desc)
        .ok_or("failed to create MTLTexture")?;

    for (mip, level) in chain.iter().enumerate() {
        unsafe {
            use objc2_metal::MTLRegion;
            let region = MTLRegion {
                origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
                size: objc2_metal::MTLSize {
                    width: level.width as usize,
                    height: level.height as usize,
                    depth: 1,
                },
            };
            let bytes_per_row = (level.width * 4) as usize;
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                mip,
                std::ptr::NonNull::new(level.pixels.as_ptr() as *mut _)
                    .ok_or("pixel slice is empty")?,
                bytes_per_row,
            );
        }
    }

    Ok(texture)
}

// Create a 1x1 opaque white RGBA texture used when no Texture asset is present.
pub(super) fn create_fallback_texture(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    upload_texture(device, 1, 1, &[255u8, 255, 255, 255])
}

// Create a 1x1 Depth32Float texture-array (one layer) with value 1.0, used
// when no ShadowStage is declared. A depth of 1.0 means "maximum depth" so
// sample_compare with LessEqual always returns 1.0 (fully lit).
//
// The fragment shader binds the shadow map as depth2d_array; using a
// 1-layer 2D-array fallback keeps the binding type identical between the
// disabled and enabled cases.
pub(super) fn create_shadow_map_fallback(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let desc = MTLTextureDescriptor::new();
    unsafe {
        desc.setTextureType(MTLTextureType::Type2DArray);
        desc.setPixelFormat(MTLPixelFormat::Depth32Float);
        desc.setWidth(1);
        desc.setHeight(1);
        desc.setArrayLength(1);
        desc.setUsage(MTLTextureUsage::ShaderRead);
        desc.setStorageMode(objc2_metal::MTLStorageMode::Shared);
    }
    let texture = device
        .newTextureWithDescriptor(&desc)
        .ok_or("failed to create shadow map fallback texture")?;
    let depth: f32 = 1.0;
    unsafe {
        use objc2_metal::MTLRegion;
        let region = MTLRegion {
            origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
            size: objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        };
        texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
            region,
            0,
            0,
            std::ptr::NonNull::new(std::ptr::addr_of!(depth) as *mut _)
                .ok_or("depth ptr is null")?,
            4,
            4,
        );
    }
    Ok(texture)
}

// Upload a six-face HDR cubemap from a CubemapTexture payload. `bytes` is the
// raw RGBA32F face-major data emitted by build/cubemap.rs::compile_cubemap_payload,
// i.e. 6 * face_size * face_size * 4 floats with face order +X, -X, +Y, -Y, +Z, -Z.
//
pub(super) fn upload_cubemap(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    face_size: u32,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let face_bytes = (face_size as usize) * (face_size as usize) * 4 * 4;
    let needed = 6 * face_bytes;
    if bytes.len() < needed {
        return Err(format!(
            "cubemap data too short for face_size {}: {} bytes, need {}",
            face_size,
            bytes.len(),
            needed
        ));
    }

    let desc = MTLTextureDescriptor::new();
    unsafe {
        desc.setTextureType(MTLTextureType::TypeCube);
        desc.setPixelFormat(MTLPixelFormat::RGBA32Float);
        desc.setWidth(face_size as usize);
        desc.setHeight(face_size as usize);
        desc.setUsage(MTLTextureUsage::ShaderRead);
        desc.setStorageMode(objc2_metal::MTLStorageMode::Shared);
    }

    let texture = device
        .newTextureWithDescriptor(&desc)
        .ok_or("failed to create MTLTextureCube")?;

    let bytes_per_row = (face_size as usize) * 4 * 4;
    let bytes_per_image = bytes_per_row * (face_size as usize);
    unsafe {
        use objc2_metal::MTLRegion;
        let region = MTLRegion {
            origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
            size: objc2_metal::MTLSize {
                width: face_size as usize,
                height: face_size as usize,
                depth: 1,
            },
        };
        for face in 0..6 {
            let face_start = face * face_bytes;
            let face_ptr = bytes.as_ptr().add(face_start) as *mut std::ffi::c_void;
            texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                region,
                0,
                face,
                std::ptr::NonNull::new(face_ptr).ok_or("cube face pointer is null")?,
                bytes_per_row,
                bytes_per_image,
            );
        }
    }
    Ok(texture)
}

// IBL textures produced by a single `EnvironmentMap` asset. Returned together
// so per-frame binding sets both with one lookup. `prefilter_mip_count == 0`
// is the runtime signal for "IBL disabled": the fragment shader keys off it
// to fall back to the legacy ambient/skybox path.
pub(super) struct EnvironmentMapTextures {
    pub irradiance: Retained<ProtocolObject<dyn MTLTexture>>,
    pub prefilter: Retained<ProtocolObject<dyn MTLTexture>>,
    pub prefilter_mip_count: u32,
}

// Create a 1x1 RGBA32Float cube of `value` for every face. Used as the
// IBL fallback when no `EnvironmentMap` is bound: the fragment shader keys
// off `prefilter_mip_count == 0` and skips IBL math, but the cube binding
// must still resolve to a valid texture.
pub(super) fn create_fallback_cubemap(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    value: [f32; 4],
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let desc = MTLTextureDescriptor::new();
    unsafe {
        desc.setTextureType(MTLTextureType::TypeCube);
        desc.setPixelFormat(MTLPixelFormat::RGBA32Float);
        desc.setWidth(1);
        desc.setHeight(1);
        desc.setUsage(MTLTextureUsage::ShaderRead);
        desc.setStorageMode(objc2_metal::MTLStorageMode::Shared);
    }
    let texture = device
        .newTextureWithDescriptor(&desc)
        .ok_or("failed to create fallback cube texture")?;
    let bytes_per_row = 4 * 4;
    let bytes_per_image = bytes_per_row;
    unsafe {
        use objc2_metal::MTLRegion;
        let region = MTLRegion {
            origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
            size: objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        };
        for face in 0..6 {
            texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                region,
                0,
                face,
                std::ptr::NonNull::new(value.as_ptr() as *mut _)
                    .ok_or("fallback cube value pointer null")?,
                bytes_per_row,
                bytes_per_image,
            );
        }
    }
    Ok(texture)
}

// Upload a 3D colour-grading LUT from a ColorLut payload. `bytes` is the raw
// RGBA8 data emitted by build/color_lut.rs: `size`³ texels ordered with the
// red axis fastest, then green, then blue. The result is sampled in the
// composite pass with the tonemapped sRGB colour as the texture coordinate.
pub(super) fn upload_color_lut(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    size: u32,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let n = size as usize;
    let needed = n * n * n * 4;
    if bytes.len() < needed {
        return Err(format!(
            "color LUT data too short for size {}: {} bytes, need {}",
            size,
            bytes.len(),
            needed
        ));
    }

    let desc = MTLTextureDescriptor::new();
    unsafe {
        desc.setTextureType(MTLTextureType::Type3D);
        desc.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        desc.setWidth(n);
        desc.setHeight(n);
        desc.setDepth(n);
        desc.setUsage(MTLTextureUsage::ShaderRead);
        desc.setStorageMode(objc2_metal::MTLStorageMode::Shared);
    }
    let texture = device
        .newTextureWithDescriptor(&desc)
        .ok_or("failed to create 3D color LUT texture")?;

    unsafe {
        use objc2_metal::MTLRegion;
        let region = MTLRegion {
            origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
            size: objc2_metal::MTLSize {
                width: n,
                height: n,
                depth: n,
            },
        };
        let bytes_per_row = n * 4;
        let bytes_per_image = bytes_per_row * n;
        texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
            region,
            0,
            0,
            std::ptr::NonNull::new(bytes.as_ptr() as *mut _).ok_or("color LUT pointer is null")?,
            bytes_per_row,
            bytes_per_image,
        );
    }
    Ok(texture)
}

// Build a 2x2x2 identity colour LUT: the eight corners of the unit RGB cube.
// Trilinear interpolation across the corners reproduces the input exactly, so
// the composite pass becomes a no-op when no `ColorLut` asset is declared.
// The 3D LUT binding must still resolve to a valid texture regardless.
pub(super) fn create_fallback_color_lut(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let mut data = Vec::with_capacity(2 * 2 * 2 * 4);
    for b in 0..2u8 {
        for g in 0..2u8 {
            for r in 0..2u8 {
                data.extend_from_slice(&[r * 255, g * 255, b * 255, 255]);
            }
        }
    }
    upload_color_lut(device, 2, &data)
}

// Upload an EnvironmentMap payload into two cube textures: a single-mip
// irradiance cube and a multi-mip prefiltered radiance cube. Both are
// `MTLTextureType::TypeCube` with `RGBA32Float` storage matching the build
// pipeline's payload format.
//
// `irradiance_face` / `prefilter_face` are the mip-0 face sizes. `mip_bytes`
// is one slice per mip in order 0..mip_count; `mip_count` must equal
// `mip_bytes.len()`.
pub(super) fn upload_environment_map(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    irradiance_face: u32,
    irradiance_bytes: &[u8],
    prefilter_face: u32,
    mip_bytes: &[&[u8]],
) -> Result<EnvironmentMapTextures, String> {
    if mip_bytes.is_empty() {
        return Err("envmap upload: prefilter mip_bytes must not be empty".into());
    }
    let irradiance = upload_cubemap(device, irradiance_face, irradiance_bytes)
        .map_err(|e| format!("envmap irradiance: {}", e))?;
    let prefilter = upload_prefilter_cube(device, prefilter_face, mip_bytes)
        .map_err(|e| format!("envmap prefilter: {}", e))?;
    Ok(EnvironmentMapTextures {
        irradiance,
        prefilter,
        prefilter_mip_count: mip_bytes.len() as u32,
    })
}

// Create a multi-mip RGBA32Float `MTLTextureType::Cube` and upload each mip
// from `mip_bytes`. `mip_bytes[m]` is expected to be 6 * (face_size >> m)² * 16 bytes.
fn upload_prefilter_cube(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    face_size: u32,
    mip_bytes: &[&[u8]],
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let mip_count = mip_bytes.len() as u32;
    let desc = MTLTextureDescriptor::new();
    unsafe {
        desc.setTextureType(MTLTextureType::TypeCube);
        desc.setPixelFormat(MTLPixelFormat::RGBA32Float);
        desc.setWidth(face_size as usize);
        desc.setHeight(face_size as usize);
        desc.setMipmapLevelCount(mip_count as usize);
        desc.setUsage(MTLTextureUsage::ShaderRead);
        desc.setStorageMode(objc2_metal::MTLStorageMode::Shared);
    }
    let texture = device
        .newTextureWithDescriptor(&desc)
        .ok_or("failed to create prefilter MTLTextureCube")?;
    for (mip, bytes) in mip_bytes.iter().enumerate() {
        let mip_face_size = face_size >> mip;
        if mip_face_size == 0 {
            return Err(format!(
                "prefilter mip {} would have zero face size (face_size {} too small)",
                mip, face_size
            ));
        }
        let face_bytes = (mip_face_size as usize) * (mip_face_size as usize) * 4 * 4;
        let needed = 6 * face_bytes;
        if bytes.len() < needed {
            return Err(format!(
                "prefilter mip {} too short: {} bytes, need {}",
                mip,
                bytes.len(),
                needed
            ));
        }
        let bytes_per_row = (mip_face_size as usize) * 4 * 4;
        let bytes_per_image = bytes_per_row * (mip_face_size as usize);
        unsafe {
            use objc2_metal::MTLRegion;
            let region = MTLRegion {
                origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
                size: objc2_metal::MTLSize {
                    width: mip_face_size as usize,
                    height: mip_face_size as usize,
                    depth: 1,
                },
            };
            for face in 0..6 {
                let face_start = face * face_bytes;
                let face_ptr = bytes.as_ptr().add(face_start) as *mut std::ffi::c_void;
                texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    region,
                    mip,
                    face,
                    std::ptr::NonNull::new(face_ptr).ok_or("prefilter face pointer null")?,
                    bytes_per_row,
                    bytes_per_image,
                );
            }
        }
    }
    Ok(texture)
}

// Create a Depth32Float Texture2DArray shadow map with `layers` cascades, each
// `size`x`size`. ShaderRead allows fragment sampling; RenderTarget allows the
// shadow pre-pass to write depth into a specific slice. StorageModePrivate
// keeps it GPU-only.
pub(super) fn create_shadow_map_array(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    size: u32,
    layers: u32,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let desc = MTLTextureDescriptor::new();
    unsafe {
        desc.setTextureType(MTLTextureType::Type2DArray);
        desc.setPixelFormat(MTLPixelFormat::Depth32Float);
        desc.setWidth(size as usize);
        desc.setHeight(size as usize);
        desc.setArrayLength(layers as usize);
        // RenderTarget (0x4) | ShaderRead (0x1)
        desc.setUsage(MTLTextureUsage(
            MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0,
        ));
        desc.setStorageMode(objc2_metal::MTLStorageMode::Private);
    }
    device
        .newTextureWithDescriptor(&desc)
        .ok_or("failed to create shadow map array texture".to_string())
}

// Off-screen HDR render targets for the post-process pipeline. The main pass
// renders linear-light RGBA16Float into `hdr_color` (MSAA) which resolves
// into `hdr_resolve` at end-of-pass; the composite pass then samples
// `hdr_resolve` for tonemap + FXAA. `depth` is the matching MSAA depth,
// kept alive after the Main pass so post-passes (decals/fog/water/raymarch
// early-out) can sample it as a read-only snapshot of the rasterised
// scene depth. `depth_resolve` is the single-sample sibling (populated by
// the Main pass via a `MultisampleResolve` store action with
// `MTLMultisampleDepthResolveFilter::Sample0`) that the raymarch pass
// uses as a writable depth attachment (and that post-Raymarch passes like
// water/decal/fog sample so they "see" raymarched surface depth alongside
// rasterised depth). The canonical post-rasterise scene depth target
// going forward; any future post-pass that needs to write depth should
// bind this rather than introduce its own depth target.
//
// `hdr_resolve_copy` is a single-sample sibling of `hdr_resolve` reserved
// for the raymarch pass's scene-copy refraction path: at the start of
// the raymarch encoder a blit copies `hdr_resolve` into this texture, so
// user SDF shaders can sample the pre-raymarch scene without violating
// Metal's attachment-aliasing rule (the raymarch pass writes the same
// `hdr_resolve` it would otherwise need to read). Same RGBA16Float
// format / ShaderRead+RenderTarget usage as `hdr_resolve`. Unbound by
// any pass when no `SdfVolume` consumes it; allocated unconditionally
// because the cost (a single full-screen RGBA16F texture ~ 18 MB at
// 1440p) is small relative to the existing HDR target footprint and
// keeps the allocation logic branch-free.
pub(super) struct HdrTargets {
    pub hdr_color: Retained<ProtocolObject<dyn MTLTexture>>,
    pub hdr_resolve: Retained<ProtocolObject<dyn MTLTexture>>,
    pub hdr_resolve_copy: Retained<ProtocolObject<dyn MTLTexture>>,
    // Scene snapshot taken at the head of the transparent pass. A blit copies
    // the latest pre-transparent scene (`scene_pre_taa`) here so translucent
    // shaders (water, glass) sample the opaque scene for refraction without
    // reading the render attachment they are writing. Same descriptor as
    // `hdr_resolve`. Distinct from `hdr_resolve_copy`, which the raymarch pass
    // fills earlier in the frame for SDF refraction.
    pub transparent_scene_copy: Retained<ProtocolObject<dyn MTLTexture>>,
    pub depth: Retained<ProtocolObject<dyn MTLTexture>>,
    pub depth_resolve: Retained<ProtocolObject<dyn MTLTexture>>,
    pub width: u32,
    pub height: u32,
}

// Create or recreate the HDR off-screen targets at `width`x`height`. The
// MSAA color/depth attachments live in private storage; the resolve target
// also lives in private storage but enables ShaderRead so the post pass
// can sample it.
pub(super) fn create_hdr_targets(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
    sample_count: u32,
) -> Result<HdrTargets, String> {
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;

    // MSAA HDR color: RGBA16Float, multi-sample 2D, render-target only.
    let color_desc = MTLTextureDescriptor::new();
    unsafe {
        color_desc.setTextureType(MTLTextureType::Type2DMultisample);
        color_desc.setPixelFormat(MTLPixelFormat::RGBA16Float);
        color_desc.setWidth(w);
        color_desc.setHeight(h);
        color_desc.setSampleCount(sample_count as usize);
        color_desc.setUsage(MTLTextureUsage::RenderTarget);
        color_desc.setStorageMode(objc2_metal::MTLStorageMode::Private);
    }
    let hdr_color = device
        .newTextureWithDescriptor(&color_desc)
        .ok_or("failed to create MSAA HDR color texture")?;

    // Single-sample resolve target: same RGBA16Float; sampled by the post pass.
    let resolve_desc = MTLTextureDescriptor::new();
    unsafe {
        resolve_desc.setTextureType(MTLTextureType::Type2D);
        resolve_desc.setPixelFormat(MTLPixelFormat::RGBA16Float);
        resolve_desc.setWidth(w);
        resolve_desc.setHeight(h);
        resolve_desc.setUsage(MTLTextureUsage(
            MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0,
        ));
        resolve_desc.setStorageMode(objc2_metal::MTLStorageMode::Private);
    }
    let hdr_resolve = device
        .newTextureWithDescriptor(&resolve_desc)
        .ok_or("failed to create HDR resolve texture")?;

    // Scene-copy sibling of `hdr_resolve`. The raymarch pass blits
    // `hdr_resolve` here before drawing so user SDF shaders can sample
    // the pre-raymarch scene as a regular texture without aliasing the
    // render attachment. Same descriptor as `hdr_resolve` so the blit
    // is a plain copy_from_texture.
    let hdr_resolve_copy = device
        .newTextureWithDescriptor(&resolve_desc)
        .ok_or("failed to create HDR resolve-copy texture")?;

    // Scene snapshot for the transparent pass. The transparent encoder blits
    // `scene_pre_taa` here before drawing so water / glass refraction reads a
    // stable copy regardless of whether SSR produced a distinct `scene_pre_taa`
    // or it aliases `hdr_resolve`. Same descriptor as `hdr_resolve`.
    let transparent_scene_copy = device
        .newTextureWithDescriptor(&resolve_desc)
        .ok_or("failed to create transparent scene-copy texture")?;

    // MSAA depth: matches the color sample count. `ShaderRead` is enabled so
    // the decal pass (and any future post-pass that needs scene depth) can
    // sample it as a `depth2d_ms<float>` after the main pass stores it.
    let depth_desc = MTLTextureDescriptor::new();
    unsafe {
        depth_desc.setTextureType(MTLTextureType::Type2DMultisample);
        depth_desc.setPixelFormat(MTLPixelFormat::Depth32Float);
        depth_desc.setWidth(w);
        depth_desc.setHeight(h);
        depth_desc.setSampleCount(sample_count as usize);
        depth_desc.setUsage(MTLTextureUsage(
            MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0,
        ));
        depth_desc.setStorageMode(objc2_metal::MTLStorageMode::Private);
    }
    let depth = device
        .newTextureWithDescriptor(&depth_desc)
        .ok_or("failed to create MSAA depth texture")?;

    // Single-sample depth resolve: populated by the Main pass via a
    // `MTLStoreAction::MultisampleResolve` with depth filter Sample0. The
    // raymarch pass binds this as its writable depth attachment; water /
    // decal / fog sample it so they see raymarched surface depth alongside
    // rasterised depth.
    let depth_resolve_desc = MTLTextureDescriptor::new();
    unsafe {
        depth_resolve_desc.setTextureType(MTLTextureType::Type2D);
        depth_resolve_desc.setPixelFormat(MTLPixelFormat::Depth32Float);
        depth_resolve_desc.setWidth(w);
        depth_resolve_desc.setHeight(h);
        depth_resolve_desc.setSampleCount(1);
        depth_resolve_desc.setUsage(MTLTextureUsage(
            MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0,
        ));
        depth_resolve_desc.setStorageMode(objc2_metal::MTLStorageMode::Private);
    }
    let depth_resolve = device
        .newTextureWithDescriptor(&depth_resolve_desc)
        .ok_or("failed to create single-sample depth resolve texture")?;

    Ok(HdrTargets {
        hdr_color,
        hdr_resolve,
        hdr_resolve_copy,
        transparent_scene_copy,
        depth,
        depth_resolve,
        width: w as u32,
        height: h as u32,
    })
}
