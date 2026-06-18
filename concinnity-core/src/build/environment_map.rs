// src/build/environment_map.rs
//
// Compiles an EnvironmentMap component's args into a payload bundling two
// precomputed IBL cubemaps:
//
//   - **Irradiance cubemap.** Low-resolution (32x32 per face by default)
//     cosine-weighted hemisphere integral of the source. Used by the shader's
//     diffuse ambient term: `diffuse = (1-F)(1-metallic) * irradiance * albedo / π`.
//   - **Prefiltered radiance cubemap.** A mip chain where mip 0 = source and
//     mip N = source convolved with the GGX lobe at roughness = N / (mip_count - 1).
//     Used with the Karis env-BRDF analytic fit (already in every fragment shader
//     as `env_brdf_approx`) for the specular ambient term.
//
// A BRDF LUT is deliberately NOT shipped: the Karis polynomial fit
// (`env_brdf_approx` in default.metal / default_frag.hlsl / FRAG_GLSL) replaces
// it analytically. That keeps one binding slot free and dodges a build step.
//
// Source format: equirectangular Radiance HDR (.hdr), same as CubemapTexture.
// Sampling: Hammersley QMC + GGX importance sampling for prefilter, uniform
// (phi, theta) grid for irradiance.
//
// Payload format (little-endian):
//   u32  magic              = b"ENVM" = 0x4D564E45
//   u32  format_id          = 0  (RGBA32F)
//   u32  irradiance_face    (e.g. 32)
//   u32  prefilter_face     (mip 0 size, e.g. 512)
//   u32  prefilter_mips     (e.g. 5)
//   u32  _pad
//   ... irradiance cube         (6 * irradiance_face² * 16 bytes)
//   ... prefilter mip 0         (6 * prefilter_face² * 16 bytes)
//   ... prefilter mip 1         (6 * (prefilter_face/2)² * 16 bytes)
//   ...
//   ... prefilter mip (mips-1)  (6 * (prefilter_face >> (mips-1))² * 16 bytes)
//
// Face order matches CubemapTexture: +X, -X, +Y, -Y, +Z, -Z.

use crate::build::cubemap::{HdrImage, decode_hdr, equirect_to_cube};
use rayon::prelude::*;
use std::io::Read;

pub const ENVMAP_PAYLOAD_MAGIC: u32 = u32::from_le_bytes(*b"ENVM");
pub const ENVMAP_FORMAT_RGBA32F: u32 = 0;
pub const ENVMAP_PAYLOAD_HEADER_BYTES: usize = 24;

pub const DEFAULT_IRRADIANCE_PHI_SAMPLES: u32 = 64;
pub const DEFAULT_IRRADIANCE_THETA_SAMPLES: u32 = 16;

// Number of mip levels for a square cube face of `face_size` pixels. The
// smallest mip is clamped to 4×4 to keep the prefilter convolution sensible
// at high roughness.
pub fn max_mip_count(face_size: u32) -> u32 {
    let mut mips = 0u32;
    let mut s = face_size;
    while s >= 4 {
        mips += 1;
        s /= 2;
    }
    mips
}

// Decode an EnvironmentMap source path the same way
// `compile_environment_map_payload` does at build time, returning the
// serialised payload (header + irradiance + prefilter mips). Exposed for the
// runtime asset hot-reload path (`cn debug` only); production reads the
// compiled payload from a blob locator instead. `prefilter_face`,
// `irradiance_face`, and `prefilter_samples` should be the values from the
// declared `EnvironmentMap` asset so the runtime decode produces the same
// texture sizes as the build pass. The convolutions are CPU-bound and take
// seconds at default sizes: the caller pays this on the render thread.
pub fn decode_source(
    source: &str,
    prefilter_face: u32,
    irradiance_face: u32,
    prefilter_samples: u32,
) -> Result<Vec<u8>, String> {
    let resolved = resolve_hdr_source(source);
    let hdr = load_hdr_file(&resolved)?;
    let source_cube = equirect_to_cube(&hdr, prefilter_face);
    let prefilter_mips = max_mip_count(prefilter_face);
    let irradiance = compute_irradiance(
        &source_cube,
        prefilter_face,
        irradiance_face,
        DEFAULT_IRRADIANCE_PHI_SAMPLES,
        DEFAULT_IRRADIANCE_THETA_SAMPLES,
    );
    let prefilter = compute_prefilter(
        &source_cube,
        prefilter_face,
        prefilter_mips,
        prefilter_samples,
    );
    Ok(serialise_payload(
        irradiance_face,
        prefilter_face,
        prefilter_mips,
        &irradiance,
        &prefilter,
    ))
}

// Resolve an EnvironmentMap `source` string into the actual file path on
// disk. The runtime hot-reload watcher needs the resolved path so it can
// subscribe to the correct parent directory; bare filenames are otherwise
// unfindable after the build pipeline runs.
pub fn resolve_source_path(source: &str) -> String {
    resolve_hdr_source(source)
}

// Resolve an EnvironmentMap source string into a filesystem path. Bare
// filenames are searched under `.concinnity/assets/` (recursively) so worlds
// can reference HDRIs by filename only, matching the lookup semantics of
// `ShaderStage` source paths. Anything containing a directory separator is
// returned unchanged, so absolute or relative paths still work.
pub fn resolve_hdr_source(source: &str) -> String {
    let p = std::path::Path::new(source);
    let is_bare = p.parent().map(|d| d.as_os_str().is_empty()).unwrap_or(true);
    if !is_bare {
        return source.to_string();
    }
    if let Some(path) = crate::world::preset::find_in_assets(source) {
        return path;
    }
    format!("{}/{source}", crate::world::CONCINNITY_ASSETS_DIR)
}

pub fn load_hdr_file(path: &str) -> Result<HdrImage, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("failed to open HDR source '{}': {}", path, e))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("failed to read HDR source '{}': {}", path, e))?;
    decode_hdr(&bytes).map_err(|e| format!("failed to decode HDR '{}': {}", path, e))
}

// Cube sampling

// Cube-face sampler: project a normalised direction onto the dominant axis
// to pick a face, then bilinearly sample within that face. Edges are clamped
// per-face (no seamless filtering).
fn sample_cube(faces: &[Vec<f32>; 6], face_size: u32, dir: [f32; 3]) -> [f32; 3] {
    let ax = dir[0].abs();
    let ay = dir[1].abs();
    let az = dir[2].abs();
    let (face, ma, s, t) = if ax >= ay && ax >= az {
        if dir[0] > 0.0 {
            (0usize, ax, -dir[2], -dir[1])
        } else {
            (1, ax, dir[2], -dir[1])
        }
    } else if ay >= az {
        if dir[1] > 0.0 {
            (2usize, ay, dir[0], dir[2])
        } else {
            (3, ay, dir[0], -dir[2])
        }
    } else if dir[2] > 0.0 {
        (4usize, az, dir[0], -dir[1])
    } else {
        (5, az, -dir[0], -dir[1])
    };
    let inv = 0.5 / ma.max(1e-20);
    let fs = face_size as f32;
    // s, t in [-1, 1] after multiplying by inv*2; map to pixel coords.
    let fx = (s * inv + 0.5) * fs - 0.5;
    let fy = (t * inv + 0.5) * fs - 0.5;
    let x0 = (fx.floor() as i32).clamp(0, face_size as i32 - 1);
    let y0 = (fy.floor() as i32).clamp(0, face_size as i32 - 1);
    let x1 = (x0 + 1).clamp(0, face_size as i32 - 1);
    let y1 = (y0 + 1).clamp(0, face_size as i32 - 1);
    let dx = (fx - fx.floor()).clamp(0.0, 1.0);
    let dy = (fy - fy.floor()).clamp(0.0, 1.0);
    let p = |x: i32, y: i32| -> [f32; 3] {
        let off = ((y as usize) * face_size as usize + x as usize) * 4;
        let face_data = &faces[face];
        [face_data[off], face_data[off + 1], face_data[off + 2]]
    };
    let p00 = p(x0, y0);
    let p10 = p(x1, y0);
    let p01 = p(x0, y1);
    let p11 = p(x1, y1);
    let w00 = (1.0 - dx) * (1.0 - dy);
    let w10 = dx * (1.0 - dy);
    let w01 = (1.0 - dx) * dy;
    let w11 = dx * dy;
    [
        p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11,
        p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11,
        p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11,
    ]
}

// Map a (face, x, y) cube texel to its world-space direction (unit vector).
fn cube_texel_dir(face: usize, x: u32, y: u32, face_size: u32) -> [f32; 3] {
    let u = (x as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;
    let v = (y as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;
    let d = match face {
        0 => [1.0, -v, -u],
        1 => [-1.0, -v, u],
        2 => [u, 1.0, v],
        3 => [u, -1.0, -v],
        4 => [u, -v, 1.0],
        5 => [-u, -v, -1.0],
        _ => unreachable!("invalid cube face index {}", face),
    };
    normalize3(d)
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-20);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// Build an orthonormal basis around `n` (N = up axis). Returns (tangent, bitangent).
fn make_tbn(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let up = if n[2].abs() < 0.999 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let t = normalize3(cross3(up, n));
    let b = cross3(n, t);
    (t, b)
}

// Hammersley + GGX importance sampling

// Hammersley quasi-random 2D sequence over `n` samples. Used to drive the
// GGX importance sampler for prefilter convolution.
pub fn hammersley(i: u32, n: u32) -> [f32; 2] {
    let mut bits = i;
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    let radical_inverse = (bits as f32) * 2.328_306_4e-10; // 1 / 2^32
    [i as f32 / n as f32, radical_inverse]
}

// Sample the GGX distribution in world space around normal `n`. Returns a
// half-vector H.
fn importance_sample_ggx(xi: [f32; 2], n: [f32; 3], roughness: f32) -> [f32; 3] {
    let a = roughness * roughness;
    let phi = 2.0 * std::f32::consts::PI * xi[0];
    let cos_theta = ((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1])).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let h_local = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];
    let (t, b) = make_tbn(n);
    normalize3([
        t[0] * h_local[0] + b[0] * h_local[1] + n[0] * h_local[2],
        t[1] * h_local[0] + b[1] * h_local[1] + n[1] * h_local[2],
        t[2] * h_local[0] + b[2] * h_local[1] + n[2] * h_local[2],
    ])
}

// Irradiance

// Compute a low-resolution irradiance cubemap by uniform (phi, theta)
// integration over the upper hemisphere around each output direction.
// Returns RGBA32F face-major (alpha = 1.0). The integral includes the
// cosine + Jacobian terms so the shader can plug the sample straight in
// as `irradiance / π * albedo`.
pub fn compute_irradiance(
    source: &[Vec<f32>; 6],
    source_face_size: u32,
    output_face_size: u32,
    phi_samples: u32,
    theta_samples: u32,
) -> [Vec<f32>; 6] {
    let f = output_face_size as usize;
    let mut faces: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0; f * f * 4]);
    let inv_n_phi = 1.0 / phi_samples as f32;
    let inv_n_theta = 1.0 / theta_samples as f32;
    // discrete weight: (Δθ * Δφ) = (π/2 / N_θ) * (2π / N_φ) = π² / (N_θ N_φ)
    let weight = std::f32::consts::PI * std::f32::consts::PI * inv_n_phi * inv_n_theta;

    // Each output texel is an independent integral over the read-only source,
    // so faces and rows within a face are computed in parallel.
    faces.par_iter_mut().enumerate().for_each(|(face, out)| {
        out.par_chunks_mut(f * 4).enumerate().for_each(|(y, row)| {
            for x in 0..output_face_size {
                let n = cube_texel_dir(face, x, y as u32, output_face_size);
                let (tan, bit) = make_tbn(n);
                let mut sum = [0.0f32; 3];
                for phi_i in 0..phi_samples {
                    let phi = 2.0 * std::f32::consts::PI * (phi_i as f32 + 0.5) * inv_n_phi;
                    let sin_phi = phi.sin();
                    let cos_phi = phi.cos();
                    for theta_i in 0..theta_samples {
                        let theta =
                            0.5 * std::f32::consts::PI * (theta_i as f32 + 0.5) * inv_n_theta;
                        let sin_theta = theta.sin();
                        let cos_theta = theta.cos();
                        let l_local = [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta];
                        let dir = [
                            tan[0] * l_local[0] + bit[0] * l_local[1] + n[0] * l_local[2],
                            tan[1] * l_local[0] + bit[1] * l_local[1] + n[1] * l_local[2],
                            tan[2] * l_local[0] + bit[2] * l_local[1] + n[2] * l_local[2],
                        ];
                        let env = sample_cube(source, source_face_size, normalize3(dir));
                        // cos(θ) for the Lambert cosine, sin(θ) for the spherical
                        // area element. Both already in [0, 1] for the hemisphere.
                        let w = cos_theta * sin_theta;
                        sum[0] += env[0] * w;
                        sum[1] += env[1] * w;
                        sum[2] += env[2] * w;
                    }
                }
                let off = x as usize * 4;
                row[off] = sum[0] * weight;
                row[off + 1] = sum[1] * weight;
                row[off + 2] = sum[2] * weight;
                row[off + 3] = 1.0;
            }
        });
    });
    faces
}

// Prefiltered radiance

// Build a prefiltered radiance cube mip chain. Mip 0 is the unmodified
// source (roughness=0 → Dirac lobe). Mip N is the GGX convolution at
// roughness = N / (mip_count - 1).
pub fn compute_prefilter(
    source: &[Vec<f32>; 6],
    source_face_size: u32,
    mip_count: u32,
    samples_per_texel: u32,
) -> Vec<[Vec<f32>; 6]> {
    let mut mips: Vec<[Vec<f32>; 6]> = Vec::with_capacity(mip_count as usize);
    // Mip 0: identity (with alpha = 1.0 for RGBA32F storage).
    {
        let f = source_face_size as usize;
        let mut mip0: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0; f * f * 4]);
        for face in 0..6 {
            for i in 0..f * f {
                let off = i * 4;
                mip0[face][off] = source[face][off];
                mip0[face][off + 1] = source[face][off + 1];
                mip0[face][off + 2] = source[face][off + 2];
                mip0[face][off + 3] = 1.0;
            }
        }
        mips.push(mip0);
    }
    // Mips 1..N: GGX convolution.
    for mip in 1..mip_count {
        let face_size = source_face_size >> mip;
        let roughness = mip as f32 / (mip_count - 1) as f32;
        mips.push(convolve_ggx(
            source,
            source_face_size,
            face_size,
            roughness,
            samples_per_texel,
        ));
    }
    mips
}

fn convolve_ggx(
    source: &[Vec<f32>; 6],
    source_face_size: u32,
    output_face_size: u32,
    roughness: f32,
    samples: u32,
) -> [Vec<f32>; 6] {
    let f = output_face_size as usize;
    let mut faces: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0; f * f * 4]);
    // Each output texel is an independent GGX convolution of the read-only
    // source, so faces and rows within a face are computed in parallel.
    faces.par_iter_mut().enumerate().for_each(|(face, out)| {
        out.par_chunks_mut(f * 4).enumerate().for_each(|(y, row)| {
            for x in 0..output_face_size {
                let n = cube_texel_dir(face, x, y as u32, output_face_size);
                // Split-sum approximation: V = R = N. The light direction is
                // then L = reflect(-V, H) = 2 (N·H) H - N.
                let mut accum = [0.0f32; 3];
                let mut total_weight = 0.0f32;
                for i in 0..samples {
                    let xi = hammersley(i, samples);
                    let h = importance_sample_ggx(xi, n, roughness);
                    let ndh = dot3(n, h);
                    if ndh <= 0.0 {
                        continue;
                    }
                    let l = normalize3([
                        2.0 * ndh * h[0] - n[0],
                        2.0 * ndh * h[1] - n[1],
                        2.0 * ndh * h[2] - n[2],
                    ]);
                    let ndl = dot3(n, l).max(0.0);
                    if ndl > 0.0 {
                        let env = sample_cube(source, source_face_size, l);
                        accum[0] += env[0] * ndl;
                        accum[1] += env[1] * ndl;
                        accum[2] += env[2] * ndl;
                        total_weight += ndl;
                    }
                }
                let off = x as usize * 4;
                if total_weight > 0.0 {
                    let inv = 1.0 / total_weight;
                    row[off] = accum[0] * inv;
                    row[off + 1] = accum[1] * inv;
                    row[off + 2] = accum[2] * inv;
                } else {
                    let n_sample = sample_cube(source, source_face_size, n);
                    row[off] = n_sample[0];
                    row[off + 1] = n_sample[1];
                    row[off + 2] = n_sample[2];
                }
                row[off + 3] = 1.0;
            }
        });
    });
    faces
}

// Payload codec

pub fn serialise_payload(
    irradiance_face: u32,
    prefilter_face: u32,
    prefilter_mips: u32,
    irradiance: &[Vec<f32>; 6],
    prefilter: &[[Vec<f32>; 6]],
) -> Vec<u8> {
    debug_assert_eq!(prefilter.len(), prefilter_mips as usize);
    let mut total = ENVMAP_PAYLOAD_HEADER_BYTES + 6 * (irradiance_face as usize).pow(2) * 4 * 4;
    for mip in 0..prefilter_mips {
        let s = (prefilter_face >> mip) as usize;
        total += 6 * s * s * 4 * 4;
    }
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(&ENVMAP_PAYLOAD_MAGIC.to_le_bytes());
    buf.extend_from_slice(&ENVMAP_FORMAT_RGBA32F.to_le_bytes());
    buf.extend_from_slice(&irradiance_face.to_le_bytes());
    buf.extend_from_slice(&prefilter_face.to_le_bytes());
    buf.extend_from_slice(&prefilter_mips.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // pad
    for face in irradiance {
        for &v in face {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    for mip in prefilter {
        for face in mip {
            for &v in face {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    buf
}

// Metadata read from a serialised EnvironmentMap payload. The byte ranges
// point into the payload buffer so the runtime can upload them directly.
#[derive(Debug)]
pub struct EnvMapView<'a> {
    pub irradiance_face: u32,
    pub prefilter_face: u32,
    // Mip count parsed from the header. The slice array
    // [`Self::prefilter_mip_bytes`] carries the same length, so callers
    // generally use that instead; the field is kept for symmetry with the
    // header layout.
    #[allow(dead_code)]
    pub prefilter_mips: u32,
    pub irradiance_bytes: &'a [u8],
    // One slice per prefilter mip, ordered mip 0 → mip N-1.
    pub prefilter_mip_bytes: Vec<&'a [u8]>,
}

// Deserialise a packed EnvironmentMap payload back into byte-range views into
// the buffer. The runtime upload path uses this to feed the per-face slices
// to the GPU without copying. Called by every backend at init time, and by
// the Metal hot-reload path via `update_environment_map`.
pub fn deserialise(bytes: &[u8]) -> Result<EnvMapView<'_>, String> {
    if bytes.len() < ENVMAP_PAYLOAD_HEADER_BYTES {
        return Err(format!(
            "envmap payload too short: {} bytes (need at least {} for header)",
            bytes.len(),
            ENVMAP_PAYLOAD_HEADER_BYTES
        ));
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != ENVMAP_PAYLOAD_MAGIC {
        return Err(format!(
            "envmap magic 0x{:08x} != expected 0x{:08x}",
            magic, ENVMAP_PAYLOAD_MAGIC
        ));
    }
    let format = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if format != ENVMAP_FORMAT_RGBA32F {
        return Err(format!("envmap format_id {} unsupported", format));
    }
    let irradiance_face = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let prefilter_face = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let prefilter_mips = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    let _pad = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
    if prefilter_mips == 0 || prefilter_mips > 12 {
        return Err(format!(
            "envmap prefilter_mips {} out of range",
            prefilter_mips
        ));
    }
    let mut off = ENVMAP_PAYLOAD_HEADER_BYTES;
    let irr_size = 6 * (irradiance_face as usize).pow(2) * 4 * 4;
    if off + irr_size > bytes.len() {
        return Err("envmap payload truncated in irradiance section".into());
    }
    let irradiance_bytes = &bytes[off..off + irr_size];
    off += irr_size;
    let mut prefilter_mip_bytes = Vec::with_capacity(prefilter_mips as usize);
    for mip in 0..prefilter_mips {
        let s = (prefilter_face >> mip) as usize;
        let mip_size = 6 * s * s * 4 * 4;
        if off + mip_size > bytes.len() {
            return Err(format!("envmap payload truncated in prefilter mip {}", mip));
        }
        prefilter_mip_bytes.push(&bytes[off..off + mip_size]);
        off += mip_size;
    }
    Ok(EnvMapView {
        irradiance_face,
        prefilter_face,
        prefilter_mips,
        irradiance_bytes,
        prefilter_mip_bytes,
    })
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_cube(face_size: u32, color: [f32; 3]) -> [Vec<f32>; 6] {
        let f = face_size as usize;
        std::array::from_fn(|_| {
            let mut face = Vec::with_capacity(f * f * 4);
            for _ in 0..f * f {
                face.extend_from_slice(&[color[0], color[1], color[2], 1.0]);
            }
            face
        })
    }

    fn face_mean(face: &[f32]) -> [f32; 3] {
        let n = face.len() / 4;
        let mut m = [0.0f32; 3];
        for px in face.chunks_exact(4) {
            m[0] += px[0];
            m[1] += px[1];
            m[2] += px[2];
        }
        [m[0] / n as f32, m[1] / n as f32, m[2] / n as f32]
    }

    fn face_variance_red(face: &[f32]) -> f32 {
        let n = face.len() / 4;
        let mean = face.chunks_exact(4).map(|p| p[0]).sum::<f32>() / n as f32;

        face.chunks_exact(4)
            .map(|p| (p[0] - mean).powi(2))
            .sum::<f32>()
            / n as f32
    }

    #[test]
    fn hammersley_first_sample_is_zero() {
        let s = hammersley(0, 1024);
        assert!(s[0].abs() < 1e-6, "x was {}", s[0]);
        assert!(s[1].abs() < 1e-6, "y was {}", s[1]);
    }

    #[test]
    fn hammersley_last_sample_is_just_under_one() {
        let s = hammersley(1023, 1024);
        assert!(s[0] > 0.99 && s[0] < 1.0, "x was {}", s[0]);
    }

    #[test]
    fn importance_sample_ggx_at_xi_zero_returns_n() {
        let n = [0.0, 0.0, 1.0];
        let h = importance_sample_ggx([0.0, 0.0], n, 0.5);
        // xi=(0,0) → cos_theta = 1 → H aligns with N.
        assert!((h[0] - 0.0).abs() < 1e-5);
        assert!((h[1] - 0.0).abs() < 1e-5);
        assert!((h[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn irradiance_solid_color_is_pi_times_color() {
        // Uniform environment L = (1, 0.5, 0.25). The hemispherical integral
        // of L * cos(θ) over the upper hemisphere is π * L. The discrete
        // (phi, theta) integration should converge to that.
        let source = solid_cube(8, [1.0, 0.5, 0.25]);
        let irr = compute_irradiance(&source, 8, 4, 64, 16);
        let mean = face_mean(&irr[0]);
        let expected = [
            std::f32::consts::PI * 1.0,
            std::f32::consts::PI * 0.5,
            std::f32::consts::PI * 0.25,
        ];
        // Discrete integration loses a few percent; accept ±5%.
        for c in 0..3 {
            let delta = (mean[c] - expected[c]).abs() / expected[c];
            assert!(
                delta < 0.05,
                "channel {} mean {} expected {}",
                c,
                mean[c],
                expected[c]
            );
        }
    }

    #[test]
    fn prefilter_mip_zero_matches_source_with_alpha_one() {
        let source = solid_cube(16, [0.7, 0.3, 0.1]);
        let mips = compute_prefilter(&source, 16, 3, 16);
        for face in &mips[0] {
            for px in 0..16 * 16 {
                let off = px * 4;
                assert!((face[off] - 0.7).abs() < 1e-6);
                assert!((face[off + 1] - 0.3).abs() < 1e-6);
                assert!((face[off + 2] - 0.1).abs() < 1e-6);
                assert!((face[off + 3] - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn prefilter_solid_color_stays_solid_at_high_roughness() {
        let source = solid_cube(16, [0.5, 0.5, 0.5]);
        let mips = compute_prefilter(&source, 16, 4, 32);
        // Last mip should still be ~0.5 grey since input is uniform.
        let mean = face_mean(&mips[3][0]);
        for (c, m) in mean.iter().enumerate() {
            assert!((m - 0.5).abs() < 0.02, "channel {} mean {}", c, m);
        }
    }

    #[test]
    fn prefilter_blurs_a_red_seam() {
        // Place a bright red column on +Z face only; prefilter at roughness=1
        // should spread it across the face so the variance drops vs the input.
        let face = 16usize;
        let mut source: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0; face * face * 4]);
        for face_data in source.iter_mut() {
            for p in face_data.chunks_exact_mut(4) {
                p[3] = 1.0;
            }
        }
        // +Z face (index 4): paint x=8 column bright red.
        for y in 0..face {
            let off = (y * face + 8) * 4;
            source[4][off] = 20.0;
        }
        let mips = compute_prefilter(&source, face as u32, 3, 256);
        // Compare variance of +Z face at mip 0 vs mip 2.
        let v0 = face_variance_red(&mips[0][4]);
        let v2 = face_variance_red(&mips[2][4]);
        assert!(
            v2 < v0 * 0.5,
            "prefilter did not blur: mip 0 var={}, mip 2 var={}",
            v0,
            v2
        );
    }

    #[test]
    fn payload_round_trip() {
        let source = solid_cube(8, [0.6, 0.4, 0.2]);
        let irr = compute_irradiance(&source, 8, 4, 32, 8);
        let prefilter = compute_prefilter(&source, 8, 2, 16);
        let blob = serialise_payload(4, 8, 2, &irr, &prefilter);
        let view = deserialise(&blob).expect("deserialise");
        assert_eq!(view.irradiance_face, 4);
        assert_eq!(view.prefilter_face, 8);
        assert_eq!(view.prefilter_mips, 2);
        assert_eq!(view.irradiance_bytes.len(), 6 * 4 * 4 * 4 * 4);
        assert_eq!(view.prefilter_mip_bytes[0].len(), 6 * 8 * 8 * 4 * 4);
        assert_eq!(view.prefilter_mip_bytes[1].len(), 6 * 4 * 4 * 4 * 4);
    }

    #[test]
    fn max_mip_count_clamps_at_four_pixels() {
        assert_eq!(max_mip_count(256), 7); // 256, 128, 64, 32, 16, 8, 4
        assert_eq!(max_mip_count(16), 3); // 16, 8, 4
        assert_eq!(max_mip_count(8), 2); // 8, 4
        assert_eq!(max_mip_count(4), 1); // 4
    }

    #[test]
    fn resolve_source_path_returns_directoried_paths_unchanged() {
        // Any path with a directory component is taken verbatim (relative or
        // absolute) so the hot-reload watcher subscribes to the right parent.
        assert_eq!(
            resolve_source_path("assets/hdri/x.hdr"),
            "assets/hdri/x.hdr"
        );
        assert_eq!(resolve_source_path("/abs/path.hdr"), "/abs/path.hdr");
    }

    // Build a minimal uncompressed Radiance HDR blob of `width × height` solid
    // (r, g, b) pixels. Mirrors the helpers in `build/cubemap.rs`'s test module:
    // they're private there, so re-implement the tiny encoder here.
    fn synth_rgbe(r: f32, g: f32, b: f32) -> [u8; 4] {
        let maxv = r.max(g).max(b);
        if maxv < 1e-32 {
            return [0, 0, 0, 0];
        }
        let bits = maxv.to_bits();
        let raw_exp = ((bits >> 23) & 0xff) as i32;
        let exp = raw_exp - 126;
        let mantissa_bits = (bits & 0x7f_ffff) | (126 << 23);
        let mantissa = f32::from_bits(mantissa_bits);
        let scale = (mantissa * 256.0) / maxv;
        [
            (r * scale) as u8,
            (g * scale) as u8,
            (b * scale) as u8,
            (exp + 128) as u8,
        ]
    }

    fn raw_hdr_blob(width: u32, height: u32, rgb: [f32; 3]) -> Vec<u8> {
        let pixel = synth_rgbe(rgb[0], rgb[1], rgb[2]);
        let mut blob = Vec::new();
        blob.extend_from_slice(b"#?RADIANCE\n");
        blob.extend_from_slice(b"FORMAT=32-bit_rle_rgbe\n\n");
        blob.extend_from_slice(format!("-Y {} +X {}\n", height, width).as_bytes());
        for _ in 0..(width * height) {
            blob.extend_from_slice(&pixel);
        }
        blob
    }

    #[test]
    fn decode_source_missing_file_errors() {
        let err =
            decode_source("/definitely/does/not/exist.hdr", 16, 8, 16).expect_err("should fail");
        assert!(
            err.contains("failed to open") || err.contains("No such file"),
            "got: {}",
            err
        );
    }

    #[test]
    fn decode_source_round_trips_through_deserialise() {
        // Write a tiny solid-colour HDR into a tempfile, decode it, and verify
        // the resulting payload deserialises with the requested sizes.
        let tmp = std::env::temp_dir().join(format!(
            "concinnity_envmap_decode_test_{}.hdr",
            std::process::id()
        ));
        std::fs::write(&tmp, raw_hdr_blob(16, 8, [0.6, 0.3, 0.15])).expect("write hdr");
        let payload = decode_source(tmp.to_str().unwrap(), 16, 8, 16).expect("decode");
        let _ = std::fs::remove_file(&tmp);
        let view = deserialise(&payload).expect("deserialise");
        assert_eq!(view.irradiance_face, 8);
        assert_eq!(view.prefilter_face, 16);
        // mip chain for face_size 16: 16, 8, 4 → 3 levels.
        assert_eq!(view.prefilter_mips, 3);
        assert_eq!(view.prefilter_mip_bytes.len(), 3);
        assert_eq!(view.irradiance_bytes.len(), 6 * 8 * 8 * 4 * 4);
    }
}
