// src/environment_map.rs
//
// Compiles an EnvironmentMap component's args into a payload bundling two
// precomputed IBL cubemaps:
//
//   - **Irradiance cubemap.** Low-resolution (8x8 per face by default)
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
//   u32  irradiance_face    (e.g. 8)
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

use concinnity_core::assets::EnvironmentMap;
use concinnity_core::build::cubemap::{HdrImage, equirect_to_cube};
use concinnity_core::build::environment_map::{
    DEFAULT_IRRADIANCE_PHI_SAMPLES, DEFAULT_IRRADIANCE_THETA_SAMPLES, compute_irradiance,
    compute_prefilter, load_hdr_file, max_mip_count, resolve_hdr_source, serialise_payload,
};

// Validation + entry point
//
// The three tunables (prefilter/irradiance face size, prefilter sample count)
// have a single source of truth: the `EnvironmentMap` `Default` impl in
// concinnity-core. Args are deserialised through that struct, so a field absent
// from the JSONL inherits the core default instead of a constant duplicated here.

fn resolve_args(args: &serde_json::Value) -> Result<EnvironmentMap, String> {
    let params: EnvironmentMap = serde_json::from_value(args.clone())
        .map_err(|e| format!("invalid EnvironmentMap args: {}", e))?;
    match (params.source.is_empty(), params.generator.is_empty()) {
        (true, true) => return Err("EnvironmentMap requires either `source` or `generator`".into()),
        (false, false) => {
            return Err("EnvironmentMap takes either `source` or `generator`, not both".into());
        }
        (false, true) => {
            if !params.source.to_ascii_lowercase().ends_with(".hdr") {
                return Err(format!(
                    "EnvironmentMap source '{}' must be a Radiance .hdr file",
                    params.source
                ));
            }
        }
        (true, false) => match params.generator.as_str() {
            "sky" => {}
            other => return Err(format!("unknown EnvironmentMap generator '{}'", other)),
        },
    }
    let prefilter_face = params.prefilter_face_size;
    if !(16..=1024).contains(&prefilter_face) || !prefilter_face.is_power_of_two() {
        return Err(format!(
            "EnvironmentMap prefilter_face_size {} must be a power of two in 16..=1024",
            prefilter_face
        ));
    }
    let irradiance_face = params.irradiance_face_size;
    if !(8..=128).contains(&irradiance_face) || !irradiance_face.is_power_of_two() {
        return Err(format!(
            "EnvironmentMap irradiance_face_size {} must be a power of two in 8..=128",
            irradiance_face
        ));
    }
    if !params.prefilter_clamp.is_finite() || params.prefilter_clamp < 0.0 {
        return Err(format!(
            "EnvironmentMap prefilter_clamp {} must be a finite value >= 0 (0 disables it)",
            params.prefilter_clamp
        ));
    }
    Ok(params)
}

pub fn validate_environment_map_args(args: &serde_json::Value) -> Result<(), String> {
    resolve_args(args).map(|_| ())
}

pub fn compile_environment_map_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let params = resolve_args(args)?;
    let prefilter_face = params.prefilter_face_size;
    let irradiance_face = params.irradiance_face_size;
    let prefilter_samples = params.prefilter_samples;
    let prefilter_mips = max_mip_count(prefilter_face);

    let hdr = if !params.source.is_empty() {
        // A bare filename (no directory component) is resolved via the same
        // asset-search the build pipeline uses for shader sources: search
        // .concinnity/assets/ recursively, falling back to the raw path so an
        // absolute or relative path also works.
        let resolved = resolve_hdr_source(&params.source);
        load_hdr_file(&resolved)?
    } else {
        match params.generator.as_str() {
            "sky" => generate_sky_equirect(),
            other => return Err(format!("unknown EnvironmentMap generator '{}'", other)),
        }
    };
    let source_cube = equirect_to_cube(&hdr, prefilter_face);
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
        params.prefilter_clamp,
        // Imported environment map: mip 0 IS the on-screen skybox, keep it unclamped.
        false,
    );
    Ok(serialise_payload(
        irradiance_face,
        prefilter_face,
        prefilter_mips,
        &irradiance,
        &prefilter,
    ))
}

// Synthetic equirectangular HDR for the `generator: "sky"` source. Same
// palette as the 2D `generate_sky` texture generator, extended to a full
// sphere: top half is zenith → mid → horizon, bottom half is solid horizon
// (no ground term yet, IBL only). Slightly super-1.0 values toward the sun
// direction give the prefilter convolution something HDR-like to chew on.
fn generate_sky_equirect() -> HdrImage {
    let width = 256u32;
    let height = 128u32;
    // Linear-light approximations of the procedural sky palette.
    let zenith = [0.012, 0.105, 0.526];
    let mid = [0.142, 0.355, 0.708];
    let horizon = [0.563, 0.726, 0.857];
    // Sun direction in equirect UV space: roughly south, 30° elevation.
    let sun_u = 0.25_f32;
    let sun_v = 0.35_f32;
    let sun_color = [3.0, 2.6, 2.1];
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        let v = row as f32 / (height - 1) as f32;
        // Map v to a "sky elevation" t in [0, 1]: 0 at horizon, 1 at zenith.
        // Top half v∈[0, 0.5] maps to zenith→horizon, bottom half stays flat at horizon.
        let t = if v < 0.5 { 1.0 - v * 2.0 } else { 0.0 };
        let base = if t > 0.5 {
            let s = (t - 0.5) * 2.0;
            [
                lerp(mid[0], zenith[0], s),
                lerp(mid[1], zenith[1], s),
                lerp(mid[2], zenith[2], s),
            ]
        } else {
            let s = t * 2.0;
            let warm = (1.0 - s).powi(2) * 0.07;
            [
                lerp(horizon[0], mid[0], s) + warm * 0.5,
                lerp(horizon[1], mid[1], s) + warm * 0.25,
                lerp(horizon[2], mid[2], s),
            ]
        };
        for col in 0..width {
            let u = col as f32 / (width - 1) as f32;
            // Soft circular sun: gaussian-ish bump in equirect UV space.
            let du = (u - sun_u).abs();
            let du = du.min(1.0 - du); // wrap horizontally
            let dv = v - sun_v;
            let d2 = du * du + dv * dv;
            let sun_amt = (-d2 / 0.0006).exp();
            let r = base[0] + sun_color[0] * sun_amt;
            let g = base[1] + sun_color[1] * sun_amt;
            let b = base[2] + sun_color[2] * sun_amt;
            pixels.push([r, g, b]);
        }
    }
    HdrImage {
        width,
        height,
        pixels,
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::build::environment_map::deserialise;

    #[test]
    fn validate_environment_map_args_requires_source_or_generator() {
        let args = serde_json::json!({});
        let err = validate_environment_map_args(&args).unwrap_err();
        assert!(err.contains("source") || err.contains("generator"));
    }

    #[test]
    fn validate_environment_map_args_rejects_non_hdr() {
        let args = serde_json::json!({ "source": "studio.png" });
        let err = validate_environment_map_args(&args).unwrap_err();
        assert!(err.contains(".hdr"));
    }

    #[test]
    fn validate_environment_map_args_accepts_sky_generator() {
        let args = serde_json::json!({ "generator": "sky" });
        validate_environment_map_args(&args).expect("sky generator should validate");
    }

    #[test]
    fn validate_environment_map_args_rejects_both_source_and_generator() {
        let args = serde_json::json!({ "source": "x.hdr", "generator": "sky" });
        let err = validate_environment_map_args(&args).unwrap_err();
        assert!(err.contains("not both"));
    }

    #[test]
    fn sky_generator_compiles_into_full_payload() {
        let args = serde_json::json!({
            "generator": "sky",
            "prefilter_face_size": 16,
            "irradiance_face_size": 8,
            "prefilter_samples": 32,
        });
        let blob = compile_environment_map_payload(&args).expect("compile");
        let view = deserialise(&blob).expect("deserialise");
        assert_eq!(view.irradiance_face, 8);
        assert_eq!(view.prefilter_face, 16);
        // Prefilter mips for face_size 16: 16, 8, 4 → 3 levels.
        assert_eq!(view.prefilter_mips, 3);
    }
}
