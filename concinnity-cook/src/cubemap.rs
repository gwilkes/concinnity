// src/cubemap.rs
//
// Compiles a CubemapTexture component's args into the binary payload that the
// renderer reads at runtime. A cubemap is six square HDR faces stored as
// RGBA32F in face-major order (face 0 → face 5, each face row-major top-down).
//
// Source format: equirectangular Radiance HDR (.hdr / RGBE). The
// equirect is resampled at build time into six cube faces using bilinear
// interpolation in HDR space.
//
// Payload format (little-endian):
//   u32  magic     = b"CUBE" = 0x45425543
//   u32  face_size
//   u32  mip_count = 1
//   u32  format_id = 0  (RGBA32F)
//   6 * face_size * face_size * 4 * 4 bytes  raw RGBA32F, face-major
//
// Face order matches the standard cube convention used by Metal / Vulkan / DX:
//   0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z

use std::io::Read;

use concinnity_core::build::cubemap::{
    CUBE_FORMAT_RGBA32F, CUBE_PAYLOAD_HEADER_BYTES, CUBE_PAYLOAD_MAGIC, HdrImage, decode_hdr,
    equirect_to_cube,
};

// Validate that args specify either a supported source extension or omit it.
pub fn validate_cubemap_args(args: &serde_json::Value) -> Result<(), String> {
    let source = args.get("source").and_then(|v| v.as_str()).unwrap_or("");
    if source.is_empty() {
        return Err("CubemapTexture requires a `source` path".into());
    }
    if !source.to_ascii_lowercase().ends_with(".hdr") {
        return Err(format!(
            "CubemapTexture source '{}' must be a Radiance .hdr file",
            source
        ));
    }
    let face_size = args
        .get("face_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(256);
    if !(8..=4096).contains(&face_size) {
        return Err(format!(
            "CubemapTexture face_size {} out of range (8..=4096)",
            face_size
        ));
    }
    if !face_size.is_power_of_two() {
        return Err(format!(
            "CubemapTexture face_size {} must be a power of two",
            face_size
        ));
    }
    Ok(())
}

// Compile a CubemapTexture component's JSON args into a packed binary payload.
pub fn compile_cubemap_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    validate_cubemap_args(args)?;
    let source = args.get("source").and_then(|v| v.as_str()).unwrap();
    let face_size = args
        .get("face_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as u32;

    let hdr = load_hdr(source)?;
    let faces = equirect_to_cube(&hdr, face_size);
    Ok(serialise_faces(face_size, &faces))
}

fn serialise_faces(face_size: u32, faces: &[Vec<f32>; 6]) -> Vec<u8> {
    let face_floats = (face_size as usize) * (face_size as usize) * 4;
    let mut buf = Vec::with_capacity(CUBE_PAYLOAD_HEADER_BYTES + 6 * face_floats * 4);
    buf.extend_from_slice(&CUBE_PAYLOAD_MAGIC.to_le_bytes());
    buf.extend_from_slice(&face_size.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&CUBE_FORMAT_RGBA32F.to_le_bytes());
    for face in faces {
        debug_assert_eq!(face.len(), face_floats);
        for &v in face {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

fn load_hdr(path: &str) -> Result<HdrImage, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("failed to open HDR source '{}': {}", path, e))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("failed to read HDR source '{}': {}", path, e))?;
    decode_hdr(&bytes).map_err(|e| format!("failed to decode HDR '{}': {}", path, e))
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::build::cubemap::deserialise;

    #[test]
    fn payload_round_trip_via_deserialise() {
        let pixel = [0.7f32, 0.3, 0.2];
        let hdr = HdrImage {
            width: 16,
            height: 8,
            pixels: vec![pixel; 16 * 8],
        };
        let faces = equirect_to_cube(&hdr, 8);
        let blob = serialise_faces(8, &faces);
        let (face_size, face_bytes) = deserialise(&blob).expect("deserialise");
        assert_eq!(face_size, 8);
        assert_eq!(face_bytes.len(), 6 * 8 * 8 * 4 * 4);
        // First face, first pixel:
        let p0 = f32::from_le_bytes(face_bytes[0..4].try_into().unwrap());
        let p1 = f32::from_le_bytes(face_bytes[4..8].try_into().unwrap());
        let p2 = f32::from_le_bytes(face_bytes[8..12].try_into().unwrap());
        let p3 = f32::from_le_bytes(face_bytes[12..16].try_into().unwrap());
        assert!((p0 - pixel[0]).abs() < 1e-4);
        assert!((p1 - pixel[1]).abs() < 1e-4);
        assert!((p2 - pixel[2]).abs() < 1e-4);
        assert!((p3 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn validate_cubemap_args_rejects_bad_face_size() {
        let args = serde_json::json!({ "source": "foo.hdr", "face_size": 300 });
        let err = validate_cubemap_args(&args).unwrap_err();
        assert!(err.contains("power of two"), "got: {}", err);
    }

    #[test]
    fn validate_cubemap_args_requires_hdr_extension() {
        let args = serde_json::json!({ "source": "foo.png" });
        let err = validate_cubemap_args(&args).unwrap_err();
        assert!(err.contains(".hdr"), "got: {}", err);
    }

    #[test]
    fn validate_cubemap_args_accepts_defaults() {
        let args = serde_json::json!({ "source": "studio.hdr" });
        validate_cubemap_args(&args).expect("defaults should validate");
    }
}
