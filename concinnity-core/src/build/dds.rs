// src/build/dds.rs
//
// Decodes a DDS container into RGBA8. Only the legacy (non-DX10) header and the
// three fourCC formats the asset set uses are handled: DXT1 (BC1), DXT5 (BC3),
// and ATI2 (BC5). The top mip is decoded; any further mips in the file are
// ignored. A DX10 extended header is rejected with a clear message rather than
// silently misread.

const MAGIC: &[u8; 4] = b"DDS ";
const HEADER_LEN: usize = 124;
const PIXELDATA_OFFSET: usize = 4 + HEADER_LEN;

// Decode the top mip of a DDS file into (width, height, RGBA8 pixels).
pub fn decode_dds(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    if bytes.len() < PIXELDATA_OFFSET {
        return Err(format!("DDS too short: {} bytes", bytes.len()));
    }
    if &bytes[0..4] != MAGIC {
        return Err("not a DDS file (bad magic)".to_string());
    }

    let height = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let width = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let fourcc = &bytes[84..88];
    let data = &bytes[PIXELDATA_OFFSET..];

    if width == 0 || height == 0 {
        return Err(format!("DDS has zero dimension {}x{}", width, height));
    }

    let pixels = match fourcc {
        b"DXT1" => super::bcn::decode_bc1(data, width, height)?,
        b"DXT5" => super::bcn::decode_bc3(data, width, height)?,
        b"ATI2" => super::bcn::decode_bc5(data, width, height)?,
        b"DX10" => {
            return Err("DDS uses a DX10 extended header, which is not supported; \
                 re-export as DXT1/DXT5/ATI2"
                .to_string());
        }
        other => {
            return Err(format!(
                "unsupported DDS fourCC {:?}; only DXT1, DXT5, and ATI2 are handled",
                String::from_utf8_lossy(other)
            ));
        }
    };

    Ok((width, height, pixels))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal legacy DDS around a single block payload.
    fn wrap_dds(fourcc: &[u8; 4], width: u32, height: u32, block: &[u8]) -> Vec<u8> {
        let mut v = vec![0u8; PIXELDATA_OFFSET];
        v[0..4].copy_from_slice(MAGIC);
        v[12..16].copy_from_slice(&height.to_le_bytes());
        v[16..20].copy_from_slice(&width.to_le_bytes());
        v[84..88].copy_from_slice(fourcc);
        v.extend_from_slice(block);
        v
    }

    #[test]
    fn decodes_dxt1() {
        let block = [0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]; // solid red
        let dds = wrap_dds(b"DXT1", 4, 4, &block);
        let (w, h, px) = decode_dds(&dds).unwrap();
        assert_eq!((w, h), (4, 4));
        assert_eq!(&px[0..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn rejects_dx10_header() {
        let dds = wrap_dds(b"DX10", 4, 4, &[0u8; 16]);
        let err = decode_dds(&dds).unwrap_err();
        assert!(err.contains("DX10"), "got: {err}");
    }

    #[test]
    fn rejects_unknown_fourcc() {
        let dds = wrap_dds(b"DXT3", 4, 4, &[0u8; 16]);
        assert!(decode_dds(&dds).is_err());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut dds = wrap_dds(b"DXT1", 4, 4, &[0u8; 8]);
        dds[0] = b'X';
        assert!(decode_dds(&dds).is_err());
    }
}
