// Build-time mesh payload entry point. Most generators compile with no source
// decode and delegate straight to concinnity-core; the `heightfield` generator
// needs its source image decoded (the runtime crate links no image decoders),
// so this crate decodes it here and hands the pixels to core's
// `compile_heightfield_payload`.

// Compile a Mesh / ProceduralMesh component's JSON args into a packed binary
// payload. The `heightfield` generator's source PNG is decoded here in the
// build crate; every other generator delegates to
// `concinnity_core::geometry::compile_mesh_payload`.
pub fn compile_mesh_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    if args.get("generator").and_then(|v| v.as_str()) == Some("heightfield") {
        let source = args
            .get("source")
            .and_then(|v| v.as_str())
            .ok_or("heightfield generator requires a `source` PNG path")?;
        let (w, h, rgba) =
            crate::texture::decode_source(source, 0).map_err(|e| format!("heightfield: {e}"))?;
        concinnity_core::geometry::compile_heightfield_payload(args, w, h, rgba)
    } else {
        concinnity_core::geometry::compile_mesh_payload(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heightfield_end_to_end_via_tmp_png() {
        // Write a minimal valid PNG to a temp file and verify the build-side
        // path decodes it, generates the grid, and bakes a collider trailer
        // whose heights equal the rendered mesh's per-vertex Y.
        use std::io::Write;
        let path = std::env::temp_dir().join("concinnity_test_heightfield_build.png");
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut buf, 8, 8);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("png header");
            // A diagonal ramp 0..63 across the 8x8 grid.
            let pixels: Vec<u8> = (0..64u8).collect();
            writer.write_image_data(&pixels).expect("png data");
        }
        let mut f = std::fs::File::create(&path).expect("tmp file");
        f.write_all(&buf).expect("write png");
        drop(f);

        let args = serde_json::json!({
            "generator": "heightfield",
            "half_width": 5.0,
            "half_depth": 5.0,
            "subdivisions": 4,
            "source": path.to_str().unwrap(),
            "elevation_min": 0.0,
            "elevation_max": 10.0,
        });
        let payload = compile_mesh_payload(&args).expect("compiles");

        let grid = concinnity_core::gfx::mesh_payload::deserialise_heightfield(&payload)
            .expect("parse")
            .expect("heightfield trailer present");
        assert_eq!((grid.rows, grid.cols), (5, 5));
        let (mesh_verts, _, _) =
            concinnity_core::gfx::mesh_payload::deserialise_with_lods(&payload)
                .expect("render path");
        assert_eq!(grid.heights.len(), mesh_verts.len());
        for (h, v) in grid.heights.iter().zip(&mesh_verts) {
            assert_eq!(*h, v.pos[1]);
        }
        // Elevation is bracketed by the configured range and varies.
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for v in &mesh_verts {
            min_y = min_y.min(v.pos[1]);
            max_y = max_y.max(v.pos[1]);
        }
        assert!(min_y >= 0.0);
        assert!(max_y <= 10.0);
        assert!(
            max_y > min_y,
            "expected variation but got flat at {}",
            max_y
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn heightfield_missing_source_errors() {
        let args = serde_json::json!({ "generator": "heightfield", "elevation_max": 1.0 });
        let err = compile_mesh_payload(&args).unwrap_err();
        assert!(err.contains("source"), "got: {err}");
    }

    #[test]
    fn non_heightfield_generator_delegates_to_core() {
        let args = serde_json::json!({ "generator": "sphere", "radius": 1.0 });
        assert!(compile_mesh_payload(&args).is_ok());
    }
}
