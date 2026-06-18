// src/geometry/skybox.rs: large inside-facing cube used as a skybox.
//
// Wound CCW from the interior. The floor face is included to prevent the clear
// colour showing at the horizon or beyond terrain edges.
//
// The blue channel is set to 2.0 (outside the [0, 1] scene range) so the
// fragment shader can identify sky vertices with a simple threshold and skip
// diffuse lighting on them.
//
// UV layout is present for legacy compatibility; sky rendering now uses the
// view-direction rather than UV so the values do not affect the output.
//
// Parameters:
//   size -- half-extent on all axes (default 490.0; keep below the Camera3D far plane)

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;

// A skybox face: four corner positions, the inward normal, then the four UVs
// (one per corner).
type SkyboxFace = (
    [f32; 3],
    [f32; 3],
    [f32; 3],
    [f32; 3],
    [f32; 3],
    [[f32; 2]; 4],
);

pub(super) fn build_skybox(args: &serde_json::Value) -> Result<(Verts, Vec<u16>), String> {
    let s = args.get("size").and_then(|v| v.as_f64()).unwrap_or(490.0) as f32;
    let color = [1.0f32, 1.0, 2.0];

    let mut verts: Verts = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();

    let faces: &[SkyboxFace] = &[
        // ceiling (+Y, viewed from below) -- CCW from interior requires reversed vertex order
        (
            [-s, s, s],
            [-s, s, -s],
            [s, s, -s],
            [s, s, s],
            [0.0, -1.0, 0.0],
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        ),
        // floor (-Y, viewed from above) -- CCW from interior requires reversed vertex order
        (
            [-s, -s, -s],
            [-s, -s, s],
            [s, -s, s],
            [s, -s, -s],
            [0.0, 1.0, 0.0],
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        ),
        // north wall (+Z)
        (
            [s, -s, s],
            [-s, -s, s],
            [-s, s, s],
            [s, s, s],
            [0.0, 0.0, -1.0],
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ),
        // south wall (-Z)
        (
            [-s, -s, -s],
            [s, -s, -s],
            [s, s, -s],
            [-s, s, -s],
            [0.0, 0.0, 1.0],
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ),
        // east wall (+X)
        (
            [s, -s, -s],
            [s, -s, s],
            [s, s, s],
            [s, s, -s],
            [-1.0, 0.0, 0.0],
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ),
        // west wall (-X)
        (
            [-s, -s, s],
            [-s, -s, -s],
            [-s, s, -s],
            [-s, s, s],
            [1.0, 0.0, 0.0],
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ),
    ];

    for (a, b, c, d, normal, uvs) in faces {
        let base = verts.len() as u16;
        verts.extend_from_slice(&[
            (*a, *normal, color, uvs[0]),
            (*b, *normal, color, uvs[1]),
            (*c, *normal, color, uvs[2]),
            (*d, *normal, color, uvs[3]),
        ]);
        idxs.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    Ok((verts, idxs))
}
