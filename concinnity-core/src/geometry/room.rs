// src/geometry/room.rs: room interior geometry.
//
// Each face is two CCW triangles. Vertex colour distinguishes surfaces:
// warm dark grey floor, light grey ceiling, slightly different greys per wall.
// UV coordinates tile at one repeat per metre.

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;

// Parses room args and delegates to build_room_geometry.
pub(super) fn build_room(args: &serde_json::Value) -> Result<(Verts, Vec<u16>), String> {
    let half_width = args
        .get("half_width")
        .and_then(|v| v.as_f64())
        .unwrap_or(8.0) as f32;
    let half_depth = args
        .get("half_depth")
        .and_then(|v| v.as_f64())
        .unwrap_or(10.0) as f32;
    let ceiling_height = args
        .get("ceiling_height")
        .and_then(|v| v.as_f64())
        .unwrap_or(3.5) as f32;
    Ok(build_room_geometry(
        half_width,
        half_depth,
        0.0,
        ceiling_height,
    ))
}

// Build room geometry from explicit extents.
//
// Returns `(vertices, indices)` where each vertex is `(pos, normal, color, uv)`.
// Winding is CCW when viewed from inside. UV coordinates tile at one repeat
// per metre. Normals point inward so diffuse lighting is correct for a camera
// inside the room.
pub(super) fn build_room_geometry(
    half_width: f32,
    half_depth: f32,
    floor_y: f32,
    ceiling_y: f32,
) -> (Verts, Vec<u16>) {
    let mut verts: Verts = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();

    let (xn, xp) = (-half_width, half_width);
    let (yn, yp) = (floor_y, ceiling_y);
    let (zn, zp) = (-half_depth, half_depth);

    let width = half_width * 2.0;
    let depth = half_depth * 2.0;
    let height = yp - yn;

    let mut push_quad = |a: [f32; 3],
                         b: [f32; 3],
                         c: [f32; 3],
                         d: [f32; 3],
                         normal: [f32; 3],
                         color: [f32; 3],
                         uv_a: [f32; 2],
                         uv_b: [f32; 2],
                         uv_c: [f32; 2],
                         uv_d: [f32; 2]| {
        let base = verts.len() as u16;
        verts.extend_from_slice(&[
            (a, normal, color, uv_a),
            (b, normal, color, uv_b),
            (c, normal, color, uv_c),
            (d, normal, color, uv_d),
        ]);
        idxs.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    };

    let floor_color = [0.25, 0.22, 0.20];
    let ceiling_color = [0.70, 0.70, 0.72];
    let wall_n_color = [0.45, 0.45, 0.48];
    let wall_s_color = [0.40, 0.40, 0.43];
    let wall_e_color = [0.50, 0.48, 0.46];
    let wall_w_color = [0.42, 0.44, 0.46];

    // floor: normal points up into the room
    push_quad(
        [xn, yn, zn],
        [xp, yn, zn],
        [xp, yn, zp],
        [xn, yn, zp],
        [0.0, 1.0, 0.0],
        floor_color,
        [0.0, 0.0],
        [width, 0.0],
        [width, depth],
        [0.0, depth],
    );

    // ceiling: normal points down into the room
    push_quad(
        [xn, yp, zp],
        [xp, yp, zp],
        [xp, yp, zn],
        [xn, yp, zn],
        [0.0, -1.0, 0.0],
        ceiling_color,
        [0.0, 0.0],
        [width, 0.0],
        [width, depth],
        [0.0, depth],
    );

    // north wall (+Z face); normal points toward -Z
    push_quad(
        [xp, yn, zp],
        [xn, yn, zp],
        [xn, yp, zp],
        [xp, yp, zp],
        [0.0, 0.0, -1.0],
        wall_n_color,
        [0.0, height],
        [width, height],
        [width, 0.0],
        [0.0, 0.0],
    );

    // south wall (-Z face); normal points toward +Z
    push_quad(
        [xn, yn, zn],
        [xp, yn, zn],
        [xp, yp, zn],
        [xn, yp, zn],
        [0.0, 0.0, 1.0],
        wall_s_color,
        [0.0, height],
        [width, height],
        [width, 0.0],
        [0.0, 0.0],
    );

    // east wall (+X face); normal points toward -X
    push_quad(
        [xp, yn, zn],
        [xp, yn, zp],
        [xp, yp, zp],
        [xp, yp, zn],
        [-1.0, 0.0, 0.0],
        wall_e_color,
        [0.0, height],
        [depth, height],
        [depth, 0.0],
        [0.0, 0.0],
    );

    // west wall (-X face); normal points toward +X
    push_quad(
        [xn, yn, zp],
        [xn, yn, zn],
        [xn, yp, zn],
        [xn, yp, zp],
        [1.0, 0.0, 0.0],
        wall_w_color,
        [0.0, height],
        [depth, height],
        [depth, 0.0],
        [0.0, 0.0],
    );

    (verts, idxs)
}
