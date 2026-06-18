// src/gfx/chunk_coord.rs
//
// Integer coordinate of one chunk column in an infinite voxel world.
//
// The chunk grid is 2D: infinite in X/Z, a single fixed-height chunk in Y
// (voxel/Minecraft-style worlds bound the vertical extent). A `ChunkCoord` is
// the integer index of a chunk column; the world position of its `(0,0,0)`
// corner is `coord * chunk_world_size`.
//
// This module is written against `core` only -- no `std`, no `alloc`, no
// threads or I/O -- so it can move into a future `no_std` client runtime
// unchanged, alongside `gfx::streaming` and `gfx::range_alloc`. The
// world->chunk conversion deliberately avoids `f32::floor` (which lives in
// `std`, not `core`) by doing the floor with an integer truncation and a sign
// correction.

// Integer index of a chunk column in the infinite X/Z grid.
//
// `x` and `z` are chunk indices, not world units; the chunk's world-space
// `(0,0,0)` corner is `(x as f32 * chunk_w, 0, z as f32 * chunk_d)`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ChunkCoord {
    pub x: i32,
    pub z: i32,
}

impl ChunkCoord {
    // A chunk coordinate from raw indices.
    pub const fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    // The chunk column a world-space `(wx, wz)` position falls in.
    //
    // `chunk_w` / `chunk_d` are the chunk's world-space size on X / Z. Uses a
    // `core`-only floor (truncate-toward-zero plus a sign fix) so a negative
    // coordinate maps to the chunk *below* it rather than toward zero.
    pub fn from_world(wx: f32, wz: f32, chunk_w: f32, chunk_d: f32) -> Self {
        Self::new(floor_div(wx, chunk_w), floor_div(wz, chunk_d))
    }

    // World-space `(x, z)` of this chunk's `(0,0,0)` corner.
    pub fn origin_world(self, chunk_w: f32, chunk_d: f32) -> (f32, f32) {
        (self.x as f32 * chunk_w, self.z as f32 * chunk_d)
    }

    // This coordinate shifted by `(dx, dz)` chunks.
    pub fn offset(self, dx: i32, dz: i32) -> Self {
        Self::new(self.x + dx, self.z + dz)
    }

    // Chebyshev (square-ring) distance to `other`, in chunks.
    //
    // This is the metric the streaming window uses for radius membership: a
    // chunk is "in view" when its Chebyshev distance to the camera chunk is
    // `<= view_radius`, which selects a square block of chunks.
    pub fn chebyshev_distance(self, other: ChunkCoord) -> i32 {
        let dx = (self.x - other.x).abs();
        let dz = (self.z - other.z).abs();
        if dx > dz { dx } else { dz }
    }

    // Squared Euclidean distance to `other`, in chunk units.
    //
    // Used to prioritise loads (nearest chunk first). Squared keeps the math
    // `sqrt`-free; `i64` so a far-apart pair cannot overflow.
    pub fn sq_distance(self, other: ChunkCoord) -> i64 {
        let dx = (self.x - other.x) as i64;
        let dz = (self.z - other.z) as i64;
        dx * dx + dz * dz
    }
}

// `floor(v / size)` as an `i32`, without `f32::floor` (which is `std`-only).
//
// `as i32` truncates toward zero; for a negative non-integer quotient that is
// one too high, so the result is decremented. A `size <= 0` is meaningless
// for a chunk grid and collapses to chunk 0 rather than dividing by zero.
fn floor_div(v: f32, size: f32) -> i32 {
    if size <= 0.0 {
        return 0;
    }
    let q = v / size;
    let t = q as i32;
    if q < 0.0 && (t as f32) != q { t - 1 } else { t }
}

// Rebase a column-major view matrix onto a render `origin` for
// camera-relative rendering.
//
// The returned matrix keeps `view`'s orientation but places the camera at
// `cam_pos - origin`, so geometry expressed relative to `origin` transforms
// identically to the absolute scene -- `camera_relative_view(view, ..) *
// model_relative` equals `view * model_absolute` -- while being computed
// entirely from small coordinates, avoiding the catastrophic cancellation a
// large-magnitude `view * model` product suffers.
//
// Only the rotation (the upper-left 3x3) of `view` is used; the translation
// column is rebuilt as `rotation * (origin - cam_pos)`. That subtraction is
// exact in `f32` when the camera sits within a chunk of the origin (two
// nearby like-signed values subtract without rounding), so no precision is
// lost rebasing.
pub fn camera_relative_view(
    view: [[f32; 4]; 4],
    cam_pos: [f32; 3],
    origin: [f32; 3],
) -> [[f32; 4]; 4] {
    // Column-major: view[col][row]. The rebased translation is rotation *
    // (origin - cam_pos) -- the view-space translation that places a camera at
    // cam_pos - origin under this orientation.
    let d = [
        origin[0] - cam_pos[0],
        origin[1] - cam_pos[1],
        origin[2] - cam_pos[2],
    ];
    let tx = view[0][0] * d[0] + view[1][0] * d[1] + view[2][0] * d[2];
    let ty = view[0][1] * d[0] + view[1][1] * d[1] + view[2][1] * d[2];
    let tz = view[0][2] * d[0] + view[1][2] * d[1] + view[2][2] * d[2];
    [view[0], view[1], view[2], [tx, ty, tz, 1.0]]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Column-major 4x4 transform of a point: m * [p, 1].
    fn apply(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
        [
            m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
            m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
            m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
        ]
    }

    #[test]
    fn from_world_maps_positive_positions_to_chunks() {
        // chunk size 16: x in [0,16) -> chunk 0, [16,32) -> chunk 1.
        assert_eq!(
            ChunkCoord::from_world(0.0, 0.0, 16.0, 16.0),
            ChunkCoord::new(0, 0)
        );
        assert_eq!(
            ChunkCoord::from_world(15.9, 0.0, 16.0, 16.0),
            ChunkCoord::new(0, 0)
        );
        assert_eq!(
            ChunkCoord::from_world(16.0, 0.0, 16.0, 16.0),
            ChunkCoord::new(1, 0)
        );
        assert_eq!(
            ChunkCoord::from_world(33.0, 48.0, 16.0, 16.0),
            ChunkCoord::new(2, 3)
        );
    }

    #[test]
    fn from_world_floors_negative_positions() {
        // -0.1 belongs to chunk -1, not chunk 0 -- a truncating cast would be wrong.
        assert_eq!(
            ChunkCoord::from_world(-0.1, 0.0, 16.0, 16.0),
            ChunkCoord::new(-1, 0)
        );
        assert_eq!(
            ChunkCoord::from_world(-16.0, 0.0, 16.0, 16.0),
            ChunkCoord::new(-1, 0)
        );
        assert_eq!(
            ChunkCoord::from_world(-16.1, 0.0, 16.0, 16.0),
            ChunkCoord::new(-2, 0)
        );
        assert_eq!(
            ChunkCoord::from_world(-1.0, -1.0, 16.0, 16.0),
            ChunkCoord::new(-1, -1)
        );
    }

    #[test]
    fn from_world_tolerates_a_non_positive_chunk_size() {
        assert_eq!(
            ChunkCoord::from_world(99.0, 99.0, 0.0, 0.0),
            ChunkCoord::new(0, 0)
        );
    }

    #[test]
    fn origin_world_round_trips_through_from_world() {
        let c = ChunkCoord::new(-3, 5);
        let (ox, oz) = c.origin_world(16.0, 16.0);
        assert_eq!((ox, oz), (-48.0, 80.0));
        // a position at the origin corner maps back to the same chunk
        assert_eq!(ChunkCoord::from_world(ox, oz, 16.0, 16.0), c);
    }

    #[test]
    fn offset_shifts_the_coordinate() {
        assert_eq!(ChunkCoord::new(2, 2).offset(-3, 1), ChunkCoord::new(-1, 3));
    }

    #[test]
    fn chebyshev_distance_is_the_square_ring_metric() {
        let c = ChunkCoord::new(0, 0);
        assert_eq!(c.chebyshev_distance(ChunkCoord::new(3, 1)), 3);
        assert_eq!(c.chebyshev_distance(ChunkCoord::new(-2, 4)), 4);
        assert_eq!(c.chebyshev_distance(c), 0);
    }

    #[test]
    fn sq_distance_orders_nearer_chunks_first() {
        let cam = ChunkCoord::new(0, 0);
        assert!(cam.sq_distance(ChunkCoord::new(1, 0)) < cam.sq_distance(ChunkCoord::new(2, 0)));
        assert_eq!(cam.sq_distance(ChunkCoord::new(3, 4)), 25);
    }

    #[test]
    fn camera_relative_view_matches_the_absolute_transform() {
        // A 90-degree yaw rotation (orthonormal), column-major.
        let rot = [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let cam = [1000.0, 12.0, -500.0];
        // A correct view matrix carries translation -rotation * cam_pos.
        let t = apply(rot, [-cam[0], -cam[1], -cam[2]]);
        let view = [rot[0], rot[1], rot[2], [t[0], t[1], t[2], 1.0]];

        // Render origin at a chunk corner near the camera.
        let origin = [992.0, 0.0, -512.0];
        let rebased = camera_relative_view(view, cam, origin);

        // The rebased view transforms an origin-relative point exactly as the
        // absolute view transforms the same point in world space.
        let p = [1005.0, 3.0, -495.0];
        let abs = apply(view, p);
        let rel = apply(
            rebased,
            [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]],
        );
        for i in 0..3 {
            assert!(
                (abs[i] - rel[i]).abs() < 1e-3,
                "axis {}: absolute {} vs rebased {}",
                i,
                abs[i],
                rel[i]
            );
        }
    }

    #[test]
    fn camera_relative_view_keeps_the_orientation_columns() {
        // Rebasing changes only the translation column; the rotation is intact.
        let view = [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [42.0, 7.0, -9.0, 1.0],
        ];
        let rebased = camera_relative_view(view, [10.0, 0.0, 20.0], [16.0, 0.0, 16.0]);
        assert_eq!(rebased[0], view[0]);
        assert_eq!(rebased[1], view[1]);
        assert_eq!(rebased[2], view[2]);
        assert_eq!(rebased[3][3], 1.0);
    }
}
