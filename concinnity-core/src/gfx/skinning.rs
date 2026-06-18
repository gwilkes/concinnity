// src/gfx/skinning.rs
//
// Backend-agnostic skeletal-animation math: skeleton hierarchy resolution,
// animation-clip sampling, and the per-joint skinning matrices the vertex
// shader consumes. Pure functions only, no GPU or backend handles.
//
// A skinned mesh carries a `Skeleton` (a joint hierarchy with a bind pose) and
// is animated by `AnimationClip`s. Each frame `AnimationSystem` samples the
// active clip into per-joint local transforms, composes the hierarchy into
// world-space joint matrices, and multiplies by the inverse bind matrices to
// get skinning matrices. The vertex shader then blends up to four of those per
// vertex.
//
// Rotations are stored as YXZ Euler degrees (matching `Prop.rotation_deg`).
// Between keyframes, translation and scale interpolate linearly while rotation
// is converted to a quaternion and slerped (shortest-arc, constant angular
// velocity), so multi-axis joint rotation follows the correct path rather
// than the skewed one a component-wise Euler lerp would take.

use crate::gfx::render_types::MAX_JOINTS;

// Column-major 4x4 matrix, `m[col][row]`: the layout shared by every
// renderer uniform in this codebase.
pub type Mat4 = [[f32; 4]; 4];

// Column-major 4x4 identity.
pub const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

// Column-major 4x4 multiply: `a * b`.
pub fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            for k in 0..4 {
                out[col][row] += a[k][row] * b[col][k];
            }
        }
    }
    out
}

// Inverse of an affine matrix whose bottom row is `[0, 0, 0, 1]`. The upper
// 3x3 is inverted via the adjugate; the translation is mapped through it.
// A near-singular upper 3x3 (degenerate scale) falls back to identity rather
// than producing NaNs.
pub fn mat4_affine_inverse(m: Mat4) -> Mat4 {
    // Upper-left 3x3, addressed as a[col][row].
    let a = m;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2])
        - a[1][0] * (a[0][1] * a[2][2] - a[2][1] * a[0][2])
        + a[2][0] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]);
    if det.abs() < 1e-12 {
        return IDENTITY;
    }
    let inv_det = 1.0 / det;
    // Inverse 3x3 (cofactor transpose * 1/det), again as inv[col][row].
    let mut inv = [[0.0f32; 4]; 4];
    inv[0][0] = (a[1][1] * a[2][2] - a[2][1] * a[1][2]) * inv_det;
    inv[1][0] = -(a[1][0] * a[2][2] - a[2][0] * a[1][2]) * inv_det;
    inv[2][0] = (a[1][0] * a[2][1] - a[2][0] * a[1][1]) * inv_det;
    inv[0][1] = -(a[0][1] * a[2][2] - a[2][1] * a[0][2]) * inv_det;
    inv[1][1] = (a[0][0] * a[2][2] - a[2][0] * a[0][2]) * inv_det;
    inv[2][1] = -(a[0][0] * a[2][1] - a[2][0] * a[0][1]) * inv_det;
    inv[0][2] = (a[0][1] * a[1][2] - a[1][1] * a[0][2]) * inv_det;
    inv[1][2] = -(a[0][0] * a[1][2] - a[1][0] * a[0][2]) * inv_det;
    inv[2][2] = (a[0][0] * a[1][1] - a[1][0] * a[0][1]) * inv_det;
    // Inverse translation: -inv3x3 * t.
    let t = [m[3][0], m[3][1], m[3][2]];
    inv[3][0] = -(inv[0][0] * t[0] + inv[1][0] * t[1] + inv[2][0] * t[2]);
    inv[3][1] = -(inv[0][1] * t[0] + inv[1][1] * t[1] + inv[2][1] * t[2]);
    inv[3][2] = -(inv[0][2] * t[0] + inv[1][2] * t[1] + inv[2][2] * t[2]);
    inv[3][3] = 1.0;
    inv
}

// Column-major 3x3 rotation matrix, `m[col][row]`.
type Mat3 = [[f32; 3]; 3];

// Unit quaternion `(x, y, z, w)` representing a rotation.
type Quat = [f32; 4];

// Column-major 3x3 rotation matrix from YXZ Euler degrees. Identical trig to
// `JointPose::to_matrix`, without the scale or translation.
fn rotation_mat3(rotation_deg: [f32; 3]) -> Mat3 {
    let [pitch, yaw, roll] = rotation_deg;
    let (sp, cp) = (pitch.to_radians().sin(), pitch.to_radians().cos());
    let (syw, cyw) = (yaw.to_radians().sin(), yaw.to_radians().cos());
    let (sr, cr) = (roll.to_radians().sin(), roll.to_radians().cos());
    [
        [cyw * cr + syw * sp * sr, cp * sr, -syw * cr + cyw * sp * sr],
        [-cyw * sr + syw * sp * cr, cp * cr, syw * sr + cyw * sp * cr],
        [syw * cp, -sp, cyw * cp],
    ]
}

// Compose a column-major `T * R * S` affine matrix from a rotation 3x3,
// per-axis scale, and translation: the layout `JointPose::to_matrix`
// produces and `blend_matrix` reuses for the interpolated rotation.
fn compose(r: Mat3, scale: [f32; 3], t: [f32; 3]) -> Mat4 {
    let [sx, sy, sz] = scale;
    [
        [r[0][0] * sx, r[0][1] * sx, r[0][2] * sx, 0.0],
        [r[1][0] * sy, r[1][1] * sy, r[1][2] * sy, 0.0],
        [r[2][0] * sz, r[2][1] * sz, r[2][2] * sz, 0.0],
        [t[0], t[1], t[2], 1.0],
    ]
}

// Quaternion of a column-major rotation 3x3 (Shepperd's method: picks the
// largest-magnitude component to keep the division well-conditioned).
fn quat_from_mat3(m: Mat3) -> Quat {
    let (m00, m11, m22) = (m[0][0], m[1][1], m[2][2]);
    let trace = m00 + m11 + m22;
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (m[1][2] - m[2][1]) / s,
            (m[2][0] - m[0][2]) / s,
            (m[0][1] - m[1][0]) / s,
            0.25 * s,
        ]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [
            0.25 * s,
            (m[1][0] + m[0][1]) / s,
            (m[2][0] + m[0][2]) / s,
            (m[1][2] - m[2][1]) / s,
        ]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [
            (m[1][0] + m[0][1]) / s,
            0.25 * s,
            (m[2][1] + m[1][2]) / s,
            (m[2][0] - m[0][2]) / s,
        ]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [
            (m[2][0] + m[0][2]) / s,
            (m[2][1] + m[1][2]) / s,
            0.25 * s,
            (m[0][1] - m[1][0]) / s,
        ]
    }
}

// Column-major rotation 3x3 of a unit quaternion.
fn quat_to_mat3(q: Quat) -> Mat3 {
    let [x, y, z, w] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn quat_normalize(q: Quat) -> Quat {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len < 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
}

// Spherical linear interpolation between two unit quaternions. Negates `b`
// when the pair points to opposite hemispheres so the interpolation always
// takes the shorter arc, and falls back to a normalised lerp when the two
// rotations are nearly parallel (the slerp denominator approaches zero there
// and nlerp is visually identical at that angle). `f` is clamped to `[0, 1]`.
fn quat_slerp(a: Quat, mut b: Quat, f: f32) -> Quat {
    let f = f.clamp(0.0, 1.0);
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if dot < 0.0 {
        b = [-b[0], -b[1], -b[2], -b[3]];
        dot = -dot;
    }
    if dot > 0.9995 {
        return quat_normalize([
            a[0] + (b[0] - a[0]) * f,
            a[1] + (b[1] - a[1]) * f,
            a[2] + (b[2] - a[2]) * f,
            a[3] + (b[3] - a[3]) * f,
        ]);
    }
    let theta_0 = dot.clamp(-1.0, 1.0).acos();
    let sin_0 = theta_0.sin();
    let s_a = ((1.0 - f) * theta_0).sin() / sin_0;
    let s_b = (f * theta_0).sin() / sin_0;
    [
        a[0] * s_a + b[0] * s_b,
        a[1] * s_a + b[1] * s_b,
        a[2] * s_a + b[2] * s_b,
        a[3] * s_a + b[3] * s_b,
    ]
}

// Decompose a column-major affine matrix into translation, a unit rotation
// quaternion, and per-axis scale: the inverse of `compose` for a
// positive-scale `T * R * S` matrix. Scale is recovered as the length of each
// rotation column; a zero-length column yields a zero scale axis and the
// rotation falls back to identity for that axis. Used by `blend_locals` to
// interpolate two joint matrices in TRS space.
pub fn decompose(m: Mat4) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let t = [m[3][0], m[3][1], m[3][2]];
    let col_len = |c: usize| (m[c][0] * m[c][0] + m[c][1] * m[c][1] + m[c][2] * m[c][2]).sqrt();
    let scale = [col_len(0), col_len(1), col_len(2)];
    let norm = |c: usize| {
        let s = scale[c];
        if s < 1e-12 {
            [0.0, 0.0, 0.0]
        } else {
            [m[c][0] / s, m[c][1] / s, m[c][2] / s]
        }
    };
    let r: Mat3 = [norm(0), norm(1), norm(2)];
    (t, quat_normalize(quat_from_mat3(r)), scale)
}

// YXZ Euler angles in degrees recovered from a unit rotation quaternion: the
// inverse of `rotation_mat3` composed with `quat_to_mat3`. glTF stores node
// rotations as quaternions; the glTF importer (concinnity-cook) converts them to
// the Euler `JointPose` representation this engine's joints use. The
// conversion is matrix-exact for non-degenerate rotations; at gimbal lock
// (pitch ±90°) it folds the rotation onto the yaw axis with zero roll.
pub fn euler_yxz_from_quat(q: [f32; 4]) -> [f32; 3] {
    let m = quat_to_mat3(quat_normalize(q));
    let sp = (-m[2][1]).clamp(-1.0, 1.0);
    // cos(pitch) is taken straight from the matrix (the column-2 length in the
    // XZ plane) rather than `sqrt(1 - sp*sp)`, which loses nearly all its
    // precision to catastrophic cancellation as pitch approaches ±90°.
    let cp = (m[2][0] * m[2][0] + m[2][2] * m[2][2]).sqrt();
    let pitch = sp.atan2(cp);
    let (yaw, roll) = if cp > 1e-4 {
        (m[2][0].atan2(m[2][2]), m[0][1].atan2(m[1][1]))
    } else {
        ((sp * m[1][0]).atan2(m[0][0]), 0.0)
    };
    [pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees()]
}

// A joint's local transform: translation, YXZ Euler rotation in degrees, and
// per-axis scale. Used both for the bind pose and for animation keyframes.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct JointPose {
    pub translation: [f32; 3],
    pub rotation_deg: [f32; 3],
    pub scale: [f32; 3],
}

impl Default for JointPose {
    fn default() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl JointPose {
    // Column-major local matrix `T * R(YXZ) * S`. Same construction as
    // `Prop::model_matrix` so joint and prop transforms compose consistently.
    pub fn to_matrix(&self) -> Mat4 {
        compose(
            rotation_mat3(self.rotation_deg),
            self.scale,
            self.translation,
        )
    }

    // Interpolate two poses into a column-major local matrix. Translation and
    // scale blend linearly; rotation is quaternion-slerped (shortest-arc,
    // constant angular velocity) rather than Euler-lerped, so multi-axis
    // joint rotation follows the correct path. `f` in `[0, 1]`.
    pub fn blend_matrix(&self, other: &JointPose, f: f32) -> Mat4 {
        let mix = |a: [f32; 3], b: [f32; 3]| {
            [
                a[0] + (b[0] - a[0]) * f,
                a[1] + (b[1] - a[1]) * f,
                a[2] + (b[2] - a[2]) * f,
            ]
        };
        let qa = quat_from_mat3(rotation_mat3(self.rotation_deg));
        let qb = quat_from_mat3(rotation_mat3(other.rotation_deg));
        let rotation = quat_to_mat3(quat_slerp(qa, qb, f));
        compose(
            rotation,
            mix(self.scale, other.scale),
            mix(self.translation, other.translation),
        )
    }
}

// One joint in a skeleton: a parent link and a local bind transform.
#[derive(Debug, Clone)]
pub struct Joint {
    // Index of the parent joint, or `None` for a root. Parents must appear
    // before their children so a single forward pass resolves the hierarchy.
    pub parent: Option<usize>,
    // Local bind-pose transform relative to the parent.
    pub bind: JointPose,
}

// A joint hierarchy plus the bind pose. The inverse bind matrices are
// precomputed once on construction.
#[derive(Debug, Clone)]
pub struct Skeleton {
    joints: Vec<Joint>,
    // World-space inverse bind matrix per joint.
    inverse_bind: Vec<Mat4>,
}

impl Skeleton {
    // Build a skeleton, resolving world bind matrices and inverting them.
    // Joints referencing a parent that does not precede them are treated as
    // roots (a forward pass cannot resolve them otherwise).
    pub fn new(joints: Vec<Joint>) -> Self {
        let mut world_bind: Vec<Mat4> = Vec::with_capacity(joints.len());
        for (i, joint) in joints.iter().enumerate() {
            let local = joint.bind.to_matrix();
            let world = match joint.parent {
                Some(p) if p < i => mat4_mul(world_bind[p], local),
                _ => local,
            };
            world_bind.push(world);
        }
        let inverse_bind = world_bind.iter().map(|m| mat4_affine_inverse(*m)).collect();
        Self {
            joints,
            inverse_bind,
        }
    }

    pub fn len(&self) -> usize {
        self.joints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    // Compose `local_poses` (one local matrix per joint) into world-space
    // joint matrices, then multiply by the inverse bind matrices to produce
    // the skinning matrices the vertex shader applies. `local_poses` shorter
    // than the skeleton has its missing tail filled from the bind pose.
    //
    // The result is capped at `MAX_JOINTS` entries (the GPU joint buffer is
    // fixed-size) and is always at least one matrix so the buffer is never
    // empty.
    pub fn skinning_matrices(&self, local_poses: &[Mat4]) -> Vec<Mat4> {
        let n = self.joints.len();
        let mut world: Vec<Mat4> = Vec::with_capacity(n);
        for (i, joint) in self.joints.iter().enumerate() {
            let local = local_poses
                .get(i)
                .copied()
                .unwrap_or_else(|| joint.bind.to_matrix());
            let world_mat = match joint.parent {
                Some(p) if p < i => mat4_mul(world[p], local),
                _ => local,
            };
            world.push(world_mat);
        }
        let mut out: Vec<Mat4> = world
            .iter()
            .zip(&self.inverse_bind)
            .map(|(w, ib)| mat4_mul(*w, *ib))
            .take(MAX_JOINTS)
            .collect();
        if out.is_empty() {
            out.push(IDENTITY);
        }
        out
    }

    // Skinning matrices for the rest (bind) pose: every joint's local
    // transform is its bind transform, so every skinning matrix is identity.
    // Used to seed a `SkeletonPose` before the first animation tick.
    pub fn bind_skinning_matrices(&self) -> Vec<Mat4> {
        let locals: Vec<Mat4> = self.joints.iter().map(|j| j.bind.to_matrix()).collect();
        self.skinning_matrices(&locals)
    }
}

// A single keyframe: a joint pose sampled at a point in time.
#[derive(Debug, Clone, Copy)]
pub struct Keyframe {
    pub time: f32,
    pub pose: JointPose,
}

// An animation channel for one joint: a time-ordered list of keyframes.
#[derive(Debug, Clone)]
pub struct JointTrack {
    pub joint: usize,
    pub keys: Vec<Keyframe>,
}

impl JointTrack {
    // Sample this track at time `t` (seconds), returning the joint's local
    // matrix. Times outside the keyframe range clamp to the nearest end key;
    // between keys translation/scale lerp and rotation slerps.
    fn sample(&self, t: f32) -> Mat4 {
        match self.keys.as_slice() {
            [] => IDENTITY,
            [only] => only.pose.to_matrix(),
            keys => {
                if t <= keys[0].time {
                    return keys[0].pose.to_matrix();
                }
                let last = keys[keys.len() - 1];
                if t >= last.time {
                    return last.pose.to_matrix();
                }
                // Linear scan: joint tracks have a handful of keys.
                for w in keys.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    if t >= a.time && t <= b.time {
                        let span = (b.time - a.time).max(1e-6);
                        let f = (t - a.time) / span;
                        return a.pose.blend_matrix(&b.pose, f);
                    }
                }
                last.pose.to_matrix()
            }
        }
    }
}

// One animation clip: a fixed-length set of per-joint keyframe tracks.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    // Total clip length in seconds.
    pub duration: f32,
    // When true, sampling past `duration` wraps; otherwise it holds the end.
    pub looping: bool,
    pub tracks: Vec<JointTrack>,
}

impl AnimationClip {
    // Sample the clip at time `t` against `skeleton`, returning one local
    // matrix per joint. Joints with no track keep their bind transform.
    pub fn sample(&self, t: f32, skeleton: &Skeleton) -> Vec<Mat4> {
        let local_t = if self.looping && self.duration > 1e-6 {
            t.rem_euclid(self.duration)
        } else {
            t.clamp(0.0, self.duration)
        };
        let mut locals: Vec<Mat4> = skeleton
            .joints()
            .iter()
            .map(|j| j.bind.to_matrix())
            .collect();
        for track in &self.tracks {
            if track.joint < locals.len() {
                locals[track.joint] = track.sample(local_t);
            }
        }
        locals
    }
}

// Blend two arrays of local joint matrices by weight `f`, clamped to `[0, 1]`:
// `f = 0` returns `a`, `f = 1` returns `b`. Each joint is decomposed into TRS,
// translation and scale are linearly interpolated, and rotation is
// quaternion-slerped (shortest-arc): the same interpolation a single clip
// uses between keyframes, so a blended pose is continuous with a clip's own
// sampling. Arrays of unequal length blend the common prefix and copy the
// longer array's tail through unchanged.
pub fn blend_locals(a: &[Mat4], b: &[Mat4], f: f32) -> Vec<Mat4> {
    let f = f.clamp(0.0, 1.0);
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        match (a.get(i), b.get(i)) {
            (Some(&ma), Some(&mb)) => {
                let (ta, qa, sa) = decompose(ma);
                let (tb, qb, sb) = decompose(mb);
                let mix = |x: [f32; 3], y: [f32; 3]| {
                    [
                        x[0] + (y[0] - x[0]) * f,
                        x[1] + (y[1] - x[1]) * f,
                        x[2] + (y[2] - x[2]) * f,
                    ]
                };
                let r = quat_to_mat3(quat_slerp(qa, qb, f));
                out.push(compose(r, mix(sa, sb), mix(ta, tb)));
            }
            (Some(&ma), None) => out.push(ma),
            (None, Some(&mb)) => out.push(mb),
            (None, None) => unreachable!("index below max of both lengths"),
        }
    }
    out
}

// Blend N arrays of local joint matrices into a single normalised weighted
// average. Implemented as an incremental normalised fold of `blend_locals`:
// after folding in array `i` the accumulator equals the weighted blend of
// arrays `0..=i`. Negative weights clamp to 0 and a 0-weight array is skipped;
// when every weight is 0 (or only one array is given) the first array is
// returned unchanged, so a single-clip mesh is unaffected. An empty input
// yields an empty result.
pub fn blend_many(poses: &[Vec<Mat4>], weights: &[f32]) -> Vec<Mat4> {
    let Some(first) = poses.first() else {
        return Vec::new();
    };
    let mut acc = first.clone();
    let mut acc_w = weights.first().copied().unwrap_or(1.0).max(0.0);
    for (i, pose) in poses.iter().enumerate().skip(1) {
        let w = weights.get(i).copied().unwrap_or(1.0).max(0.0);
        if w <= 0.0 {
            continue;
        }
        let total = acc_w + w;
        let f = if total > 1e-6 { w / total } else { 0.0 };
        acc = blend_locals(&acc, pose, f);
        acc_w = total;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    // A two-joint vertical chain: root at origin, child one unit up in y.
    fn chain() -> Skeleton {
        Skeleton::new(vec![
            Joint {
                parent: None,
                bind: JointPose::default(),
            },
            Joint {
                parent: Some(0),
                bind: JointPose {
                    translation: [0.0, 1.0, 0.0],
                    ..JointPose::default()
                },
            },
        ])
    }

    #[test]
    fn bind_pose_skinning_matrices_are_identity() {
        let sk = chain();
        for m in sk.bind_skinning_matrices() {
            for col in 0..4 {
                for row in 0..4 {
                    assert!(approx(m[col][row], IDENTITY[col][row]));
                }
            }
        }
    }

    #[test]
    fn affine_inverse_round_trips() {
        let m = JointPose {
            translation: [3.0, -2.0, 5.0],
            rotation_deg: [0.0, 30.0, 0.0],
            scale: [2.0, 2.0, 2.0],
        }
        .to_matrix();
        let id = mat4_mul(m, mat4_affine_inverse(m));
        for col in 0..4 {
            for row in 0..4 {
                assert!(approx(id[col][row], IDENTITY[col][row]));
            }
        }
    }

    #[test]
    fn rotating_child_joint_moves_a_bound_point() {
        // Rotate the child joint 90 deg yaw. A point at the child's origin in
        // bind space (0,1,0) should be carried by the child's skinning matrix
        // but the joint origin itself is the rotation pivot, so it stays put.
        // A point offset +x from the child should swing to -z.
        let sk = chain();
        let mut locals: Vec<Mat4> = sk.joints().iter().map(|j| j.bind.to_matrix()).collect();
        locals[1] = JointPose {
            translation: [0.0, 1.0, 0.0],
            rotation_deg: [0.0, 90.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
        .to_matrix();
        let skin = sk.skinning_matrices(&locals);
        // Bind-space point one unit +x of the child joint origin: (1, 1, 0).
        let p = [1.0f32, 1.0, 0.0, 1.0];
        let m = skin[1];
        let out = [
            m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0] * p[3],
            m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1] * p[3],
            m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2] * p[3],
        ];
        // +x swings to -z under a +90 deg yaw; y unchanged.
        assert!(approx(out[0], 0.0), "x was {}", out[0]);
        assert!(approx(out[1], 1.0), "y was {}", out[1]);
        assert!(approx(out[2], -1.0), "z was {}", out[2]);
    }

    #[test]
    fn clip_sampling_interpolates_between_keys() {
        let sk = chain();
        let clip = AnimationClip {
            duration: 2.0,
            looping: true,
            tracks: vec![JointTrack {
                joint: 1,
                keys: vec![
                    Keyframe {
                        time: 0.0,
                        pose: JointPose {
                            translation: [0.0, 1.0, 0.0],
                            ..JointPose::default()
                        },
                    },
                    Keyframe {
                        time: 2.0,
                        pose: JointPose {
                            translation: [0.0, 1.0, 0.0],
                            rotation_deg: [0.0, 90.0, 0.0],
                            ..JointPose::default()
                        },
                    },
                ],
            }],
        };
        // Halfway through: yaw should be 45 deg.
        let locals = clip.sample(1.0, &sk);
        // Recover yaw: for a pure yaw the first column is (cos, 0, -sin).
        let yaw = (-locals[1][0][2]).atan2(locals[1][0][0]).to_degrees();
        assert!(approx(yaw, 45.0), "yaw was {}", yaw);
    }

    #[test]
    fn looping_clip_wraps_past_duration() {
        let sk = chain();
        let clip = AnimationClip {
            duration: 2.0,
            looping: true,
            tracks: vec![JointTrack {
                joint: 1,
                keys: vec![Keyframe {
                    time: 0.5,
                    pose: JointPose {
                        translation: [9.0, 1.0, 0.0],
                        ..JointPose::default()
                    },
                }],
            }],
        };
        // t = 2.5 wraps to 0.5: identical sample.
        let a = clip.sample(0.5, &sk);
        let b = clip.sample(2.5, &sk);
        assert_eq!(a[1], b[1]);
    }

    #[test]
    fn unparented_joint_is_treated_as_root() {
        // A joint whose parent index does not precede it must not panic and
        // must behave as a root.
        let sk = Skeleton::new(vec![Joint {
            parent: Some(5),
            bind: JointPose::default(),
        }]);
        assert_eq!(sk.len(), 1);
        assert_eq!(sk.bind_skinning_matrices().len(), 1);
    }

    #[test]
    fn quat_mat3_round_trips() {
        // A rotation 3x3 -> quaternion -> rotation 3x3 must reproduce itself,
        // across the diagonal-dominant and trace-positive branches of
        // Shepperd's method. This is what makes blend_matrix's endpoints exact.
        for e in [
            [0.0, 0.0, 0.0],
            [30.0, 50.0, 20.0],
            [-80.0, 140.0, -25.0],
            [90.0, 0.0, 0.0],
            [0.0, 180.0, 0.0],
        ] {
            let r = rotation_mat3(e);
            let r2 = quat_to_mat3(quat_from_mat3(r));
            for c in 0..3 {
                for row in 0..3 {
                    assert!(
                        approx(r[c][row], r2[c][row]),
                        "e={:?} [{}][{}]: {} vs {}",
                        e,
                        c,
                        row,
                        r[c][row],
                        r2[c][row]
                    );
                }
            }
        }
    }

    #[test]
    fn blend_matrix_endpoints_match_keyframe_poses() {
        // At f=0 / f=1 the interpolated matrix must equal the keyframe pose's
        // own matrix, so a clip is continuous across keyframe boundaries.
        let a = JointPose {
            translation: [1.0, 2.0, 3.0],
            rotation_deg: [10.0, 20.0, 30.0],
            scale: [1.0, 1.5, 2.0],
        };
        let b = JointPose {
            translation: [-4.0, 0.0, 5.0],
            rotation_deg: [70.0, -40.0, 15.0],
            scale: [2.0, 1.0, 0.5],
        };
        let at0 = a.blend_matrix(&b, 0.0);
        let at1 = a.blend_matrix(&b, 1.0);
        let ma = a.to_matrix();
        let mb = b.to_matrix();
        for c in 0..4 {
            for row in 0..4 {
                assert!(approx(at0[c][row], ma[c][row]), "f=0 [{}][{}]", c, row);
                assert!(approx(at1[c][row], mb[c][row]), "f=1 [{}][{}]", c, row);
            }
        }
    }

    #[test]
    fn slerp_midpoint_splits_the_arc_equally() {
        // The defining property of slerp: the f=0.5 quaternion is equidistant
        // (equal rotation angle) from both endpoints. A component-wise Euler
        // lerp does not satisfy this for a multi-axis rotation difference.
        let qa = quat_from_mat3(rotation_mat3([10.0, 20.0, 30.0]));
        let qb = quat_from_mat3(rotation_mat3([70.0, -40.0, 80.0]));
        let qm = quat_slerp(qa, qb, 0.5);
        let angle = |x: Quat, y: Quat| {
            let d = (x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3])
                .abs()
                .min(1.0);
            2.0 * d.acos()
        };
        assert!(
            approx(angle(qa, qm), angle(qm, qb)),
            "arcs {} vs {}",
            angle(qa, qm),
            angle(qm, qb)
        );
    }

    #[test]
    fn decompose_round_trips_a_composed_matrix() {
        // decompose must invert compose for a positive-scale TRS matrix, so
        // blend_locals interpolates the same transform a clip sampled.
        let pose = JointPose {
            translation: [3.0, -2.0, 5.0],
            rotation_deg: [25.0, -60.0, 40.0],
            scale: [1.5, 0.5, 2.0],
        };
        let m = pose.to_matrix();
        let (t, q, s) = decompose(m);
        let rebuilt = compose(quat_to_mat3(q), s, t);
        for c in 0..4 {
            for row in 0..4 {
                assert!(
                    approx(rebuilt[c][row], m[c][row]),
                    "[{}][{}]: {} vs {}",
                    c,
                    row,
                    rebuilt[c][row],
                    m[c][row]
                );
            }
        }
    }

    #[test]
    fn blend_locals_endpoints_are_exact() {
        // f=0 must equal a, f=1 must equal b: a cross-fade is continuous with
        // each source clip at its extremes.
        let a = vec![
            JointPose {
                translation: [1.0, 0.0, 0.0],
                rotation_deg: [10.0, 20.0, 30.0],
                scale: [1.0, 2.0, 1.0],
            }
            .to_matrix(),
        ];
        let b = vec![
            JointPose {
                translation: [0.0, 4.0, -1.0],
                rotation_deg: [-50.0, 70.0, 5.0],
                scale: [2.0, 1.0, 0.5],
            }
            .to_matrix(),
        ];
        let at0 = blend_locals(&a, &b, 0.0);
        let at1 = blend_locals(&a, &b, 1.0);
        for c in 0..4 {
            for row in 0..4 {
                assert!(approx(at0[0][c][row], a[0][c][row]), "f=0 [{}][{}]", c, row);
                assert!(approx(at1[0][c][row], b[0][c][row]), "f=1 [{}][{}]", c, row);
            }
        }
    }

    #[test]
    fn blend_locals_midpoint_slerps_rotation() {
        // The f=0.5 blend of two pure yaws is the yaw midpoint: rotation is
        // slerped, not matrix-lerped (which would shrink the rotation).
        let a = vec![
            JointPose {
                rotation_deg: [0.0, 0.0, 0.0],
                ..JointPose::default()
            }
            .to_matrix(),
        ];
        let b = vec![
            JointPose {
                rotation_deg: [0.0, 90.0, 0.0],
                ..JointPose::default()
            }
            .to_matrix(),
        ];
        let mid = blend_locals(&a, &b, 0.5);
        // For a pure yaw the first column is (cos, 0, -sin).
        let yaw = (-mid[0][0][2]).atan2(mid[0][0][0]).to_degrees();
        assert!(approx(yaw, 45.0), "yaw was {}", yaw);
    }

    #[test]
    fn blend_locals_unequal_lengths_keep_the_longer_tail() {
        let a = vec![IDENTITY];
        let b = vec![IDENTITY, IDENTITY, IDENTITY];
        let out = blend_locals(&a, &b, 0.5);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn blend_many_normalises_weights() {
        // Three pure-yaw poses at 0/30/90 deg blended 1:1:2 must land on the
        // weighted-average yaw 52.5 deg. Equal scaling of every weight (the
        // normalisation) must not move the result.
        let yaw = |deg: f32| {
            vec![
                JointPose {
                    rotation_deg: [0.0, deg, 0.0],
                    ..JointPose::default()
                }
                .to_matrix(),
            ]
        };
        let poses = vec![yaw(0.0), yaw(30.0), yaw(90.0)];
        let recover = |out: &[Mat4]| (-out[0][0][2]).atan2(out[0][0][0]).to_degrees();
        let a = blend_many(&poses, &[1.0, 1.0, 2.0]);
        let b = blend_many(&poses, &[5.0, 5.0, 10.0]);
        assert!(approx(recover(&a), 52.5), "yaw was {}", recover(&a));
        assert!(
            approx(recover(&a), recover(&b)),
            "weight scaling moved blend"
        );
    }

    #[test]
    fn blend_many_skips_zero_weight_and_falls_back_to_first() {
        let poses = vec![vec![IDENTITY], vec![IDENTITY]];
        // A second clip at weight 0 leaves the first untouched.
        let zeroed = blend_many(&poses, &[1.0, 0.0]);
        assert_eq!(zeroed, poses[0]);
        // All-zero weights also fall back to the first array.
        let all_zero = blend_many(&poses, &[0.0, 0.0]);
        assert_eq!(all_zero, poses[0]);
    }

    #[test]
    fn blend_many_with_zero_first_weight_picks_up_later_clip() {
        // A 0-weight first clip must not poison the fold: the second clip
        // (weight 1) should win outright.
        let a = vec![
            JointPose {
                translation: [9.0, 9.0, 9.0],
                ..JointPose::default()
            }
            .to_matrix(),
        ];
        let b = vec![
            JointPose {
                translation: [1.0, 2.0, 3.0],
                ..JointPose::default()
            }
            .to_matrix(),
        ];
        let out = blend_many(&[a, b.clone()], &[0.0, 1.0]);
        assert_eq!(out, b);
    }

    #[test]
    fn euler_from_quat_round_trips_through_the_rotation_matrix() {
        // quat -> YXZ Euler must reproduce the original rotation matrix, so
        // the glTF importer's quaternion node rotations land losslessly in the
        // Euler JointPose representation. Checked across multi-axis rotations.
        for e in [
            [0.0, 0.0, 0.0],
            [25.0, -60.0, 40.0],
            [-80.0, 140.0, -25.0],
            [10.0, 200.0, -170.0],
        ] {
            let r = rotation_mat3(e);
            let q = quat_from_mat3(r);
            let e2 = euler_yxz_from_quat(q);
            let r2 = rotation_mat3(e2);
            for c in 0..3 {
                for row in 0..3 {
                    assert!(
                        approx(r[c][row], r2[c][row]),
                        "e={:?} [{}][{}]: {} vs {}",
                        e,
                        c,
                        row,
                        r[c][row],
                        r2[c][row]
                    );
                }
            }
        }
    }

    #[test]
    fn euler_from_quat_handles_gimbal_lock() {
        // At pitch ±90° the conversion must stay finite and reproduce the
        // rotation matrix (with roll folded onto yaw).
        for e in [[90.0, 35.0, 0.0], [-90.0, -110.0, 0.0]] {
            let r = rotation_mat3(e);
            let e2 = euler_yxz_from_quat(quat_from_mat3(r));
            assert!(e2.iter().all(|v| v.is_finite()), "non-finite for {:?}", e);
            let r2 = rotation_mat3(e2);
            for c in 0..3 {
                for row in 0..3 {
                    assert!(approx(r[c][row], r2[c][row]), "e={:?} [{}][{}]", e, c, row);
                }
            }
        }
    }

    #[test]
    fn blend_matrix_lerps_translation_and_scale() {
        // Translation and scale stay linearly interpolated: only rotation
        // moved to the quaternion path.
        let a = JointPose {
            translation: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        };
        let b = JointPose {
            translation: [4.0, 8.0, -2.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [3.0, 3.0, 3.0],
        };
        let m = a.blend_matrix(&b, 0.25);
        assert!(approx(m[3][0], 1.0));
        assert!(approx(m[3][1], 2.0));
        assert!(approx(m[3][2], -0.5));
        // No rotation: the diagonal carries the lerped scale 1 + 0.25*2 = 1.5.
        assert!(approx(m[0][0], 1.5));
        assert!(approx(m[1][1], 1.5));
        assert!(approx(m[2][2], 1.5));
    }
}
