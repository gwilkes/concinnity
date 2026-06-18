// src/gfx/camera.rs
//
// Pure math for a right-handed first-person camera. All functions are
// stateless -- callers own position, yaw, and pitch directly (e.g. on
// Camera3D) and pass them in as needed.

// Build a column-major view matrix from position, yaw, and pitch.
//
// Convention matches the GLSL mat4 layout used by the shader UBOs:
// view[column][row], right-handed, Y-up.
pub fn view_matrix(position: [f32; 3], yaw: f32, pitch: f32) -> [[f32; 4]; 4] {
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();

    let fwd = [-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch];
    let right = normalize(cross(fwd, [0.0, 1.0, 0.0]));
    let up = cross(right, fwd);

    let [rx, ry, rz] = right;
    let [ux, uy, uz] = up;
    let [fx, fy, fz] = fwd;
    let [px, py, pz] = position;

    [
        [rx, ux, -fx, 0.0],
        [ry, uy, -fy, 0.0],
        [rz, uz, -fz, 0.0],
        [
            -(rx * px + ry * py + rz * pz),
            -(ux * px + uy * py + uz * pz),
            fx * px + fy * py + fz * pz,
            1.0,
        ],
    ]
}

// First-person input math, Vulkan/GLFW only. Currently unreferenced (the
// Vulkan input path drives the camera through GraphicsSystem directly) but
// kept as the reference implementation for a future GLFW-side controller.
#[cfg(backend_vk)]
#[allow(dead_code)]
pub fn look(yaw: f32, pitch: f32, dyaw: f32, dpitch: f32) -> (f32, f32) {
    let new_yaw = yaw + dyaw;
    let new_pitch = (pitch + dpitch).clamp(
        -std::f32::consts::FRAC_PI_2 + 0.01,
        std::f32::consts::FRAC_PI_2 - 0.01,
    );
    (new_yaw, new_pitch)
}

// Horizontal forward vector (ignores pitch so WASD movement stays on the ground plane).
#[cfg(backend_vk)]
#[allow(dead_code)]
pub fn forward_xz(yaw: f32) -> [f32; 3] {
    [-yaw.sin(), 0.0, -yaw.cos()]
}

// Horizontal right vector.
#[cfg(backend_vk)]
#[allow(dead_code)]
pub fn right_xz(yaw: f32) -> [f32; 3] {
    [yaw.cos(), 0.0, -yaw.sin()]
}

// Move `position` by `delta`, clamping to the AABB defined by `bounds_min`/`bounds_max`.
// `radius` is the player half-width applied on X and Z.
#[cfg(backend_vk)]
#[allow(dead_code)]
pub fn move_and_collide(
    position: [f32; 3],
    delta: [f32; 3],
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    radius: f32,
) -> [f32; 3] {
    [
        (position[0] + delta[0]).clamp(bounds_min[0] + radius, bounds_max[0] - radius),
        (position[1] + delta[1]).clamp(bounds_min[1], bounds_max[1]),
        (position[2] + delta[2]).clamp(bounds_min[2] + radius, bounds_max[2] - radius),
    ]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-7 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}
