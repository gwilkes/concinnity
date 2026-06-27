// src/assets/transform.rs

use crate::ecs::{AssetOrigin, Component};

/// World-space placement of an entity: translation, rotation, and scale.
///
/// Runtime-only placement state. Physics and interaction systems mutate it and
/// the renderer reads it to position draws. Not authored directly in a world
/// file; it carries the same transform fields a `Prop` declares.
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    /// World-space position [x, y, z].
    pub position: [f32; 3],
    /// Euler rotation in degrees [pitch, yaw, roll], applied in YXZ order.
    pub rotation_deg: [f32; 3],
    /// Non-uniform scale [x, y, z].
    pub scale: [f32; 3],
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl Transform {
    /// Build a column-major model matrix from this transform.
    /// Order: scale, then YXZ Euler rotation, then translation.
    pub fn model_matrix(&self) -> [[f32; 4]; 4] {
        let [px, py, pz] = self.position;
        let [pitch_deg, yaw_deg, roll_deg] = self.rotation_deg;
        let [sx, sy, sz] = self.scale;

        let (pr, yr, rr) = (
            pitch_deg.to_radians(),
            yaw_deg.to_radians(),
            roll_deg.to_radians(),
        );
        let (sp, cp) = (pr.sin(), pr.cos());
        let (sy_, cy) = (yr.sin(), yr.cos());
        let (sr, cr) = (rr.sin(), rr.cos());

        // YXZ rotation: R = Ry * Rx * Rz
        // Combined and scaled, column-major storage: out[col][row].
        [
            [
                sx * (cy * cr + sy_ * sp * sr),
                sx * (cp * sr),
                sx * (-sy_ * cr + cy * sp * sr),
                0.0,
            ],
            [
                sy * (-cy * sr + sy_ * sp * cr),
                sy * (cp * cr),
                sy * (sy_ * sr + cy * sp * cr),
                0.0,
            ],
            [sz * (sy_ * cp), sz * (-sp), sz * (cy * cp), 0.0],
            [px, py, pz, 1.0],
        ]
    }
}

/// `Transform` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformArgs {}

impl Component for Transform {
    const NAME: &'static str = "Transform";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = TransformArgs;

    fn to_args(&self) -> TransformArgs {
        TransformArgs {}
    }
    fn from_args(_: TransformArgs) -> Self {
        Self::default()
    }
}
