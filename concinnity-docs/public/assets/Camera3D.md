<!-- Auto-generated - do not edit. -->

# Camera3D

Declares the 3D camera. One per scene.

```jsonl
{
  "name": "main_camera",
  "type": "Camera3D",
  "args": {
    "fov_y_degrees": 80.0,
    "near": 0.05,
    "far": 500.0,
    "position": [0.0, 4.0, 0.0]
  }
}
```

## Parameters

- `fov_y_degrees`: A float. Vertical field-of-view in degrees. Defaults to `75.0`.
- `near`: A float. Near clip plane distance. Defaults to `0.05`.
- `far`: A float. Far clip plane distance. Defaults to `200.0`.
- `position`: An array of 3 floats. Initial eye position in world space [x, y, z]. Defaults to `[0.0, 1.7, 0.0]`.
- `yaw`: A float. Initial yaw in radians (0 = looking toward -Z). Defaults to `0.0`.
- `pitch`: A float. Initial pitch in radians. Defaults to `0.0`.
- `controller`: A [CameraController](CameraController.md) object. Input controller settings, or `null` to leave the camera uncontrolled (driven by a [CameraShot](CameraShot.md) / [SceneReel](SceneReel.md) cutscene). Omitted defaults to a free-fly inspector controller.
