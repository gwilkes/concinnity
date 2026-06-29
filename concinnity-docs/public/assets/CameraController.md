<!-- Auto-generated - do not edit. -->

# CameraController

First-person / fly-through controller settings carried on a `Camera3D`.

A `Camera3D` whose `controller` is set (the default) is driven each frame by
the internal camera controller, which turns mouse/keyboard input into a
camera orientation and a movement intent. Set `controller` to `null` for a
camera driven by something else (a `CameraShot` / `SceneReel` cutscene).

## Parameters

- `free_fly`: A boolean. Direct 6-DoF flight mode. WASD moves along the camera's full forward vector (yaw + pitch) and jump rises along world +Y; the controller writes the new position straight onto Camera3D, bypassing the physics step and the bounds box. Used for inspector / fly-through cameras (the default, e.g. the `cn add foo.glb` scaffold). Set `false` for the FPS-style ground walker.
- `move_speed`: A float. Walk / fly speed in world units per second. Defaults to `1.0`.
- `sprint_multiplier`: A float. Sprint multiplier applied when the sprint key is held. Defaults to `3.0`.
- `mouse_sensitivity`: A float. Mouse look sensitivity in radians per pixel. Defaults to `0.0015`.
- `player_radius`: A float. Margin kept between the camera and the bounds box (world units). Defaults to `0.3`.
- `bounds_min`: An array of 3 floats. AABB minimum corner the camera centre must stay inside [x, y, z]. Defaults to `[—, —, —]`.
- `bounds_max`: An array of 3 floats. AABB maximum corner the camera centre must stay inside [x, y, z]. Defaults to `[—, —, —]`.
