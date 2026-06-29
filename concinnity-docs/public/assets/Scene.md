<!-- Auto-generated - do not edit. -->

# Scene

A named group marker with timing and transition settings for
[SceneReel](SceneReel.md).

[Prop](Prop.md)s belong to a Scene by naming convention: props whose `name`
begins with `<scene_name>_` are associated with that Scene.

```jsonl
{"name":"day",  "type":"Scene","args":{"duration_secs":5.0,"transition":"FadeBlack"}}
{"name":"night","type":"Scene","args":{"duration_secs":5.0,"transition":"FadeBlack"}}
// Props named "day_*" belong to Scene "day"; "night_*" to Scene "night"
```

## Parameters

- `duration_secs`: A float. Seconds to hold this scene before advancing. None = hold indefinitely. Optional.
- `transition`: A string. Transition into this scene: "Cut" (immediate) or "FadeBlack" (dip to black). Defaults to `"Cut"`.
- `camera_shot`: A string. A [CameraShot](CameraShot.md) or [Camera3D](Camera3D.md) to activate when this scene becomes active. `None` keeps the current camera unchanged. Optional.
