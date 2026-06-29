<!-- Auto-generated - do not edit. -->

# Animation

A skeletal animation clip that animates one [SkinnedMesh](SkinnedMesh.md).

The clip plays every frame, sampling each track and deforming the target
mesh's skeleton. Joints with no track hold their bind pose.

Several `Animation` assets may target the same [SkinnedMesh](SkinnedMesh.md);
they are then blended into one pose, weighted by each clip's `weight` (a
normalised weighted average). A single clip plays at full strength
regardless of its `weight`.

**glTF import.** A clip may be authored entirely by hand (`tracks` filled
out, `source` left empty) or imported from the same `.glb` that backs the
target [SkinnedMesh](SkinnedMesh.md). Set `source` to the `.glb` path and the
build imports `duration` + `tracks` from it. `animation_index` picks one
clip when the file contains several (default 0); `animation_name` names it
for matching against the file's clip names: when set it takes precedence
over the index. Channels whose target node is not a joint of the file's
first skinned node are dropped. The same `.glb` should back the target
[SkinnedMesh](SkinnedMesh.md) so the joint indices agree.

```jsonl
// Inline:
{"name":"flag_wave","type":"Animation","args":{"target":"flag","duration":2.0,"tracks":[{"joint":1,"keyframes":[{"time":0.0,"rotation_deg":[0,0,0]},{"time":1.0,"rotation_deg":[0,30,0]},{"time":2.0,"rotation_deg":[0,0,0]}]}]}}
// From glTF:
{"name":"hero_walk","type":"Animation","args":{"target":"hero","source":"models/hero.glb","animation_name":"Walk","looping":true}}
```

## Parameters

- `target`: A string. The [SkinnedMesh](SkinnedMesh.md) asset this clip animates. Optional.
- `source`: A string. Optional path to a `.glb` file. When set, the build imports `duration` + `tracks` from it; inline-authored clips leave this empty.
- `animation_index`: An integer. Index of the animation to import when `source` is set and the file contains several. Ignored when `animation_name` is non-empty. Defaults to `0`.
- `animation_name`: A string. Name of the animation to import. When set, the matching glTF animation is looked up by name; takes precedence over `animation_index`.
- `duration`: A float. Clip length in seconds. Overridden by glTF import. Defaults to `1.0`.
- `looping`: A boolean. When true, playback wraps after `duration`. Defaults to `true`.
- `weight`: A float. Blend weight used when several clips target the same [SkinnedMesh](SkinnedMesh.md). Ignored when this is the only clip on its target. Defaults to `1.0`.
- `fade_in_secs`: A float. When non-zero, the clip's contribution ramps from 0 to its declared `weight` over this many seconds after the world starts. Zero (the default) plays the clip at full `weight` from the first frame.
- `tracks`: An array of [AnimationTrack](AnimationTrack.md) objects. Per-joint keyframe channels.
