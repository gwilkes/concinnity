<!-- Auto-generated - do not edit. -->

# Keyframe

One keyframe in an animation track: a joint pose sampled at `time` seconds.
The pose fields (`translation`, `rotation_deg`, `scale`) are given directly
on the keyframe, each defaulting to the identity transform when omitted.

## Parameters

- `time`: A float. Time of this keyframe in seconds from the clip start.
- `pose`: An object. The joint's transform at this keyframe.
