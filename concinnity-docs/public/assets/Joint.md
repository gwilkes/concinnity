<!-- Auto-generated - do not edit. -->

# Joint

A physics constraint connecting two [Prop](Prop.md)s that own a `collider`.

The joint pins `anchor_a` on `body_a` to `anchor_b` on `body_b` and locks
the relative motion of the two bodies according to its `kind`. Anchors are
in each body's local frame: `[0, 0, 0]` is the body's own pivot.

To anchor a body to "the world" (no second prop), leave `body_b` empty: a
hidden static anchor is created at `anchor_b` (interpreted as world space in
that case) and the body joints to it. This is the pendulum / lamp / trapeze
pattern.

`axis` only applies to `revolute` and `prismatic`: it is the single free
axis (rotation or translation) in each body's local frame. The vector is
normalised on load; a zero axis falls back to `[0, 1, 0]`.

`limits_enabled` clamps the free axis: angle in degrees for revolute,
distance in world units for prismatic. `motor_target_velocity` and
`motor_max_force` drive the free axis when `motor_max_force > 0`; the
velocity is in degrees/sec for revolute, units/sec for prismatic.

```jsonl
// Pendulum: a dynamic ball hanging 2 m below a world anchor, hinged on +Z.
{"name":"pendulum_joint","type":"Joint","args":{
  "kind":"revolute","body_a":"pendulum_bob",
  "anchor_a":[0,2,0],"anchor_b":[0,5,0],"axis":[0,0,1]
}}

// Door: hinged on a wall, swing limited to ±90°.
{"name":"door_hinge","type":"Joint","args":{
  "kind":"revolute","body_a":"wall","body_b":"door",
  "anchor_a":[1,1,0],"anchor_b":[-0.5,0,0],"axis":[0,1,0],
  "limits_enabled":true,"limits":[-90,90]
}}
```

## Parameters

- `kind`: A string. Constraint shape; defaults to "fixed".
- `body_a`: A string. First body: a [Prop](Prop.md) name. Required. Optional.
- `body_b`: A string. Second body: a [Prop](Prop.md) name. Empty means "world anchor", in which case `anchor_b` is interpreted as a world-space position. Optional.
- `anchor_a`: An array of 3 floats. Attach point in `body_a`'s local frame. Defaults to `[0.0, 0.0, 0.0]`.
- `anchor_b`: An array of 3 floats. Attach point in `body_b`'s local frame (or world space if `body_b` is empty). Defaults to `[0.0, 0.0, 0.0]`.
- `axis`: An array of 3 floats. Free axis for revolute/prismatic, in each body's local frame. Defaults to `[0.0, 1.0, 0.0]`.
- `limits_enabled`: A boolean. Whether the `limits` clamp is enforced. Defaults to `false`.
- `limits`: An array of 2 floats. `[min, max]` clamp on the free axis: degrees for revolute, world units for prismatic. Ignored unless `limits_enabled` is true. Defaults to `[0.0, 0.0]`.
- `motor_target_velocity`: A float. Motor target velocity: degrees/sec for revolute, world units/sec for prismatic. Ignored unless `motor_max_force > 0`. Defaults to `0.0`.
- `motor_max_force`: A float. Motor force budget. The motor is inactive when this is 0. Defaults to `0.0`.
