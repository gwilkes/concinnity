<!-- Auto-generated - do not edit. -->

# RigidBody

Gives a player [Camera3D](Camera3D.md) gravity, jumping, and a grounded
character body.

Every [Camera3D](Camera3D.md) already collides with the world as a capsule.
Adding a RigidBody upgrades that camera from a free-flying spectator to a
grounded character: it falls under gravity, lands on surfaces, climbs steps,
slides off steep slopes, and can jump. The capsule size is configured here
too.

```json
{ "name": "player_body", "type": "RigidBody", "args": { "jump_height": 1.4 } }
```

## Parameters

- `gravity_scale`: A float. Multiplier applied to the global gravity constant. 1.0 = normal gravity. Defaults to `1.0`.
- `capsule_radius`: A float. Radius of the player capsule used for collision, in world units. Defaults to `0.3`.
- `capsule_height`: A float. Total height of the player capsule. The camera eye sits at the top. Defaults to `1.7`.
- `jump_height`: A float. Apex height of a jump in world units. 0 disables jumping. Defaults to `1.1`.
- `max_slope_deg`: A float. Steepest slope the player can walk up, in degrees. Defaults to `50.0`.
- `step_height`: A float. Tallest obstacle the controller auto-steps over, in world units. Defaults to `0.3`.
