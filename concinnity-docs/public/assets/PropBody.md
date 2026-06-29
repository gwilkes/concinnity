<!-- Auto-generated - do not edit. -->

# PropBody

Makes a companion [Prop](Prop.md) a dynamic physics body.

Attach a PropBody to give a [Prop](Prop.md) real physics: it falls, collides,
stacks, tumbles, and (with `pickup: true` on the prop) can be carried and
thrown. A Prop with a `collider` but no PropBody is a static, immovable
obstacle.

```json
{
  "name": "crate_a_body",
  "type": "PropBody",
  "args": { "prop_name": "crate_a", "mass": 4.0, "friction": 0.6 }
}
```

## Parameters

- `prop_name`: A string. The [Prop](Prop.md) this body drives. Must match a Prop declared in the same world. Optional.
- `mass`: A float. Mass in kilograms. 0 lets the simulation derive mass from the collider shape and a default density.
- `friction`: A float. Friction coefficient used for contacts with this body. Defaults to `0.5`.
- `restitution`: A float. Bounciness in [0, 1]. 0 is fully inelastic. Defaults to `0.0`.
- `gravity_scale`: A float. Multiplier applied to world gravity for this body. 1.0 is normal. Defaults to `1.0`.
- `linear_damping`: A float. Linear velocity damping, modelling air drag. Defaults to `0.05`.
