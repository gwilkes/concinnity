<!-- Auto-generated - do not edit. -->

# PropCollider

Collision volume attached to a [Prop](Prop.md).

The shape dimensions are in the prop's local space and are scaled by the
prop's `scale`. `ball` and `capsule` use the X scale component (they assume
uniform scaling).

## Parameters

- `shape`: A string. Collision shape: "aabb" (alias "cuboid"), "ball", or "capsule". Defaults to `"cuboid"`.
- `half_extents`: An array of 3 floats. Box half-extents in local space [x, y, z]. Used by cuboid shapes. Defaults to `[0.5, 0.5, 0.5]`.
- `radius`: A float. Radius in local space. Used by ball and capsule shapes. Defaults to `0.5`.
- `half_height`: A float. Half the cylinder height in local space. Used by capsule shapes. Defaults to `0.5`.
