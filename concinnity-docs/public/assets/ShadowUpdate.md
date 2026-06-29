<!-- Auto-generated - do not edit. -->

# ShadowUpdate

How often each cascaded-shadow-map slice is re-rendered. The shadow pass
re-rasterizes all scene geometry into every cascade, so it is one of the
heavier passes; updating distant cascades less often cuts that cost.

`hybrid` (the default) re-renders the nearest cascade every frame (so close
shadows stay crisp) and rotates through the farther cascades one per frame.
Distant shadows then lag a few frames while the camera moves, which is
imperceptible at that range. `every_frame` re-renders all cascades every
frame: pick it for scenes with fast-moving shadow casters where even distant
shadow lag is unacceptable. Each cascade is always primed (rendered once)
before it is sampled, so there is never missing shadow data.

## Values

- `every_frame`
- `hybrid`
