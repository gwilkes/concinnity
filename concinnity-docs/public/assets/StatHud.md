<!-- Auto-generated - do not edit. -->

# StatHud

Requests the default on-screen stats HUD. Drives a set of
[TextLabel](TextLabel.md) chips with live engine stats, refreshed on a fixed
interval.

Each label field, when set, receives one chip: `fps_label` the averaged
frame rate, `vram_label` the GPU-memory use, `ev_label` the auto-exposure
value, and `edr_label` the HDR headroom multiplier. Chips whose stat is
unavailable stay blank. The frame-rate and GPU-memory chips are shown or
hidden from the in-game video settings ("Display performance stats"); the
exposure and HDR chips show whenever their feature is active.

The chips are packed into a tight strip anchored at the top-left of the
window, left to right in the order fps, vram, ev, edr; a blank chip reserves
no width, so hidden readouts leave no gap. Their on-screen position is fixed
by the engine rather than the authored coordinates.

Developer-facing readouts (per-pass GPU timings, cursor position, live
camera pose) live on the separate [DebugHud](DebugHud.md), toggled with F1.

A world that declares a [MainMenu](MainMenu.md) receives a `StatHud`, its
chip labels, and their font at build time when it declares none (the
menu's performance-stats toggles drive the chips), so the example below is
only needed to restyle the chips or run a HUD without a menu. Declare an
[EngineDefaults](EngineDefaults.md) with `"hud": false` to remove the
injection entirely.

```jsonl
{"type":"Font","name":"hud_font","args":{"size_px":20}}
{"type":"TextLabel","name":"fps_chip","args":{"font":"hud_font","x":10,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.18,0.32,0.85],"padding":5}}
{"type":"TextLabel","name":"vram_chip","args":{"font":"hud_font","x":92,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.18,0.32,0.85],"padding":5}}
{"type":"TextLabel","name":"ev_chip","args":{"font":"hud_font","x":192,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.18,0.32,0.85],"padding":5}}
{"type":"TextLabel","name":"edr_chip","args":{"font":"hud_font","x":272,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.18,0.32,0.85],"padding":5}}
{"type":"StatHud","name":"hud","args":{"fps_label":"fps_chip","vram_label":"vram_chip","ev_label":"ev_chip","edr_label":"edr_chip"}}
```

## Parameters

- `fps_label`: A string. [TextLabel](TextLabel.md) that receives the frame-rate chip text. Optional.
- `vram_label`: A string. [TextLabel](TextLabel.md) that receives the GPU-memory chip text. Optional.
- `ev_label`: A string. [TextLabel](TextLabel.md) that receives the auto-exposure chip text. Optional.
- `edr_label`: A string. [TextLabel](TextLabel.md) that receives the HDR-headroom chip text. Optional.
