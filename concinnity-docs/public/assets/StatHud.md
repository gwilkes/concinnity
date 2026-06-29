<!-- Auto-generated - do not edit. -->

# StatHud

Requests an on-screen performance HUD. Drives a set of
[TextLabel](TextLabel.md) chips with live engine stats, refreshed on a fixed
interval and toggled with F1.

Each label field, when set, receives one chip: `fps_label` the averaged
frame rate, `vram_label` the GPU-memory use, `ev_label` the auto-exposure
value, `edr_label` the HDR headroom multiplier, `passes_label` a multi-line
list of the heaviest rendering steps of the last frame, `mouse_label` the
cursor position in window pixels, and `camera_label` the live camera pose
(position, yaw, pitch) in the exact form a fixed viewpoint is reproduced
with. Chips whose stat is unavailable stay blank.

```jsonl
{"type":"Font","name":"hud_font","args":{"size_px":20}}
{"type":"TextLabel","name":"fps_chip","args":{"font":"hud_font","x":10,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
{"type":"TextLabel","name":"vram_chip","args":{"font":"hud_font","x":92,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
{"type":"TextLabel","name":"ev_chip","args":{"font":"hud_font","x":192,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
{"type":"TextLabel","name":"edr_chip","args":{"font":"hud_font","x":272,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
{"type":"TextLabel","name":"passes_chip","args":{"font":"hud_font","x":10,"y":36,"scale":0.6,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
{"type":"TextLabel","name":"mouse_chip","args":{"font":"hud_font","x":352,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
{"type":"TextLabel","name":"camera_chip","args":{"font":"hud_font","x":352,"y":36,"scale":0.6,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
{"type":"StatHud","name":"hud","args":{"fps_label":"fps_chip","vram_label":"vram_chip","ev_label":"ev_chip","edr_label":"edr_chip","passes_label":"passes_chip","mouse_label":"mouse_chip","camera_label":"camera_chip"}}
```

## Parameters

- `fps_label`: A string. [TextLabel](TextLabel.md) that receives the frame-rate chip text. Optional.
- `vram_label`: A string. [TextLabel](TextLabel.md) that receives the GPU-memory chip text. Optional.
- `ev_label`: A string. [TextLabel](TextLabel.md) that receives the auto-exposure chip text. Optional.
- `edr_label`: A string. [TextLabel](TextLabel.md) that receives the HDR-headroom chip text. Optional.
- `passes_label`: A string. [TextLabel](TextLabel.md) that receives the per-step GPU-timing chip text. Optional.
- `mouse_label`: A string. [TextLabel](TextLabel.md) that receives the cursor-position chip text. Optional.
- `camera_label`: A string. [TextLabel](TextLabel.md) that receives the live camera-pose chip text. Optional.
