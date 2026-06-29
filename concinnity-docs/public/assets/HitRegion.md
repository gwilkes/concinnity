<!-- Auto-generated - do not edit. -->

# HitRegion

A responsive invisible rectangular region in screen space.

When clicked, fires an `action`. When hovered, it optionally restyles a
referenced [TextLabel](TextLabel.md) (colour and/or scale).

The cursor must be free (not captured for camera control) for events to fire.

```jsonl
{
  "name": "btn_start",
  "type": "HitRegion",
  "args": {
    "x": 430, "y": 330, "width": 220, "height": 40,
    "label": "scene_menu_start",
    "hover_color": [1.0, 0.85, 0.3],
    "hover_scale": 1.08,
    "action": "scene:scene_game"
  }
}
```

## Parameters

- `x`: A float. Left edge of the region in window pixels. Defaults to `0.0`.
- `y`: A float. Top edge of the region in window pixels. Defaults to `0.0`.
- `width`: A float. Width of the region in window pixels. Defaults to `100.0`.
- `height`: A float. Height of the region in window pixels. Defaults to `40.0`.
- `label`: A string. A [TextLabel](TextLabel.md) to style on hover. `None` = no label effect. Optional.
- `hover_color`: An array of 3 floats. RGB colour applied to the label while hovered. `None` = no change. Optional.
- `hover_scale`: A float. Scale applied to the label while hovered. None = no change. Optional.
- `action`: A string. Action to fire on click. Recognised forms: `"scene:<name>"`, `"quit"`, `"view:show:<name>"`, `"view:hide"`, `"view:toggle:<name>"`.
- `drag_handle`: A string. The [Sprite](Sprite.md) a [Slider](Slider.md) drag region moves along its track. `None` for ordinary regions. Set automatically when a `Slider` expands; you don't set this directly. Optional.
- `view`: A string. [View](View.md) this region belongs to. Resolved automatically from the naming convention (a region named `<view>_*` belongs to view `<view>`); you don't set this directly. While a view is active, only its regions fire; when no view is active, only view-less regions fire.
- `disabled`: A boolean. Whether this region is inert. A disabled region never hovers or fires. Set by the engine at runtime (e.g. a settings row whose feature the GPU cannot provide is disabled and grayed out); you don't set this directly. Defaults to `false`.
