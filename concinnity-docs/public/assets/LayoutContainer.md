<!-- Auto-generated - do not edit. -->

# LayoutContainer

Positions a set of [TextLabel](TextLabel.md)s as a stack of rows, so a HUD does
not have to hand-place every chip. Each row lays its labels out left to
right; rows stack top to bottom. The container owns the labels' on-screen
position: the labels keep their own styling (font, colour, background,
padding) but their `x`/`y` are overwritten each frame.

Sizing is content-driven: a label is measured at its current text, so the
layout reflows as live HUD values change width. A row with a single label
sits on its own line beneath the previous row, which is how a wide chip
(e.g. a multi-pass timing line) ends up spanning the width of the row above.

Labels referenced by `cols` are matched by name; a label whose font is not
loaded, or which is hidden, is skipped and reserves no space.

```jsonl
{"name":"hud_layout","type":"LayoutContainer","args":{"x":10,"y":10,"col_gap":6,"row_gap":6,"rows":[{"cols":["fps_chip","vram_chip","ev_chip","edr_chip"]},{"cols":["passes_chip"]}]}}
```

## Parameters

- `x`: A float. Left edge of the container in window pixels. Defaults to `10.0`.
- `y`: A float. Top edge of the container in window pixels. Defaults to `10.0`.
- `col_gap`: A float. Pixels between adjacent labels in a row, measured between their background boxes. Defaults to `6.0`.
- `row_gap`: A float. Pixels between adjacent rows, measured between their background boxes. Defaults to `6.0`.
- `rows`: An array of [LayoutRow](LayoutRow.md) objects. Rows of labels, top to bottom.
- `visible`: A boolean. When false, the container leaves its labels where they are instead of repositioning them. Defaults to `true`.
