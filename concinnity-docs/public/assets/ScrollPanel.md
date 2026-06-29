<!-- Auto-generated - do not edit. -->

# ScrollPanel

Runtime model that makes a band of UI rows scrollable and (optionally)
collapsible.

A `ScrollPanel` is emitted by the build (e.g. by a settings menu) and read
by the UI at runtime; it is not hand-authored. It names a content band (a
fixed rectangle in the menu's reference canvas), the ordered rows that live
inside it, the collapsible groups some rows belong to, and the scrollbar
thumb/track sprites. The UI lays the rows out each frame: a collapsed group's
body rows hide and the rows below them move up; when the visible stack is
taller than the band it scrolls (mouse wheel or thumb drag) and rows outside
the band are clipped.

All pixel fields are in the same reference-space coordinates as the View's
other UI (see the overlay scaling notes on [MainMenu](MainMenu.md)).

## Parameters

- `view`: A string. [View](View.md) this panel belongs to. Resolved automatically from the `<view>_*` naming convention; you don't set this directly. The panel is only live while its view is active. Optional.
- `x`: A float. Left edge of the content band in reference pixels.
- `y`: A float. Top edge of the content band in reference pixels.
- `width`: A float. Width of the content band in reference pixels.
- `height`: A float. Height of the content band (the visible window) in reference pixels.
- `rows`: An array of [ScrollRow](ScrollRow.md) objects. The rows in the band, top to bottom.
- `groups`: An array of [ScrollGroup](ScrollGroup.md) objects. Collapsible groups, referenced by index from [ScrollRow::group].
- `thumb`: A string. Scrollbar thumb [Sprite](Sprite.md) the UI moves and resizes. `None` for a panel with no scrollbar. Optional.
- `track`: A string. Scrollbar track [Sprite](Sprite.md). Hidden along with the thumb when the content fits the band. Optional.
- `track_x`: A float. Left edge of the scrollbar track in reference pixels.
- `track_y`: A float. Top edge of the scrollbar track in reference pixels.
- `track_w`: A float. Width of the scrollbar track in reference pixels.
- `track_h`: A float. Height of the scrollbar track in reference pixels (the thumb travels within it).
