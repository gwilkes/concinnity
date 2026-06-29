<!-- Auto-generated - do not edit. -->

# ScrollRow

One row inside a [ScrollPanel](ScrollPanel.md): the elements that move
together, the row's height, and the collapsible group it belongs to.

## Parameters

- `elements`: An array of strings. The [Sprite](Sprite.md)/[TextLabel](TextLabel.md) ids that make up this row and move (and clip) together. Click regions are matched to their row by position, so they are not listed here.
- `base_y`: A float. The row's authored top edge in reference pixels (its build-time, all groups expanded, unscrolled position). Defaults to `0.0`.
- `height`: A float. The row's height in reference pixels (its vertical pitch in the stack). Defaults to `0.0`.
- `group`: An integer. Index into [ScrollPanel::groups] of the group whose collapsed state hides this row, or `-1` for a row that is always shown (a group header or an ungrouped row). Defaults to `-1`.
