// HitRegion / View / KeyBinding input dispatch. An internal system (not a
// declarable asset): `World::start` constructs one whenever the world contains
// any `HitRegion`, `View`, or `KeyBinding`, then it processes hover/click,
// view overlays, and key bindings each frame.

use crate::assets::{
    FrameInput, HitRegion, KeyBinding, SceneCommand, ScrollPanel, SettingCommand, SettingOp,
    Sprite, TextLabel, View, ViewCommand,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use concinnity_core::gfx::overlay::OverlayTransform;
use concinnity_core::gfx::scroll_layout::{self, RowSpec};
use std::collections::HashMap;

// How many reference-space pixels one unit of scroll-wheel delta moves a panel.
const WHEEL_SCROLL_SPEED: f32 = 2.0;
// Shown in a rebind row's value label while it waits for the user to press a key.
const REBIND_PROMPT: &str = "Press a key...";

// Per-hit-region bookkeeping stored after init().
#[derive(Debug)]
struct RegionEntry {
    region: HitRegion,
    // Original TextLabel color, captured at init() for hover-out restore.
    original_color: Option<[f32; 3]>,
    // Original TextLabel scale, captured at init() for hover-out restore.
    original_scale: Option<f32>,
    // Whether this region was hovered last frame (to detect transitions).
    was_hovered: bool,
    // The view this region belongs to (derived from its name prefix at
    // init()), or `None` if it belongs to no view. Regions in a view only
    // fire while that view is active; regions outside any view only fire
    // when no view is active.
    view: Option<AssetId>,
    // For a slider drag region (action `setting:<key>:drag`), the setting key.
    // `None` for an ordinary click region. A slider region is driven by the
    // drag pass, not the click-to-fire path.
    slider_key: Option<String>,
    // The scroll panel + row this region belongs to, if it sits in a panel's
    // content band (resolved by position at init). Such a region reflows with
    // its row each frame and only fires while its row is shown and inside the
    // band. `None` for chrome (tab bar, Back) and non-panel regions.
    scroll_row: Option<(usize, usize)>,
    // The region's authored y, kept so the scroll reflow can set
    // `region.y = base_y + dy` from a fresh delta each frame.
    region_base_y: f32,
    // The collapsible group index this region's click toggles (action
    // `group:toggle:<gid>`), or `None`. A group-toggle region flips its panel's
    // group instead of firing an action.
    group_toggle: Option<usize>,
    // Set by the scroll reflow when this region's row is hidden (its group is
    // collapsed); a hidden region never hovers or fires.
    hidden: bool,
}

// Per-view bookkeeping.
#[derive(Debug, Default)]
struct ViewRegistry {
    // Every declared View id. The dispatch path warns when a `view:*` action
    // resolves to an id not in this set.
    known: std::collections::HashSet<AssetId>,
    // Currently-active view (single slot, no stacking).
    active: Option<AssetId>,
    // Previously-active view saved on each Show; restored on Hide. Lets a
    // modal-style overlay (pause, options) dismiss back to the page the user
    // was on without scene tracking.
    prev: Option<AssetId>,
}

// One row of a scroll panel: the elements that move together, their authored
// y's (snapshot at init so the reflow is `base + dy`), the row's top + height
// (for bucketing regions to rows), and the collapsible group it belongs to.
#[derive(Debug)]
struct RowState {
    elements: Vec<AssetId>,
    base_ys: Vec<f32>,
    base_y: f32,
    height: f32,
    group: Option<usize>,
}

// A collapsible group's runtime state: whether it is collapsed and the header
// label whose `+`/`-` prefix reflects it.
#[derive(Debug)]
struct GroupState {
    collapsed: bool,
    header: Option<AssetId>,
    title: String,
}

// Runtime state for one scroll panel, drained from a `ScrollPanel` at init.
#[derive(Debug)]
struct PanelState {
    view: Option<AssetId>,
    // Content band [x, y, width, height] in reference space.
    band: [f32; 4],
    rows: Vec<RowState>,
    groups: Vec<GroupState>,
    thumb: Option<AssetId>,
    track: Option<AssetId>,
    track_x: f32,
    track_y: f32,
    track_w: f32,
    track_h: f32,
    // Current scroll offset (reference pixels), clamped by the solver.
    scroll: f32,
    // Last solve outputs, kept for the thumb-drag cursor->scroll mapping.
    content_height: f32,
    thumb_h: f32,
}

// An in-progress key rebind: a Controls-tab rebind row was clicked and is
// waiting for the user to press a key. The next `FrameInput.captured_key` binds
// it; Escape cancels and restores the row's previous value text.
#[derive(Debug)]
struct Capture {
    // The rebind setting key, e.g. `"key_forward"`.
    setting_key: String,
    // The value `TextLabel` showing the bound key (set to a prompt while
    // capturing; GraphicsSystem rewrites it after the bind).
    value_label: Option<AssetId>,
    // The label's text before capture began, restored if the user cancels.
    prev_text: String,
}

// HitRegion / View / KeyBinding input dispatch behavior. Constructed
// internally by `World::start` when the world declares any `HitRegion`,
// `View`, or `KeyBinding`; never a world-declared asset, so it carries no
// config.
#[derive(Debug)]
pub struct UiInputSystem {
    regions: Vec<RegionEntry>,
    bindings: Vec<KeyBinding>,
    views: ViewRegistry,
    // asset_id of UI elements (Sprite, TextLabel) by their owning view.
    // Built at init() from `<view_name>_*` name prefixes.
    sprites_by_view: HashMap<AssetId, Vec<AssetId>>,
    labels_by_view: HashMap<AssetId, Vec<AssetId>>,
    // Index (into `regions`) of the slider currently being dragged, or `None`.
    // Set on the press edge over a slider track, cleared on button release.
    dragging: Option<usize>,
    // Scroll panels in the world, drained at init. Driven each frame: collapse
    // state + scroll offset are solved into per-row positions written back onto
    // the elements + regions.
    panels: Vec<PanelState>,
    // `(panel index, grab offset)` while the scrollbar thumb is being dragged.
    // The grab offset keeps the thumb from jumping under the cursor on grab.
    thumb_drag: Option<(usize, f32)>,
    // A pending key rebind (a Controls-tab rebind row is capturing), or `None`.
    // While set, the menu consumes the frame for capture: the next pressed key
    // binds it and Escape cancels.
    capturing: Option<Capture>,
}

impl UiInputSystem {
    // Empty dispatch state. The world's `HitRegion` / `View` / `KeyBinding`
    // components are drained into it in [`System::init`].
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            bindings: Vec::new(),
            views: ViewRegistry::default(),
            sprites_by_view: HashMap::new(),
            labels_by_view: HashMap::new(),
            dragging: None,
            panels: Vec::new(),
            thumb_drag: None,
            capturing: None,
        }
    }
}

impl System for UiInputSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        // Drain View assets, record every id, and pick the one flagged
        // `initial` as the active view at world start.
        let mut initial: Option<AssetId> = None;
        for v in ctx.drain::<View>() {
            self.views.known.insert(v.asset_id);
            if v.initial && initial.is_none() {
                initial = Some(v.asset_id);
            }
        }

        // Drain KeyBindings: they aren't iterated each frame on the world,
        // we just match the pulse against this snapshot.
        self.bindings = ctx.drain::<KeyBinding>();

        // Drain HitRegions, capture per-region hover restore state, and
        // assign each region to a view (or none) based on the resolved
        // `view` field that the build pipeline writes from the name prefix.
        let hit_regions = ctx.drain::<HitRegion>();
        for region in hit_regions {
            // A region disabled by the engine (e.g. a capability-gated settings
            // row grayed out at init) is inert: dropping it here means it never
            // hovers, fires, drags, or reflows. Its labels are styled + reflowed
            // independently (by GraphicsSystem and the scroll panel).
            if region.disabled {
                continue;
            }
            let (original_color, original_scale) = match region.label {
                None => (None, None),
                Some(label_id) => ctx
                    .query::<TextLabel>()
                    .find(|l| l.asset_id == label_id)
                    .map(|l| (Some(l.color), Some(l.scale)))
                    .unwrap_or((None, None)),
            };
            let view = region.view;
            let slider_key = slider_key_from_action(&region.action);
            let group_toggle = group_toggle_from_action(&region.action);
            let region_base_y = region.y;
            self.regions.push(RegionEntry {
                region,
                original_color,
                original_scale,
                was_hovered: false,
                view,
                slider_key,
                scroll_row: None,
                region_base_y,
                group_toggle,
                hidden: false,
            });
        }

        // Build view → UI-element maps by reading each Sprite/TextLabel's
        // resolved `view` field (the build pipeline writes it from the
        // <view>_* name prefix).
        for s in ctx.query::<Sprite>() {
            if let Some(view_id) = s.view {
                self.sprites_by_view
                    .entry(view_id)
                    .or_default()
                    .push(s.asset_id);
            }
        }
        for l in ctx.query::<TextLabel>() {
            if let Some(view_id) = l.view {
                self.labels_by_view
                    .entry(view_id)
                    .or_default()
                    .push(l.asset_id);
            }
        }

        // Drain ScrollPanels into runtime state and bucket the regions into
        // their rows (uses the regions drained just above).
        self.init_panels(ctx);

        // Views start hidden: zero out the visibility of every view-owned
        // Sprite and TextLabel.
        for ids in self.sprites_by_view.values() {
            for &id in ids {
                for sp in ctx.query_mut::<Sprite>() {
                    if sp.asset_id == id {
                        sp.visible = false;
                        break;
                    }
                }
            }
        }
        for ids in self.labels_by_view.values() {
            for &id in ids {
                for lbl in ctx.query_mut::<TextLabel>() {
                    if lbl.asset_id == id {
                        lbl.visible = false;
                        break;
                    }
                }
            }
        }

        // Activate the initial view (if any) by showing its elements.
        if let Some(id) = initial {
            self.set_view_visibility(id, true, ctx);
            self.views.active = Some(id);
        }

        // Solve the initial scroll layout so frame 0 already shows the right
        // collapsed/scrolled positions (a default-collapsed group starts shut).
        self.apply_scroll_layout(ctx);
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // Drain accumulated ViewCommands first so a click last frame takes
        // effect before this frame's hit-testing reads `active`.
        for cmd in ctx.drain::<ViewCommand>() {
            self.apply_view_command(cmd, ctx);
        }

        // Read (not drain) the per-frame input snapshot so this system can
        // coexist with Camera3DSystem (both query it; GraphicsSystem clears it
        // before the next push). Take the most recent if more than one exists.
        let input = match ctx.query::<FrameInput>().last().cloned() {
            Some(i) => i,
            None => return StepResult::Continue,
        };

        // A pending key rebind (a Controls-tab rebind row was clicked) consumes
        // the whole frame: the next pressed key binds it, Escape cancels (and
        // restores the row's previous text), otherwise it keeps waiting. No
        // clicks, hover, or other key bindings fire while capturing.
        if self.capturing.is_some() {
            if input.escape {
                self.cancel_capture(ctx);
            } else if let Some(key) = input.captured_key {
                let cap = self.capturing.take().expect("capturing is some");
                ctx.push(SettingCommand {
                    setting: cap.setting_key,
                    op: SettingOp::Rebind(key),
                    value_label: cap.value_label,
                    persist: true,
                });
                // GraphicsSystem rewrites the value label to the bound key when
                // it drains the command next tick; the prompt shows until then.
            }
            return StepResult::Continue;
        }

        // Handle KeyBindings before HitRegion clicks so an Esc-toggle-pause
        // beats a click that landed on the same frame.
        if input.escape {
            for kb in &self.bindings {
                if kb.key == "Escape" && !kb.action.is_empty() {
                    // KeyBindings carry no label (no settings row binds a key).
                    if let Some(result) = fire_action(&kb.action, None, ctx) {
                        return result;
                    }
                    break;
                }
            }
        }

        let mx = input.mouse_x;
        let my = input.mouse_y;
        let clicked = input.left_click;
        let down = input.left_button_down;
        let active_view = self.views.active;
        // View-owned regions are overlay UI authored in the reference canvas and
        // scaled onto the window; map the live cursor back into reference space
        // before testing it against their (reference-space) rects. View-less
        // regions stay in window pixels (see crate::gfx::overlay).
        let overlay = OverlayTransform::from_viewport(input.viewport);

        // Scroll-wheel + scrollbar-thumb input for the active view's panel; both
        // adjust the panel's scroll offset (clamped later in the apply pass). A
        // thumb drag suppresses the slider + click passes so the gutter doesn't
        // double as a control.
        let thumb_active = self.handle_scroll_input(&input, mx, my, active_view, &overlay);

        // Per-panel bands (reference space), so a scroll-content region only
        // fires while the cursor is inside its panel window.
        let panel_bands: Vec<[f32; 4]> = self.panels.iter().map(|p| p.band).collect();

        // Slider drag pass. A slider's track region is driven here, not by the
        // click-to-fire loop below: the press edge (`clicked`) over a track
        // begins a drag, the held button (`down`) tracks the cursor each frame,
        // and release commits the final value. The dragged region is remembered
        // so the drag continues even when the cursor leaves the track.
        if !thumb_active && !down {
            // Release: commit the dragged slider's final position (persists).
            if let Some(i) = self.dragging.take()
                && self.regions[i].view == active_view
                && let Some(key) = self.regions[i].slider_key.clone()
            {
                // Slider tracks are overlay UI: map the cursor to reference space.
                let (qx, _) = overlay.inverse(mx, my);
                let r = &self.regions[i].region;
                let frac = ((qx - r.x) / r.width).clamp(0.0, 1.0);
                let label = r.label;
                ctx.push(SettingCommand {
                    setting: key,
                    op: SettingOp::SetFraction(frac),
                    value_label: label,
                    persist: true,
                });
            }
        } else if !thumb_active {
            // Slider tracks are overlay UI: map the cursor to reference space.
            let (qx, qy) = overlay.inverse(mx, my);
            for i in 0..self.regions.len() {
                if self.regions[i].view != active_view {
                    continue;
                }
                let Some(key) = self.regions[i].slider_key.clone() else {
                    continue;
                };
                let (rx, ry, rw, rh, label) = {
                    let r = &self.regions[i].region;
                    (r.x, r.y, r.width, r.height, r.label)
                };
                let over = qx >= rx && qx < rx + rw && qy >= ry && qy < ry + rh;
                if self.dragging.is_none() && clicked && over {
                    self.dragging = Some(i);
                }
                if self.dragging == Some(i) {
                    let frac = ((qx - rx) / rw).clamp(0.0, 1.0);
                    // In-progress: apply live but skip the disk write (persist
                    // only on release, above).
                    ctx.push(SettingCommand {
                        setting: key,
                        op: SettingOp::SetFraction(frac),
                        value_label: label,
                        persist: false,
                    });
                }
            }
        }

        // A group-toggle click recorded here is applied after the loop (the loop
        // borrows the regions mutably; the panels are mutated below).
        let mut toggle_group: Option<usize> = None;
        // A rebind-row click recorded here (setting key + value label) starts a
        // capture after the loop, for the same borrow reason.
        let mut start_capture: Option<(String, Option<AssetId>)> = None;
        for entry in &mut self.regions {
            // While dragging the scrollbar thumb, no region hovers or fires.
            if thumb_active {
                entry.was_hovered = false;
                continue;
            }
            // Slider regions are driven by the drag pass above, not by clicks.
            if entry.slider_key.is_some() {
                entry.was_hovered = false;
                continue;
            }
            // Filter by active view: when a view is shown, only its regions
            // fire; otherwise only view-less regions fire.
            if entry.view != active_view {
                // Also clear stale hover state so a label doesn't keep its
                // hover styling while behind an overlay.
                entry.was_hovered = false;
                continue;
            }
            // A scroll-content region whose row is collapsed never hovers/fires.
            if entry.scroll_row.is_some() && entry.hidden {
                entry.was_hovered = false;
                continue;
            }

            // Overlay (view-owned) regions hit-test in reference space; HUD
            // regions in window pixels.
            let (qx, qy) = if entry.view.is_some() {
                overlay.inverse(mx, my)
            } else {
                (mx, my)
            };
            let group_toggle = entry.group_toggle;
            let r = &entry.region;
            let mut hovered = qx >= r.x && qx < r.x + r.width && qy >= r.y && qy < r.y + r.height;
            // A scroll-content region only counts as hovered inside its band, so
            // a row scrolled past the edge does not catch clicks over the chrome.
            if let Some((pi, _)) = entry.scroll_row
                && let Some(band) = panel_bands.get(pi)
            {
                hovered = hovered && point_in_rect(qx, qy, *band);
            }

            // Apply / restore hover styling on the referenced label.
            if let Some(label_id) = r.label {
                if hovered && !entry.was_hovered {
                    let hover_color = r.hover_color;
                    let hover_scale = r.hover_scale;
                    for lbl in ctx.query_mut::<TextLabel>() {
                        if lbl.asset_id == label_id {
                            if let Some(c) = hover_color {
                                lbl.color = c;
                            }
                            if let Some(s) = hover_scale {
                                lbl.scale = s;
                            }
                            break;
                        }
                    }
                } else if !hovered && entry.was_hovered {
                    let orig_color = entry.original_color;
                    let orig_scale = entry.original_scale;
                    for lbl in ctx.query_mut::<TextLabel>() {
                        if lbl.asset_id == label_id {
                            if let Some(c) = orig_color {
                                lbl.color = c;
                            }
                            if let Some(s) = orig_scale {
                                lbl.scale = s;
                            }
                            break;
                        }
                    }
                }
            }

            entry.was_hovered = hovered;

            if hovered && clicked {
                // A group header toggles its panel's group (handled after the
                // loop) instead of firing an action.
                if let Some(gid) = group_toggle {
                    toggle_group = Some(gid);
                } else if let Some(key) = rebind_key_from_action(&r.action) {
                    // A rebind row enters capture (started after the loop)
                    // instead of firing an action immediately.
                    start_capture = Some((key.to_string(), r.label));
                } else if !r.action.is_empty()
                    && let Some(result) = fire_action(&r.action, r.label, ctx)
                {
                    return result;
                }
            }
        }

        // Begin a key rebind capture for a clicked rebind row: stash the value
        // label's current text (to restore on cancel) and show the prompt.
        if let Some((setting_key, value_label)) = start_capture {
            let prev_text = value_label
                .and_then(|id| {
                    ctx.query::<TextLabel>()
                        .find(|l| l.asset_id == id)
                        .map(|l| l.content.clone())
                })
                .unwrap_or_default();
            if let Some(id) = value_label {
                self.set_label_text(ctx, id, REBIND_PROMPT);
            }
            self.capturing = Some(Capture {
                setting_key,
                value_label,
                prev_text,
            });
        }

        // Apply a recorded group toggle to the active view's panel, then solve
        // every panel so the next frame draws + hit-tests the reflowed layout.
        if let Some(gid) = toggle_group
            && let Some(panel) = self.panels.iter_mut().find(|p| p.view == active_view)
            && let Some(g) = panel.groups.get_mut(gid)
        {
            g.collapsed = !g.collapsed;
        }
        self.apply_scroll_layout(ctx);

        StepResult::Continue
    }
}

impl UiInputSystem {
    fn apply_view_command(&mut self, cmd: ViewCommand, ctx: &mut PipelineContext) {
        // Semantics:
        //   Show(X)     : plain navigation: active=X, prev cleared.
        //   Hide        : restore prev (modal dismiss); both cleared after.
        //   Toggle(X)   : if active==X, dismiss back to prev; else open X
        //                 modally, saving current active as prev.
        //
        // Only Toggle saves prev, so navigation (Show) doesn't pollute the
        // pause-restore slot.
        let (new_active, new_prev) = match cmd {
            ViewCommand::Hide => (self.views.prev, None),
            ViewCommand::Show(id) => {
                if !self.views.known.contains(&id) {
                    tracing::warn!("ViewCommand::Show: unknown view {}", id);
                    return;
                }
                if self.views.active == Some(id) {
                    return;
                }
                (Some(id), None)
            }
            ViewCommand::Toggle(id) => {
                if !self.views.known.contains(&id) {
                    tracing::warn!("ViewCommand::Toggle: unknown view {}", id);
                    return;
                }
                if self.views.active == Some(id) {
                    (self.views.prev, None)
                } else {
                    (Some(id), self.views.active)
                }
            }
        };

        if new_active == self.views.active {
            return;
        }

        if let Some(prev) = self.views.active {
            self.set_view_visibility(prev, false, ctx);
        }
        if let Some(next) = new_active {
            self.set_view_visibility(next, true, ctx);
        }
        self.views.active = new_active;
        self.views.prev = new_prev;
    }

    // Cancel a pending rebind capture, restoring the row's previous value text.
    fn cancel_capture(&mut self, ctx: &mut PipelineContext) {
        if let Some(cap) = self.capturing.take()
            && let Some(id) = cap.value_label
        {
            let prev = cap.prev_text.clone();
            self.set_label_text(ctx, id, &prev);
        }
    }

    // Overwrite the content of the TextLabel with the given id, if present.
    fn set_label_text(&self, ctx: &mut PipelineContext, id: AssetId, text: &str) {
        for l in ctx.query_mut::<TextLabel>() {
            if l.asset_id == id {
                l.content = text.to_string();
                break;
            }
        }
    }

    fn set_view_visibility(&self, view_id: AssetId, visible: bool, ctx: &mut PipelineContext) {
        if let Some(ids) = self.sprites_by_view.get(&view_id) {
            for &id in ids {
                for s in ctx.query_mut::<Sprite>() {
                    if s.asset_id == id {
                        s.visible = visible;
                        break;
                    }
                }
            }
        }
        if let Some(ids) = self.labels_by_view.get(&view_id) {
            for &id in ids {
                for l in ctx.query_mut::<TextLabel>() {
                    if l.asset_id == id {
                        l.visible = visible;
                        break;
                    }
                }
            }
        }
    }

    // Drain the world's ScrollPanels into runtime state: snapshot each row
    // element's authored y (so the reflow is `base + dy`), translate the
    // `i32` group index into an `Option`, and bucket each HitRegion into the
    // panel row whose band its centre falls in (so the region reflows + gates
    // with that row). Runs once at init, after HitRegions are drained.
    fn init_panels(&mut self, ctx: &mut PipelineContext) {
        let panels = ctx.drain::<ScrollPanel>();
        if panels.is_empty() {
            return;
        }
        // Snapshot the authored y of every element any panel row references.
        let wanted: std::collections::HashSet<AssetId> = panels
            .iter()
            .flat_map(|p| p.rows.iter().flat_map(|r| r.elements.iter().copied()))
            .collect();
        let mut elem_y: HashMap<AssetId, f32> = HashMap::new();
        for s in ctx.query::<Sprite>() {
            if wanted.contains(&s.asset_id) {
                elem_y.insert(s.asset_id, s.y);
            }
        }
        for l in ctx.query::<TextLabel>() {
            if wanted.contains(&l.asset_id) {
                elem_y.insert(l.asset_id, l.y);
            }
        }

        for p in panels {
            let rows = p
                .rows
                .iter()
                .map(|r| {
                    let base_ys = r
                        .elements
                        .iter()
                        .map(|id| elem_y.get(id).copied().unwrap_or(r.base_y))
                        .collect();
                    RowState {
                        elements: r.elements.clone(),
                        base_ys,
                        base_y: r.base_y,
                        height: r.height,
                        group: (r.group >= 0).then_some(r.group as usize),
                    }
                })
                .collect();
            let groups = p
                .groups
                .iter()
                .map(|g| GroupState {
                    collapsed: g.collapsed,
                    header: g.header,
                    title: g.title.clone(),
                })
                .collect();
            self.panels.push(PanelState {
                view: p.view,
                band: [p.x, p.y, p.width, p.height],
                rows,
                groups,
                thumb: p.thumb,
                track: p.track,
                track_x: p.track_x,
                track_y: p.track_y,
                track_w: p.track_w,
                track_h: p.track_h,
                scroll: 0.0,
                content_height: 0.0,
                thumb_h: 0.0,
            });
        }

        // Bucket each panel-content region into its row by centre y. Only
        // content regions (a settings action or a group toggle) are bucketed;
        // chrome regions (tabs, Back -- `view:show`) are left fixed even when an
        // overflow row's authored y reaches their position. Panels read
        // immutably while the regions are mutated (disjoint fields).
        let panels = &self.panels;
        for entry in self.regions.iter_mut() {
            let is_content =
                entry.region.action.starts_with("setting:") || entry.group_toggle.is_some();
            if !is_content {
                continue;
            }
            let cy = entry.region.y + entry.region.height * 0.5;
            'find: for (pi, panel) in panels.iter().enumerate() {
                if panel.view != entry.view {
                    continue;
                }
                for (ri, row) in panel.rows.iter().enumerate() {
                    if cy >= row.base_y && cy < row.base_y + row.height {
                        entry.scroll_row = Some((pi, ri));
                        break 'find;
                    }
                }
            }
        }
    }

    // The reference-space rectangle of a panel's scrollbar thumb at its current
    // scroll, or `None` if the panel does not overflow (no thumb to grab).
    fn thumb_rect(panel: &PanelState) -> Option<[f32; 4]> {
        if panel.content_height <= 0.0 || panel.thumb_h >= panel.track_h {
            return None;
        }
        let offset_frac = (panel.scroll / panel.content_height).clamp(0.0, 1.0);
        let thumb_y = panel.track_y + offset_frac * panel.track_h;
        Some([panel.track_x, thumb_y, panel.track_w, panel.thumb_h])
    }

    // Apply scroll-wheel + scrollbar-thumb input to the active view's panel.
    // Returns true while the thumb is being dragged so the caller suppresses the
    // slider + click passes. The solver clamps the resulting scroll offset.
    fn handle_scroll_input(
        &mut self,
        input: &FrameInput,
        mx: f32,
        my: f32,
        active_view: Option<AssetId>,
        overlay: &OverlayTransform,
    ) -> bool {
        let (qx, qy) = overlay.inverse(mx, my);
        let active_panel = self.panels.iter().position(|p| p.view == active_view);

        // Wheel: scroll the active panel while the cursor is over its band.
        if input.scroll_delta != 0.0
            && let Some(pi) = active_panel
            && point_in_rect(qx, qy, self.panels[pi].band)
        {
            self.panels[pi].scroll += input.scroll_delta * WHEEL_SCROLL_SPEED;
        }

        // Thumb drag: begin on the press edge over the thumb, then map the
        // cursor's y to a scroll offset for the rest of the press.
        if !input.left_button_down {
            self.thumb_drag = None;
        } else {
            if self.thumb_drag.is_none()
                && input.left_click
                && let Some(pi) = active_panel
                && let Some(rect) = Self::thumb_rect(&self.panels[pi])
                && point_in_rect(qx, qy, rect)
            {
                self.thumb_drag = Some((pi, qy - rect[1]));
            }
            if let Some((pi, grab)) = self.thumb_drag {
                let panel = &mut self.panels[pi];
                let travel = (panel.track_h - panel.thumb_h).max(0.0);
                let max_scroll = (panel.content_height - panel.band[3]).max(0.0);
                if travel > 0.0 && max_scroll > 0.0 {
                    let thumb_top = (qy - grab).clamp(panel.track_y, panel.track_y + travel);
                    let frac = (thumb_top - panel.track_y) / travel;
                    panel.scroll = frac * max_scroll;
                }
            }
        }
        self.thumb_drag.is_some()
    }

    // Solve every panel's vertical layout and write the result back: element y +
    // visibility, region reflow + hidden flag, the scrollbar thumb position +
    // size, and each group header's `+`/`-` prefix. Only the active view's panel
    // writes (an inactive view's elements stay hidden by the view system). Runs
    // at init and at the end of each step so the next frame draws + hit-tests the
    // reflowed positions consistently.
    fn apply_scroll_layout(&mut self, ctx: &mut PipelineContext) {
        if self.panels.is_empty() {
            return;
        }
        let active = self.views.active;

        // Accumulate component writes, then apply in single passes.
        let mut sprite_updates: HashMap<AssetId, (f32, Option<f32>, bool)> = HashMap::new();
        let mut label_updates: HashMap<AssetId, (f32, bool)> = HashMap::new();
        let mut track_visible: Vec<(AssetId, bool)> = Vec::new();
        let mut header_text: Vec<(AssetId, String)> = Vec::new();
        // Per-panel `(active, row placements)` for the region reflow below.
        let mut solved_rows: Vec<(bool, Vec<scroll_layout::RowPlacement>)> =
            Vec::with_capacity(self.panels.len());

        for panel in self.panels.iter_mut() {
            let panel_active = panel.view == active;
            let collapsed: Vec<bool> = panel.groups.iter().map(|g| g.collapsed).collect();
            let specs: Vec<RowSpec> = panel
                .rows
                .iter()
                .map(|r| RowSpec {
                    height: r.height,
                    group: r.group,
                })
                .collect();
            let solved = scroll_layout::solve(&specs, &collapsed, panel.band[3], panel.scroll);
            panel.scroll = solved.scroll;
            panel.content_height = solved.content_height;
            panel.thumb_h = solved.thumb_frac * panel.track_h;

            if panel_active {
                for (ri, row) in panel.rows.iter().enumerate() {
                    let pl = solved.rows[ri];
                    for (k, id) in row.elements.iter().enumerate() {
                        let y = row.base_ys[k] + pl.dy;
                        sprite_updates.insert(*id, (y, None, pl.visible));
                        label_updates.insert(*id, (y, pl.visible));
                    }
                }
                let scrollable = solved.scrollable();
                if let Some(thumb) = panel.thumb {
                    let thumb_y = panel.track_y + solved.thumb_offset_frac * panel.track_h;
                    sprite_updates.insert(thumb, (thumb_y, Some(panel.thumb_h), scrollable));
                }
                if let Some(track) = panel.track {
                    track_visible.push((track, scrollable));
                }
                for g in &panel.groups {
                    if let Some(h) = g.header {
                        let prefix = if g.collapsed { "+ " } else { "- " };
                        header_text.push((h, format!("{prefix}{}", g.title)));
                    }
                }
            }
            solved_rows.push((panel_active, solved.rows));
        }

        // Reflow each panel-owned region in memory (positions the click loop
        // hit-tests against next frame).
        for entry in self.regions.iter_mut() {
            if let Some((pi, ri)) = entry.scroll_row
                && let Some((panel_active, rows)) = solved_rows.get(pi)
                && *panel_active
            {
                let pl = rows[ri];
                entry.region.y = entry.region_base_y + pl.dy;
                entry.hidden = !pl.visible;
            }
        }

        // Apply the accumulated component writes.
        for s in ctx.query_mut::<Sprite>() {
            if let Some(&(y, h, vis)) = sprite_updates.get(&s.asset_id) {
                s.y = y;
                if let Some(hh) = h {
                    s.height = hh;
                }
                s.visible = vis;
            }
        }
        for (id, vis) in &track_visible {
            for s in ctx.query_mut::<Sprite>() {
                if s.asset_id == *id {
                    s.visible = *vis;
                    break;
                }
            }
        }
        for l in ctx.query_mut::<TextLabel>() {
            if let Some(&(y, vis)) = label_updates.get(&l.asset_id) {
                l.y = y;
                l.visible = vis;
            }
        }
        for (id, text) in &header_text {
            for l in ctx.query_mut::<TextLabel>() {
                if l.asset_id == *id {
                    l.content = text.clone();
                    break;
                }
            }
        }
    }
}

// The setting key of a slider drag action (`setting:<key>:drag`), or `None`
// for any other action. A region with `Some` here is a slider track, driven by
// the drag pass rather than the click-to-fire path.
fn slider_key_from_action(action: &str) -> Option<String> {
    let rest = action.strip_prefix("setting:")?;
    let key = rest.strip_suffix(":drag")?;
    (!key.is_empty()).then(|| key.to_string())
}

// The setting key of a key-rebind action (`setting:<key>:rebind`), or `None`
// for any other action. A region with `Some` here enters capture mode on click
// instead of firing an action.
fn rebind_key_from_action(action: &str) -> Option<&str> {
    let rest = action.strip_prefix("setting:")?;
    let key = rest.strip_suffix(":rebind")?;
    (!key.is_empty()).then_some(key)
}

// The collapsible-group index of a group-toggle action (`group:toggle:<gid>`),
// or `None`. A region with `Some` here flips its panel's group instead of
// firing an action.
fn group_toggle_from_action(action: &str) -> Option<usize> {
    action.strip_prefix("group:toggle:")?.parse::<usize>().ok()
}

// Whether a point lies inside an `[x, y, width, height]` rectangle.
fn point_in_rect(x: f32, y: f32, rect: [f32; 4]) -> bool {
    x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3]
}

// Parse and execute an action string. Returns Some(StepResult) when the
// action produces an engine-level result (e.g. Quit), None otherwise. `label`
// is the firing region's referenced TextLabel (the value display for a
// settings row), forwarded so GraphicsSystem can update it.
fn fire_action(
    action: &str,
    label: Option<AssetId>,
    ctx: &mut PipelineContext,
) -> Option<StepResult> {
    if action == "quit" {
        return Some(StepResult::Stop);
    }
    if let Some(scene_ref) = action.strip_prefix("scene:") {
        // The build rewrites `scene:<name>` to `scene:<id>` so the target is
        // a plain integer here (see concinnity_cook::pipeline::resolve_scene_refs).
        match scene_ref.parse::<u32>() {
            Ok(id) => {
                ctx.push(SceneCommand {
                    scene: AssetId(id),
                    transition: "FadeBlack".to_string(),
                });
                // Hide any active view on a scene change: the user has
                // chosen a new context, so the overlay is dismissed.
                ctx.push(ViewCommand::Hide);
            }
            Err(_) => tracing::warn!("UiInputSystem: unresolved scene action '{}'", action),
        }
        return None;
    }
    if action == "view:hide" {
        ctx.push(ViewCommand::Hide);
        return None;
    }
    if let Some(view_ref) = action.strip_prefix("view:show:") {
        match view_ref.parse::<u32>() {
            Ok(id) => ctx.push(ViewCommand::Show(AssetId(id))),
            Err(_) => tracing::warn!("UiInputSystem: unresolved view action '{}'", action),
        }
        return None;
    }
    if let Some(view_ref) = action.strip_prefix("view:toggle:") {
        match view_ref.parse::<u32>() {
            Ok(id) => ctx.push(ViewCommand::Toggle(AssetId(id))),
            Err(_) => tracing::warn!("UiInputSystem: unresolved view action '{}'", action),
        }
        return None;
    }
    // setting:<key>:next|prev -- cycle a graphics setting. GraphicsSystem
    // drains the SettingCommand to apply, persist, and refresh the value label.
    if let Some(rest) = action.strip_prefix("setting:") {
        match rest.rsplit_once(':') {
            Some((key, "next")) | Some((key, "prev")) if !key.is_empty() => {
                let op = if rest.ends_with(":prev") {
                    SettingOp::Prev
                } else {
                    SettingOp::Next
                };
                ctx.push(SettingCommand {
                    setting: key.to_string(),
                    op,
                    value_label: label,
                    // A cycle is one discrete change: always persisted.
                    persist: true,
                });
            }
            // Slider drags and key rebinds are driven by their own passes (the
            // drag pass and the capture flow), not the click-to-fire path, so
            // they never reach here from a HitRegion click; recognise them so a
            // stray binding does not log a false "malformed" warning.
            Some((key, "drag")) | Some((key, "rebind")) if !key.is_empty() => {}
            _ => tracing::warn!("UiInputSystem: malformed setting action '{}'", action),
        }
        return None;
    }
    tracing::warn!("UiInputSystem: unknown action '{}'", action);
    None
}

#[cfg(test)]
mod tests {
    // UiInputSystem is internal: each test seeds the gating components
    // (HitRegion / View / KeyBinding) before `world.start()`, which constructs
    // the system from them via the build schedule.
    use super::*;
    use crate::assets::{HitRegion, ScrollGroup, ScrollRow, TextLabel};
    use crate::ecs::World;

    fn make_frame_input(mx: f32, my: f32, clicked: bool) -> FrameInput {
        FrameInput {
            mouse_x: mx,
            mouse_y: my,
            left_click: clicked,
            ..Default::default()
        }
    }

    // A view-owned TextLabel used as a scroll-panel element.
    fn panel_label(id: u32, y: f32, view: AssetId, content: &str) -> TextLabel {
        TextLabel {
            asset_id: AssetId(id),
            font: None,
            content: content.to_string(),
            x: 0.0,
            y,
            color: [1.0, 1.0, 1.0],
            scale: 1.0,
            centered: false,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            view: Some(view),
        }
    }

    fn label_field<T>(world: &World, id: AssetId, f: impl Fn(&TextLabel) -> T) -> T {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .map(f)
            .unwrap()
    }

    #[test]
    fn hover_applies_and_restores_label_style() {
        let mut world = World::new_empty();

        world.add_component(TextLabel {
            asset_id: AssetId(1),
            font: None,
            content: "Hello".to_string(),
            x: 0.0,
            y: 0.0,
            color: [1.0, 1.0, 1.0],
            scale: 1.0,
            centered: false,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            view: None,
        });
        world.add_component(HitRegion {
            x: 10.0,
            y: 10.0,
            width: 100.0,
            height: 40.0,
            label: Some(AssetId(1)),
            hover_color: Some([1.0, 0.0, 0.0]),
            hover_scale: Some(2.0),
            action: String::new(),
            drag_handle: None,
            view: None,
            disabled: false,
        });
        world.start().unwrap();

        // Hover over the region.
        world.add_component(make_frame_input(50.0, 30.0, false));
        world.step();

        // Label should be styled.
        let lbl_color = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(1))
            .map(|l| l.color)
            .unwrap();
        assert_eq!(lbl_color, [1.0, 0.0, 0.0]);

        // Move cursor away.
        world.add_component(make_frame_input(0.0, 0.0, false));
        world.step();

        let lbl_color_after = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(1))
            .map(|l| l.color)
            .unwrap();
        assert_eq!(lbl_color_after, [1.0, 1.0, 1.0]);
    }

    // When a region's hover_scale equals its label's scale (what the generated
    // settings menu emits), hovering changes only the color: the label keeps its
    // size, so it does not grow or shift out of its row. This is the runtime end
    // of the build-side `default_menu_hover_is_color_only` guarantee.
    #[test]
    fn hover_with_matching_scale_changes_color_only() {
        let mut world = World::new_empty();

        world.add_component(TextLabel {
            asset_id: AssetId(1),
            font: None,
            content: "Vsync".to_string(),
            x: 0.0,
            y: 0.0,
            color: [0.85, 0.85, 0.85],
            scale: 0.66,
            centered: false,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            view: None,
        });
        world.add_component(HitRegion {
            x: 10.0,
            y: 10.0,
            width: 100.0,
            height: 40.0,
            label: Some(AssetId(1)),
            hover_color: Some([1.0, 0.85, 0.3]),
            // Matches the label's scale, so hover must not resize it.
            hover_scale: Some(0.66),
            action: String::new(),
            drag_handle: None,
            view: None,
            disabled: false,
        });
        world.start().unwrap();

        world.add_component(make_frame_input(50.0, 30.0, false));
        world.step();

        let lbl = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(1))
            .map(|l| (l.color, l.scale))
            .unwrap();
        assert_eq!(lbl.0, [1.0, 0.85, 0.3], "hover should change color");
        assert_eq!(lbl.1, 0.66, "hover must not change the label scale");
    }

    #[test]
    fn click_pushes_scene_command() {
        let mut world = World::new_empty();

        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "scene:3".to_string(),
            drag_handle: None,
            view: None,
            disabled: false,
        });
        world.start().unwrap();

        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();

        let has_cmd = world.query::<SceneCommand>().next().is_some();
        assert!(has_cmd);
    }

    #[test]
    fn quit_action_returns_stop() {
        let mut world = World::new_empty();

        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "quit".to_string(),
            drag_handle: None,
            view: None,
            disabled: false,
        });
        world.start().unwrap();

        world.add_component(make_frame_input(50.0, 50.0, true));
        let result = world.step();
        assert_eq!(result, StepResult::Stop);
    }

    // Showing a view makes its sprites visible and hides them again on Hide.
    #[test]
    fn view_show_and_hide_toggles_sprite_visibility() {
        let mut world = World::new_empty();

        let view_id = AssetId(10);
        world.add_component(View {
            asset_id: view_id,
            initial: false,
            fade_in_secs: 0.0,
        });
        world.add_component(Sprite {
            asset_id: AssetId(11),
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            texture: None,
            tint: [0.0, 0.0, 0.0, 0.5],
            follow_cursor: false,
            visible: true, // intentionally true to confirm init hides it
            view: Some(view_id),
        });
        world.start().unwrap();

        // init() hides view elements.
        let visible_after_init = world
            .query::<Sprite>()
            .find(|s| s.asset_id == AssetId(11))
            .map(|s| s.visible)
            .unwrap();
        assert!(!visible_after_init, "view starts hidden after init");

        // Show the view.
        world.add_component(ViewCommand::Show(view_id));
        world.add_component(FrameInput::default());
        world.step();

        let visible_after_show = world
            .query::<Sprite>()
            .find(|s| s.asset_id == AssetId(11))
            .map(|s| s.visible)
            .unwrap();
        assert!(visible_after_show, "view sprite is visible after Show");

        // Hide it again.
        world.add_component(ViewCommand::Hide);
        world.add_component(FrameInput::default());
        world.step();

        let visible_after_hide = world
            .query::<Sprite>()
            .find(|s| s.asset_id == AssetId(11))
            .map(|s| s.visible)
            .unwrap();
        assert!(!visible_after_hide, "view sprite is hidden after Hide");
    }

    // A view-owned region is overlay UI authored in the reference canvas; when
    // the window differs from the reference the region is scaled, and the live
    // cursor must be mapped back into reference space to hit it. At a 2x
    // viewport, a click at the scaled on-screen rect fires; a click at the raw
    // reference coordinates (which no longer overlap the scaled rect) does not.
    fn frame_input_at(mx: f32, my: f32, viewport: [f32; 2]) -> FrameInput {
        FrameInput {
            mouse_x: mx,
            mouse_y: my,
            left_click: true,
            viewport,
            ..Default::default()
        }
    }

    fn overlay_region_world() -> World {
        let mut world = World::new_empty();
        let view_id = AssetId(30);
        world.add_component(View {
            asset_id: view_id,
            initial: true,
            fade_in_secs: 0.0,
        });
        // Reference-space rect [200,400] x [200,260].
        world.add_component(HitRegion {
            x: 200.0,
            y: 200.0,
            width: 200.0,
            height: 60.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "scene:7".to_string(),
            drag_handle: None,
            view: Some(view_id),
            disabled: false,
        });
        world.start().unwrap();
        world
    }

    #[test]
    fn view_owned_region_hit_tests_in_reference_space_when_scaled() {
        // 2x reference viewport: the reference center (300,230) maps on-screen
        // to (600,460). A click there inverse-maps back inside the rect → fires.
        let mut world = overlay_region_world();
        world.add_component(frame_input_at(600.0, 460.0, [2560.0, 1440.0]));
        world.step();
        assert!(
            world.query::<SceneCommand>().next().is_some(),
            "click at the scaled rect should fire the region"
        );

        // A click at the raw reference coordinates lands outside the scaled
        // rect at 2x, so it must not fire.
        let mut world = overlay_region_world();
        world.add_component(frame_input_at(300.0, 230.0, [2560.0, 1440.0]));
        world.step();
        assert!(
            world.query::<SceneCommand>().next().is_none(),
            "click at the unscaled coords should miss the scaled region"
        );
    }

    // While a view is active, underlying scene HitRegions don't fire.
    #[test]
    fn hit_region_filtered_when_view_is_active() {
        let mut world = World::new_empty();

        let view_id = AssetId(20);
        world.add_component(View {
            asset_id: view_id,
            initial: false,
            fade_in_secs: 0.0,
        });
        // A scene-level region (no view) that would normally fire.
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "scene:7".to_string(),
            drag_handle: None,
            view: None,
            disabled: false,
        });
        world.start().unwrap();

        // Show the view, then click where the scene-region is.
        world.add_component(ViewCommand::Show(view_id));
        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();

        let has_cmd = world.query::<SceneCommand>().next().is_some();
        assert!(
            !has_cmd,
            "scene-level region should not fire while view is active"
        );
    }

    #[test]
    fn fire_action_dispatches_view_variants() {
        // view:hide → ViewCommand::Hide
        let mut world = World::new_empty();
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "view:hide".to_string(),
            drag_handle: None,
            view: None,
            disabled: false,
        });
        world.start().unwrap();
        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();
        assert!(matches!(
            world.query::<ViewCommand>().next(),
            Some(ViewCommand::Hide)
        ));

        // view:show:42 → ViewCommand::Show(42)
        let mut world = World::new_empty();
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "view:show:42".to_string(),
            drag_handle: None,
            view: None,
            disabled: false,
        });
        world.start().unwrap();
        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();
        let cmd = world.query::<ViewCommand>().next().cloned();
        assert!(matches!(cmd, Some(ViewCommand::Show(AssetId(42)))));

        // view:toggle:43 → ViewCommand::Toggle(43)
        let mut world = World::new_empty();
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "view:toggle:43".to_string(),
            drag_handle: None,
            view: None,
            disabled: false,
        });
        world.start().unwrap();
        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();
        let cmd = world.query::<ViewCommand>().next().cloned();
        assert!(matches!(cmd, Some(ViewCommand::Toggle(AssetId(43)))));
    }

    #[test]
    fn fire_action_dispatches_setting_with_value_label() {
        // setting:vsync:next → SettingCommand carrying the region's label as the
        // value-label to update, and the parsed direction.
        let mut world = World::new_empty();
        let value_label = AssetId(99);
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: Some(value_label),
            hover_color: None,
            hover_scale: None,
            action: "setting:vsync:next".to_string(),
            drag_handle: None,
            view: None,
            disabled: false,
        });
        world.start().unwrap();
        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();
        let cmd = world.query::<SettingCommand>().next().cloned().unwrap();
        assert_eq!(cmd.setting, "vsync");
        assert_eq!(cmd.op, SettingOp::Next);
        assert_eq!(cmd.value_label, Some(value_label));

        // The :prev suffix parses to the reverse direction. The default
        // HitRegion is 100x40, so click within those bounds.
        let mut world = World::new_empty();
        world.add_component(HitRegion {
            action: "setting:vsync:prev".to_string(),
            ..Default::default()
        });
        world.start().unwrap();
        world.add_component(make_frame_input(50.0, 20.0, true));
        world.step();
        let cmd = world.query::<SettingCommand>().next().cloned().unwrap();
        assert_eq!(cmd.op, SettingOp::Prev);
    }

    // A region the engine disabled (e.g. a capability-gated settings row grayed
    // out at init) is inert: clicking where it sits fires nothing.
    #[test]
    fn disabled_region_does_not_fire() {
        let mut world = World::new_empty();
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "setting:ray_traced_reflections:next".to_string(),
            drag_handle: None,
            view: None,
            disabled: true,
        });
        world.start().unwrap();

        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();
        assert!(
            world.query::<SettingCommand>().next().is_none(),
            "a disabled region must not fire its action"
        );
    }

    #[test]
    fn slider_drag_pushes_set_fraction_then_persists_on_release() {
        let mut world = World::new_empty();
        let value_label = AssetId(7);
        world.add_component(HitRegion {
            x: 100.0,
            y: 0.0,
            width: 200.0,
            height: 40.0,
            label: Some(value_label),
            hover_color: None,
            hover_scale: None,
            action: "setting:exposure:drag".to_string(),
            drag_handle: Some(AssetId(8)),
            view: None,
            disabled: false,
        });
        world.start().unwrap();

        // Press at x=150 (25% across the [100, 300) track) with the button held:
        // a live, non-persisting fraction.
        world.add_component(FrameInput {
            mouse_x: 150.0,
            mouse_y: 20.0,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        });
        world.step();
        let cmd = world.query::<SettingCommand>().last().cloned().unwrap();
        assert_eq!(cmd.setting, "exposure");
        assert!(matches!(cmd.op, SettingOp::SetFraction(f) if (f - 0.25).abs() < 1.0e-5));
        assert_eq!(cmd.value_label, Some(value_label));
        assert!(
            !cmd.persist,
            "an in-progress drag applies live but does not persist"
        );

        // Release at x=250 (75%): the button up commits the final value and persists.
        world.add_component(FrameInput {
            mouse_x: 250.0,
            mouse_y: 20.0,
            left_click: false,
            left_button_down: false,
            ..Default::default()
        });
        world.step();
        let cmd = world.query::<SettingCommand>().last().cloned().unwrap();
        assert!(matches!(cmd.op, SettingOp::SetFraction(f) if (f - 0.75).abs() < 1.0e-5));
        assert!(cmd.persist, "release commits and persists the final value");
    }

    // A group-toggle click collapses the group's body rows (hiding their
    // elements) and flips the header's `+`/`-` prefix; the body's click region
    // then goes inert.
    #[test]
    fn group_toggle_collapses_body_and_updates_header() {
        let mut world = World::new_empty();
        let view = AssetId(50);
        let (header, body) = (AssetId(51), AssetId(52));
        world.add_component(View {
            asset_id: view,
            initial: true,
            fade_in_secs: 0.0,
        });
        world.add_component(panel_label(51, 100.0, view, "- Adv"));
        world.add_component(panel_label(52, 140.0, view, "Body"));
        // Header click region (toggles group 0).
        world.add_component(HitRegion {
            x: 0.0,
            y: 100.0,
            width: 300.0,
            height: 40.0,
            label: Some(header),
            hover_color: None,
            hover_scale: None,
            action: "group:toggle:0".to_string(),
            drag_handle: None,
            view: Some(view),
            disabled: false,
        });
        // Body click region (a settings action; a content region, so it is
        // bucketed into its row and gated by the collapse).
        world.add_component(HitRegion {
            x: 0.0,
            y: 140.0,
            width: 300.0,
            height: 40.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "setting:vsync:next".to_string(),
            drag_handle: None,
            view: Some(view),
            disabled: false,
        });
        world.add_component(ScrollPanel {
            view: Some(view),
            x: 0.0,
            y: 100.0,
            width: 300.0,
            height: 100.0,
            rows: vec![
                ScrollRow {
                    elements: vec![header],
                    base_y: 100.0,
                    height: 40.0,
                    group: -1,
                },
                ScrollRow {
                    elements: vec![body],
                    base_y: 140.0,
                    height: 40.0,
                    group: 0,
                },
            ],
            groups: vec![ScrollGroup {
                collapsed: false,
                header: Some(header),
                title: "Adv".to_string(),
            }],
            thumb: None,
            track: None,
            track_x: 0.0,
            track_y: 0.0,
            track_w: 0.0,
            track_h: 0.0,
        });
        world.start().unwrap();

        // Expanded after init: body shown, header reads "- Adv".
        assert!(label_field(&world, body, |l| l.visible));
        assert_eq!(label_field(&world, header, |l| l.content.clone()), "- Adv");

        // Click the header to collapse.
        world.add_component(make_frame_input(10.0, 120.0, true));
        world.step();
        assert!(!label_field(&world, body, |l| l.visible), "body hides");
        assert_eq!(label_field(&world, header, |l| l.content.clone()), "+ Adv");

        // The body's region is now inert: clicking where it was fires nothing.
        world.add_component(make_frame_input(10.0, 160.0, true));
        world.step();
        assert!(
            world.query::<SettingCommand>().next().is_none(),
            "a collapsed row's region does not fire"
        );
    }

    // The mouse wheel, with the cursor over the panel band, scrolls the content
    // up: the top row's element moves up by wheel-delta * speed (clamped).
    #[test]
    fn wheel_scrolls_panel_content() {
        let mut world = World::new_empty();
        let view = AssetId(60);
        let e0 = AssetId(61);
        world.add_component(View {
            asset_id: view,
            initial: true,
            fade_in_secs: 0.0,
        });
        world.add_component(panel_label(61, 0.0, view, "Row0"));
        // Three 40px rows (120px) in a 60px band -> overflows by 60px.
        world.add_component(ScrollPanel {
            view: Some(view),
            x: 0.0,
            y: 0.0,
            width: 300.0,
            height: 60.0,
            rows: vec![
                ScrollRow {
                    elements: vec![e0],
                    base_y: 0.0,
                    height: 40.0,
                    group: -1,
                },
                ScrollRow {
                    elements: vec![],
                    base_y: 40.0,
                    height: 40.0,
                    group: -1,
                },
                ScrollRow {
                    elements: vec![],
                    base_y: 80.0,
                    height: 40.0,
                    group: -1,
                },
            ],
            groups: vec![],
            thumb: None,
            track: None,
            track_x: 0.0,
            track_y: 0.0,
            track_w: 0.0,
            track_h: 0.0,
        });
        world.start().unwrap();
        assert_eq!(label_field(&world, e0, |l| l.y), 0.0);

        // Wheel down with the cursor inside the band: scroll = 10 * speed (2.0)
        // = 20 (within the 60px max), so the top row moves up by 20.
        world.add_component(FrameInput {
            mouse_x: 10.0,
            mouse_y: 10.0,
            scroll_delta: 10.0,
            ..Default::default()
        });
        world.step();
        assert_eq!(label_field(&world, e0, |l| l.y), -20.0);

        // Wheel far past the end clamps to the max (60px up), not further.
        world.add_component(FrameInput {
            mouse_x: 10.0,
            mouse_y: 10.0,
            scroll_delta: 1000.0,
            ..Default::default()
        });
        world.step();
        assert_eq!(label_field(&world, e0, |l| l.y), -60.0);
    }

    // A rebind row: a value TextLabel showing the current key + a HitRegion over
    // it whose action enters capture mode.
    fn rebind_world() -> (World, AssetId) {
        let mut world = World::new_empty();
        let value = AssetId(7);
        world.add_component(TextLabel {
            asset_id: value,
            font: None,
            content: "W".to_string(),
            x: 0.0,
            y: 0.0,
            color: [1.0, 1.0, 1.0],
            scale: 1.0,
            centered: false,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            view: None,
        });
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 40.0,
            label: Some(value),
            hover_color: None,
            hover_scale: None,
            action: "setting:key_forward:rebind".to_string(),
            drag_handle: None,
            view: None,
            disabled: false,
        });
        world.start().unwrap();
        (world, value)
    }

    // Clicking a rebind row enters capture (the value shows the prompt and no
    // command fires); the next pressed key binds it via a Rebind SettingCommand.
    #[test]
    fn rebind_click_captures_then_binds_next_key() {
        use crate::assets::Key;
        let (mut world, value) = rebind_world();

        // Click the rebind row: enters capture, value shows the prompt, and no
        // command is pushed yet.
        world.add_component(make_frame_input(50.0, 20.0, true));
        world.step();
        assert_eq!(
            label_field(&world, value, |l| l.content.clone()),
            REBIND_PROMPT
        );
        assert!(
            world.query::<SettingCommand>().next().is_none(),
            "no command until a key is pressed"
        );

        // Press a key: it binds, pushing a Rebind command carrying the key.
        world.add_component(FrameInput {
            captured_key: Some(Key::Q),
            ..Default::default()
        });
        world.step();
        let cmd = world.query::<SettingCommand>().next().cloned().unwrap();
        assert_eq!(cmd.setting, "key_forward");
        assert_eq!(cmd.value_label, Some(value));
        assert!(matches!(cmd.op, SettingOp::Rebind(Key::Q)));
        assert!(cmd.persist);
    }

    // Escape while capturing cancels and restores the row's previous value text.
    #[test]
    fn rebind_escape_cancels_and_restores() {
        let (mut world, value) = rebind_world();
        world.add_component(make_frame_input(50.0, 20.0, true));
        world.step();
        assert_eq!(
            label_field(&world, value, |l| l.content.clone()),
            REBIND_PROMPT
        );

        // Escape cancels: the previous text returns and nothing is bound.
        world.add_component(FrameInput {
            escape: true,
            ..Default::default()
        });
        world.step();
        assert_eq!(label_field(&world, value, |l| l.content.clone()), "W");
        assert!(world.query::<SettingCommand>().next().is_none());
    }

    // A captured key with no active capture binds nothing.
    #[test]
    fn captured_key_without_capture_is_ignored() {
        use crate::assets::Key;
        let (mut world, _value) = rebind_world();
        world.add_component(FrameInput {
            captured_key: Some(Key::Q),
            ..Default::default()
        });
        world.step();
        assert!(world.query::<SettingCommand>().next().is_none());
    }

    #[test]
    fn escape_key_binding_fires_action() {
        let mut world = World::new_empty();

        let view_id = AssetId(50);
        world.add_component(View {
            asset_id: view_id,
            initial: false,
            fade_in_secs: 0.0,
        });
        world.add_component(KeyBinding {
            key: "Escape".to_string(),
            action: "view:toggle:50".to_string(),
        });
        world.start().unwrap();

        // Press Escape.
        world.add_component(FrameInput {
            escape: true,
            ..Default::default()
        });
        world.step();

        let cmd = world.query::<ViewCommand>().next().cloned();
        assert!(matches!(cmd, Some(ViewCommand::Toggle(AssetId(50)))));
    }

    // A gating component (here a View) spawns the internal UiInputSystem.
    #[test]
    fn ui_component_spawns_internal_system() {
        let mut world = World::new_empty();
        world.add_component(View {
            asset_id: AssetId(1),
            initial: false,
            fade_in_secs: 0.0,
        });
        world.start().unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["UiInputSystem"]);
    }

    // No HitRegion / View / KeyBinding means no UiInputSystem.
    #[test]
    fn no_ui_components_means_no_system() {
        let mut world = World::new_empty();
        world.start().unwrap();
        assert!(world.systems().is_empty());
    }
}
