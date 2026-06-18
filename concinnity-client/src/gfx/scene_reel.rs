// src/gfx/scene_reel.rs
//
// Platform-agnostic SceneReel state and transition logic. The SceneControl
// trait decouples this module from any specific backend; callers supply a
// concrete backend that implements the two mutation methods.

use crate::ecs::asset_id::AssetId;

const FADE_HALF_SECS: f32 = 0.3;

pub(crate) struct ReelEntry {
    pub(crate) scene: AssetId,
    pub(crate) duration_secs: Option<f32>,
    pub(crate) transition: String,
}

pub(crate) struct ReelState {
    pub(crate) entries: Vec<ReelEntry>,
    pub(crate) current_idx: usize,
    pub(crate) looping: bool,
    // Value of elapsed at the moment the current scene became active.
    pub(crate) scene_started_at: f32,
    pub(crate) fade: FadePhase,
    // Clear colour before any fade was applied (restored after fade-in).
    pub(crate) base_clear_color: [f32; 4],
}

pub(crate) enum FadePhase {
    None,
    // Fading clear_color toward black; new_idx is the scene to activate mid-fade.
    ToBlack { started_at: f32, new_idx: usize },
    // New scene is active; fading clear_color back from black.
    FromBlack { started_at: f32 },
}

// Backend operations required to drive scene visibility and fade transitions.
pub trait SceneControl {
    fn update_visibility(&mut self, draw_idx: usize, visible: bool);
    fn update_clear_color(&mut self, color: [f32; 4]);
}

// Set draw-object visibility according to which scene is currently active.
// Props with no scene association (prop_scene[i] == None) are always visible.
pub(crate) fn set_scene_visibility<B: SceneControl + ?Sized>(
    prop_draw_indices: &[Vec<usize>],
    prop_scene: &[Option<AssetId>],
    active_scene: AssetId,
    backend: &mut B,
) {
    for (prop_idx, scene_opt) in prop_scene.iter().enumerate() {
        let visible = match scene_opt {
            None => true,
            Some(s) => *s == active_scene,
        };
        if let Some(draw_idxs) = prop_draw_indices.get(prop_idx) {
            for &draw_idx in draw_idxs {
                backend.update_visibility(draw_idx, visible);
            }
        }
    }
}

// Advance SceneReel state, update clear colour for fades, and switch
// visibility when the active scene changes.
pub(crate) fn tick_reel<B: SceneControl + ?Sized>(
    reel_opt: &mut Option<ReelState>,
    prop_draw_indices: &[Vec<usize>],
    prop_scene: &[Option<AssetId>],
    elapsed: f32,
    backend: &mut B,
) {
    let reel = match reel_opt {
        Some(r) => r,
        None => return,
    };

    match reel.fade {
        FadePhase::ToBlack {
            started_at,
            new_idx,
        } => {
            let t = ((elapsed - started_at) / FADE_HALF_SECS).clamp(0.0, 1.0);
            let [r, g, b, a] = reel.base_clear_color;
            backend.update_clear_color([r * (1.0 - t), g * (1.0 - t), b * (1.0 - t), a]);
            if t >= 1.0 {
                let new_scene = reel.entries[new_idx].scene;
                reel.current_idx = new_idx;
                reel.scene_started_at = elapsed;
                reel.fade = FadePhase::FromBlack {
                    started_at: elapsed,
                };
                set_scene_visibility(prop_draw_indices, prop_scene, new_scene, backend);
                tracing::debug!("SceneReel: switched to scene {}", new_scene);
            }
            return;
        }
        FadePhase::FromBlack { started_at } => {
            let t = ((elapsed - started_at) / FADE_HALF_SECS).clamp(0.0, 1.0);
            let [r, g, b, a] = reel.base_clear_color;
            backend.update_clear_color([r * t, g * t, b * t, a]);
            if t >= 1.0 {
                backend.update_clear_color(reel.base_clear_color);
                reel.fade = FadePhase::None;
            }
            return;
        }
        FadePhase::None => {}
    }

    let duration = match reel.entries[reel.current_idx].duration_secs {
        Some(d) => d,
        None => return, // hold indefinitely
    };
    if elapsed - reel.scene_started_at < duration {
        return;
    }

    let next_idx = reel.current_idx + 1;
    let next_idx = if next_idx >= reel.entries.len() {
        if reel.looping {
            0
        } else {
            return;
        }
    } else {
        next_idx
    };

    let transition = reel.entries[next_idx].transition.clone();
    match transition.as_str() {
        "FadeBlack" => {
            reel.fade = FadePhase::ToBlack {
                started_at: elapsed,
                new_idx: next_idx,
            };
        }
        _ => {
            let new_scene = reel.entries[next_idx].scene;
            reel.current_idx = next_idx;
            reel.scene_started_at = elapsed;
            set_scene_visibility(prop_draw_indices, prop_scene, new_scene, backend);
            tracing::debug!("SceneReel: cut to scene {}", new_scene);
        }
    }
}

// Imperatively jump to a named scene, bypassing the timed SceneReel advance.
// Ignored with a warning if the target scene is not in the reel, or no reel exists.
pub(crate) fn jump_to_scene<B: SceneControl + ?Sized>(
    reel_opt: &mut Option<ReelState>,
    prop_draw_indices: &[Vec<usize>],
    prop_scene: &[Option<AssetId>],
    elapsed: f32,
    target_scene: AssetId,
    transition: &str,
    backend: &mut B,
) {
    let reel = match reel_opt {
        Some(r) => r,
        None => {
            tracing::warn!(
                "SceneCommand: jump to {} ignored -- no SceneReel in world",
                target_scene
            );
            return;
        }
    };

    let new_idx = match reel.entries.iter().position(|e| e.scene == target_scene) {
        Some(i) => i,
        None => {
            tracing::warn!(
                "SceneCommand: jump to {} ignored -- scene not found in reel",
                target_scene
            );
            return;
        }
    };

    if new_idx == reel.current_idx {
        return;
    }

    match transition {
        "FadeBlack" => {
            reel.fade = FadePhase::ToBlack {
                started_at: elapsed,
                new_idx,
            };
        }
        _ => {
            let new_scene = reel.entries[new_idx].scene;
            reel.current_idx = new_idx;
            reel.scene_started_at = elapsed;
            reel.fade = FadePhase::None;
            set_scene_visibility(prop_draw_indices, prop_scene, new_scene, backend);
            tracing::debug!("SceneCommand: cut to scene {}", new_scene);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal SceneControl implementation that records every call.
    #[derive(Default)]
    struct TestBackend {
        visibility: Vec<(usize, bool)>,
        clear_colors: Vec<[f32; 4]>,
    }

    impl SceneControl for TestBackend {
        fn update_visibility(&mut self, draw_idx: usize, visible: bool) {
            self.visibility.push((draw_idx, visible));
        }
        fn update_clear_color(&mut self, color: [f32; 4]) {
            self.clear_colors.push(color);
        }
    }

    fn make_reel(scenes: &[AssetId], looping: bool) -> ReelState {
        ReelState {
            entries: scenes
                .iter()
                .map(|&s| ReelEntry {
                    scene: s,
                    duration_secs: Some(1.0),
                    transition: "Cut".to_string(),
                })
                .collect(),
            current_idx: 0,
            looping,
            scene_started_at: 0.0,
            fade: FadePhase::None,
            base_clear_color: [1.0, 1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn set_visibility_active_scene_visible_others_hidden() {
        // Three props: one in "a", one with no scene, one in "b".
        let indices: Vec<Vec<usize>> = vec![vec![0], vec![1], vec![2]];
        let scenes: Vec<Option<AssetId>> = vec![Some(AssetId(0)), None, Some(AssetId(1))];
        let mut backend = TestBackend::default();
        set_scene_visibility(&indices, &scenes, AssetId(0), &mut backend);

        assert!(
            backend.visibility.contains(&(0, true)),
            "prop in 'a' should be visible"
        );
        assert!(
            backend.visibility.contains(&(1, true)),
            "scene-less prop always visible"
        );
        assert!(
            backend.visibility.contains(&(2, false)),
            "prop in 'b' should be hidden"
        );
    }

    #[test]
    fn set_visibility_no_scene_always_visible_regardless_of_active() {
        let indices: Vec<Vec<usize>> = vec![vec![0]];
        let scenes: Vec<Option<AssetId>> = vec![None];
        let mut backend = TestBackend::default();
        set_scene_visibility(&indices, &scenes, AssetId(99), &mut backend);
        assert_eq!(backend.visibility, vec![(0, true)]);
    }

    #[test]
    fn tick_reel_none_duration_holds_indefinitely() {
        let mut reel = make_reel(&[AssetId(0)], false);
        reel.entries[0].duration_secs = None;
        let mut opt = Some(reel);
        let mut backend = TestBackend::default();
        tick_reel(&mut opt, &[], &[], 999.0, &mut backend);
        // No scene switch, no visibility changes.
        assert!(backend.visibility.is_empty());
        assert_eq!(opt.as_ref().unwrap().current_idx, 0);
    }

    #[test]
    fn tick_reel_holds_until_duration_exceeded() {
        let mut opt = Some(make_reel(&[AssetId(0), AssetId(1)], false));
        let mut backend = TestBackend::default();
        tick_reel(&mut opt, &[], &[], 0.5, &mut backend); // 0.5 < 1.0
        assert_eq!(opt.as_ref().unwrap().current_idx, 0);
        assert!(backend.visibility.is_empty());
    }

    #[test]
    fn tick_reel_cut_advances_to_next_scene() {
        // Two props, one per scene, to verify visibility is updated on cut.
        let indices: Vec<Vec<usize>> = vec![vec![0], vec![1]];
        let scenes: Vec<Option<AssetId>> = vec![Some(AssetId(0)), Some(AssetId(1))];
        let mut opt = Some(make_reel(&[AssetId(0), AssetId(1)], false));
        let mut backend = TestBackend::default();
        tick_reel(&mut opt, &indices, &scenes, 2.0, &mut backend); // 2.0 > 1.0
        assert_eq!(opt.as_ref().unwrap().current_idx, 1);
        assert!(backend.visibility.contains(&(0, false)), "old scene hidden");
        assert!(backend.visibility.contains(&(1, true)), "new scene visible");
    }

    #[test]
    fn tick_reel_looping_wraps_to_start() {
        let mut reel = make_reel(&[AssetId(0), AssetId(1)], true);
        reel.current_idx = 1; // at last entry
        let mut opt = Some(reel);
        let mut backend = TestBackend::default();
        tick_reel(&mut opt, &[], &[], 2.0, &mut backend);
        assert_eq!(opt.as_ref().unwrap().current_idx, 0);
    }

    #[test]
    fn tick_reel_non_looping_stops_at_last() {
        let mut reel = make_reel(&[AssetId(0), AssetId(1)], false);
        reel.current_idx = 1;
        let mut opt = Some(reel);
        let mut backend = TestBackend::default();
        tick_reel(&mut opt, &[], &[], 2.0, &mut backend);
        assert_eq!(opt.as_ref().unwrap().current_idx, 1);
    }

    #[test]
    fn tick_reel_fade_to_black_darkens_clear_color() {
        let mut reel = make_reel(&[AssetId(0), AssetId(1)], false);
        reel.base_clear_color = [1.0, 0.0, 0.0, 1.0];
        reel.fade = FadePhase::ToBlack {
            started_at: 0.0,
            new_idx: 1,
        };
        let mut opt = Some(reel);
        let mut backend = TestBackend::default();
        // elapsed = FADE_HALF_SECS / 2 → t = 0.5
        tick_reel(&mut opt, &[], &[], FADE_HALF_SECS * 0.5, &mut backend);
        assert_eq!(backend.clear_colors.len(), 1);
        let [r, _g, _b, a] = backend.clear_colors[0];
        assert!((r - 0.5).abs() < 1e-5, "red should be half-dimmed");
        assert!((a - 1.0).abs() < 1e-5, "alpha unchanged");
        // Still in ToBlack, no scene switch yet.
        assert!(matches!(
            opt.as_ref().unwrap().fade,
            FadePhase::ToBlack { .. }
        ));
    }

    #[test]
    fn tick_reel_fade_to_black_completes_and_enters_from_black() {
        let mut reel = make_reel(&[AssetId(0), AssetId(1)], false);
        reel.fade = FadePhase::ToBlack {
            started_at: 0.0,
            new_idx: 1,
        };
        let mut opt = Some(reel);
        let mut backend = TestBackend::default();
        // elapsed = FADE_HALF_SECS → t = 1.0, scene switches
        tick_reel(&mut opt, &[], &[], FADE_HALF_SECS, &mut backend);
        let r = opt.as_ref().unwrap();
        assert_eq!(r.current_idx, 1);
        assert!(matches!(r.fade, FadePhase::FromBlack { .. }));
    }

    #[test]
    fn tick_reel_fade_from_black_restores_clear_color() {
        let mut reel = make_reel(&[AssetId(0)], false);
        reel.base_clear_color = [1.0, 1.0, 1.0, 1.0];
        reel.fade = FadePhase::FromBlack { started_at: 0.0 };
        let mut opt = Some(reel);
        let mut backend = TestBackend::default();
        // elapsed = FADE_HALF_SECS → t = 1.0, fade ends
        tick_reel(&mut opt, &[], &[], FADE_HALF_SECS, &mut backend);
        assert!(matches!(opt.as_ref().unwrap().fade, FadePhase::None));
        // The final clear_color call restores the base color.
        assert_eq!(*backend.clear_colors.last().unwrap(), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn jump_to_scene_no_reel_is_no_op() {
        let mut opt: Option<ReelState> = None;
        let mut backend = TestBackend::default();
        jump_to_scene(&mut opt, &[], &[], 0.0, AssetId(99), "Cut", &mut backend);
        assert!(backend.visibility.is_empty());
    }

    #[test]
    fn jump_to_scene_same_scene_is_no_op() {
        let mut opt = Some(make_reel(&[AssetId(0), AssetId(1)], false));
        let mut backend = TestBackend::default();
        jump_to_scene(&mut opt, &[], &[], 0.0, AssetId(0), "Cut", &mut backend);
        assert_eq!(opt.as_ref().unwrap().current_idx, 0);
        assert!(backend.visibility.is_empty());
    }

    #[test]
    fn jump_to_scene_cut_switches_immediately() {
        let indices: Vec<Vec<usize>> = vec![vec![0], vec![1]];
        let scenes: Vec<Option<AssetId>> = vec![Some(AssetId(0)), Some(AssetId(1))];
        let mut opt = Some(make_reel(&[AssetId(0), AssetId(1)], false));
        let mut backend = TestBackend::default();
        jump_to_scene(
            &mut opt,
            &indices,
            &scenes,
            1.0,
            AssetId(1),
            "Cut",
            &mut backend,
        );
        assert_eq!(opt.as_ref().unwrap().current_idx, 1);
        assert!(matches!(opt.as_ref().unwrap().fade, FadePhase::None));
        assert!(backend.visibility.contains(&(1, true)));
    }

    #[test]
    fn jump_to_scene_fade_black_starts_to_black_phase() {
        let mut opt = Some(make_reel(&[AssetId(0), AssetId(1)], false));
        let mut backend = TestBackend::default();
        jump_to_scene(
            &mut opt,
            &[],
            &[],
            5.0,
            AssetId(1),
            "FadeBlack",
            &mut backend,
        );
        // current_idx not changed yet; scene switches mid-fade
        assert_eq!(opt.as_ref().unwrap().current_idx, 0);
        assert!(matches!(
            opt.as_ref().unwrap().fade,
            FadePhase::ToBlack { started_at, new_idx: 1 } if (started_at - 5.0).abs() < 1e-6
        ));
    }
}
