// src/anim_reload.rs
//
// Animation clip hot-reload: re-import each captured file-backed Animation from
// its source .glb and push the rebuilt clip into the live AnimationSystem. The
// GLB decode lives here (editor side) because the runtime crate links no image
// decoders; the runtime crate exposes only the catalogue (`reload_entries`) and
// the setter (`apply_reloaded_clip`). Mirrors the path the desugar pass takes at
// build time, so a hot-reloaded clip is byte-identical to a fresh `cn build`.

use std::collections::HashMap;

use crate::gfx::animation::AnimationSystem;
use crate::gfx::skinning::{AnimationClip, JointTrack, Keyframe};

// Re-import every file-backed clip when an asset-source change is pending.
// Driven by the debug server's per-frame tick. No-op when no file-backed clips
// were captured or no source change is pending.
pub(crate) fn reload_clips_if_pending(anim: &mut AnimationSystem) {
    if anim.reload_entries().is_empty()
        || !concinnity_client::app::dev_flags::take_pending_animations()
    {
        return;
    }
    reload_clips(anim);
}

fn reload_clips(anim: &mut AnimationSystem) {
    // Parse each unique source .glb once per reload; many clips can target the
    // same character file.
    let mut parsed_cache: HashMap<String, gltf::Gltf> = HashMap::new();
    let mut reloaded = 0usize;
    let mut failed = 0usize;

    // Snapshot the catalogue so the &mut setter can run while we iterate.
    let entries = anim.reload_entries().to_vec();
    for entry in &entries {
        let doc = match parsed_cache.get(&entry.source) {
            Some(d) => d,
            None => match concinnity_cook::glb::parse_glb(&entry.source) {
                Ok(d) => parsed_cache.entry(entry.source.clone()).or_insert(d),
                Err(e) => {
                    tracing::error!(
                        "animation hot-reload: failed to parse '{}': {} \
                         (clip slot {}:{} kept its old keyframes)",
                        entry.source,
                        e,
                        entry.target,
                        entry.clip_index
                    );
                    failed += 1;
                    continue;
                }
            },
        };
        let imported = match concinnity_cook::glb::import_glb_animation_from_doc(
            doc,
            &entry.source,
            entry.animation_index,
            &entry.animation_name,
        ) {
            Ok(a) => a,
            Err(e) => {
                tracing::error!(
                    "animation hot-reload: failed to import animation from '{}': {} \
                     (clip slot {}:{} kept its old keyframes)",
                    entry.source,
                    e,
                    entry.target,
                    entry.clip_index
                );
                failed += 1;
                continue;
            }
        };
        let clip = imported_to_clip(&imported, entry.looping);
        if anim.apply_reloaded_clip(entry.target, entry.clip_index, clip, entry.weight) {
            reloaded += 1;
        } else {
            tracing::error!(
                "animation hot-reload: target {} clip {} no longer present (skipped)",
                entry.target,
                entry.clip_index
            );
            failed += 1;
        }
    }
    tracing::info!(
        "animation hot-reload: reloaded {} clip(s) ({} failed)",
        reloaded,
        failed
    );
}

// Convert a build-side ImportedAnimation into the runtime AnimationClip form.
fn imported_to_clip(
    imported: &concinnity_cook::glb::ImportedAnimation,
    looping: bool,
) -> AnimationClip {
    AnimationClip {
        duration: imported.duration.max(1e-3),
        looping,
        tracks: imported
            .tracks
            .iter()
            .map(|t| JointTrack {
                joint: t.joint,
                keys: t
                    .keys
                    .iter()
                    .map(|k| Keyframe {
                        time: k.time,
                        pose: k.pose,
                    })
                    .collect(),
            })
            .collect(),
    }
}
