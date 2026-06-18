// src/audio/mod.rs
//
// A thin wrapper around the kira audio engine for 3D positional sound. kira
// (and its cpal / symphonia dependencies) is confined to this module: callers
// work entirely in the engine's `[f32; 3]` representation and the opaque
// `EmitterId`.
//
// `AudioEngine` owns one kira `AudioManager`, a single listener, and one
// spatial track per emitter. `AudioSystem` builds it at init and updates the
// listener / emitter poses every frame.
//
// When no audio output device is available the engine is built in a disabled
// state and every method becomes a no-op. This keeps headless / CI runs (which
// may have no sound card) from failing.

// The internal positional-audio system that drives `AudioEngine` from the
// world's `AudioEmitter` / `AudioClip` components.
pub(crate) mod system;

use std::io::Cursor;

use kira::listener::ListenerHandle;
use kira::sound::static_sound::StaticSoundData;
use kira::track::{SpatialTrackBuilder, SpatialTrackHandle};
use kira::{AudioManager, AudioManagerSettings, Decibels, DefaultBackend, Tween};

// Opaque handle to a spatial emitter inside an [`AudioEngine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EmitterId(usize);

// The live kira state. Present only when an output device was acquired.
struct Active {
    manager: AudioManager<DefaultBackend>,
    listener: ListenerHandle,
    // One spatial track per emitter, indexed by `EmitterId`.
    emitters: Vec<SpatialTrackHandle>,
}

// A 3D positional audio engine.
pub struct AudioEngine {
    // `None` when no output device was available; the engine is then inert.
    active: Option<Active>,
    // The last master gain requested via `set_master_volume` (linear; 1.0 =
    // unity). Recorded even when the engine is disabled, so it reflects the
    // requested master regardless of whether a device is present.
    last_master_volume: f32,
}

impl std::fmt::Debug for AudioEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioEngine")
            .field("enabled", &self.active.is_some())
            .field(
                "emitters",
                &self.active.as_ref().map_or(0, |a| a.emitters.len()),
            )
            .field("master_volume", &self.last_master_volume)
            .finish()
    }
}

impl AudioEngine {
    // Build the engine, acquiring the default output device. Returns a
    // disabled (no-op) engine when no device is available or kira fails to
    // start, so the caller never has to handle an error.
    pub fn new() -> AudioEngine {
        match Self::try_start() {
            Ok(active) => AudioEngine {
                active: Some(active),
                last_master_volume: 1.0,
            },
            Err(e) => {
                tracing::warn!("AudioEngine disabled: {e}");
                AudioEngine::disabled()
            }
        }
    }

    // An engine with no output device. Every method is a no-op.
    fn disabled() -> AudioEngine {
        AudioEngine {
            active: None,
            last_master_volume: 1.0,
        }
    }

    fn try_start() -> Result<Active, String> {
        let mut manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default())
            .map_err(|e| format!("audio manager init failed: {e}"))?;
        // The listener starts at the origin facing -Z (identity orientation);
        // AudioSystem moves it onto the camera on the first step.
        let listener = manager
            .add_listener(ORIGIN, IDENTITY_ORIENTATION)
            .map_err(|e| format!("listener init failed: {e}"))?;
        Ok(Active {
            manager,
            listener,
            emitters: Vec::new(),
        })
    }

    // Whether the engine acquired an output device.
    pub fn is_enabled(&self) -> bool {
        self.active.is_some()
    }

    // Add a spatial emitter at `position`. Returns `None` on a disabled
    // engine or when kira's emitter limit is reached.
    pub fn add_emitter(&mut self, position: [f32; 3]) -> Option<EmitterId> {
        let active = self.active.as_mut()?;
        let track = active
            .manager
            .add_spatial_sub_track(&active.listener, vec3(position), SpatialTrackBuilder::new())
            .map_err(|e| tracing::warn!("audio emitter add failed: {e}"))
            .ok()?;
        let id = EmitterId(active.emitters.len());
        active.emitters.push(track);
        Some(id)
    }

    // Start playing an encoded audio clip on an emitter. `encoded` is a whole
    // audio file (ogg / wav / flac / ...); `gain` is a linear volume
    // multiplier (1.0 leaves the clip unchanged). Returns false when the
    // engine is disabled or the clip could not be decoded / played.
    pub fn play_clip(&mut self, id: EmitterId, encoded: &[u8], looping: bool, gain: f32) -> bool {
        let Some(active) = self.active.as_mut() else {
            return false;
        };
        let Some(track) = active.emitters.get_mut(id.0) else {
            return false;
        };
        let mut data = match StaticSoundData::from_cursor(Cursor::new(encoded.to_vec())) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("audio clip decode failed: {e}");
                return false;
            }
        };
        data = data.volume(Decibels(gain_to_db(gain)));
        if looping {
            data = data.loop_region(..);
        }
        match track.play(data) {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!("audio clip play failed: {e}");
                false
            }
        }
    }

    // Move an emitter. No-op on a disabled engine or an unknown id.
    pub fn set_emitter_position(&mut self, id: EmitterId, position: [f32; 3]) {
        if let Some(active) = self.active.as_mut()
            && let Some(track) = active.emitters.get_mut(id.0)
        {
            track.set_position(vec3(position), Tween::default());
        }
    }

    // Set the master output volume as a linear gain (1.0 = unchanged). Scales
    // every emitter by adjusting the main mix track, so it applies to clips
    // already playing as well as future ones. No-op on a disabled engine. The
    // short default tween makes a live change click-free.
    pub fn set_master_volume(&mut self, gain: f32) {
        self.last_master_volume = gain;
        if let Some(active) = self.active.as_mut() {
            active
                .manager
                .main_track()
                .set_volume(Decibels(gain_to_db(gain)), Tween::default());
        }
    }

    // Update the listener pose from a camera position and yaw / pitch
    // (radians). No-op on a disabled engine.
    pub fn set_listener(&mut self, position: [f32; 3], yaw: f32, pitch: f32) {
        if let Some(active) = self.active.as_mut() {
            active
                .listener
                .set_position(vec3(position), Tween::default());
            active
                .listener
                .set_orientation(orientation_quat(yaw, pitch), Tween::default());
        }
    }
}

// Listener spawn position.
const ORIGIN: mint::Vector3<f32> = mint::Vector3 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

// Unrotated listener orientation (faces -Z).
const IDENTITY_ORIENTATION: mint::Quaternion<f32> = mint::Quaternion { s: 1.0, v: ORIGIN };

fn vec3(p: [f32; 3]) -> mint::Vector3<f32> {
    mint::Vector3 {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

// Convert a linear gain multiplier to decibels. A gain of 1.0 maps to 0 dB;
// gains at or below ~-80 dB are clamped so silence does not yield `-inf`.
fn gain_to_db(gain: f32) -> f32 {
    20.0 * gain.max(1.0e-4).log10()
}

// Build the listener orientation quaternion from camera yaw / pitch (radians).
//
// An unrotated kira listener faces -Z with +X right and +Y up, which matches
// the engine's camera basis (yaw 0, pitch 0 looks toward -Z). The result is
// the yaw rotation about +Y composed with the pitch rotation about +X.
fn orientation_quat(yaw: f32, pitch: f32) -> mint::Quaternion<f32> {
    let hy = yaw * 0.5;
    let hp = -pitch * 0.5;
    let (sy, cy) = hy.sin_cos();
    let (sp, cp) = hp.sin_cos();
    mint::Quaternion {
        s: cy * cp,
        v: mint::Vector3 {
            x: cy * sp,
            y: sy * cp,
            z: -sy * sp,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_engine_is_inert() {
        let mut engine = AudioEngine::disabled();
        assert!(!engine.is_enabled());
        assert_eq!(engine.add_emitter([0.0; 3]), None);
        // None of these may panic on a disabled engine.
        engine.set_emitter_position(EmitterId(0), [1.0, 2.0, 3.0]);
        engine.set_listener([4.0, 5.0, 6.0], 1.0, 0.5);
        // A disabled engine still records the requested master gain (it just
        // cannot apply it to a device).
        assert!((engine.last_master_volume - 1.0).abs() < 1.0e-6);
        engine.set_master_volume(0.5);
        assert!((engine.last_master_volume - 0.5).abs() < 1.0e-6);
        assert!(!engine.play_clip(EmitterId(0), &[], true, 1.0));
    }

    #[test]
    fn identity_orientation_at_zero_yaw_pitch() {
        let q = orientation_quat(0.0, 0.0);
        assert!((q.s - 1.0).abs() < 1.0e-6);
        assert!(q.v.x.abs() < 1.0e-6 && q.v.y.abs() < 1.0e-6 && q.v.z.abs() < 1.0e-6);
    }

    #[test]
    fn orientation_quaternion_stays_unit_length() {
        for &(yaw, pitch) in &[(0.5, 0.3), (-1.2, 0.8), (3.0, -0.6), (-2.7, -1.1)] {
            let q = orientation_quat(yaw, pitch);
            let len = (q.s * q.s + q.v.x * q.v.x + q.v.y * q.v.y + q.v.z * q.v.z).sqrt();
            assert!(
                (len - 1.0).abs() < 1.0e-5,
                "not unit ({len}) at {yaw}/{pitch}"
            );
        }
    }

    #[test]
    fn gain_to_db_reference_points() {
        assert!(gain_to_db(1.0).abs() < 1.0e-4); // unity -> 0 dB
        assert!((gain_to_db(0.5) - (-6.0206)).abs() < 0.01); // half -> ~-6 dB
        assert!(gain_to_db(0.0) < -60.0); // silence clamped low, not -inf
        assert!(gain_to_db(0.0).is_finite());
    }
}
