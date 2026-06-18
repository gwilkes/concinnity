// src/gfx/particles.rs
//
// Backend-agnostic resolution of `ParticleEmitter` components into the
// `ParticleEmitterRecord`s the backends consume. Each record carries the
// clamped emitter tunables, the resolved texture pool slot, and the per-frame
// uniform builder the GPU compute + render passes share. Pure CPU; the
// per-emitter GPU buffers themselves are allocated by the backend at init.

use crate::assets::ParticleEmitter;
use crate::gfx::render_types::ParticleParams;

// Upper bound on the per-emitter pool the backend will allocate. Each slot
// is 32 bytes on the GPU (matching `Particle` in `metal/shaders/particle.metal`),
// so 65 536 slots = 2 MiB per emitter, already well past the visual point
// of diminishing returns for a billboard pool.
pub const MAX_PARTICLES_PER_EMITTER: u32 = 65_536;

// Hard floor on lifetime so the per-particle `age / lifetime` ratio never
// divides by zero in the render kernel.
const MIN_LIFETIME: f32 = 0.001;

// Resolved per-emitter state threaded into the backend at init. The backend
// allocates one GPU particle pool of `max_particles` slots per record and
// drives its compute + render passes from these fields each frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleEmitterRecord {
    // Index of the albedo texture in the renderer's bindless / per-frame
    // texture pool. `0` means "no texture authored": the renderer's white
    // fallback at slot 0 is sampled and the colour gradient still shows.
    pub texture_slot: usize,
    // World-space spawn origin.
    pub position: [f32; 3],
    // Mean emission direction, unit-length. The compute kernel samples a
    // new particle's initial velocity from the cone of half-angle
    // `spread_cos` around this vector.
    pub direction: [f32; 3],
    // Cosine of the cone half-angle. `1.0` = straight jet, `-1.0` = full
    // sphere. Pre-computed so the kernel does not call `cos()` per spawn.
    pub spread_cos: f32,
    // Inclusive lower bound on the initial particle speed (m/s).
    pub speed_min: f32,
    // Inclusive upper bound on the initial particle speed (m/s).
    pub speed_max: f32,
    // Inclusive lower bound on the particle lifetime (seconds).
    pub lifetime_min: f32,
    // Inclusive upper bound on the particle lifetime (seconds).
    pub lifetime_max: f32,
    // Constant acceleration applied each frame, in m/s².
    pub gravity: [f32; 3],
    // Particles spawned per second.
    pub spawn_rate: f32,
    // Pool size in slots. Live + dead particles share this fixed pool.
    pub max_particles: u32,
    // World-space billboard side length at `age = 0` (m).
    pub size_start: f32,
    // World-space billboard side length at `age = lifetime` (m).
    pub size_end: f32,
    // Linear-space RGBA at `age = 0`.
    pub color_start: [f32; 4],
    // Linear-space RGBA at `age = lifetime`.
    pub color_end: [f32; 4],
}

#[allow(dead_code)] // Metal-only particle pipeline consumer; DirectX / Vulkan don't draw particles yet.
impl ParticleEmitterRecord {
    // Conservative world-space AABB enclosing every particle this emitter
    // could spawn over its full lifetime. Used by the per-frame frustum-cull
    // skip so an off-screen emitter pays no _render_ cost; the compute
    // kernel still ticks so the pool keeps evolving while the camera looks
    // away.
    //
    // The bound is a sphere centred on the emission point and is intentionally
    // loose: it ignores the cone-spread restriction (`spread_cos`) so the
    // same AABB also covers full-sphere emitters, and it sums the worst-case
    // ballistic terms: `speed_max * lifetime_max` (straight-line reach),
    // `0.5 * |gravity| * lifetime_max²` (gravity drift), and a
    // `max(size_start, size_end) * sqrt(2) / 2` half-diagonal for the
    // camera-facing billboard quad. A tighter cone-aware bound is a future
    // refinement; this version is correct (never false-cull) and cheap.
    pub fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        let speed_reach = self.speed_max * self.lifetime_max;
        let gx = self.gravity[0];
        let gy = self.gravity[1];
        let gz = self.gravity[2];
        let g_mag = (gx * gx + gy * gy + gz * gz).sqrt();
        let g_drift = 0.5 * g_mag * self.lifetime_max * self.lifetime_max;
        let max_size = self.size_start.max(self.size_end);
        // The billboard quad is a square of side `size`, viewed any way; the
        // bounding sphere of a unit square has radius sqrt(2)/2.
        let billboard_radius = 0.5 * max_size * std::f32::consts::SQRT_2;
        let r = speed_reach + g_drift + billboard_radius;
        let c = self.position;
        (
            [c[0] - r, c[1] - r, c[2] - r],
            [c[0] + r, c[1] + r, c[2] + r],
        )
    }

    // Build the per-frame compute + render uniform from this record's static
    // fields and the dynamic spawn / time state the runtime carries.
    //
    // `dt` is the elapsed seconds since the previous compute dispatch (the
    // integration step); `spawn_budget` is `floor(spawn_accumulator)`, the
    // integer count of fresh particles the kernel may emit this frame; and
    // `random_seed` is the per-frame seed the kernel mixes with the thread
    // id to drive its cheap on-GPU RNG.
    pub fn params(&self, dt: f32, spawn_budget: u32, random_seed: u32) -> ParticleParams {
        ParticleParams {
            position: self.position,
            spread_cos: self.spread_cos,
            direction: self.direction,
            speed_min: self.speed_min,
            gravity: self.gravity,
            speed_max: self.speed_max,
            color_start: self.color_start,
            color_end: self.color_end,
            lifetime_min: self.lifetime_min,
            lifetime_max: self.lifetime_max,
            size_start: self.size_start,
            size_end: self.size_end,
            dt: dt.max(0.0),
            spawn_budget,
            random_seed,
            max_particles: self.max_particles,
        }
    }
}

// Resolve a list of `ParticleEmitter` components into `ParticleEmitterRecord`s
// the backend can consume. Skips invisible emitters and emitters whose pool
// would be empty. An emitter referencing an unknown texture is logged and
// dropped; an emitter with no `texture` falls back to slot 0 (white).
pub fn build_particle_records(
    emitters: &[&ParticleEmitter],
    texture_name_to_slot: &std::collections::HashMap<crate::ecs::asset_id::AssetId, usize>,
) -> Vec<ParticleEmitterRecord> {
    let mut out = Vec::new();
    for e in emitters {
        if !e.visible {
            continue;
        }
        let max_particles = e.max_particles.clamp(1, MAX_PARTICLES_PER_EMITTER);
        let slot = match e.texture {
            None => 0,
            Some(tex_id) => match texture_name_to_slot.get(&tex_id) {
                Some(&s) => s,
                None => {
                    tracing::error!(
                        "GraphicsSystem: ParticleEmitter {} references unknown texture {}",
                        e.asset_id,
                        tex_id
                    );
                    continue;
                }
            },
        };
        let direction = normalise_direction(e.direction);
        let spread_cos = (e.spread_deg.clamp(0.0, 180.0).to_radians()).cos();
        let lifetime_min = e.lifetime_min.max(MIN_LIFETIME);
        let lifetime_max = e.lifetime_max.max(lifetime_min);
        let speed_min = e.speed_min.max(0.0);
        let speed_max = e.speed_max.max(speed_min);
        out.push(ParticleEmitterRecord {
            texture_slot: slot,
            position: e.position,
            direction,
            spread_cos,
            speed_min,
            speed_max,
            lifetime_min,
            lifetime_max,
            gravity: e.gravity,
            spawn_rate: e.spawn_rate.max(0.0),
            max_particles,
            size_start: e.size_start.max(0.0),
            size_end: e.size_end.max(0.0),
            color_start: sanitised_color(e.color_start),
            color_end: sanitised_color(e.color_end),
        });
    }
    out
}

fn normalise_direction(d: [f32; 3]) -> [f32; 3] {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if !len.is_finite() || len < 1e-6 {
        // A zero / non-finite direction falls back to world-up so the cone
        // still has a well-defined axis. The asset-side default is `[0, 1, 0]`,
        // so this only matters for hand-built records or pathological JSON.
        [0.0, 1.0, 0.0]
    } else {
        [d[0] / len, d[1] / len, d[2] / len]
    }
}

fn sanitised_color(c: [f32; 4]) -> [f32; 4] {
    let mut out = c;
    for x in out.iter_mut() {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
    out
}

// Per-emitter spawn accumulator. The runtime keeps one of these per record
// and feeds the integer overflow into the compute kernel as `spawn_budget`
// each frame. Fractional carry-over keeps low spawn rates honest even when
// the frame-time is below the per-particle interval.
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)] // Metal-only particle runtime; DirectX / Vulkan ignore.
pub struct ParticleSpawnState {
    // Fractional particles owed by this emitter, carried forward across
    // frames. Cleared by `take_budget` after harvesting the integer part.
    pub accumulator: f32,
}

#[allow(dead_code)] // see ParticleSpawnState: Metal-only consumer.
impl ParticleSpawnState {
    // Add this frame's spawn allotment and pop off the integer part. Returns
    // `0` when the emitter is paused (`spawn_rate <= 0`).
    pub fn take_budget(&mut self, dt: f32, spawn_rate: f32, max_particles: u32) -> u32 {
        if spawn_rate <= 0.0 || dt <= 0.0 || !dt.is_finite() {
            return 0;
        }
        self.accumulator += spawn_rate * dt;
        // A pool of `max_particles` slots cannot absorb more than that many
        // spawns in a single dispatch; extra budget would just churn the RNG
        // for nothing.
        let max_per_frame = max_particles as f32;
        if self.accumulator > max_per_frame {
            self.accumulator = max_per_frame;
        }
        let whole = self.accumulator.floor() as u32;
        self.accumulator -= whole as f32;
        whole
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::ParticleEmitter;

    #[test]
    fn invisible_emitter_is_skipped() {
        let e = ParticleEmitter {
            visible: false,
            ..Default::default()
        };
        let names = std::collections::HashMap::new();
        assert!(build_particle_records(&[&e], &names).is_empty());
    }

    #[test]
    fn missing_texture_drops_emitter() {
        let e = ParticleEmitter {
            texture: Some(crate::ecs::asset_id::AssetId(999)),
            ..Default::default()
        };
        let names = std::collections::HashMap::new();
        assert!(build_particle_records(&[&e], &names).is_empty());
    }

    #[test]
    fn emitter_without_texture_uses_fallback_slot() {
        let e = ParticleEmitter::default();
        let names = std::collections::HashMap::new();
        let recs = build_particle_records(&[&e], &names);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].texture_slot, 0);
    }

    #[test]
    fn build_normalises_direction() {
        let e = ParticleEmitter {
            direction: [0.0, 5.0, 0.0],
            ..Default::default()
        };
        let names = std::collections::HashMap::new();
        let recs = build_particle_records(&[&e], &names);
        let d = recs[0].direction;
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn build_falls_back_for_zero_direction() {
        let e = ParticleEmitter {
            direction: [0.0, 0.0, 0.0],
            ..Default::default()
        };
        let names = std::collections::HashMap::new();
        let recs = build_particle_records(&[&e], &names);
        assert_eq!(recs[0].direction, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn build_clamps_max_particles_to_engine_cap() {
        let e = ParticleEmitter {
            max_particles: u32::MAX,
            ..Default::default()
        };
        let names = std::collections::HashMap::new();
        let recs = build_particle_records(&[&e], &names);
        assert_eq!(recs[0].max_particles, MAX_PARTICLES_PER_EMITTER);
    }

    #[test]
    fn build_precomputes_spread_cosine() {
        let e = ParticleEmitter {
            spread_deg: 0.0,
            ..Default::default()
        };
        let names = std::collections::HashMap::new();
        let recs = build_particle_records(&[&e], &names);
        assert!((recs[0].spread_cos - 1.0).abs() < 1e-6);
    }

    #[test]
    fn build_lifts_lifetime_max_to_min() {
        let e = ParticleEmitter {
            lifetime_min: 3.0,
            lifetime_max: 0.1,
            ..Default::default()
        };
        let names = std::collections::HashMap::new();
        let recs = build_particle_records(&[&e], &names);
        assert!(recs[0].lifetime_max >= recs[0].lifetime_min);
    }

    #[test]
    fn spawn_state_emits_integer_budget() {
        let mut s = ParticleSpawnState::default();
        let b = s.take_budget(0.5, 10.0, 100);
        assert_eq!(b, 5);
        let b = s.take_budget(0.5, 10.0, 100);
        assert_eq!(b, 5);
    }

    #[test]
    fn spawn_state_carries_fraction_across_frames() {
        let mut s = ParticleSpawnState::default();
        // 1.5 particles/frame; should alternate 1, 2, 1, 2 in the budget.
        let dt = 1.0;
        let rate = 1.5;
        let cap = 100;
        let mut total = 0;
        for _ in 0..4 {
            total += s.take_budget(dt, rate, cap);
        }
        assert_eq!(total, 6);
    }

    #[test]
    fn spawn_state_zero_rate_returns_zero() {
        let mut s = ParticleSpawnState::default();
        assert_eq!(s.take_budget(1.0, 0.0, 100), 0);
        assert_eq!(s.accumulator, 0.0);
    }

    #[test]
    fn spawn_state_caps_at_pool_capacity() {
        let mut s = ParticleSpawnState::default();
        // 1000 particles/sec, but pool only holds 10 slots: the kernel can
        // never absorb more than `max_particles` in a single dispatch.
        let b = s.take_budget(1.0, 1000.0, 10);
        assert!(b <= 10);
    }

    fn make_record(position: [f32; 3]) -> ParticleEmitterRecord {
        ParticleEmitterRecord {
            texture_slot: 0,
            position,
            direction: [0.0, 1.0, 0.0],
            spread_cos: 1.0,
            speed_min: 1.0,
            speed_max: 2.0,
            lifetime_min: 1.0,
            lifetime_max: 2.0,
            gravity: [0.0, 0.0, 0.0],
            spawn_rate: 32.0,
            max_particles: 64,
            size_start: 0.1,
            size_end: 0.1,
            color_start: [1.0; 4],
            color_end: [1.0; 4],
        }
    }

    #[test]
    fn aabb_centres_on_emission_origin() {
        let r = make_record([3.0, 4.0, -5.0]);
        let (mn, mx) = r.aabb();
        let cx = 0.5 * (mn[0] + mx[0]);
        let cy = 0.5 * (mn[1] + mx[1]);
        let cz = 0.5 * (mn[2] + mx[2]);
        assert!((cx - 3.0).abs() < 1e-5);
        assert!((cy - 4.0).abs() < 1e-5);
        assert!((cz + 5.0).abs() < 1e-5);
    }

    #[test]
    fn aabb_radius_covers_speed_reach() {
        // No gravity, no billboard size: radius should equal speed_max *
        // lifetime_max = 2 * 2 = 4 (plus a small billboard term ~0.07).
        let mut r = make_record([0.0; 3]);
        r.size_start = 0.0;
        r.size_end = 0.0;
        r.gravity = [0.0; 3];
        let (mn, mx) = r.aabb();
        // Half-extent on each axis equals the radius.
        let radius = 0.5 * (mx[0] - mn[0]);
        let expected = r.speed_max * r.lifetime_max; // = 4.0
        assert!((radius - expected).abs() < 1e-5);
    }

    #[test]
    fn aabb_includes_gravity_drift() {
        // Gravity 10 m/s² for 2 s integrates to 0.5 * 10 * 4 = 20 m drift,
        // dominates over the 4 m speed reach. Total radius ≈ 24.
        let mut r = make_record([0.0; 3]);
        r.size_start = 0.0;
        r.size_end = 0.0;
        r.gravity = [0.0, -10.0, 0.0];
        let (mn, mx) = r.aabb();
        let radius = 0.5 * (mx[0] - mn[0]);
        let expected = r.speed_max * r.lifetime_max + 0.5 * 10.0 * r.lifetime_max.powi(2);
        assert!((radius - expected).abs() < 1e-4);
    }

    #[test]
    fn aabb_includes_billboard_size() {
        // No motion, no gravity: only the billboard half-diagonal matters.
        // size = 1.0 → half-diag = sqrt(2)/2 ≈ 0.707.
        let mut r = make_record([0.0; 3]);
        r.speed_min = 0.0;
        r.speed_max = 0.0;
        r.gravity = [0.0; 3];
        r.size_start = 1.0;
        r.size_end = 1.0;
        let (mn, mx) = r.aabb();
        let radius = 0.5 * (mx[0] - mn[0]);
        let expected = 0.5 * 1.0 * std::f32::consts::SQRT_2;
        assert!((radius - expected).abs() < 1e-5);
    }
}
