// src/app/sources/mod.rs
// `cn add` scaffold presets for 3D scene targets. Per-format entry generation
// (the `.glb` / `.fbx` fan-out) now lives in `concinnity_core::build::import`
// and runs at build time from a `SceneImport` asset; this module only holds the
// renderer scaffold a fresh scene add injects (see `glb::scaffold`).

pub mod glb;
