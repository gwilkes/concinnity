// src/check.rs
// Cross-asset reference declarations. The `CrossReferenced` trait and its data
// kinds live here so asset files in this crate can declare their references;
// the resolver, leaf checkers, and dispatch live in the build crate.

pub mod cross_reference;
