// src/build/mod.rs
//
// Asset payload format helpers + the shared build-time types. The asset COMPILE
// pipeline (importers, encoders, image/glTF decoders, shader compilation, the
// world expansion + check front-half, and blob writing) lives in the
// `concinnity-cook` crate; this module keeps only what a running engine needs:
// the pre-compiled payload `deserialise` family, the payload-format types and
// consts, the built-in shader sources, and the `BuildCtx` / `Platform` /
// `SourceBacked` types shared with the build crate. Submodules stay `pub` so
// both the client runtime and the build crate can reach them across the
// workspace split.
pub mod asset;
pub mod bcn;
pub mod color_lut;
pub mod cubemap;
pub mod dds;
pub mod environment_map;
pub mod font;
pub mod shader;
pub mod texture;
pub mod tga;

pub use asset::{BuildCtx, Platform, SourceBacked};
