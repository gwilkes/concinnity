// `BuildAsset` is the build-time counterpart to `Component`. Components whose
// `PAYLOAD = AssetPayload::Compiled` implement this trait to turn their args
// into a binary payload (mesh vertices, shader bytecode, decoded image, etc.).
// The build pipeline calls `<T as BuildAsset>::compile_payload` for each
// declared asset and packs the resulting bytes into a blob.
//
// `BuildCtx`, `Platform`, and the companion `SourceBacked` trait stay in
// concinnity-core: the runtime and world layers, and several core asset-file
// helpers whose signatures take `&BuildCtx`, depend on them. `BuildCtx` is
// re-exported here so the trait + the moved impls can keep naming it as
// `crate::asset::BuildCtx`.

use crate::ecs::Component;
pub use concinnity_core::build::BuildCtx;

// A component that compiles to a binary payload at build time.
//
// Only types whose `Component::PAYLOAD` is `AssetPayload::Compiled` should
// implement this. The build pipeline dispatches via a match on
// `ComponentType` in [`crate::pipeline`].
pub trait BuildAsset: Component {
    fn compile_payload(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> std::io::Result<Vec<u8>>;

    // On-disk files this asset's `compile_payload` reads, beyond what the
    // payload cache can derive from the args JSON. The cache layer mixes the
    // contents-hash of each returned path into the per-asset cache key so an
    // edit to one of those files invalidates the cached payload.
    //
    // Default is empty: appropriate for assets whose only inputs are the
    // args themselves (or whose source paths are resolved by the cache's
    // generic JSON string walk). Override when `compile_payload` reads a
    // file at a path the generic walk would miss, e.g. `SdfVolume` reading
    // `assets/shaders/<name>.metal` from the source tree, or any asset
    // whose resolution rules differ from the cache's default lookup.
    //
    // Return only paths that exist on disk. The cache silently drops paths
    // it can't read, so returning a placeholder is safe but pointless.
    fn source_files(_args: &serde_json::Value, _ctx: &BuildCtx<'_>) -> Vec<String> {
        Vec::new()
    }
}
