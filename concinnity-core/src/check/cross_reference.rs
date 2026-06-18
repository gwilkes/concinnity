// src/check/cross_reference.rs
//
// Cross-asset name reference declarations. Every referencing asset (Prop ->
// Mesh, Material -> Texture, etc.) declares its own references by implementing
// `CrossReferenced` in its asset file. The build crate's resolver consumes
// these declarations: it resolves each `RefKind` to the matching set of asset
// names and detects Prop parent cycles. Adding a new referencing asset means
// writing one impl plus one dispatch arm in the build crate.

// The category of asset a name reference must resolve to. Reference kinds are
// deliberately not 1:1 with asset types: `MeshSource` accepts several types
// and `CameraShot` resolves to `Camera3D` names.
#[derive(Debug, Clone, Copy)]
pub enum RefKind {
    // Mesh, ProceduralMesh, VoxelChunk, or a mesh-kind File.
    MeshSource,
    Texture,
    Material,
    Model,
    Prop,
    Scene,
    // A camera shot target, resolves to declared `Camera3D` names.
    CameraShot,
    BlockType,
}

// One item produced by a referencing asset's `cross_refs`.
pub enum CrossRef {
    // `target` must resolve to an asset in `kind`'s name-set; if it does not,
    // `error` is collected verbatim.
    Resolve {
        kind: RefKind,
        target: String,
        error: String,
    },
    // A problem the asset detected on its own: a missing required field, a
    // malformed array entry, an empty list. Collected verbatim.
    Issue(String),
}

// Implemented by every asset type that references other assets by name.
// `cross_refs` extracts those references (and any structural problems) from
// the asset's args; the resolver resolves each `Resolve` against the world.
pub trait CrossReferenced {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef>;
}
