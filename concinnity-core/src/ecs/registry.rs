// src/ecs/registry.rs
//
// Single source of truth for the renderer-free half of the engine's asset
// registry: every Component type paired with its stable u8 discriminant.
//
// Components are pure data, registered with one call (`define_components!`).
// There is no system registry: every system is internal client code,
// constructed at runtime from world content (see the client's
// `World::build_internal_systems`), never declared in a world or serialized to
// a blob. The runtime `SystemAsset` enum that holds the constructed systems is
// generated client-side from each system's `System` behavior impl, in the
// client crate's `ecs::registry`.
//
// Component discriminants live in 0..128. Discriminants are stable on disk --
// do not reorder or repurpose existing entries. The 128..255 range was once
// used by declarable systems; those are all retired (see the note below) and
// must not be reused.

#[allow(unused_imports)]
use crate::assets;
use crate::ecs::{AssetKind, BlobAssetDef, Component, PayloadLocator, Registration};
use crate::result::CnResult;

crate::define_components! {
        Window            => assets::Window,            1,
        GraphicsConfig    => assets::GraphicsConfig,    2,
        ShaderStage       => assets::ShaderStage,       3,
        Camera3D          => assets::Camera3D,          4,
        Mesh              => assets::Mesh,              5,
        FrameInput        => assets::FrameInput,        6,
        Texture           => assets::Texture,           7,
        Prop              => assets::Prop,              8,
        RigidBody         => assets::RigidBody,         9,
        PropBody          => assets::PropBody,          10,
        Room              => assets::Room,              11,
        Material          => assets::Material,          12,
        DirectionalLight  => assets::DirectionalLight,  13,
        PointLight        => assets::PointLight,        14,
        ProceduralMesh    => assets::ProceduralMesh,    15,
        Model             => assets::Model,             16,
        Scene             => assets::Scene,             17,
        SceneReel         => assets::SceneReel,         18,
        Font              => assets::Font,              19,
        TextLabel         => assets::TextLabel,         20,
        LightRig          => assets::LightRig,          21,
        MaterialPalette   => assets::MaterialPalette,   22,
        CameraShot        => assets::CameraShot,        23,
        Prefab            => assets::Prefab,            24,
        HitRegion         => assets::HitRegion,         25,
        File              => assets::File,              27,
        BlockType         => assets::BlockType,         28,
        VoxelChunk        => assets::VoxelChunk,        29,
        InstancedProp     => assets::InstancedProp,     30,
        CubemapTexture    => assets::CubemapTexture,    31,
        EnvironmentMap    => assets::EnvironmentMap,    32,
        PostProcessConfig => assets::PostProcessConfig, 33,
        ColorLut          => assets::ColorLut,          34,
        SkinnedMesh       => assets::SkinnedMesh,       35,
        Animation         => assets::Animation,         36,
        SkeletonPose      => assets::SkeletonPose,      37,
        StreamingConfig   => assets::StreamingConfig,   38,
        VoxelWorld        => assets::VoxelWorld,        39,
        AudioClip         => assets::AudioClip,         40,
        AudioEmitter      => assets::AudioEmitter,      41,
        Sprite            => assets::Sprite,            42,
        KeyBinding        => assets::KeyBinding,        43,
        View              => assets::View,              44,
        Decal             => assets::Decal,             46,
        VolumetricFog     => assets::VolumetricFog,     47,
        Joint             => assets::Joint,             48,
        ParticleEmitter   => assets::ParticleEmitter,   49,
        WaterSurface      => assets::WaterSurface,      50,
        SdfVolume         => assets::SdfVolume,         51,
        GlassPanel        => assets::GlassPanel,        52,
        LayoutContainer   => assets::LayoutContainer,   53,
        PhysicsConfig     => assets::PhysicsConfig,     54,
        FpsCounter        => assets::FpsCounter,        55,
        StatHud           => assets::StatHud,           56,
        SceneImport       => assets::SceneImport,       57,
        MainMenu          => assets::MainMenu,          58,
        OptionSelect      => assets::OptionSelect,      59,
        Slider            => assets::Slider,            61,
        ScrollPanel       => assets::ScrollPanel,       62,
        ReflectionProbe   => assets::ReflectionProbe,   65,
        Transform         => assets::Transform,         66,
        MeshRenderer      => assets::MeshRenderer,      67,
        ModelRenderer     => assets::ModelRenderer,     68,
        Collider          => assets::Collider,          69,
        Interactable      => assets::Interactable,      70,
        Pickup            => assets::Pickup,            71,
        Parent            => assets::Parent,            72,
        Children          => assets::Children,          73,
        SceneMember       => assets::SceneMember,       74,
        GlobalTransform   => assets::GlobalTransform,   75,
        RenderHandle      => assets::RenderHandle,      76,
        Held              => assets::Held,              77,
}

#[cfg(test)]
mod tests {
    use crate::ecs::ComponentType;

    // Convention guard for the asset-reference contract: a user-declarable
    // asset's `args` is its public JSON schema: always a JSON object of common
    // types, never a bare scalar or enum. `Component::Args` must therefore
    // serialize to a JSON object, and its `Default` must construct and
    // serialize cleanly (so the generated reference always has fields to list
    // and the engine has defaults to ship). Internal/runtime-only assets (e.g.
    // command enums) are exempt; they are never declared by hand.
    #[test]
    fn declarable_assets_have_object_args_schemas() {
        for &(ty, reg_fn) in ComponentType::all() {
            let reg = reg_fn();
            if !reg.addable() {
                continue;
            }
            let default_args = reg.default_args.as_ref().unwrap_or_else(|| {
                panic!(
                    "{}: Args::default() failed to serialize to JSON",
                    ty.as_str()
                )
            });
            assert!(
                default_args.is_object(),
                "{}: args schema is not a JSON object (got {default_args}). A declarable \
                 asset's args must be a JSON object of common types.",
                ty.as_str()
            );
        }
    }

    // The per-instance components an entity is composed from are RuntimeOnly:
    // never authored in a world, never in the asset reference, and exempt from
    // the declarable-args contract above. Guard that they stay that way so a
    // stray `External` origin can't leak one into the authoring surface.
    #[test]
    fn per_instance_components_are_runtime_only() {
        for ty in [
            ComponentType::Transform,
            ComponentType::MeshRenderer,
            ComponentType::ModelRenderer,
            ComponentType::Collider,
            ComponentType::Interactable,
            ComponentType::Pickup,
            ComponentType::Parent,
            ComponentType::Children,
            ComponentType::SceneMember,
            ComponentType::GlobalTransform,
            ComponentType::RenderHandle,
            ComponentType::Held,
        ] {
            assert!(
                !ty.registration().addable(),
                "{} must be RuntimeOnly (not declarable)",
                ty.as_str()
            );
        }
    }
}

// Retired component discriminants (0..128), stable on disk; never reuse. Each
// is now an Events<T> queue, not a component; all were RuntimeOnly (never
// serialized), so no blob references the gaps:
//   26 SceneCommand, 45 ViewCommand, 60 SettingCommand, 63 ControlsCommand,
//   64 AudioCommand
//
// Retired system discriminants (128..255), stable on disk; never reuse:
//   130 GraphicsSystem, 131 FpsCounter, 141 Camera3DSystem, 142 PhysicsSystem,
//   143 UiInputSystem, 145 AnimationSystem, 146 AudioSystem, 147 StatHud
// These were declarable systems before systems became internal. FpsCounter /
// StatHud became components (discs 55 / 56); PhysicsSystem's world config became
// the `PhysicsConfig` component (disc 54).
