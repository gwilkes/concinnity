// src/assets/mod.rs
//
// Asset type definitions: one pure-data component per file. Systems are not
// assets: every system is internal client code (see the client's
// `World::build_internal_systems`), driven by the presence of the components
// defined here. The client re-exports this module under the historical
// `crate::assets::*` paths.

// Component data types.
mod animation;
pub mod audio_clip;
mod audio_command;
mod audio_emitter;
mod block_type;
mod camera3d;
mod camera_shot;
mod color_lut;
mod controls_command;
mod cubemap_texture;
mod decal;
mod despawn_request;
mod directional_light;
mod environment_map;
mod file;
mod font;
mod frame_input;
mod glass_panel;
mod graphics_config;
mod hit_region;
mod input_key;
pub mod instanced_prop;
mod joint;
mod key_binding;
mod layout_container;
mod lifetime;
mod light_rig;
mod main_menu;
mod material;
mod material_palette;
mod mesh;
mod model;
mod option_select;
mod particle_emitter;
mod physics_config;
mod point_light;
mod post_process_config;
mod prefab;
pub mod procedural_mesh;
mod prop;
mod prop_body;
mod reflection_probe;
mod reparent_request;
mod rigid_body;
mod room;
mod scene;
mod scene_command;
mod scene_import;
mod scene_reel;
mod scroll_panel;
pub mod sdf_volume;
mod setting_command;
pub mod shader_stage;
mod skeleton_pose;
mod skinned_mesh;
mod slider;
mod spawn_request;
mod spawner;
mod sprite;
mod streaming_config;
mod text_label;
mod texture;
mod view;
mod view_command;
mod volumetric_fog;
mod voxel_chunk;
mod voxel_world;
mod water_surface;
mod window;

// Per-instance components an entity is composed from: its placement, render
// description, collision, hierarchy, and gameplay tags.
mod children;
mod collider;
mod global_transform;
mod held;
mod interactable;
mod mesh_renderer;
mod model_renderer;
mod parent;
mod pickup;
mod render_handle;
mod scene_member;
mod transform;

// HUD-overlay request components. Declaring one runs the matching internal
// overlay behavior (in the client crate); both are pure data here.
mod fps_counter;
mod stat_hud;

pub use animation::Animation;
pub use audio_clip::AudioClip;
pub use audio_command::AudioCommand;
pub use audio_emitter::AudioEmitter;
pub use block_type::BlockType;
pub use camera_shot::CameraShot;
pub use camera3d::{Camera3D, CameraController};
pub use color_lut::ColorLut;
pub use controls_command::ControlsCommand;
pub use cubemap_texture::CubemapTexture;
pub use decal::Decal;
pub use despawn_request::DespawnRequest;
pub use directional_light::DirectionalLight;
pub use environment_map::EnvironmentMap;
pub use file::{File, FileKind};
pub use font::Font;
pub use frame_input::FrameInput;
pub use glass_panel::GlassPanel;
pub use graphics_config::GraphicsConfig;
#[allow(unused_imports)]
pub use graphics_config::ShadowUpdate;
pub use hit_region::HitRegion;
pub use input_key::Key;
pub use instanced_prop::InstancedProp;
pub use joint::{Joint, JointKind};
pub use key_binding::KeyBinding;
pub use layout_container::{Justify, LabelBox, LayoutContainer, LayoutRow, Placement};
pub use lifetime::Lifetime;
pub use light_rig::LightRig;
pub use main_menu::{MainMenu, MainMenuItem};
pub use material::Material;
pub use material_palette::MaterialPalette;
pub use mesh::{Mesh, VertexData};
pub use model::{Model, SubMeshRef};
pub use option_select::OptionSelect;
pub use particle_emitter::ParticleEmitter;
pub use physics_config::PhysicsConfig;
pub use point_light::PointLight;
#[allow(unused_imports)]
pub use post_process_config::IndirectLighting;
pub use post_process_config::PostProcessConfig;
#[allow(unused_imports)]
pub use post_process_config::ReflectionBlurResolution;
#[allow(unused_imports)]
pub use post_process_config::SsgiResolution;
#[allow(unused_imports)]
pub use post_process_config::UpscaleQuality;
#[allow(unused_imports)]
pub use post_process_config::UpscalerBackend;
pub use prefab::Prefab;
pub use procedural_mesh::ProceduralMesh;
pub use prop::Prop;
// `PropCollider` is re-exported for tests / future consumers; the crate
// currently only uses it through `Prop.collider`, so the re-export is unused
// at compile time outside of the test module.
#[allow(unused_imports)]
pub use prop::PropCollider;
pub use prop_body::PropBody;
pub use reflection_probe::ReflectionProbe;
pub use reparent_request::ReparentRequest;
pub use rigid_body::RigidBody;
pub use room::Room;
pub use scene::Scene;
pub use scene_command::SceneCommand;
pub use scene_import::SceneImport;
pub use scene_reel::SceneReel;
pub use scroll_panel::{ScrollGroup, ScrollPanel, ScrollRow};
pub use sdf_volume::SdfVolume;
pub use setting_command::{SettingCommand, SettingOp};
// Re-exported for the Metal raymarch encoder; non-Metal builds reach
// the asset through `SdfVolume` only.
#[cfg(backend_metal)]
#[allow(unused_imports)]
pub use sdf_volume::{SDF_MAX_STEPS_CEILING, SDF_MAX_STEPS_FLOOR, SDF_PARAMS_LEN};
pub use shader_stage::{ShaderKind, ShaderStage};
pub use skeleton_pose::SkeletonPose;
pub use skinned_mesh::{JointDef, SkinnedMesh, SkinnedVertexData, build_skeleton_from_joint_defs};
pub use slider::Slider;
pub use spawn_request::SpawnRequest;
pub use spawner::Spawner;
pub use sprite::Sprite;
pub use streaming_config::StreamingConfig;
pub use text_label::TextLabel;
pub use texture::Texture;
pub use view::View;
pub use view_command::ViewCommand;
pub use volumetric_fog::VolumetricFog;
pub use voxel_chunk::VoxelChunk;
pub use voxel_world::VoxelWorld;
pub use water_surface::WaterSurface;
// Re-exported for the Metal water encoder; non-Metal builds reach the
// asset through `WaterSurface` only.
#[cfg(backend_metal)]
#[allow(unused_imports)]
pub use water_surface::{MAX_WATER_WAVES, WaterWave};
pub use window::{Window, WindowArgs, WindowMode};

// Per-instance components an entity is composed from.
pub use children::Children;
pub use collider::Collider;
pub use global_transform::GlobalTransform;
pub use held::Held;
pub use interactable::Interactable;
pub use mesh_renderer::MeshRenderer;
pub use model_renderer::ModelRenderer;
pub use parent::Parent;
pub use pickup::Pickup;
pub use render_handle::RenderHandle;
pub use scene_member::SceneMember;
pub use transform::Transform;

// HUD-overlay request components; their behavior lives in the client crate.
pub use fps_counter::FpsCounter;
pub use stat_hud::StatHud;
