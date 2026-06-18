// src/metal/shader_layout.rs
//
// The engine's buffer-binding contract for user-authored Metal shaders, plus
// the pure comparison that catches CPU/GPU struct-layout mismatches. A custom
// ShaderStage is linked into the engine's standard pipeline and inherits the
// engine's buffer bindings (per-frame view uniforms, per-object data, lights,
// shadow cascades). If the user declares one of those structs with a different
// layout than the engine's `#[repr(C)]` struct, the GPU reads the engine's
// bytes through the wrong offsets: garbage, and a GPU fault when a wrong stride
// walks a binding off the end of its buffer (the `RtGeomEntry` failure mode).
//
// This module is deliberately free of any Metal API: it defines what the engine
// expects (built from the real `#[repr(C)]` structs via `offset_of!`) and how to
// compare a backend-neutral reflected layout against it. The Metal reflection
// that produces the reflected layout lives in `shader_reflect.rs`; keeping the
// comparison separate makes it unit-testable without a GPU device.

use std::collections::HashMap;
use std::mem::{offset_of, size_of};

use crate::gfx::render_types::{
    GpuObjectData, LightUniforms, MaterialUniforms, ShadowPassPush, ShadowUniforms,
};

use super::uniforms::{ModelUniforms, ViewUniforms};

// One field the engine guarantees at a fixed byte offset inside an
// engine-provided buffer struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExpectedField {
    pub name: &'static str,
    pub offset: usize,
}

// The engine's authoritative layout for one buffer struct a user shader may
// bind. `size` is the `#[repr(C)]` `size_of`; for a buffer bound as an array /
// pointer (e.g. `GpuObjectData`) it is the per-element stride, which is what a
// wrong-stride bug corrupts.
#[derive(Clone, Debug)]
pub(super) struct ExpectedStruct {
    pub name: &'static str,
    pub size: usize,
    pub fields: Vec<ExpectedField>,
}

// Which engine pipeline stage an entry point belongs to. A custom ShaderStage
// declared `kind: "vertex"` can be a main vertex shader or a shadow caster;
// they bind different engine buffers, so the reflector resolves the stage from
// the entry-point name (see `shader_reflect.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineStage {
    Vertex,
    Fragment,
    Shadow,
}

// A struct layout as reflected from a compiled user shader. Backend-neutral so
// the comparison stays Metal-free; `shader_reflect.rs` fills it from Metal
// pipeline reflection, and the unit tests fill it by hand.
#[derive(Clone, Debug, PartialEq)]
pub struct ReflectedStruct {
    // The struct type name the shader declared at this binding.
    pub name: String,
    // The binding's data size in bytes (struct size, or element stride for a
    // pointer/array binding).
    pub size: usize,
    pub fields: Vec<ReflectedField>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReflectedField {
    pub name: String,
    pub offset: usize,
}

// Build an ExpectedField from a real Rust struct field.
macro_rules! field {
    ($t:ty, $f:ident) => {
        ExpectedField {
            name: stringify!($f),
            offset: offset_of!($t, $f),
        }
    };
}

// The engine-owned buffer bindings for a stage: `(buffer_index, expected
// layout)`. Only indices the engine itself binds are listed: buffers the user
// fully owns, the `Vertex` stage-in at buffer(1), and the bindless texture
// argument buffer are intentionally absent and never validated.
//
// The indices mirror the binds in `metal/draw/main.rs` and the shadow shader;
// the layouts are derived from the real `#[repr(C)]` structs (the single source
// of truth, the same ones the `*_layout_matches_msl` tests pin to MSL).
pub(super) fn engine_buffers(stage: EngineStage) -> Vec<(u32, ExpectedStruct)> {
    match stage {
        EngineStage::Vertex => vec![
            (0, view_uniforms_layout()),
            (2, model_uniforms_layout()),
            (9, gpu_object_data_layout()),
        ],
        EngineStage::Fragment => vec![
            (0, view_uniforms_layout()),
            (3, material_uniforms_layout()),
            (4, light_uniforms_layout()),
            (5, shadow_uniforms_layout()),
            (9, gpu_object_data_layout()),
        ],
        EngineStage::Shadow => vec![
            (0, shadow_uniforms_layout()),
            (2, model_uniforms_layout()),
            (7, shadow_pass_push_layout()),
        ],
    }
}

fn view_uniforms_layout() -> ExpectedStruct {
    ExpectedStruct {
        name: "ViewUniforms",
        size: size_of::<ViewUniforms>(),
        fields: vec![
            field!(ViewUniforms, vp),
            field!(ViewUniforms, view),
            field!(ViewUniforms, elapsed),
            field!(ViewUniforms, cam_pos),
            field!(ViewUniforms, prefilter_mip_count),
        ],
    }
}

fn model_uniforms_layout() -> ExpectedStruct {
    ExpectedStruct {
        name: "ModelUniforms",
        size: size_of::<ModelUniforms>(),
        fields: vec![field!(ModelUniforms, model)],
    }
}

fn gpu_object_data_layout() -> ExpectedStruct {
    ExpectedStruct {
        name: "GpuObjectData",
        size: size_of::<GpuObjectData>(),
        fields: vec![
            field!(GpuObjectData, model),
            field!(GpuObjectData, tint),
            field!(GpuObjectData, roughness),
            field!(GpuObjectData, emissive),
            field!(GpuObjectData, metallic),
            field!(GpuObjectData, albedo_index),
            field!(GpuObjectData, normal_index),
            field!(GpuObjectData, macro_variation),
            field!(GpuObjectData, terrain_blend),
            field!(GpuObjectData, bb_min),
            field!(GpuObjectData, cull_distance),
            field!(GpuObjectData, bb_max),
        ],
    }
}

fn material_uniforms_layout() -> ExpectedStruct {
    ExpectedStruct {
        name: "MaterialUniforms",
        size: size_of::<MaterialUniforms>(),
        fields: vec![
            field!(MaterialUniforms, roughness),
            field!(MaterialUniforms, metallic),
            field!(MaterialUniforms, macro_variation),
            field!(MaterialUniforms, terrain_blend),
            field!(MaterialUniforms, tint),
            field!(MaterialUniforms, emissive),
        ],
    }
}

fn light_uniforms_layout() -> ExpectedStruct {
    ExpectedStruct {
        name: "LightUniforms",
        size: size_of::<LightUniforms>(),
        fields: vec![
            field!(LightUniforms, directional),
            field!(LightUniforms, point),
            field!(LightUniforms, num_directional),
            field!(LightUniforms, num_point),
        ],
    }
}

fn shadow_uniforms_layout() -> ExpectedStruct {
    ExpectedStruct {
        name: "ShadowUniforms",
        size: size_of::<ShadowUniforms>(),
        fields: vec![
            field!(ShadowUniforms, light_vps),
            field!(ShadowUniforms, cascade_splits),
        ],
    }
}

fn shadow_pass_push_layout() -> ExpectedStruct {
    ExpectedStruct {
        name: "ShadowPassPush",
        size: size_of::<ShadowPassPush>(),
        fields: vec![field!(ShadowPassPush, cascade_idx)],
    }
}

// Compare the engine's expected layout against the shader's reflected layout
// for one binding. Returns `Err(message)` on a mismatch, naming the binding,
// the field, and the expected-vs-actual offset.
//
// Two checks, complementary:
//   * The binding's data size must match the engine struct's size. A wrong
//     field type (`float3` where the engine packs `[f32; 3]`) changes the
//     stride even when every named offset still lines up: this is the check
//     that would have caught the `RtGeomEntry` fault.
//   * Every engine field the shader also declares (matched by name) must sit at
//     the engine's offset. Fields the shader renames or omits are skipped: it
//     only has to read the fields it uses from where the engine put them. The
//     size check remains the backstop for the renamed-field case.
pub(super) fn compare_binding(
    index: u32,
    expected: &ExpectedStruct,
    reflected: &ReflectedStruct,
) -> Result<(), String> {
    if expected.size != reflected.size {
        return Err(format!(
            "buffer({index}) binding '{}' is {} bytes but the engine's '{}' is {} bytes \
             (a field-type mismatch such as `float3` vs `packed_float3` changes the stride \
             and corrupts every following field / array element)",
            reflected.name, reflected.size, expected.name, expected.size
        ));
    }
    for ef in &expected.fields {
        if let Some(rf) = reflected.fields.iter().find(|rf| rf.name == ef.name)
            && rf.offset != ef.offset
        {
            return Err(format!(
                "buffer({index}) binding '{}': field '{}' is at offset {} but the engine's \
                     '{}' puts it at offset {}",
                reflected.name, ef.name, rf.offset, expected.name, ef.offset
            ));
        }
    }
    Ok(())
}

// Validate every engine-owned binding a shader stage uses against the engine's
// contract. `reflected` maps buffer index → the layout reflected at that index;
// indices the shader does not bind are simply absent and skipped. Returns the
// first mismatch, or `Ok(())` if every engine binding the shader uses matches.
pub fn validate_stage(
    stage: EngineStage,
    reflected: &HashMap<u32, ReflectedStruct>,
) -> Result<(), String> {
    for (index, expected) in engine_buffers(stage) {
        if let Some(found) = reflected.get(&index) {
            compare_binding(index, &expected, found)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Turn an ExpectedStruct into a faithful ReflectedStruct (what reflection
    // would report for a correct user shader copying the engine struct).
    fn faithful(expected: &ExpectedStruct) -> ReflectedStruct {
        ReflectedStruct {
            name: expected.name.to_string(),
            size: expected.size,
            fields: expected
                .fields
                .iter()
                .map(|f| ReflectedField {
                    name: f.name.to_string(),
                    offset: f.offset,
                })
                .collect(),
        }
    }

    fn view_expected() -> ExpectedStruct {
        engine_buffers(EngineStage::Fragment)
            .into_iter()
            .find(|(i, _)| *i == 0)
            .unwrap()
            .1
    }

    #[test]
    fn faithful_copy_passes_every_stage() {
        // A shader that declares every engine struct exactly as the engine does
        // validates clean on all three stages.
        for stage in [
            EngineStage::Vertex,
            EngineStage::Fragment,
            EngineStage::Shadow,
        ] {
            let reflected: HashMap<u32, ReflectedStruct> = engine_buffers(stage)
                .iter()
                .map(|(i, e)| (*i, faithful(e)))
                .collect();
            assert!(validate_stage(stage, &reflected).is_ok());
        }
    }

    #[test]
    fn unused_bindings_are_skipped() {
        // A shader that binds none of the engine buffers has nothing to check.
        let reflected = HashMap::new();
        assert!(validate_stage(EngineStage::Fragment, &reflected).is_ok());
    }

    #[test]
    fn wrong_field_offset_is_rejected() {
        let expected = view_expected();
        let mut reflected = faithful(&expected);
        // Shift cam_pos as a float3-vs-padded-vec mistake would.
        let cam = reflected
            .fields
            .iter_mut()
            .find(|f| f.name == "cam_pos")
            .unwrap();
        cam.offset += 4;
        let err = compare_binding(0, &expected, &reflected).expect_err("must reject");
        assert!(err.contains("cam_pos"), "message names the field: {err}");
        assert!(err.contains("offset"));
    }

    #[test]
    fn wrong_struct_size_is_rejected() {
        // The RtGeomEntry failure mode: same named offsets, larger stride.
        let expected = view_expected();
        let mut reflected = faithful(&expected);
        reflected.size += 16;
        let err = compare_binding(0, &expected, &reflected).expect_err("must reject");
        assert!(err.contains("bytes"), "message mentions the size: {err}");
        assert!(err.contains("stride"));
    }

    #[test]
    fn renamed_field_is_skipped_but_size_still_guards() {
        // A renamed field can't be offset-checked, but a layout change that
        // renames AND resizes is still caught by the size check.
        let expected = view_expected();
        let mut reflected = faithful(&expected);
        for f in &mut reflected.fields {
            if f.name == "cam_pos" {
                f.name = "camera_position".to_string();
            }
        }
        // Pure rename, same size: passes (we only check fields present on both).
        assert!(compare_binding(0, &expected, &reflected).is_ok());
        // Rename plus a stride change: rejected by the size guard.
        reflected.size += 16;
        assert!(compare_binding(0, &expected, &reflected).is_err());
    }
}
