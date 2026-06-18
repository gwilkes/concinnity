// Metal pipeline-reflection adapter for build-time shader layout validation.
// This is the only file that touches the Metal reflection API; the engine
// binding contract and the comparison live in `shader_layout.rs` (Metal-free,
// unit-tested without a GPU). It reflects a user-authored `.metal` stage's
// engine-provided buffer bindings into the backend-neutral `ReflectedStruct`
// form and compares them against the engine's `#[repr(C)]` layouts. Registered
// with the core build pipeline via `ShaderBuildValidator` so a layout mismatch
// fails `cn build` with a clear message instead of faulting the GPU at run time.
//
// Reflection needs a live pipeline. A vertex/shadow stage reflects through a
// vertex-only pipeline (`rasterizationEnabled = false`, no fragment function); a
// fragment stage is paired with the engine's built-in `vertex_main` so its
// `[[stage_in]]` links. Anything that prevents reflection (no device, a pipeline
// that won't create for an unrelated reason) is reported as an infrastructure
// issue and fails open: only a layout mismatch we actually observed fails the
// build.

#![allow(clippy::incompatible_msrv)]
// Driven only by the binary's build chain (cn build / cn debug); the FFI lib
// never registers the validator, so these items read as dead under the lib.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Once;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString};
use objc2_metal::{
    MTLBinding, MTLBindingType, MTLBufferBinding, MTLCompileOptions, MTLCreateSystemDefaultDevice,
    MTLDevice, MTLFunction, MTLLibrary, MTLPipelineOption, MTLPixelFormat,
    MTLRenderPipelineDescriptor, MTLRenderPipelineReflection, MTLVertexDescriptor, MTLVertexFormat,
    MTLVertexStepFunction,
};

use concinnity_cook::shader::{ShaderBuildValidator, set_shader_build_validator};

use crate::metal::shader_layout::{EngineStage, ReflectedField, ReflectedStruct, validate_stage};

// A no-input fragment used only to make vertex/shadow reflection pipelines
// link. Declaring no `[[stage_in]]` means it imposes no constraint on the
// vertex stage's outputs, so any real vertex/shadow entry pairs with it.
const STUB_FRAGMENT_SRC: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
    fragment float4 __reflect_stub_fragment() { return float4(0.0); }\n";

// Register the Metal shader-layout validator with the core build pipeline. Safe
// to call from every build entry point; only the first call installs it (the
// underlying registration is itself first-wins).
pub(crate) fn register_shader_layout_validator() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        set_shader_build_validator(Box::new(MetalShaderValidator));
    });
}

struct MetalShaderValidator;

impl ShaderBuildValidator for MetalShaderValidator {
    fn validate_metal(&self, source: &str, kind: &str, asset_name: &str) -> Result<(), String> {
        match validate_metal_source(source, kind) {
            Ok(()) => Ok(()),
            Err(Issue::Mismatch(msg)) => Err(format!(
                "shader asset '{asset_name}': {msg}\nThe shader declares an engine-provided buffer \
                 struct with a different memory layout than the engine's, so the GPU would read the \
                 engine's data through the wrong offsets. Match the documented layout (see the \
                 ShaderStage asset reference)."
            )),
            Err(Issue::Infra(reason)) => {
                // Fail open: never break a build over a reflection-infrastructure
                // problem. A missed check is recoverable; a spurious build break
                // erodes trust in the build.
                tracing::warn!("shader asset '{asset_name}': skipped layout validation ({reason})");
                Ok(())
            }
        }
    }
}

// The outcome of reflecting a shader. A `Mismatch` is a real layout error that
// fails the build; an `Infra` issue is a reflection problem we fail open on.
#[derive(Debug)]
enum Issue {
    Mismatch(String),
    Infra(String),
}

// Reflect a compiled user `.metal` source and validate every engine-provided
// buffer struct it binds. `kind` is the compile kind (`"vertex"` | `"fragment"`);
// a `"vertex"` source may carry a main vertex shader, a shadow caster, or both,
// disambiguated by entry-point name.
fn validate_metal_source(source: &str, kind: &str) -> Result<(), Issue> {
    objc2::rc::autoreleasepool(|_| {
        let device =
            MTLCreateSystemDefaultDevice().ok_or_else(|| Issue::Infra("no Metal device".into()))?;
        let user_lib = compile_library(&device, source)
            .map_err(|e| Issue::Infra(format!("source did not compile for reflection: {e}")))?;
        let names = function_names(&user_lib);

        // Each (stage, entry point) we recognise gets reflected and checked. A
        // source that exposes no engine entry point is skipped (fail open).
        let mut targets: Vec<(EngineStage, &str)> = Vec::new();
        if kind == "fragment" {
            if names.iter().any(|n| n == "fragment_main") {
                targets.push((EngineStage::Fragment, "fragment_main"));
            }
        } else {
            if names.iter().any(|n| n == "vertex_main") {
                targets.push((EngineStage::Vertex, "vertex_main"));
            } else if names.iter().any(|n| n == "vertex_main_instanced") {
                targets.push((EngineStage::Vertex, "vertex_main_instanced"));
            }
            if names.iter().any(|n| n == "shadow_vertex_main") {
                targets.push((EngineStage::Shadow, "shadow_vertex_main"));
            }
        }
        if targets.is_empty() {
            return Err(Issue::Infra(format!(
                "no recognised engine entry point for kind '{kind}'"
            )));
        }

        for (stage, entry) in targets {
            let reflected = reflect_stage(&device, &user_lib, entry, stage)
                .map_err(|e| Issue::Infra(format!("reflection of '{entry}' failed: {e}")))?;
            validate_stage(stage, &reflected).map_err(Issue::Mismatch)?;
        }
        Ok(())
    })
}

// Reflect one stage's engine buffer bindings into `index -> ReflectedStruct`.
fn reflect_stage(
    device: &ProtocolObject<dyn MTLDevice>,
    user_lib: &ProtocolObject<dyn MTLLibrary>,
    entry: &str,
    stage: EngineStage,
) -> Result<HashMap<u32, ReflectedStruct>, String> {
    let entry_fn = function(user_lib, entry)?;
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(&standard_vertex_descriptor()));

    let is_fragment = matches!(stage, EngineStage::Fragment);
    if is_fragment {
        // A fragment pipeline needs a vertex function for its `[[stage_in]]` to
        // link; pair with the engine's built-in `vertex_main`, exactly what the
        // fragment runs against at draw time.
        let builtin_src = concinnity_core::build::shader::builtin_shader_source("default.metal")
            .ok_or("built-in default.metal source unavailable")?;
        let builtin_lib = compile_library(device, builtin_src)?;
        let vert_fn = function(&builtin_lib, "vertex_main")?;
        desc.setVertexFunction(Some(&vert_fn));
        desc.setFragmentFunction(Some(&entry_fn));
    } else {
        // A vertex/shadow pipeline needs a fragment to link, but we only care
        // about the vertex bindings. Pair with a trivial stub fragment that has
        // no `[[stage_in]]`: it imposes no constraint on the vertex's outputs,
        // so any real vertex/shadow entry links regardless of what it returns.
        let stub_lib = compile_library(device, STUB_FRAGMENT_SRC)?;
        let stub_fn = function(&stub_lib, "__reflect_stub_fragment")?;
        desc.setVertexFunction(Some(&entry_fn));
        desc.setFragmentFunction(Some(&stub_fn));
    }
    unsafe {
        desc.colorAttachments()
            .objectAtIndexedSubscript(0)
            .setPixelFormat(MTLPixelFormat::RGBA16Float);
    }

    let reflection = create_reflection(device, &desc)?;
    let bindings = if is_fragment {
        reflection.fragmentBindings()
    } else {
        reflection.vertexBindings()
    };
    Ok(bindings_to_map(&bindings))
}

// Create a pipeline with binding reflection enabled and return the reflection.
fn create_reflection(
    device: &ProtocolObject<dyn MTLDevice>,
    desc: &MTLRenderPipelineDescriptor,
) -> Result<Retained<MTLRenderPipelineReflection>, String> {
    let mut reflection: Option<Retained<MTLRenderPipelineReflection>> = None;
    device
        .newRenderPipelineStateWithDescriptor_options_reflection_error(
            desc,
            MTLPipelineOption::BindingInfo,
            Some(&mut reflection),
        )
        .map_err(|e| format!("pipeline creation failed: {e:?}"))?;
    reflection.ok_or_else(|| "pipeline returned no reflection".to_string())
}

// Collect the buffer bindings into `index -> ReflectedStruct`, keeping only
// bindings backed by a struct (a `constant X&` struct or a `constant X*`
// pointer to one). Other buffer/texture/sampler bindings are ignored: they are
// either user-owned or outside the engine contract.
fn bindings_to_map(
    bindings: &NSArray<ProtocolObject<dyn MTLBinding>>,
) -> HashMap<u32, ReflectedStruct> {
    let mut map = HashMap::new();
    for binding in bindings.iter() {
        let binding: &ProtocolObject<dyn MTLBinding> = &binding;
        if binding.r#type() != MTLBindingType::Buffer {
            continue;
        }
        // The binding conforms to MTLBufferBinding once it is the Buffer type.
        // ProtocolObject is a transparent wrapper over the same object, so this
        // cast just re-views it through the buffer sub-protocol.
        let buf: &ProtocolObject<dyn MTLBufferBinding> = unsafe {
            &*(binding as *const ProtocolObject<dyn MTLBinding>
                as *const ProtocolObject<dyn MTLBufferBinding>)
        };

        // A struct binding carries its layout directly; a pointer/array binding
        // (e.g. `constant GpuObjectData*`) carries it on the pointee, whose
        // dataSize is the per-element stride.
        let (struct_ty, size) = if let Some(st) = buf.bufferStructType() {
            (Some(st), buf.bufferDataSize())
        } else if let Some(ptr) = buf.bufferPointerType() {
            (ptr.elementStructType(), ptr.dataSize())
        } else {
            (None, 0)
        };
        let Some(st) = struct_ty else {
            continue;
        };

        let fields = st
            .members()
            .iter()
            .map(|m| ReflectedField {
                name: m.name().to_string(),
                offset: m.offset(),
            })
            .collect();
        map.insert(
            binding.index() as u32,
            ReflectedStruct {
                name: binding.name().to_string(),
                size,
                fields,
            },
        );
    }
    map
}

fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    let options = MTLCompileOptions::new();
    device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .map_err(|e| format!("{e:?}"))
}

fn function(
    lib: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLFunction>>, String> {
    lib.newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| format!("entry point '{name}' not found"))
}

fn function_names(lib: &ProtocolObject<dyn MTLLibrary>) -> Vec<String> {
    lib.functionNames().iter().map(|n| n.to_string()).collect()
}

// The engine's standard five-attribute mesh vertex descriptor (the `Vertex`
// layout), with the vertex stream at buffer index 1 so it does not collide with
// the engine's `ViewUniforms` at buffer 0. Required for the `[[stage_in]]` of
// the vertex/shadow stages to link during reflection.
fn standard_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
    const STREAM: usize = 1;
    let vd = MTLVertexDescriptor::new();
    let attrs = [
        (0u32, MTLVertexFormat::Float3, 0usize),
        (1, MTLVertexFormat::Float3, 12),
        (2, MTLVertexFormat::Float3, 24),
        (3, MTLVertexFormat::Float3, 36),
        (4, MTLVertexFormat::Float2, 48),
    ];
    unsafe {
        for (idx, fmt, offset) in attrs {
            let a = vd.attributes().objectAtIndexedSubscript(idx as usize);
            a.setFormat(fmt);
            a.setOffset(offset);
            a.setBufferIndex(STREAM);
        }
        let layout = vd.layouts().objectAtIndexedSubscript(STREAM);
        layout.setStride(std::mem::size_of::<crate::gfx::mesh_payload::Vertex>());
        layout.setStepFunction(MTLVertexStepFunction::PerVertex);
    }
    vd
}

#[cfg(test)]
mod tests {
    use super::*;

    // A correct user vertex shader: declares ViewUniforms exactly as the engine
    // does (packed_float3 cam_pos) and binds it at buffer(0).
    const GOOD_VERTEX: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct ViewUniforms {
            float4x4 vp;
            float4x4 view;
            float elapsed;
            float _pad;
            packed_float3 cam_pos;
            float prefilter_mip_count;
        };
        struct VIn { float3 pos [[attribute(0)]]; };
        vertex float4 vertex_main(VIn in [[stage_in]],
                                  constant ViewUniforms& view [[buffer(0)]]) {
            float3 p = in.pos + float3(view.cam_pos) * view.prefilter_mip_count * view.elapsed;
            return view.vp * view.view * float4(p, 1.0);
        }
    "#;

    // The same shader but with `float3 cam_pos` (16-byte aligned, size 16)
    // instead of packed_float3: exactly the float3-vs-[f32;3] class of bug. It
    // grows the struct's stride past the engine's 160 bytes, so the size check
    // catches it (the `RtGeomEntry` failure mode).
    const BAD_SIZE_VERTEX: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct ViewUniforms {
            float4x4 vp;
            float4x4 view;
            float elapsed;
            float _pad;
            float3 cam_pos;
            float prefilter_mip_count;
        };
        struct VIn { float3 pos [[attribute(0)]]; };
        vertex float4 vertex_main(VIn in [[stage_in]],
                                  constant ViewUniforms& view [[buffer(0)]]) {
            float3 p = in.pos + view.cam_pos * view.prefilter_mip_count * view.elapsed;
            return view.vp * view.view * float4(p, 1.0);
        }
    "#;

    // `vp` and `view` swapped: the total size is unchanged (two float4x4 + the
    // tail), but every named field lands at the wrong offset: exercises the
    // field-offset check rather than the size check.
    const BAD_OFFSET_VERTEX: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct ViewUniforms {
            float4x4 view;
            float4x4 vp;
            float elapsed;
            float _pad;
            packed_float3 cam_pos;
            float prefilter_mip_count;
        };
        struct VIn { float3 pos [[attribute(0)]]; };
        vertex float4 vertex_main(VIn in [[stage_in]],
                                  constant ViewUniforms& view [[buffer(0)]]) {
            float3 p = in.pos + float3(view.cam_pos) * view.prefilter_mip_count * view.elapsed;
            return view.vp * view.view * float4(p, 1.0);
        }
    "#;

    // Headless CI may have no Metal device; skip the device-backed assertions
    // there rather than fail.
    fn have_device() -> bool {
        MTLCreateSystemDefaultDevice().is_some()
    }

    #[test]
    fn faithful_view_uniforms_validate() {
        if !have_device() {
            return;
        }
        assert!(
            matches!(validate_metal_source(GOOD_VERTEX, "vertex"), Ok(())),
            "a faithful ViewUniforms copy must validate"
        );
    }

    #[test]
    fn wrong_struct_size_is_rejected() {
        if !have_device() {
            return;
        }
        // The float3-vs-packed bug grows the struct stride; caught by the size
        // check (MSL `float3` is 16 bytes, pushing ViewUniforms past 160).
        match validate_metal_source(BAD_SIZE_VERTEX, "vertex") {
            Err(Issue::Mismatch(msg)) => {
                assert!(
                    msg.contains("ViewUniforms"),
                    "names the engine struct: {msg}"
                );
                assert!(
                    msg.contains("bytes") && msg.contains("stride"),
                    "reports the size: {msg}"
                );
            }
            other => panic!("expected a layout mismatch, got {other:?}"),
        }
    }

    #[test]
    fn wrong_field_offset_is_rejected() {
        if !have_device() {
            return;
        }
        // Swapped fields keep the size but move every offset.
        match validate_metal_source(BAD_OFFSET_VERTEX, "vertex") {
            Err(Issue::Mismatch(msg)) => {
                assert!(msg.contains("offset"), "reports the offset: {msg}");
                assert!(
                    msg.contains("vp") || msg.contains("view"),
                    "names a shifted field: {msg}"
                );
            }
            other => panic!("expected a layout mismatch, got {other:?}"),
        }
    }

    #[test]
    fn validator_fails_the_build_with_asset_context() {
        if !have_device() {
            return;
        }
        // The build-facing entry point wraps the mismatch with the asset name so
        // `cn build` reports which shader to fix.
        let err = MetalShaderValidator
            .validate_metal(BAD_SIZE_VERTEX, "vertex", "my_custom_vert")
            .expect_err("a mismatched shader must fail the build");
        assert!(err.contains("my_custom_vert"), "names the asset: {err}");
        assert!(
            err.contains("ViewUniforms"),
            "names the engine struct: {err}"
        );
    }

    #[test]
    fn faithful_shader_passes_the_build_entry() {
        if !have_device() {
            return;
        }
        MetalShaderValidator
            .validate_metal(GOOD_VERTEX, "vertex", "ok_vert")
            .expect("a faithful shader must pass the build");
    }
}
