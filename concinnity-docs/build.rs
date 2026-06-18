// Walks the concinnity-core asset source files at ../concinnity-core/src/assets/*.rs
// and emits a static table at
// $OUT_DIR/assets_doc.rs. For each asset struct that implements Component the
// emitted AssetDoc contains:
//   - summary:  first paragraph of the struct-level rustdoc
//   - full_doc: struct-level rustdoc (hand-written table lines stripped)
//               followed by a `#### Parameters` bullet list generated from the
//               asset's `args` fields. Each bullet states the field's JSON type
//               in prose (so no Rust type name, enum, struct, or otherwise,
//               ever reaches the user), folds in the field's own rustdoc, and
//               appends the default unless the prose already covers it.
//
// Nested objects a field embeds (a Prop's collider, a Camera's controller, the
// element type of an array, …) are documented once under a trailing "Value
// types" category and linked from the fields that use them, the way a JSON
// schema separates `$defs` from the objects that reference them.
//
// The categorisation list below controls which assets appear in the LLM-facing
// reference and in what order. Assets discovered in the source tree but not
// listed here are silently ignored; assets listed here but missing from the
// source (or lacking rustdoc) cause a compile-time panic.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs;
use std::path::Path;

// Shared with the crate; renders type phrases, field bullets, and the markdown
// document written to docs/asset-reference.md.
#[path = "src/render.rs"]
mod render;
use render::{FieldEntry, FieldType, RefEntry, VALUE_TYPES_CATEGORY, render_parameters};

const ASSETS_DIR: &str = "../concinnity-core/src/assets";

// Markdown reference written into the source tree, relative to this crate.
const REFERENCE_DOC: &str = "../concinnity-docs/asset-reference.md";

const CATEGORIES: &[(&str, &[&str])] = &[
    (
        "World setup",
        &[
            "GraphicsConfig",
            "ShaderStage",
            "Window",
            "PhysicsConfig",
            "StreamingConfig",
        ],
    ),
    ("Scene organization", &["Scene", "SceneReel"]),
    (
        "Geometry",
        &[
            "SceneImport",
            "BlockType",
            "Mesh",
            "Model",
            "ProceduralMesh",
            "Room",
            "VoxelChunk",
            "VoxelWorld",
        ],
    ),
    (
        "Materials and textures",
        &["Material", "MaterialPalette", "Texture", "CubemapTexture"],
    ),
    (
        "Lighting",
        &[
            "DirectionalLight",
            "LightRig",
            "PointLight",
            "EnvironmentMap",
        ],
    ),
    ("Camera", &["Camera3D", "CameraShot"]),
    (
        "Objects",
        &["Prop", "InstancedProp", "Prefab", "RigidBody", "PropBody"],
    ),
    ("Animation", &["Animation", "Joint", "SkinnedMesh"]),
    ("Audio", &["AudioClip", "AudioEmitter"]),
    (
        "Effects and atmosphere",
        &[
            "Decal",
            "GlassPanel",
            "WaterSurface",
            "VolumetricFog",
            "ParticleEmitter",
            "SdfVolume",
        ],
    ),
    ("Post-processing", &["PostProcessConfig", "ColorLut"]),
    (
        "UI",
        &[
            "Font",
            "TextLabel",
            "Sprite",
            "FpsCounter",
            "StatHud",
            "HitRegion",
            "KeyBinding",
            "View",
            "LayoutContainer",
        ],
    ),
    ("Utilities", &["FrameInput", "File"]),
];

// Cross-file indices over the parsed asset sources. Enums, value-type structs,
// and `impl Default` blocks can each live in a different file from the asset
// that references them, so every lookup goes through these.
struct Ctx<'a> {
    files: &'a [syn::File],
    // Enum ident -> its serialized string variants (string-valued enums only).
    enums: &'a HashMap<String, Vec<String>>,
    // Named-field struct ident -> index of the file it is defined in.
    struct_file_idx: &'a HashMap<String, usize>,
    // Component struct ident -> its declarable NAME (for linking to assets).
    comp_by_struct: &'a HashMap<String, String>,
}

struct ComponentMeta {
    name: String,
    struct_ident: String,
    args_struct: String,
}

struct Rendered {
    summary: String,
    full_doc: String,
}

fn main() {
    println!("cargo:rerun-if-changed={ASSETS_DIR}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/render.rs");

    let files = parse_asset_files();
    let enums = collect_enums(&files);
    let struct_file_idx = collect_structs(&files);
    let components = collect_components(&files, &struct_file_idx);
    let comp_by_struct: HashMap<String, String> = components
        .iter()
        .map(|c| (c.struct_ident.clone(), c.name.clone()))
        .collect();

    let ctx = Ctx {
        files: &files,
        enums: &enums,
        struct_file_idx: &struct_file_idx,
        comp_by_struct: &comp_by_struct,
    };

    // Render every component, collecting the nested value types its fields use.
    let mut by_name: BTreeMap<String, Rendered> = BTreeMap::new();
    let mut value_types: BTreeSet<String> = BTreeSet::new();
    for c in &components {
        let (summary, full_doc) =
            render_doc_entry(&c.struct_ident, &c.args_struct, &ctx, &mut value_types);
        by_name.insert(c.name.clone(), Rendered { summary, full_doc });
    }

    // Render value types to a fixpoint (one may embed another).
    let value_type_docs = render_value_types(value_types, &ctx);

    let mut out = String::new();
    out.push_str("// Auto-generated by concinnity-docs/build.rs - do not edit.\n\n");
    out.push_str("pub struct AssetDoc {\n");
    out.push_str("    pub type_name: &'static str,\n");
    out.push_str("    pub category:  &'static str,\n");
    out.push_str("    pub summary:   &'static str,\n");
    out.push_str("    pub full_doc:  &'static str,\n");
    out.push_str("}\n\n");
    out.push_str("pub static ASSET_DOCS: &[AssetDoc] = &[\n");

    // Collected in document order so the markdown reference and the embedded
    // table stay in the same order.
    let mut ref_entries: Vec<RefEntry> = Vec::new();

    for (cat, names) in CATEGORIES {
        for name in *names {
            let entry = by_name.get(*name).unwrap_or_else(|| {
                panic!(
                    "build.rs: no rustdoc found for asset `{name}` (category `{cat}`). \
                     Either the rustdoc on its struct is empty or its NAME constant \
                     differs from the string in CATEGORIES."
                );
            });
            push_doc(&mut out, name, cat, &entry.summary, &entry.full_doc);
            ref_entries.push(RefEntry {
                category: cat,
                type_name: name,
                full_doc: entry.full_doc.as_str(),
            });
        }
    }

    // Value types last, alphabetically (BTreeMap order).
    for (name, entry) in &value_type_docs {
        push_doc(
            &mut out,
            name,
            VALUE_TYPES_CATEGORY,
            &entry.summary,
            &entry.full_doc,
        );
        ref_entries.push(RefEntry {
            category: VALUE_TYPES_CATEGORY,
            type_name: name,
            full_doc: entry.full_doc.as_str(),
        });
    }
    out.push_str("];\n");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest = Path::new(&out_dir).join("assets_doc.rs");
    fs::write(&dest, out).expect("write assets_doc.rs");

    let markdown = render::render_reference_md(&ref_entries);
    write_reference_doc(&markdown);
}

fn push_doc(out: &mut String, type_name: &str, category: &str, summary: &str, full_doc: &str) {
    out.push_str("    AssetDoc {\n");
    out.push_str(&format!("        type_name: {type_name:?},\n"));
    out.push_str(&format!("        category:  {category:?},\n"));
    out.push_str(&format!("        summary:   {summary:?},\n"));
    out.push_str(&format!("        full_doc:  {full_doc:?},\n"));
    out.push_str("    },\n");
}

// Write the rendered reference into the source tree. The markdown is an
// auxiliary artifact (the embedded ASSET_DOCS table is the build's real
// output), so an unwritable docs/ warns rather than failing the whole build.
// Skips the write when the content is unchanged to avoid churning the file's
// mtime (and the git working tree) on every build.
fn write_reference_doc(content: &str) {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let path = Path::new(&manifest_dir).join(REFERENCE_DOC);

    if fs::read_to_string(&path).ok().as_deref() == Some(content) {
        return;
    }
    if let Some(parent) = path.parent()
        && let Err(e) = fs::create_dir_all(parent)
    {
        println!("cargo:warning=asset reference: {}: {e}", parent.display());
        return;
    }
    if let Err(e) = fs::write(&path, content) {
        println!("cargo:warning=asset reference: {}: {e}", path.display());
    }
}

// Source parsing and cross-file indices

fn parse_asset_files() -> Vec<syn::File> {
    let mut out = Vec::new();
    let entries = fs::read_dir(ASSETS_DIR)
        .unwrap_or_else(|e| panic!("build.rs: could not read {ASSETS_DIR}: {e}"));
    for entry in entries {
        let path = entry.expect("read_dir entry").path();
        if path.is_dir() {
            let sub = fs::read_dir(&path)
                .unwrap_or_else(|e| panic!("build.rs: could not read {}: {e}", path.display()));
            for s in sub {
                push_parsed(&s.expect("read_dir entry").path(), &mut out);
            }
        } else {
            push_parsed(&path, &mut out);
        }
    }
    out
}

fn push_parsed(path: &Path, out: &mut Vec<syn::File>) {
    if path.extension().and_then(|e| e.to_str()) != Some("rs") {
        return;
    }
    let src = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("build.rs: read {}: {e}", path.display()));
    let file =
        syn::parse_file(&src).unwrap_or_else(|e| panic!("build.rs: parse {}: {e}", path.display()));
    out.push(file);
}

// Enum ident -> serialized variants, for enums that serialize to a plain string
// (all variants are unit). Data-carrying enums become JSON objects, so they are
// left out here and fall through to the generic object rendering.
fn collect_enums(files: &[syn::File]) -> HashMap<String, Vec<String>> {
    let mut out = HashMap::new();
    for file in files {
        for item in &file.items {
            let e = match item {
                syn::Item::Enum(e) => e,
                _ => continue,
            };
            if e.variants
                .iter()
                .any(|v| !matches!(v.fields, syn::Fields::Unit))
            {
                continue;
            }
            let rule = serde_kv(&e.attrs, "rename_all");
            let variants = e
                .variants
                .iter()
                .map(|v| {
                    serde_kv(&v.attrs, "rename")
                        .unwrap_or_else(|| apply_case(&v.ident.to_string(), rule.as_deref()))
                })
                .collect();
            out.insert(e.ident.to_string(), variants);
        }
    }
    out
}

// Every named-field struct ident -> the index of the file defining it.
fn collect_structs(files: &[syn::File]) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    for (i, file) in files.iter().enumerate() {
        for item in &file.items {
            if let syn::Item::Struct(s) = item
                && matches!(s.fields, syn::Fields::Named(_))
            {
                out.insert(s.ident.to_string(), i);
            }
        }
    }
    out
}

fn collect_components(
    files: &[syn::File],
    struct_file_idx: &HashMap<String, usize>,
) -> Vec<ComponentMeta> {
    let mut out = Vec::new();
    for file in files {
        for item in &file.items {
            let imp = match item {
                syn::Item::Impl(i) => i,
                _ => continue,
            };
            let trait_name = match &imp.trait_ {
                Some((_, p, _)) => p
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default(),
                None => continue,
            };
            // `Component` is implemented by the data struct itself and carries
            // the asset metadata surfaced in the reference. Systems are internal
            // engine code with no declarable asset, so they have no metadata.
            if trait_name != "Component" {
                continue;
            }
            let struct_ident = match imp.self_ty.as_ref() {
                syn::Type::Path(tp) => tp
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default(),
                _ => continue,
            };
            let name = name_const(imp).unwrap_or_else(|| struct_ident.clone());
            // The field table is built from the asset's `args`, not its runtime
            // struct. Most assets use `type Args = Self`; a few (Camera3D, Room,
            // File) keep runtime state on the Component and declare a separate
            // args struct, document that one when it exists.
            let args_struct = component_args_struct(imp)
                .filter(|a| a != "Self" && *a != struct_ident && struct_file_idx.contains_key(a))
                .unwrap_or_else(|| struct_ident.clone());
            out.push(ComponentMeta {
                name,
                struct_ident,
                args_struct,
            });
        }
    }
    out
}

// Doc entry rendering

// Render one entry: the description comes from `doc_ident`'s rustdoc, the
// parameter bullets from `args_ident`'s fields. For `type Args = Self` assets
// the two are the same struct; for value types both are the value type itself.
fn render_doc_entry(
    doc_ident: &str,
    args_ident: &str,
    ctx: &Ctx,
    value_types: &mut BTreeSet<String>,
) -> (String, String) {
    let doc = struct_doc(doc_ident, ctx);
    let cleaned = strip_table_lines(&doc);
    let fields = build_fields(args_ident, ctx, value_types);
    let params = render_parameters(&fields);
    let full_doc = match (cleaned.is_empty(), params.is_empty()) {
        (_, true) => cleaned,
        (true, false) => params,
        (false, false) => format!("{}\n\n{}", cleaned, params.trim_end()),
    };
    (first_paragraph(&doc), full_doc)
}

fn render_value_types(seed: BTreeSet<String>, ctx: &Ctx) -> BTreeMap<String, Rendered> {
    let mut done: BTreeMap<String, Rendered> = BTreeMap::new();
    let mut queue: Vec<String> = seed.into_iter().collect();
    while let Some(name) = queue.pop() {
        if done.contains_key(&name) || !ctx.struct_file_idx.contains_key(&name) {
            continue;
        }
        let mut more = BTreeSet::new();
        let (summary, full_doc) = render_doc_entry(&name, &name, ctx, &mut more);
        let summary = if summary.is_empty() {
            "Nested object embedded by other assets.".to_string()
        } else {
            summary
        };
        done.insert(name, Rendered { summary, full_doc });
        for m in more {
            if !done.contains_key(&m) {
                queue.push(m);
            }
        }
    }
    done
}

fn build_fields(
    args_ident: &str,
    ctx: &Ctx,
    value_types: &mut BTreeSet<String>,
) -> Vec<FieldEntry> {
    let file = match ctx.struct_file_idx.get(args_ident) {
        Some(i) => &ctx.files[*i],
        None => return Vec::new(),
    };
    let st = match find_struct(file, args_ident) {
        Some(s) => s,
        None => return Vec::new(),
    };
    let named = match &st.fields {
        syn::Fields::Named(n) => n,
        _ => return Vec::new(),
    };
    let defaults = extract_defaults(file, args_ident);
    let mut out = Vec::new();
    for field in &named.named {
        if has_serde_skip(&field.attrs) {
            continue;
        }
        let ident = match &field.ident {
            Some(i) => i.to_string(),
            None => continue,
        };
        let key = get_serde_rename(&field.attrs).unwrap_or_else(|| ident.clone());
        let (ty, optional) = match option_inner(&field.ty) {
            Some(inner) => (map_type(inner, ctx, value_types), true),
            None => (map_type(&field.ty, ctx, value_types), false),
        };
        let doc = collapse_doc(&extract_doc(&field.attrs));
        // An em-dash means no default was discoverable (derived Default, or a
        // non-literal initializer), leave it off rather than guess.
        let default = defaults
            .get(&ident)
            .filter(|d| d.as_str() != "\u{2014}")
            .cloned();
        out.push(FieldEntry {
            key,
            ty,
            optional,
            default,
            doc,
        });
    }
    out
}

// Translate a Rust field type to a JSON-shaped FieldType. Records any nested
// non-asset struct it links to in `value_types` so it gets its own entry.
fn map_type(ty: &syn::Type, ctx: &Ctx, value_types: &mut BTreeSet<String>) -> FieldType {
    match ty {
        syn::Type::Path(tp) => {
            let seg = match tp.path.segments.last() {
                Some(s) => s,
                None => return FieldType::Object,
            };
            let id = seg.ident.to_string();
            match id.as_str() {
                "f32" | "f64" => FieldType::Float,
                "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "usize" | "isize" => {
                    FieldType::Integer
                }
                "bool" => FieldType::Bool,
                "String" | "AssetId" => FieldType::Str,
                // serde_json::Value and maps are open-ended JSON objects.
                "Value" | "HashMap" | "BTreeMap" => FieldType::Object,
                "Option" | "Box" => first_generic(seg)
                    .map(|inner| map_type(inner, ctx, value_types))
                    .unwrap_or(FieldType::Object),
                "Vec" => {
                    let elem = first_generic(seg)
                        .map(|inner| map_type(inner, ctx, value_types))
                        .unwrap_or(FieldType::Object);
                    FieldType::Array {
                        elem: Box::new(elem),
                        len: None,
                    }
                }
                _ => {
                    if let Some(variants) = ctx.enums.get(&id) {
                        FieldType::Enum(variants.clone())
                    } else if let Some(name) = ctx.comp_by_struct.get(&id) {
                        // A field embedding another asset's struct links to that
                        // asset; it is documented in its own category already.
                        FieldType::Named(name.clone())
                    } else if ctx.struct_file_idx.contains_key(&id) {
                        value_types.insert(id.clone());
                        FieldType::Named(id)
                    } else {
                        FieldType::Object
                    }
                }
            }
        }
        syn::Type::Array(arr) => {
            let elem = map_type(&arr.elem, ctx, value_types);
            FieldType::Array {
                elem: Box::new(elem),
                len: array_len(&arr.len),
            }
        }
        syn::Type::Reference(r) => map_type(&r.elem, ctx, value_types),
        _ => FieldType::Object,
    }
}

// syn helpers

fn find_struct<'a>(file: &'a syn::File, ident: &str) -> Option<&'a syn::ItemStruct> {
    file.items.iter().find_map(|item| match item {
        syn::Item::Struct(s) if s.ident == ident => Some(s),
        _ => None,
    })
}

fn struct_doc(ident: &str, ctx: &Ctx) -> String {
    match ctx.struct_file_idx.get(ident) {
        Some(i) => find_struct(&ctx.files[*i], ident)
            .map(|s| extract_doc(&s.attrs))
            .unwrap_or_default(),
        None => String::new(),
    }
}

fn first_generic(seg: &syn::PathSegment) -> Option<&syn::Type> {
    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
        for a in &args.args {
            if let syn::GenericArgument::Type(t) = a {
                return Some(t);
            }
        }
    }
    None
}

fn option_inner(ty: &syn::Type) -> Option<&syn::Type> {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last()
        && seg.ident == "Option"
    {
        return first_generic(seg);
    }
    None
}

fn array_len(expr: &syn::Expr) -> Option<usize> {
    if let syn::Expr::Lit(l) = expr
        && let syn::Lit::Int(i) = &l.lit
    {
        return i.base10_parse::<usize>().ok();
    }
    None
}

// The string literal NAME constant on a Component impl, if any.
fn name_const(imp: &syn::ItemImpl) -> Option<String> {
    imp.items.iter().find_map(|it| {
        let c = match it {
            syn::ImplItem::Const(c) if c.ident == "NAME" => c,
            _ => return None,
        };
        if let syn::Expr::Lit(lit) = &c.expr
            && let syn::Lit::Str(s) = &lit.lit
        {
            return Some(s.value());
        }
        None
    })
}

// The struct named by a Component impl's `type Args = …`, if any.
fn component_args_struct(imp: &syn::ItemImpl) -> Option<String> {
    imp.items.iter().find_map(|it| match it {
        syn::ImplItem::Type(t) if t.ident == "Args" => match &t.ty {
            syn::Type::Path(tp) => tp.path.segments.last().map(|s| s.ident.to_string()),
            _ => None,
        },
        _ => None,
    })
}

// True for fields that carry #[serde(skip)] (exact token, not skip_serializing).
fn has_serde_skip(attrs: &[syn::Attribute]) -> bool {
    for attr in attrs {
        if !attr.path().is_ident("serde") {
            continue;
        }
        if let syn::Meta::List(list) = &attr.meta {
            for part in list.tokens.to_string().split(',') {
                if part.trim() == "skip" {
                    return true;
                }
            }
        }
    }
    false
}

// Returns the value of #[serde(rename = "...")] if present on a field.
fn get_serde_rename(attrs: &[syn::Attribute]) -> Option<String> {
    serde_kv(attrs, "rename")
}

// Returns the string value of a `key = "..."` pair inside any #[serde(...)]
// attribute. The `= ` boundary check keeps `rename` from matching `rename_all`.
fn serde_kv(attrs: &[syn::Attribute], key: &str) -> Option<String> {
    for attr in attrs {
        if !attr.path().is_ident("serde") {
            continue;
        }
        let list = match &attr.meta {
            syn::Meta::List(list) => list,
            _ => continue,
        };
        let tokens = list.tokens.to_string();
        for part in tokens.split(',') {
            let part = part.trim();
            let rest = match part.strip_prefix(key) {
                Some(r) => r.trim_start(),
                None => continue,
            };
            let rest = match rest.strip_prefix('=') {
                Some(r) => r.trim(),
                None => continue,
            };
            if let (Some(a), Some(b)) = (rest.find('"'), rest.rfind('"'))
                && a != b
            {
                return Some(rest[a + 1..b].to_string());
            }
        }
    }
    None
}

// Apply a serde `rename_all` rule to a PascalCase variant ident.
fn apply_case(ident: &str, rule: Option<&str>) -> String {
    match rule {
        None | Some("PascalCase") => ident.to_string(),
        Some("lowercase") => ident.to_lowercase(),
        Some("UPPERCASE") => ident.to_uppercase(),
        Some("snake_case") => split_words(ident).join("_"),
        Some("SCREAMING_SNAKE_CASE") => split_words(ident).join("_").to_uppercase(),
        Some("kebab-case") => split_words(ident).join("-"),
        Some("camelCase") => {
            let words = split_words(ident);
            let mut s = String::new();
            for (i, w) in words.iter().enumerate() {
                if i == 0 {
                    s.push_str(w);
                } else {
                    let mut chars = w.chars();
                    if let Some(f) = chars.next() {
                        s.push(f.to_ascii_uppercase());
                        s.push_str(chars.as_str());
                    }
                }
            }
            s
        }
        Some(_) => ident.to_string(),
    }
}

// Split a PascalCase ident into lowercase words: "VertexInstanced" -> [vertex,
// instanced]. Acronym runs are not special-cased (none occur in asset enums).
fn split_words(ident: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    for (i, c) in ident.chars().enumerate() {
        if c.is_uppercase() && i != 0 && !cur.is_empty() {
            words.push(std::mem::take(&mut cur));
        }
        cur.push(c.to_ascii_lowercase());
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

// Default extraction (from impl Default)

// Render an expression from a Default::default() body to a display string.
fn render_expr(expr: &syn::Expr) -> String {
    match expr {
        syn::Expr::Lit(l) => match &l.lit {
            syn::Lit::Float(f) => f.to_string(),
            syn::Lit::Int(i) => i.base10_digits().to_string(),
            syn::Lit::Bool(b) => b.value.to_string(),
            syn::Lit::Str(s) => format!("\"{}\"", s.value()),
            _ => "\u{2014}".to_string(),
        },
        syn::Expr::Array(arr) => {
            let items: Vec<String> = arr.elems.iter().map(render_expr).collect();
            format!("[{}]", items.join(", "))
        }
        syn::Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => {
            let inner = render_expr(&u.expr);
            if inner != "\u{2014}" {
                format!("-{}", inner)
            } else {
                "\u{2014}".to_string()
            }
        }
        // "string".to_string()
        syn::Expr::MethodCall(mc) if mc.method == "to_string" => render_expr(&mc.receiver),
        // None, or other path-expressions
        syn::Expr::Path(p) => {
            let s = p
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            if s == "None" {
                "null".to_string()
            } else {
                "\u{2014}".to_string()
            }
        }
        _ => "\u{2014}".to_string(),
    }
}

// Extract field_name -> default_value_string from impl Default for struct_name.
fn extract_defaults(file: &syn::File, struct_name: &str) -> HashMap<String, String> {
    let mut defaults = HashMap::new();
    for item in &file.items {
        let imp = match item {
            syn::Item::Impl(i) => i,
            _ => continue,
        };
        let is_default_trait = imp
            .trait_
            .as_ref()
            .map(|(_, p, _)| {
                p.segments
                    .last()
                    .map(|s| s.ident == "Default")
                    .unwrap_or(false)
            })
            .unwrap_or(false);
        if !is_default_trait {
            continue;
        }
        let self_name = match imp.self_ty.as_ref() {
            syn::Type::Path(tp) => tp
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default(),
            _ => continue,
        };
        if self_name != struct_name {
            continue;
        }
        for impl_item in &imp.items {
            let f = match impl_item {
                syn::ImplItem::Fn(f) if f.sig.ident == "default" => f,
                _ => continue,
            };
            for stmt in &f.block.stmts {
                collect_fields_from_stmt(stmt, &mut defaults);
            }
        }
    }
    defaults
}

fn collect_fields_from_stmt(stmt: &syn::Stmt, out: &mut HashMap<String, String>) {
    if let syn::Stmt::Expr(e, _) = stmt {
        collect_fields_from_expr(e, out);
    }
}

fn collect_fields_from_expr(expr: &syn::Expr, out: &mut HashMap<String, String>) {
    match expr {
        syn::Expr::Struct(es) => {
            for fv in &es.fields {
                if let syn::Member::Named(n) = &fv.member {
                    out.insert(n.to_string(), render_expr(&fv.expr));
                }
            }
        }
        syn::Expr::Block(eb) => {
            for stmt in &eb.block.stmts {
                collect_fields_from_stmt(stmt, out);
            }
        }
        _ => {}
    }
}

// Doc-comment helpers

fn extract_doc(attrs: &[syn::Attribute]) -> String {
    let mut doc = String::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        let nv = match &attr.meta {
            syn::Meta::NameValue(nv) => nv,
            _ => continue,
        };
        let lit_str = match &nv.value {
            syn::Expr::Lit(l) => match &l.lit {
                syn::Lit::Str(s) => s,
                _ => continue,
            },
            _ => continue,
        };
        let line = lit_str.value();
        let line = line.strip_prefix(' ').unwrap_or(&line);
        doc.push_str(line);
        doc.push('\n');
    }
    while doc.ends_with('\n') {
        doc.pop();
    }
    doc
}

// Collapse a multi-line field doc to a single line for a bullet.
fn collapse_doc(doc: &str) -> String {
    doc.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn first_paragraph(doc: &str) -> String {
    let para = doc.split("\n\n").next().unwrap_or("");
    para.split('\n')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

// Remove markdown table lines (starting with '|') from a doc string.
// Collapses the resulting double-blank lines left behind.
fn strip_table_lines(doc: &str) -> String {
    let mut out = String::new();
    let mut prev_blank = false;
    for line in doc.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('|') {
            continue;
        }
        let is_blank = trimmed.is_empty();
        if is_blank && prev_blank {
            continue;
        }
        out.push_str(line);
        out.push('\n');
        prev_blank = is_blank;
    }
    out.trim_end().to_string()
}
