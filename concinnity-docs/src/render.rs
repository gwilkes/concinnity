// Renders the per-asset reference pages from the docs extracted at build time.
// Each page's body is the same `full_doc` string surfaced by the
// describe_asset_type tool, so a generated public/assets/<Name>.md matches what
// the server serves for that type.
//
// Shared by two compilation units:
//   - build.rs includes it via #[path] to write the public/assets/*.md pages.
//   - the crate compiles it normally, so the lib tests can re-render from
//     ASSET_DOCS and assert the committed pages are in sync.
// Kept std-only so the build script can include it without extra dependencies.

use std::collections::HashMap;

// Leading marker on every generated page. The browser pipeline strips it before
// rendering, but it warns a human (or an AI) editing the file by hand.
pub const AUTOGEN_MARKER: &str = "<!-- Auto-generated - do not edit. -->";

// Field type rendering
//
// build.rs translates each Rust field type into one of these JSON-shaped
// descriptors, then this module renders it to an English phrase. Keeping the
// rendering here (std-only, no syn) lets it be unit-tested in the crate's test
// build, away from the build script.

// A JSON-shaped description of a field's type. `Enum` carries the accepted
// string values; `Named`/`NamedEnum` carry a documented type's name, which is
// also its relative `Name.md` link target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldType {
    Bool,
    Float,
    Integer,
    Str,
    // A string restricted to a closed set of values (a Rust enum that
    // serializes to a string) with no per-value documentation. Holds the
    // serialized variant strings in order, rendered inline.
    Enum(Vec<String>),
    // A free-form JSON object: `serde_json::Value`, a map, or an unrecognised
    // type with no documented shape.
    Object,
    // A documented value-type struct that has its own page, by name (also its
    // route). Rendered as a JSON object.
    Named(String),
    // A documented string enum that has its own page (its values carry their
    // own docs there), by name. Rendered as a string that links to its page,
    // not as an object.
    NamedEnum(String),
    // `Some(n)` for a fixed-size `[T; n]` array, `None` for a variable `Vec<T>`.
    Array {
        elem: Box<FieldType>,
        len: Option<usize>,
    },
}

// A single documented field of an asset or value type.
pub struct FieldEntry {
    pub key: String,
    pub ty: FieldType,
    // True for `Option<T>` fields.
    pub optional: bool,
    // Rendered default value (e.g. `2048`, `true`, `[0.0, 0.0]`, `"metal"`),
    // or `None` when no default is discoverable (derived `Default`, or absent).
    pub default: Option<String>,
    pub doc: String,
}

// One value of a documented enum: its serialized string and its own doc line.
pub struct EnumValue {
    pub value: String,
    pub doc: String,
}

// An entry in the index table of contents: a type's name (and route) plus its
// one-line summary.
pub struct IndexEntry {
    pub name: String,
    pub summary: String,
}

// In-page-safe slug for a type name (lowercase, alphanumerics and hyphens).
// Used to resolve hand-written `](#slug)` cross-references back to a type name.
pub fn slug(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect::<String>()
        .to_lowercase()
}

// The link target for a documented type: a sibling `.md` file. Keeping the
// links relative and self-contained means the generated docs cross-link
// correctly when browsed as plain markdown (e.g. on GitHub); a docs viewer is
// free to rewrite the `.md` suffix to its own routes at render time.
fn doc_link(name: &str) -> String {
    format!("{name}.md")
}

// `a` / `a` or `b` / `a`, `b`, or `c` (each value in code ticks).
fn one_of(values: &[String]) -> String {
    let ticked: Vec<String> = values.iter().map(|v| format!("`{v}`")).collect();
    match ticked.as_slice() {
        [] => String::new(),
        [a] => a.clone(),
        [a, b] => format!("{a} or {b}"),
        [rest @ .., last] => format!("{}, or {}", rest.join(", "), last),
    }
}

// The pluralised noun phrase for an array element, e.g. `floats`, `strings`,
// `[WaterWave](WaterWave.md) objects`.
fn elem_plural(t: &FieldType) -> String {
    match t {
        FieldType::Bool => "booleans".to_string(),
        FieldType::Float => "floats".to_string(),
        FieldType::Integer => "integers".to_string(),
        FieldType::Str => "strings".to_string(),
        FieldType::Enum(values) => format!("strings (each one of {})", one_of(values)),
        FieldType::Object => "objects".to_string(),
        FieldType::Named(name) => format!("[{name}]({}) objects", doc_link(name)),
        FieldType::NamedEnum(name) => format!("strings (see [{name}]({}))", doc_link(name)),
        FieldType::Array { elem, len } => match len {
            Some(n) => format!("arrays of {n} {}", elem_plural(elem)),
            None => format!("arrays of {}", elem_plural(elem)),
        },
    }
}

// The capitalised, sentence-leading phrase for a field type, e.g. `A string`,
// `An array of 4 floats`, `A [PropCollider](PropCollider.md) object`.
pub fn type_phrase(t: &FieldType) -> String {
    match t {
        FieldType::Bool => "A boolean".to_string(),
        FieldType::Float => "A float".to_string(),
        FieldType::Integer => "An integer".to_string(),
        FieldType::Str => "A string".to_string(),
        FieldType::Enum(values) => format!("A string (one of {})", one_of(values)),
        FieldType::Object => "An object".to_string(),
        FieldType::Named(name) => format!("A [{name}]({}) object", doc_link(name)),
        FieldType::NamedEnum(name) => format!("A string (see [{name}]({}))", doc_link(name)),
        FieldType::Array { elem, len } => match len {
            Some(n) => format!("An array of {n} {}", elem_plural(elem)),
            None => format!("An array of {}", elem_plural(elem)),
        },
    }
}

// True when the prose already states a concrete default, so appending one would
// be redundant. Kept narrow ("default") to avoid suppressing real defaults whose
// docs merely contain a stray "when no …" / "leave …".
fn doc_states_default(doc_lower: &str) -> bool {
    doc_lower.contains("default")
}

// True when the prose already conveys that the field is optional / omittable, so
// a trailing "Optional." would be redundant.
fn doc_states_optional(doc_lower: &str) -> bool {
    [
        "default",
        "if omitted",
        "when absent",
        "when no",
        "when unset",
        "optional",
        "unset",
        "or null",
        "leave",
        "omit",
    ]
    .iter()
    .any(|needle| doc_lower.contains(needle))
}

// Render one field as a markdown bullet: type phrase, the field's own doc, and
// a default/optional clause unless the doc already states it.
pub fn render_field_bullet(f: &FieldEntry) -> String {
    let mut s = format!("- `{}`: {}.", f.key, type_phrase(&f.ty));
    let doc = f.doc.trim();
    if !doc.is_empty() {
        s.push(' ');
        s.push_str(doc);
        if !doc.ends_with(['.', '!', '?', ':']) {
            s.push('.');
        }
    }
    let lower = doc.to_lowercase();
    match &f.default {
        Some(def) if def != "null" && !doc_states_default(&lower) => {
            s.push_str(&format!(" Defaults to `{def}`."));
        }
        _ if f.optional && !doc_states_optional(&lower) => {
            s.push_str(" Optional.");
        }
        _ => {}
    }
    s
}

// Render the `## Parameters` section for a set of fields, or an empty string
// when the type has no documented fields.
pub fn render_parameters(fields: &[FieldEntry]) -> String {
    if fields.is_empty() {
        return String::new();
    }
    let mut s = String::from("## Parameters\n\n");
    for f in fields {
        s.push_str(&render_field_bullet(f));
        s.push('\n');
    }
    s
}

// Render the `## Values` section for a documented enum, one bullet per
// serialized value with its own doc line. Empty when the enum has no values.
pub fn render_values(values: &[EnumValue]) -> String {
    if values.is_empty() {
        return String::new();
    }
    let mut s = String::from("## Values\n\n");
    for v in values {
        s.push_str(&format!("- `{}`", v.value));
        let doc = v.doc.trim();
        if !doc.is_empty() {
            s.push_str(": ");
            s.push_str(doc);
            if !doc.ends_with(['.', '!', '?', ':']) {
                s.push('.');
            }
        }
        s.push('\n');
    }
    s
}

// Rewrite a doc body's cross-references to documented types into the relative
// `](Name.md)` form, so they cross-link when browsed as plain markdown. Two
// source forms are recognised, resolving through `name_for_slug`:
//   - a hand-written `[Text](#slug)` anchor (the single-page workaround), and
//   - an idiomatic rustdoc shortcut link `[Type]` (no target), where `Type` is
//     a documented name.
// Anything that does not resolve to a documented type is left untouched.
pub fn rewrite_doc_links(doc: &str, name_for_slug: &HashMap<String, String>) -> String {
    let names: std::collections::HashSet<&str> =
        name_for_slug.values().map(String::as_str).collect();
    rewrite_shortcut_links(&rewrite_anchor_links(doc, name_for_slug), &names)
}

// `[Text](#slug)` -> `[Text](Name.md)` for a known slug.
fn rewrite_anchor_links(doc: &str, name_for_slug: &HashMap<String, String>) -> String {
    const NEEDLE: &str = "](#";
    let mut out = String::with_capacity(doc.len());
    let mut rest = doc;
    while let Some(pos) = rest.find(NEEDLE) {
        let after = &rest[pos + NEEDLE.len()..];
        if let Some(end) = after.find(')') {
            let anchor = &after[..end];
            if let Some(name) = name_for_slug.get(anchor) {
                out.push_str(&rest[..pos]);
                out.push_str("](");
                out.push_str(&doc_link(name));
                out.push(')');
                rest = &after[end + 1..];
                continue;
            }
        }
        // No closing paren or unknown anchor: emit through the needle and move on.
        out.push_str(&rest[..pos + NEEDLE.len()]);
        rest = &rest[pos + NEEDLE.len()..];
    }
    out.push_str(rest);
    out
}

// `[Type]` -> `[Type](Type.md)` when `Type` is a documented name and the
// brackets are a bare shortcut link: not already a link (`[Type](...)`), not a
// reference label (`[text][Type]` or `[Type]: …`).
fn rewrite_shortcut_links(doc: &str, names: &std::collections::HashSet<&str>) -> String {
    let mut out = String::with_capacity(doc.len());
    let mut rest = doc;
    let mut prev = '\0';
    while let Some(pos) = rest.find('[') {
        out.push_str(&rest[..pos]);
        if pos > 0 {
            prev = rest[..pos].chars().next_back().unwrap();
        }
        let after_open = &rest[pos + 1..];
        let handled = match after_open.find(']') {
            Some(end) => {
                let inner = &after_open[..end];
                let next = after_open[end + 1..].chars().next().unwrap_or(' ');
                if names.contains(inner) && prev != ']' && next != '(' && next != '[' && next != ':'
                {
                    out.push('[');
                    out.push_str(inner);
                    out.push_str("](");
                    out.push_str(&doc_link(inner));
                    out.push(')');
                    prev = ')';
                    rest = &after_open[end + 1..];
                    true
                } else {
                    false
                }
            }
            None => false,
        };
        if !handled {
            out.push('[');
            prev = '[';
            rest = after_open;
        }
    }
    out.push_str(rest);
    out
}

// Assemble a full page: the auto-generated marker, the `# Name` heading, then
// the body (description plus the generated Parameters/Values section).
pub fn render_page(name: &str, body: &str) -> String {
    let mut out = String::new();
    out.push_str(AUTOGEN_MARKER);
    out.push_str("\n\n# ");
    out.push_str(name);
    out.push_str("\n\n");
    out.push_str(body.trim());
    while out.ends_with('\n') {
        out.pop();
    }
    out.push('\n');
    out
}

// Render the index page: an alphabetical list of every asset, then a list of
// the referenced value types and enums, each linking to its own page.
pub fn render_index(assets: &[IndexEntry], ref_types: &[IndexEntry]) -> String {
    let mut out = String::new();
    out.push_str(AUTOGEN_MARKER);
    out.push_str("\n\n# Assets\n\n");
    for a in assets {
        out.push_str(&index_line(a));
    }
    if !ref_types.is_empty() {
        out.push_str("\n## Reference types\n\n");
        for t in ref_types {
            out.push_str(&index_line(t));
        }
    }
    while out.ends_with('\n') {
        out.pop();
    }
    out.push('\n');
    out
}

fn index_line(e: &IndexEntry) -> String {
    let summary = e.summary.trim();
    if summary.is_empty() {
        format!("- [{}]({})\n", e.name, doc_link(&e.name))
    } else {
        format!("- [{}]({}) - {}\n", e.name, doc_link(&e.name), summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(
        key: &str,
        ty: FieldType,
        optional: bool,
        default: Option<&str>,
        doc: &str,
    ) -> FieldEntry {
        FieldEntry {
            key: key.to_string(),
            ty,
            optional,
            default: default.map(str::to_string),
            doc: doc.to_string(),
        }
    }

    #[test]
    fn type_phrase_covers_scalars_and_enums() {
        assert_eq!(type_phrase(&FieldType::Bool), "A boolean");
        assert_eq!(type_phrase(&FieldType::Float), "A float");
        assert_eq!(type_phrase(&FieldType::Integer), "An integer");
        assert_eq!(type_phrase(&FieldType::Str), "A string");
        assert_eq!(type_phrase(&FieldType::Object), "An object");
        assert_eq!(
            type_phrase(&FieldType::Array {
                elem: Box::new(FieldType::Object),
                len: None,
            }),
            "An array of objects"
        );
        assert_eq!(
            type_phrase(&FieldType::Enum(vec!["vertex".into(), "fragment".into()])),
            "A string (one of `vertex` or `fragment`)"
        );
        assert_eq!(
            type_phrase(&FieldType::Enum(vec![
                "quality".into(),
                "balanced".into(),
                "performance".into()
            ])),
            "A string (one of `quality`, `balanced`, or `performance`)"
        );
    }

    #[test]
    fn type_phrase_covers_arrays_and_named() {
        let arr4 = FieldType::Array {
            elem: Box::new(FieldType::Float),
            len: Some(4),
        };
        assert_eq!(type_phrase(&arr4), "An array of 4 floats");
        let vec_str = FieldType::Array {
            elem: Box::new(FieldType::Str),
            len: None,
        };
        assert_eq!(type_phrase(&vec_str), "An array of strings");
        let vec_named = FieldType::Array {
            elem: Box::new(FieldType::Named("WaterWave".into())),
            len: None,
        };
        assert_eq!(
            type_phrase(&vec_named),
            "An array of [WaterWave](WaterWave.md) objects"
        );
        assert_eq!(
            type_phrase(&FieldType::Named("PropCollider".into())),
            "A [PropCollider](PropCollider.md) object"
        );
        assert_eq!(
            type_phrase(&FieldType::NamedEnum("ShaderKind".into())),
            "A string (see [ShaderKind](ShaderKind.md))"
        );
        let nested = FieldType::Array {
            elem: Box::new(FieldType::Array {
                elem: Box::new(FieldType::Float),
                len: Some(2),
            }),
            len: None,
        };
        assert_eq!(type_phrase(&nested), "An array of arrays of 2 floats");
    }

    #[test]
    fn slug_matches_lowercase_anchor() {
        assert_eq!(slug("PropCollider"), "propcollider");
        assert_eq!(slug("Camera3D"), "camera3d");
    }

    #[test]
    fn bullet_appends_default_when_doc_silent() {
        let b = render_field_bullet(&field(
            "frames_in_flight",
            FieldType::Integer,
            false,
            Some("2"),
            "Preferred number of frames in flight",
        ));
        assert_eq!(
            b,
            "- `frames_in_flight`: An integer. Preferred number of frames in flight. Defaults to `2`."
        );
    }

    #[test]
    fn bullet_skips_default_when_doc_mentions_it() {
        let b = render_field_bullet(&field(
            "validation",
            FieldType::Bool,
            true,
            Some("null"),
            "Enable validation. Defaults to true in debug builds.",
        ));
        // No appended "Defaults to" / "Optional." (the doc already covers it).
        assert_eq!(
            b,
            "- `validation`: A boolean. Enable validation. Defaults to true in debug builds."
        );
    }

    #[test]
    fn bullet_marks_optional_when_null_default() {
        let b = render_field_bullet(&field(
            "collider",
            FieldType::Named("PropCollider".into()),
            true,
            Some("null"),
            "Collision volume",
        ));
        assert_eq!(
            b,
            "- `collider`: A [PropCollider](PropCollider.md) object. Collision volume. Optional."
        );
    }

    #[test]
    fn parameters_section_empty_for_no_fields() {
        assert_eq!(render_parameters(&[]), "");
    }

    #[test]
    fn values_section_renders_each_value() {
        let vals = vec![
            EnumValue {
                value: "left".into(),
                doc: "Pack against the left edge".into(),
            },
            EnumValue {
                value: "center".into(),
                doc: String::new(),
            },
        ];
        let s = render_values(&vals);
        assert!(s.starts_with("## Values\n\n"));
        assert!(s.contains("- `left`: Pack against the left edge."));
        assert!(s.contains("- `center`\n"));
    }

    #[test]
    fn values_section_empty_for_no_values() {
        assert_eq!(render_values(&[]), "");
    }

    #[test]
    fn rewrite_doc_links_resolves_known_anchors_only() {
        let mut map = HashMap::new();
        map.insert("audioemitter".to_string(), "AudioEmitter".to_string());
        map.insert("camera3d".to_string(), "Camera3D".to_string());
        let doc = "Played by an [AudioEmitter](#audioemitter); see [Camera3D](#camera3d) \
                   and an [Unknown](#unknown) anchor.";
        let out = rewrite_doc_links(doc, &map);
        assert!(out.contains("[AudioEmitter](AudioEmitter.md)"));
        assert!(out.contains("[Camera3D](Camera3D.md)"));
        // Unknown anchors are left untouched.
        assert!(out.contains("[Unknown](#unknown)"));
    }

    #[test]
    fn rewrite_doc_links_leaves_relative_links_alone() {
        let map = HashMap::new();
        let doc = "A [PropCollider](PropCollider.md) object.";
        assert_eq!(rewrite_doc_links(doc, &map), doc);
    }

    #[test]
    fn rewrite_doc_links_rewrites_shortcut_type_links() {
        let mut map = HashMap::new();
        map.insert("shadowupdate".to_string(), "ShadowUpdate".to_string());
        let doc = "How often. See [ShadowUpdate]. Also [ShadowUpdate](ShadowUpdate.md) stays.";
        let out = rewrite_doc_links(doc, &map);
        assert!(out.contains("See [ShadowUpdate](ShadowUpdate.md)."));
        // The already-linked occurrence is not double-wrapped.
        assert!(!out.contains(".md.md"));
        assert!(!out.contains(".md)](ShadowUpdate.md)"));
    }

    #[test]
    fn rewrite_doc_links_skips_non_type_brackets() {
        let mut map = HashMap::new();
        map.insert("prop".to_string(), "Prop".to_string());
        let doc = "An array [0, 1] and a [Prop] and a label [text][Prop].";
        let out = rewrite_doc_links(doc, &map);
        assert!(out.contains("and a [Prop](Prop.md) and"));
        assert!(out.contains("[0, 1]")); // not a documented type
        assert!(out.contains("[text][Prop]")); // collapsed-reference label left alone
    }

    #[test]
    fn render_page_has_marker_then_h1_then_body() {
        let page = render_page("Prop", "A prop.\n\n## Parameters\n\n- `x`: A float.");
        let mut lines = page.lines();
        assert_eq!(lines.next(), Some(AUTOGEN_MARKER));
        assert_eq!(lines.next(), Some(""));
        assert_eq!(lines.next(), Some("# Prop"));
        assert!(page.contains("A prop."));
        assert!(page.contains("## Parameters"));
        assert!(page.ends_with('\n'));
        assert!(!page.ends_with("\n\n"));
    }

    fn idx(name: &str, summary: &str) -> IndexEntry {
        IndexEntry {
            name: name.to_string(),
            summary: summary.to_string(),
        }
    }

    #[test]
    fn render_index_lists_assets_then_reference_types() {
        let assets = vec![idx("Camera3D", "A camera."), idx("Prop", "A prop.")];
        let refs = vec![idx("PropCollider", "A collision volume.")];
        let md = render_index(&assets, &refs);
        assert!(md.starts_with(AUTOGEN_MARKER));
        assert!(md.contains("# Assets"));
        assert!(md.contains("- [Camera3D](Camera3D.md) - A camera."));
        assert!(md.contains("- [Prop](Prop.md) - A prop."));
        assert!(md.contains("## Reference types"));
        assert!(md.contains("- [PropCollider](PropCollider.md) - A collision volume."));
        let assets_pos = md.find("# Assets").unwrap();
        let refs_pos = md.find("## Reference types").unwrap();
        assert!(assets_pos < refs_pos);
    }

    #[test]
    fn render_index_omits_reference_section_when_empty() {
        let md = render_index(&[idx("Prop", "A prop.")], &[]);
        assert!(!md.contains("Reference types"));
        assert!(md.ends_with('\n'));
    }
}
