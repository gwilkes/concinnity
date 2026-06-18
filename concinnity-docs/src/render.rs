// Renders the asset reference document from the per-asset docs extracted at
// build time. The body of each entry is the same `full_doc` string surfaced by
// the describe_asset_type tool, so the generated docs/asset-reference.md matches
// what the server serves per type.
//
// Shared by two compilation units:
//   - build.rs includes it via #[path] to write docs/asset-reference.md.
//   - the crate compiles it normally, so the lib tests can re-render from
//     ASSET_DOCS and assert the committed file is in sync.
// Kept std-only so the build script can include it without extra dependencies.

// One asset's rendered entry: its category, type name, and full doc body
// (struct rustdoc followed by the generated parameter list).
pub struct RefEntry<'a> {
    pub category: &'a str,
    pub type_name: &'a str,
    pub full_doc: &'a str,
}

// Category heading the synthetic value-type entries are grouped under. These
// document the nested objects that asset fields embed (e.g. a Prop's collider)
// rather than user-declarable top-level assets, so they sort after every real
// category and are excluded from the chat-start declarable summary.
pub const VALUE_TYPES_CATEGORY: &str = "Value types";

// Field type rendering
//
// build.rs translates each Rust field type into one of these JSON-shaped
// descriptors, then this module renders it to an English phrase. Keeping the
// rendering here (std-only, no syn) lets it be unit-tested in the crate's test
// build, away from the build script.

// A JSON-shaped description of a field's type. `Enum` carries the accepted
// string values; `Named` carries a documented nested object's type name (which
// is also its in-page anchor).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldType {
    Bool,
    Float,
    Integer,
    Str,
    // A string restricted to a closed set of values (a Rust enum that
    // serializes to a string). Holds the serialized variant strings in order.
    Enum(Vec<String>),
    // A free-form JSON object: `serde_json::Value`, a map, or an unrecognised
    // type with no documented shape.
    Object,
    // A documented nested object, by value-type name (also its anchor).
    Named(String),
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

// In-page anchor for a value-type name, matching the React docNav `slugify`
// for the single-token PascalCase names used here (lowercase, alphanumerics
// and hyphens only).
pub fn slug(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect::<String>()
        .to_lowercase()
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
// `[WaterWave](#waterwave) objects`.
fn elem_plural(t: &FieldType) -> String {
    match t {
        FieldType::Bool => "booleans".to_string(),
        FieldType::Float => "floats".to_string(),
        FieldType::Integer => "integers".to_string(),
        FieldType::Str => "strings".to_string(),
        FieldType::Enum(values) => format!("strings (each one of {})", one_of(values)),
        FieldType::Object => "objects".to_string(),
        FieldType::Named(name) => format!("[{name}](#{}) objects", slug(name)),
        FieldType::Array { elem, len } => match len {
            Some(n) => format!("arrays of {n} {}", elem_plural(elem)),
            None => format!("arrays of {}", elem_plural(elem)),
        },
    }
}

// The capitalised, sentence-leading phrase for a field type, e.g. `A string`,
// `An array of 4 floats`, `A [PropCollider](#propcollider) object`.
pub fn type_phrase(t: &FieldType) -> String {
    match t {
        FieldType::Bool => "A boolean".to_string(),
        FieldType::Float => "A float".to_string(),
        FieldType::Integer => "An integer".to_string(),
        FieldType::Str => "A string".to_string(),
        FieldType::Enum(values) => format!("A string (one of {})", one_of(values)),
        FieldType::Object => "An object".to_string(),
        FieldType::Named(name) => format!("A [{name}](#{}) object", slug(name)),
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

// Render the `#### Parameters` section for a set of fields, or an empty string
// when the type has no documented fields.
pub fn render_parameters(fields: &[FieldEntry]) -> String {
    if fields.is_empty() {
        return String::new();
    }
    let mut s = String::from("#### Parameters\n\n");
    for f in fields {
        s.push_str(&render_field_bullet(f));
        s.push('\n');
    }
    s
}

// const INTRO: &str = "Every asset type the world build understands, grouped by category. Each \
//     entry gives the asset's purpose and its `args` fields with types and defaults. Generated \
//     from the rustdoc on each asset struct in the concinnity-core crate.";

// Render the full asset reference markdown. `entries` must already be ordered
// so that all entries sharing a category are contiguous; a `## category`
// heading is emitted each time the category changes.
pub fn render_reference_md(entries: &[RefEntry]) -> String {
    let mut out = String::new();
    out.push_str("<!-- Auto-generated by concinnity-docs/build.rs - do not edit. -->\n\n");
    out.push_str("# Asset Reference\n\n");

    let mut current_cat: Option<&str> = None;
    for e in entries {
        if current_cat != Some(e.category) {
            out.push_str("## ");
            out.push_str(e.category);
            out.push_str("\n\n");
            current_cat = Some(e.category);
        }
        out.push_str("### ");
        out.push_str(e.type_name);
        out.push_str("\n\n");
        out.push_str(e.full_doc.trim());
        out.push_str("\n\n");
    }

    // Normalise to a single trailing newline.
    while out.ends_with('\n') {
        out.pop();
    }
    out.push('\n');
    out
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
            "An array of [WaterWave](#waterwave) objects"
        );
        assert_eq!(
            type_phrase(&FieldType::Named("PropCollider".into())),
            "A [PropCollider](#propcollider) object"
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
            "- `collider`: A [PropCollider](#propcollider) object. Collision volume. Optional."
        );
    }

    #[test]
    fn parameters_section_empty_for_no_fields() {
        assert_eq!(render_parameters(&[]), "");
    }

    fn sample() -> Vec<RefEntry<'static>> {
        vec![
            RefEntry {
                category: "Geometry",
                type_name: "Mesh",
                full_doc: "A mesh.\n\n#### Parameters\n\n- `foo`: A string.",
            },
            RefEntry {
                category: "Geometry",
                type_name: "Model",
                full_doc: "A model.",
            },
            RefEntry {
                category: "Lighting",
                type_name: "PointLight",
                full_doc: "A point light.",
            },
        ]
    }

    #[test]
    fn emits_each_category_heading_once() {
        let md = render_reference_md(&sample());
        assert_eq!(md.matches("## Geometry").count(), 1);
        assert_eq!(md.matches("## Lighting").count(), 1);
    }

    #[test]
    fn emits_each_type_with_its_full_doc() {
        let md = render_reference_md(&sample());
        assert!(md.contains("### Mesh"));
        assert!(md.contains("### Model"));
        assert!(md.contains("### PointLight"));
        assert!(md.contains("A mesh."));
        assert!(md.contains("#### Parameters"));
    }

    #[test]
    fn category_heading_precedes_its_types() {
        let md = render_reference_md(&sample());
        let geometry = md.find("## Geometry").unwrap();
        let mesh = md.find("### Mesh").unwrap();
        let lighting = md.find("## Lighting").unwrap();
        let point = md.find("### PointLight").unwrap();
        assert!(geometry < mesh);
        assert!(mesh < lighting);
        assert!(lighting < point);
    }

    #[test]
    fn ends_with_single_newline() {
        let md = render_reference_md(&sample());
        assert!(md.ends_with('\n'));
        assert!(!md.ends_with("\n\n"));
    }

    #[test]
    fn starts_with_autogen_tag_then_title() {
        let md = render_reference_md(&sample());
        let mut lines = md.lines();
        assert_eq!(
            lines.next(),
            Some("<!-- Auto-generated by concinnity-docs/build.rs - do not edit. -->")
        );
        // The `# Asset Reference` title precedes the first category heading.
        let title = md.find("# Asset Reference").unwrap();
        let first_cat = md.find("## ").unwrap();
        assert!(title < first_cat);
    }

    #[test]
    fn empty_input_still_has_title() {
        let md = render_reference_md(&[]);
        assert!(md.starts_with("<!-- Auto-generated"));
        assert!(md.contains("# Asset Reference"));
        assert!(md.ends_with('\n'));
    }
}
