// LLM-facing asset reference. Built at compile time by this crate's build.rs,
// which parses the concinnity-core asset modules with syn and extracts the
// rustdoc on each Component struct. The table is embedded in the binary; no
// runtime file I/O.
//
// Two consumers:
//   - concinnity-ai's describe_asset_type tool returns the full_doc field for a
//     single type on demand.
//   - concinnity-infra's new-chat context emits the per-category one-liner list
//     (the summary field).

// The rendering library, shared with build.rs via #[path]. Compiled only under
// #[cfg(test)] here so its build-time helpers carry no runtime weight; the lib
// tests below re-render from ASSET_DOCS to assert the committed markdown is in
// sync.
#[cfg(test)]
#[path = "render.rs"]
mod render;

include!(concat!(env!("OUT_DIR"), "/assets_doc.rs"));

// Mirrors render::VALUE_TYPES_CATEGORY. That module is compiled only under
// #[cfg(test)], so the runtime filter below keeps its own copy;
// value_types_category_label_matches asserts the two never drift.
const VALUE_TYPES_CATEGORY: &str = "Value types";

// Look up the full doc for an asset type by NAME (case-insensitive).
pub fn describe(type_name: &str) -> Option<&'static AssetDoc> {
    ASSET_DOCS
        .iter()
        .find(|d| d.type_name.eq_ignore_ascii_case(type_name))
}

// Render the per-category list of `Name - summary` lines, the shape
// `gather_new_chat_context` wants. Categories appear in CATEGORIES order
// from build.rs; assets within a category appear in the order listed there.
// The synthetic "Value types" category is excluded (those document the
// nested objects assets embed, not user-declarable top-level assets).
pub fn category_summary_block() -> String {
    let declarable = || {
        ASSET_DOCS
            .iter()
            .filter(|d| d.category != VALUE_TYPES_CATEGORY)
    };
    let mut out = String::new();
    let mut current_cat: Option<&str> = None;
    let max_name = declarable().map(|d| d.type_name.len()).max().unwrap_or(0);
    for d in declarable() {
        if current_cat != Some(d.category) {
            if current_cat.is_some() {
                out.push('\n');
            }
            out.push_str("### ");
            out.push_str(d.category);
            out.push('\n');
            current_cat = Some(d.category);
        }
        // Pad type names so summaries align, purely cosmetic for the LLM.
        out.push_str(&format!(
            "{:<width$} - {}\n",
            d.type_name,
            d.summary,
            width = max_name
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_finds_known_type() {
        let d = describe("Texture").expect("Texture should be in ASSET_DOCS");
        assert_eq!(d.type_name, "Texture");
        assert!(!d.summary.is_empty());
        assert!(!d.full_doc.is_empty());
        assert!(d.full_doc.contains(d.summary));
    }

    #[test]
    fn describe_is_case_insensitive() {
        assert!(describe("texture").is_some());
        assert!(describe("TEXTURE").is_some());
    }

    #[test]
    fn describe_returns_none_for_unknown_type() {
        assert!(describe("NotARealAsset").is_none());
    }

    #[test]
    fn category_summary_block_contains_all_declarable_assets() {
        let block = category_summary_block();
        for d in ASSET_DOCS {
            if d.category == VALUE_TYPES_CATEGORY {
                continue;
            }
            assert!(
                block.contains(d.type_name),
                "block missing {}: {block}",
                d.type_name
            );
        }
    }

    #[test]
    fn value_types_category_label_matches() {
        assert_eq!(VALUE_TYPES_CATEGORY, crate::render::VALUE_TYPES_CATEGORY);
    }

    #[test]
    fn category_summary_block_excludes_value_types() {
        let block = category_summary_block();
        assert!(!block.contains("### Value types"));
    }

    #[test]
    fn category_summary_block_has_category_headers() {
        let block = category_summary_block();
        assert!(block.contains("### Geometry"));
        assert!(block.contains("### Lighting"));
    }

    #[test]
    fn asset_reference_md_matches_generated_docs() {
        use crate::render::{RefEntry, render_reference_md};

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/asset-reference.md");
        let on_disk = std::fs::read_to_string(path).unwrap_or_else(|e| {
            panic!("read {path}: {e}; rebuild concinnity-infra to generate it")
        });

        let entries: Vec<RefEntry> = ASSET_DOCS
            .iter()
            .map(|d| RefEntry {
                category: d.category,
                type_name: d.type_name,
                full_doc: d.full_doc,
            })
            .collect();
        let expected = render_reference_md(&entries);

        assert_eq!(
            on_disk, expected,
            "docs/asset-reference.md is out of date - rebuild concinnity-infra to regenerate it"
        );
    }

    #[test]
    fn all_summaries_are_single_line() {
        for d in ASSET_DOCS {
            assert!(
                !d.summary.contains('\n'),
                "{}'s summary spans multiple lines: {:?}",
                d.type_name,
                d.summary
            );
            assert!(!d.summary.is_empty(), "{} has empty summary", d.type_name);
        }
    }
}
