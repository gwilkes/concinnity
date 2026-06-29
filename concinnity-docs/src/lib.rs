// LLM-facing asset reference. Built at compile time by this crate's build.rs,
// which parses the concinnity-core asset modules with syn and extracts the
// rustdoc on each Component struct. The table is embedded in the binary; no
// runtime file I/O. The same data is also written to public/assets/*.md, one
// page per type, for a docs site to fetch and render.
//
// Two consumers:
//   - concinnity-ai's describe_asset_type tool returns the full_doc field for a
//     single type on demand.
//   - concinnity-infra's new-chat context emits the flat asset list (the
//     summary field).

// The rendering library, shared with build.rs via #[path]. Compiled only under
// #[cfg(test)] here so its build-time helpers carry no runtime weight; the lib
// tests below re-render from ASSET_DOCS to assert the committed pages are in
// sync.
#[cfg(test)]
#[path = "render.rs"]
mod render;

include!(concat!(env!("OUT_DIR"), "/assets_doc.rs"));

// Look up the full doc for an asset type by NAME (case-insensitive). Finds both
// top-level assets and the reference types (value-type structs, documented
// enums) they embed.
pub fn describe(type_name: &str) -> Option<&'static AssetDoc> {
    ASSET_DOCS
        .iter()
        .find(|d| d.type_name.eq_ignore_ascii_case(type_name))
}

// Render the flat list of `Name - summary` lines for every authorable asset,
// the shape `gather_new_chat_context` wants. Assets appear alphabetically (the
// order build.rs emits them in). Reference types are excluded: they document
// the nested objects and enums assets embed, not user-declarable assets.
pub fn asset_summary_block() -> String {
    let assets = || ASSET_DOCS.iter().filter(|d| !d.is_reference_type);
    let max_name = assets().map(|d| d.type_name.len()).max().unwrap_or(0);
    let mut out = String::new();
    for d in assets() {
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
    fn describe_finds_reference_types() {
        // A nested value type embedded by an asset (Prop.collider) is documented
        // and reachable by name, not just inlined into the asset page.
        let d = describe("PropCollider").expect("PropCollider should be in ASSET_DOCS");
        assert!(d.is_reference_type);
    }

    #[test]
    fn asset_summary_block_contains_all_assets() {
        let block = asset_summary_block();
        for d in ASSET_DOCS {
            if d.is_reference_type {
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
    fn asset_summary_block_excludes_reference_types() {
        let block = asset_summary_block();
        for d in ASSET_DOCS {
            if d.is_reference_type {
                assert!(
                    !block.lines().any(|l| l.starts_with(d.type_name)),
                    "block should not list reference type {}",
                    d.type_name
                );
            }
        }
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

    // No `](#anchor)` cross-references survive into the embedded docs: every one
    // is rewritten to a relative `Name.md` link at build time.
    #[test]
    fn no_in_page_anchor_links_remain() {
        for d in ASSET_DOCS {
            assert!(
                !d.full_doc.contains("](#"),
                "{} still has an in-page anchor link: {:?}",
                d.type_name,
                d.full_doc
            );
        }
    }

    fn pages_dir() -> std::path::PathBuf {
        std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/public/assets")).to_path_buf()
    }

    #[test]
    fn each_page_matches_generated_docs() {
        use crate::render::render_page;
        for d in ASSET_DOCS {
            let path = pages_dir().join(format!("{}.md", d.type_name));
            let on_disk = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!(
                    "read {}: {e}; rebuild concinnity-infra to regenerate the asset pages",
                    path.display()
                )
            });
            let expected = render_page(d.type_name, d.full_doc);
            assert_eq!(
                on_disk, expected,
                "public/assets/{}.md is out of date - rebuild concinnity-infra to regenerate it",
                d.type_name
            );
        }
    }

    #[test]
    fn index_matches_generated_docs() {
        use crate::render::{IndexEntry, render_index};
        let entry = |d: &AssetDoc| IndexEntry {
            name: d.type_name.to_string(),
            summary: d.summary.to_string(),
        };
        let assets: Vec<IndexEntry> = ASSET_DOCS
            .iter()
            .filter(|d| !d.is_reference_type)
            .map(entry)
            .collect();
        let ref_types: Vec<IndexEntry> = ASSET_DOCS
            .iter()
            .filter(|d| d.is_reference_type)
            .map(entry)
            .collect();
        let expected = render_index(&assets, &ref_types);

        let path = pages_dir().join("index.md");
        let on_disk = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        assert_eq!(
            on_disk, expected,
            "public/assets/index.md is out of date - rebuild concinnity-infra to regenerate it"
        );
    }
}
