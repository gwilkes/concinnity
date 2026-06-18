// src/ecs/asset_id.rs
//
// Dense u32 asset identity. Asset names declared in world.jsonl are interned
// to an AssetId at build time; the blob and the runtime carry only the
// integer. Cross-references between assets (Prop -> Mesh, Material -> Texture,
// ...) are likewise resolved to AssetId during the build, so every runtime
// lookup is an integer compare instead of a string compare.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cell::RefCell;
use std::collections::HashMap;

// A dense integer handle for one asset. Assigned by the build-time interner
// in world.jsonl declaration order. Equality and hashing are integer ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct AssetId(pub u32);

impl std::fmt::Display for AssetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

// AssetId serializes as a plain u32 in every format. It deserializes from a
// u32 in non-self-describing formats (bincode, used for the blob defs table)
// and from either an integer or a name string in human-readable formats
// (JSON, used for `args_bytes`). A name string is resolved through the
// thread-local interner -- this is the build-time path that turns a
// world.jsonl reference into an id.
impl Serialize for AssetId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u32(self.0)
    }
}

struct AssetIdVisitor;

impl serde::de::Visitor<'_> for AssetIdVisitor {
    type Value = AssetId;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an asset id integer or a name string")
    }

    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<AssetId, E> {
        Ok(AssetId(v as u32))
    }
    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<AssetId, E> {
        Ok(AssetId(v as u32))
    }
    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<AssetId, E> {
        Ok(intern(v))
    }
    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<AssetId, E> {
        Ok(intern(&v))
    }
}

impl<'de> Deserialize<'de> for AssetId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            d.deserialize_any(AssetIdVisitor)
        } else {
            Ok(AssetId(u32::deserialize(d)?))
        }
    }
}

// `serde` `deserialize_with` helper for an optional cross-reference field.
//
// Accepts a name string (interned), an integer id, an empty string, or null;
// the latter two resolve to `None`. Apply with
// `#[serde(default, deserialize_with = "...")]` so a missing field is `None`.
pub fn de_opt_asset_ref<'de, D>(d: D) -> Result<Option<AssetId>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptVisitor;

    impl serde::de::Visitor<'_> for OptVisitor {
        type Value = Option<AssetId>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("an asset reference name string, id integer, or null")
        }

        fn visit_unit<E: serde::de::Error>(self) -> Result<Option<AssetId>, E> {
            Ok(None)
        }
        fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Option<AssetId>, E> {
            Ok(Some(AssetId(v as u32)))
        }
        fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Option<AssetId>, E> {
            Ok(Some(AssetId(v as u32)))
        }
        fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Option<AssetId>, E> {
            if v.is_empty() {
                Ok(None)
            } else {
                Ok(Some(intern(v)))
            }
        }
        fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Option<AssetId>, E> {
            self.visit_str(&v)
        }
    }

    d.deserialize_any(OptVisitor)
}

// Maps asset name strings to dense ids. Build-time only.
#[derive(Default)]
struct Interner {
    map: HashMap<String, u32>,
    names: Vec<String>,
}

impl Interner {
    fn intern(&mut self, name: &str) -> AssetId {
        if let Some(&id) = self.map.get(name) {
            return AssetId(id);
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_string());
        self.map.insert(name.to_string(), id);
        AssetId(id)
    }
}

thread_local! {
    static INTERNER: RefCell<Interner> = RefCell::new(Interner::default());
}

// Intern a name into the current thread's interner, returning its id. If the
// name was already interned the existing id is returned (idempotent).
pub fn intern(name: &str) -> AssetId {
    INTERNER.with(|i| i.borrow_mut().intern(name))
}

// Clear the thread-local interner. Call once at the start of a build so ids
// are dense and declaration-ordered for that build.
pub fn reset_interner() {
    INTERNER.with(|i| *i.borrow_mut() = Interner::default());
}

// Snapshot every interned name on the current thread, indexed by `AssetId`.
// Because ids are assigned in world.jsonl declaration order, `table[id]` is
// the declared name for that id. Used by the binary-only `crate::debug`
// module to remap runtime `AssetId`s back to their declared names.
#[allow(dead_code)] // consumed by the binary-only crate::debug module
pub fn name_table() -> Vec<String> {
    INTERNER.with(|i| i.borrow().names.clone())
}

// Pre-intern a batch of names in order so identity ids are dense and follow
// world.jsonl declaration order.
pub fn intern_all(names: &[&str]) {
    INTERNER.with(|i| {
        let mut interner = i.borrow_mut();
        for n in names {
            interner.intern(n);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_is_idempotent_and_dense() {
        reset_interner();
        intern_all(&["a", "b", "c"]);
        assert_eq!(intern("a"), AssetId(0));
        assert_eq!(intern("b"), AssetId(1));
        assert_eq!(intern("c"), AssetId(2));
        // a fresh reference past the pre-interned set gets the next id
        assert_eq!(intern("d"), AssetId(3));
        assert_eq!(intern("a"), AssetId(0));
    }

    #[test]
    fn name_table_snapshots_in_id_order() {
        reset_interner();
        intern_all(&["a", "b", "c"]);
        assert_eq!(name_table(), vec!["a", "b", "c"]);
    }

    #[test]
    fn asset_id_round_trips_through_json_as_integer() {
        let id = AssetId(7);
        let bytes = serde_json::to_vec(&id).unwrap();
        assert_eq!(bytes, b"7");
        let back: AssetId = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, id);
    }

    #[test]
    fn asset_id_deserializes_from_name_string_via_interner() {
        reset_interner();
        intern_all(&["floor", "wall"]);
        let id: AssetId = serde_json::from_str("\"wall\"").unwrap();
        assert_eq!(id, AssetId(1));
    }

    #[test]
    fn opt_ref_treats_empty_and_null_as_none() {
        reset_interner();
        intern_all(&["mesh_a"]);

        #[derive(serde::Deserialize)]
        struct Holder {
            #[serde(default, deserialize_with = "de_opt_asset_ref")]
            r: Option<AssetId>,
        }

        let empty: Holder = serde_json::from_str("{\"r\":\"\"}").unwrap();
        assert_eq!(empty.r, None);
        let null: Holder = serde_json::from_str("{\"r\":null}").unwrap();
        assert_eq!(null.r, None);
        let missing: Holder = serde_json::from_str("{}").unwrap();
        assert_eq!(missing.r, None);
        let named: Holder = serde_json::from_str("{\"r\":\"mesh_a\"}").unwrap();
        assert_eq!(named.r, Some(AssetId(0)));
        let by_id: Holder = serde_json::from_str("{\"r\":5}").unwrap();
        assert_eq!(by_id.r, Some(AssetId(5)));
    }
}
