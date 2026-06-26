// concinnity-eas/src/storage.rs
//
// The closed-world component-storage macro. Given a list of `field => type`
// pairs it generates the per-type `Column`-backed storage struct, the access
// trait that resolves a component type to its column at compile time, and the
// generic storage operations (typed push, drain, mutable access, counts).
//
// The engine pairs this with its own asset/blob codegen: concinnity-core's
// `define_components!` calls this for the storage half and adds the asset-enum
// dispatch (`push(ComponentAsset)`, `all_defs`) in a separate impl block. The
// storage layout and the access trait live here so the (eventual)
// multi-component query can share one definition, and so the engine's component
// set stays the only thing that names concrete component types.

#[macro_export]
macro_rules! define_component_storage {
    (
        storage: $storage:ident,
        slot: $slot:ident,
        $( $field:ident => $ty:path ),+ $(,)?
    ) => {
        // One `Column<T>` per registered component type, plus the entity
        // allocator that stamps each row's id and the change tick stamped on
        // every structural edit. Field names are the caller's field idents,
        // reached through the `$slot` trait; callers never name them directly.
        #[allow(non_snake_case)]
        #[derive(Default, Debug)]
        pub struct $storage {
            $( pub $field: $crate::Column<$ty>, )+
            entities: $crate::Entities,
            change_tick: $crate::Tick,
        }

        impl $storage {
            // Push a statically-typed component into its column, minting a fresh
            // Entity for the new row.
            #[allow(dead_code)]
            pub fn push_typed<C: $slot>(&mut self, c: C) {
                let entity = self.entities.alloc();
                let tick = self.change_tick.bump();
                C::slot_mut(self).push(entity, c, tick);
            }

            // Remove and return every component of type C, despawning each row's
            // Entity so the indices recycle.
            #[allow(dead_code)]
            pub fn drain<C: $slot>(&mut self) -> ::std::vec::Vec<C> {
                let owners = C::slot(self).entities().to_vec();
                for entity in owners {
                    self.entities.despawn(entity);
                }
                let tick = self.change_tick.bump();
                C::slot_mut(self).drain(tick)
            }

            // Mutable slice of every component of type C, stamping the change
            // tick because any element may be written.
            #[allow(dead_code)]
            pub fn values_mut<C: $slot>(&mut self) -> &mut [C] {
                let tick = self.change_tick.bump();
                C::slot_mut(self).values_mut(tick)
            }

            // Total number of components across all typed columns.
            #[allow(dead_code)]
            pub fn len(&self) -> usize {
                0 $( + self.$field.len() )+
            }

            #[allow(dead_code)]
            pub fn is_empty(&self) -> bool {
                true $( && self.$field.is_empty() )+
            }
        }

        // Resolves a component type to its column inside the storage at compile
        // time, so the generic storage operations above need no runtime
        // dispatch. A registered component is exactly a type with a `$slot` impl.
        // `'static`: components own their data, and the generic ops hand out
        // borrows of (and owned vectors of) the type.
        pub trait $slot: Sized + 'static {
            fn slot(s: &$storage) -> &$crate::Column<Self>;
            fn slot_mut(s: &mut $storage) -> &mut $crate::Column<Self>;
        }

        $(
            impl $slot for $ty {
                fn slot(s: &$storage) -> &$crate::Column<Self> { &s.$field }
                fn slot_mut(s: &mut $storage) -> &mut $crate::Column<Self> { &mut s.$field }
            }
        )+
    };
}

#[cfg(test)]
mod tests {
    // `pub` so the generated `pub` columns don't expose a more-private type
    // (the real engine's component types are `pub`, so this never bites there).
    #[derive(Default, Debug, PartialEq, Clone, Copy)]
    pub struct Position(u32);

    #[derive(Default, Debug, PartialEq, Clone, Copy)]
    pub struct Velocity(i32);

    define_component_storage! {
        storage: TestStorage,
        slot: TestSlot,
        Position => Position,
        Velocity => Velocity,
    }

    #[test]
    fn push_count_mutate_drain() {
        let mut s = TestStorage::default();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);

        s.push_typed(Position(1));
        s.push_typed(Position(2));
        s.push_typed(Velocity(-3));
        assert!(!s.is_empty());
        assert_eq!(s.len(), 3);

        // values_mut resolves the type to its own column.
        for p in s.values_mut::<Position>() {
            p.0 += 10;
        }

        // Draining one type leaves the other untouched.
        assert_eq!(s.drain::<Position>(), vec![Position(11), Position(12)]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.drain::<Velocity>(), vec![Velocity(-3)]);
        assert!(s.is_empty());
    }

    #[test]
    fn columns_carry_row_aligned_entities() {
        let mut s = TestStorage::default();
        s.push_typed(Position(7));
        s.push_typed(Position(8));
        // Each pushed row got a distinct Entity, aligned with the data.
        let entities = <Position as TestSlot>::slot(&s).entities();
        assert_eq!(entities.len(), 2);
        assert_ne!(entities[0], entities[1]);
    }
}
