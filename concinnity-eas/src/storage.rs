// concinnity-eas/src/storage.rs
//
// The closed-world component-storage macro. Given a list of `field => type, id`
// triples it generates the per-type `Column`-backed storage struct, the access
// trait that resolves a component type to its column (and its component id) at
// compile time, and the generic storage operations: typed push, entity-targeted
// insert/remove, whole-entity despawn, drain, mutable access, counts, and
// read-only multi-component joins.
//
// The struct also owns the `JoinIndex` keyed by entity id, kept in sync by every
// structural edit so a multi-component query can find an entity's row in each
// column without scanning. The single hazard the maintenance must respect: a
// swap-remove moves the column's last row into the freed slot, so the moved
// row's owner needs its recorded row patched, or a later join probe reads the
// wrong row.
//
// The engine pairs this with its own asset/blob codegen: concinnity-core's
// `define_components!` calls this for the storage half and adds the asset-enum
// dispatch (`push(ComponentAsset)`, `all_defs`) in a separate impl block. The
// storage layout, access trait, and join live here so the component query shares
// one definition, and so the engine's component set stays the only thing that
// names concrete component types.

#[macro_export]
macro_rules! define_component_storage {
    (
        storage: $storage:ident,
        slot: $slot:ident,
        $( $field:ident => $ty:path, $disc:expr ),+ $(,)?
    ) => {
        // One `Column<T>` per registered component type, the entity allocator
        // that stamps each row's id, the change tick stamped on every structural
        // edit, and the join index that maps an entity to its row in each
        // column. Field columns are the caller's field idents, reached through
        // the `$slot` trait; callers never name them directly.
        #[allow(non_snake_case)]
        #[derive(Default, Debug)]
        pub struct $storage {
            $( pub $field: $crate::Column<$ty>, )+
            entities: $crate::Entities,
            change_tick: $crate::Tick,
            join: $crate::JoinIndex,
        }

        impl $storage {
            // Push a statically-typed component into its column, minting a fresh
            // Entity for the new row and recording it in the join index.
            #[allow(dead_code)]
            pub fn push_typed<C: $slot>(&mut self, c: C) -> $crate::Entity {
                let entity = self.entities.alloc();
                let tick = self.change_tick.bump();
                let col = C::slot_mut(self);
                col.push(entity, c, tick);
                let row = (col.len() - 1) as u32;
                self.join.set(entity, $crate::ComponentId::new(C::DISCRIMINANT), row);
                entity
            }

            // Allocate a bare entity that owns no components yet. Useful for
            // gameplay-only entities and as the target of later `insert_typed`.
            #[allow(dead_code)]
            pub fn spawn(&mut self) -> $crate::Entity {
                self.entities.alloc()
            }

            // Whether a handle refers to a currently-live entity.
            #[allow(dead_code)]
            pub fn is_alive(&self, entity: $crate::Entity) -> bool {
                self.entities.is_alive(entity)
            }

            // Add component C to an existing entity. Unlike `push_typed` this does
            // not mint an entity: it is how an entity comes to own more than one
            // component. The entity must be alive and must not already have C
            // (a second row for the same (entity, C) would desync the join).
            #[allow(dead_code)]
            pub fn insert_typed<C: $slot>(&mut self, entity: $crate::Entity, c: C) {
                let id = $crate::ComponentId::new(C::DISCRIMINANT);
                debug_assert!(
                    self.entities.is_alive(entity),
                    "insert_typed on a despawned entity",
                );
                debug_assert!(
                    self.join.row(entity, id).is_none(),
                    "insert_typed: entity already has this component",
                );
                let tick = self.change_tick.bump();
                let col = C::slot_mut(self);
                col.push(entity, c, tick);
                let row = (col.len() - 1) as u32;
                self.join.set(entity, id, row);
            }

            // Remove component C from an entity (leaving the entity alive and any
            // other components intact), returning the value if present. Swap-
            // remove moves the column's last row into the freed slot, so the
            // moved row's owner has its recorded row patched.
            #[allow(dead_code)]
            pub fn remove_typed<C: $slot>(&mut self, entity: $crate::Entity) -> Option<C> {
                let id = $crate::ComponentId::new(C::DISCRIMINANT);
                let row = self.join.row(entity, id)? as usize;
                let tick = self.change_tick.bump();
                let col = C::slot_mut(self);
                let last = col.len() - 1;
                let moved = if row != last { Some(col.entities()[last]) } else { None };
                let value = col.swap_remove(row, tick);
                self.join.clear(entity, id);
                if let Some(moved) = moved {
                    self.join.set(moved, id, row as u32);
                }
                Some(value)
            }

            // Despawn an entity: swap-remove its row from every column it has,
            // patching each moved tail row, then recycle the entity id. This is
            // the structural-change primitive runtime despawn is built on.
            #[allow(dead_code)]
            pub fn despawn(&mut self, entity: $crate::Entity) {
                if !self.entities.is_alive(entity) {
                    return;
                }
                let tick = self.change_tick.bump();
                $(
                    {
                        let id = $crate::ComponentId::new(<$ty as $slot>::DISCRIMINANT);
                        if let Some(row) = self.join.row(entity, id) {
                            let row = row as usize;
                            let col = &mut self.$field;
                            let last = col.len() - 1;
                            let moved =
                                if row != last { Some(col.entities()[last]) } else { None };
                            col.swap_remove(row, tick);
                            if let Some(moved) = moved {
                                self.join.set(moved, id, row as u32);
                            }
                        }
                    }
                )+
                self.join.clear_entity(entity);
                self.entities.despawn(entity);
            }

            // Remove and return every component of type C. Each owner loses
            // only its C component; an owner that has no other component left is
            // despawned so its Entity recycles. An owner that still has other
            // components stays alive with those intact and join-reachable. The
            // whole C column empties at once, so no per-row tail patch is needed
            // for C; only each owner's C entry in the join is cleared.
            #[allow(dead_code)]
            pub fn drain<C: $slot>(&mut self) -> ::std::vec::Vec<C> {
                let id = $crate::ComponentId::new(C::DISCRIMINANT);
                let owners = C::slot(self).entities().to_vec();
                let tick = self.change_tick.bump();
                let drained = C::slot_mut(self).drain(tick);
                for entity in owners {
                    self.join.clear(entity, id);
                    if self.join.mask(entity).is_empty() {
                        self.entities.despawn(entity);
                    }
                }
                drained
            }

            // Mutable slice of every component of type C, stamping the change
            // tick because any element may be written.
            #[allow(dead_code)]
            pub fn values_mut<C: $slot>(&mut self) -> &mut [C] {
                let tick = self.change_tick.bump();
                C::slot_mut(self).values_mut(tick)
            }

            // Read-only join over two component types. Iterates the first type's
            // rows and, for each owning entity that also has the second type,
            // yields both component refs. This is the multi-component query for
            // read paths (the draw-list push, scene visibility): one column scan
            // plus a join probe per row, no allocation.
            #[allow(dead_code)]
            pub fn join2<'s, A: $slot, B: $slot>(
                &'s self,
            ) -> impl Iterator<Item = ($crate::Entity, &'s A, &'s B)> + 's {
                let bid = $crate::ComponentId::new(B::DISCRIMINANT);
                let bcol = B::slot(self);
                A::slot(self)
                    .iter_with_entities()
                    .filter_map(move |(entity, a)| {
                        let brow = self.join.row(entity, bid)? as usize;
                        let b = bcol.get(brow);
                        debug_assert!(
                            b.is_some(),
                            "join2: stale JoinIndex row for an entity's component",
                        );
                        Some((entity, a, b?))
                    })
            }

            // Read-only join over three component types, lead on the first.
            #[allow(dead_code)]
            pub fn join3<'s, A: $slot, B: $slot, C: $slot>(
                &'s self,
            ) -> impl Iterator<Item = ($crate::Entity, &'s A, &'s B, &'s C)> + 's {
                let bid = $crate::ComponentId::new(B::DISCRIMINANT);
                let cid = $crate::ComponentId::new(C::DISCRIMINANT);
                let bcol = B::slot(self);
                let ccol = C::slot(self);
                A::slot(self)
                    .iter_with_entities()
                    .filter_map(move |(entity, a)| {
                        let brow = self.join.row(entity, bid)? as usize;
                        let crow = self.join.row(entity, cid)? as usize;
                        let b = bcol.get(brow);
                        let c = ccol.get(crow);
                        debug_assert!(
                            b.is_some() && c.is_some(),
                            "join3: stale JoinIndex row for an entity's component",
                        );
                        Some((entity, a, b?, c?))
                    })
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
        // dispatch. A registered component is exactly a type with a `$slot` impl,
        // and `DISCRIMINANT` is its stable id, used as its `ComponentId` in the
        // join index. `'static`: components own their data, and the generic ops
        // hand out borrows of (and owned vectors of) the type.
        pub trait $slot: Sized + 'static {
            const DISCRIMINANT: u8;
            fn slot(s: &$storage) -> &$crate::Column<Self>;
            fn slot_mut(s: &mut $storage) -> &mut $crate::Column<Self>;
        }

        $(
            impl $slot for $ty {
                const DISCRIMINANT: u8 = $disc;
                fn slot(s: &$storage) -> &$crate::Column<Self> { &s.$field }
                fn slot_mut(s: &mut $storage) -> &mut $crate::Column<Self> { &mut s.$field }
            }
            // The ComponentMask is a u128, so a discriminant past 127 would
            // silently alias another component's mask bit in a release build.
            // Make that a build error at the registration site instead.
            const _: () = assert!(
                $disc <= $crate::ComponentId::MAX,
                "component discriminant exceeds the 127-bit ComponentMask ceiling",
            );
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

    #[derive(Default, Debug, PartialEq, Clone, Copy)]
    pub struct Tag;

    define_component_storage! {
        storage: TestStorage,
        slot: TestSlot,
        Position => Position, 1,
        Velocity => Velocity, 2,
        Tag => Tag, 3,
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
        let a = s.push_typed(Position(7));
        let b = s.push_typed(Position(8));
        // Each pushed row got a distinct Entity, aligned with the data.
        let entities = <Position as TestSlot>::slot(&s).entities();
        assert_eq!(entities, &[a, b]);
        assert_ne!(a, b);
    }

    #[test]
    fn insert_puts_two_components_on_one_entity() {
        let mut s = TestStorage::default();
        // push_typed mints the entity and gives it its first component.
        let e = s.push_typed(Position(5));
        // insert_typed adds a second component to the SAME entity -- the thing
        // that was impossible while every row minted its own entity.
        s.insert_typed(e, Velocity(-2));
        s.insert_typed(e, Tag);

        let joined: Vec<_> = s.join2::<Position, Velocity>().collect();
        assert_eq!(joined.len(), 1);
        assert_eq!(joined[0], (e, &Position(5), &Velocity(-2)));

        let joined3: Vec<_> = s.join3::<Position, Velocity, Tag>().collect();
        assert_eq!(joined3.len(), 1);
        assert_eq!(joined3[0], (e, &Position(5), &Velocity(-2), &Tag));
    }

    #[test]
    fn join2_only_matches_entities_with_both() {
        let mut s = TestStorage::default();
        let a = s.push_typed(Position(1));
        s.insert_typed(a, Velocity(10));
        // b has only a Position, so it must not appear in the join.
        let _b = s.push_typed(Position(2));
        let c = s.push_typed(Position(3));
        s.insert_typed(c, Velocity(30));

        let mut joined: Vec<_> = s
            .join2::<Position, Velocity>()
            .map(|(e, p, v)| (e, *p, *v))
            .collect();
        joined.sort_by_key(|(e, _, _)| e.index());
        assert_eq!(
            joined,
            vec![
                (a, Position(1), Velocity(10)),
                (c, Position(3), Velocity(30))
            ]
        );
    }

    #[test]
    fn remove_typed_patches_the_moved_tail_row() {
        let mut s = TestStorage::default();
        // Three entities each with a Velocity; removing the middle one swap-moves
        // the last row into its slot. The join must still find the moved entity.
        let a = s.push_typed(Velocity(1));
        let b = s.push_typed(Velocity(2));
        let c = s.push_typed(Velocity(3));

        let removed = s.remove_typed::<Velocity>(b);
        assert_eq!(removed, Some(Velocity(2)));
        // a and c are still readable through the join at their (possibly moved)
        // rows; b is gone.
        let joined: std::collections::HashMap<_, _> = s
            .join2::<Velocity, Velocity>() // self-join echoes the live rows
            .map(|(e, v, _)| (e, *v))
            .collect();
        assert_eq!(joined.get(&a), Some(&Velocity(1)));
        assert_eq!(joined.get(&c), Some(&Velocity(3)));
        assert_eq!(joined.get(&b), None);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn remove_typed_returns_none_when_absent() {
        let mut s = TestStorage::default();
        let e = s.push_typed(Position(1));
        // e has no Velocity.
        assert_eq!(s.remove_typed::<Velocity>(e), None);
        // A bare entity has nothing to remove either.
        let bare = s.spawn();
        assert_eq!(s.remove_typed::<Position>(bare), None);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn remove_typed_last_row_takes_the_no_move_branch() {
        let mut s = TestStorage::default();
        let a = s.push_typed(Velocity(1));
        let b = s.push_typed(Velocity(2));
        let c = s.push_typed(Velocity(3));
        // Removing the LAST row (c) means row == last, so nothing is swapped in.
        assert_eq!(s.remove_typed::<Velocity>(c), Some(Velocity(3)));
        let joined: std::collections::HashMap<_, _> = s
            .join2::<Velocity, Velocity>()
            .map(|(e, v, _)| (e, *v))
            .collect();
        assert_eq!(joined.get(&a), Some(&Velocity(1)));
        assert_eq!(joined.get(&b), Some(&Velocity(2)));
        assert_eq!(joined.get(&c), None);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn remove_one_component_keeps_siblings_on_a_multi_component_entity() {
        let mut s = TestStorage::default();
        // Two multi-component entities; remove a non-tail Velocity row from the
        // first and confirm both entities' surviving components stay joinable.
        let a = s.push_typed(Position(1));
        s.insert_typed(a, Velocity(10));
        s.insert_typed(a, Tag);
        let b = s.push_typed(Position(2));
        s.insert_typed(b, Velocity(20));

        assert_eq!(s.remove_typed::<Velocity>(a), Some(Velocity(10)));
        assert!(s.is_alive(a));
        // a kept Position + Tag; b kept Position + Velocity.
        let pos_tag: Vec<_> = s
            .join2::<Position, Tag>()
            .map(|(e, p, _)| (e, *p))
            .collect();
        assert_eq!(pos_tag, vec![(a, Position(1))]);
        let pos_vel: Vec<_> = s
            .join2::<Position, Velocity>()
            .map(|(e, p, v)| (e, *p, *v))
            .collect();
        assert_eq!(pos_vel, vec![(b, Position(2), Velocity(20))]);
    }

    #[test]
    fn remove_then_reinsert_same_component_on_live_entity() {
        let mut s = TestStorage::default();
        let a = s.push_typed(Position(1));
        let b = s.push_typed(Position(2));
        let _c = s.push_typed(Position(3));
        s.insert_typed(b, Velocity(20));

        // Remove then re-insert the same component type on the same live entity.
        // The re-insert must not trip insert_typed's "already has it" assert.
        assert_eq!(s.remove_typed::<Velocity>(b), Some(Velocity(20)));
        assert!(s.is_alive(b));
        s.insert_typed(b, Velocity(21));

        let joined: std::collections::HashMap<_, _> = s
            .join2::<Position, Velocity>()
            .map(|(e, p, v)| (e, (*p, *v)))
            .collect();
        assert_eq!(joined.get(&b), Some(&(Position(2), Velocity(21))));
        assert_eq!(joined.get(&a), None);
    }

    #[test]
    fn drain_one_type_keeps_shared_entities_and_their_other_components() {
        let mut s = TestStorage::default();
        // shared owns Position + Velocity; solo owns only Position.
        let shared = s.push_typed(Position(1));
        s.insert_typed(shared, Velocity(99));
        let solo = s.push_typed(Position(2));

        let drained = s.drain::<Position>();
        assert_eq!(drained.len(), 2);
        // solo had only Position, so it is despawned and recycled.
        assert!(!s.is_alive(solo));
        // shared still has Velocity, so it stays alive and join-reachable; no
        // orphaned Velocity row, and len() reflects exactly the one survivor.
        assert!(s.is_alive(shared));
        let vels: Vec<_> = s
            .join2::<Velocity, Velocity>()
            .map(|(e, v, _)| (e, *v))
            .collect();
        assert_eq!(vels, vec![(shared, Velocity(99))]);
        assert_eq!(s.len(), 1);
        // Draining the remaining type now despawns shared too.
        assert_eq!(s.drain::<Velocity>(), vec![Velocity(99)]);
        assert!(!s.is_alive(shared));
        assert!(s.is_empty());
    }

    #[test]
    fn despawn_removes_all_components_and_patches_tails() {
        let mut s = TestStorage::default();
        // e1 has Position+Velocity+Tag; e2 has Position+Velocity. Despawning e1
        // swap-removes from three columns; e2's rows (the tails) must be patched.
        let e1 = s.push_typed(Position(1));
        s.insert_typed(e1, Velocity(11));
        s.insert_typed(e1, Tag);
        let e2 = s.push_typed(Position(2));
        s.insert_typed(e2, Velocity(22));

        s.despawn(e1);
        assert!(!s.is_alive(e1));
        assert!(s.is_alive(e2));

        // e2 still joins correctly after the swap-remove reordering.
        let joined: Vec<_> = s.join2::<Position, Velocity>().collect();
        assert_eq!(joined, vec![(e2, &Position(2), &Velocity(22))]);
        // e1 contributed one row to each column; all three are gone.
        assert_eq!(<Position as TestSlot>::slot(&s).len(), 1);
        assert_eq!(<Velocity as TestSlot>::slot(&s).len(), 1);
        assert_eq!(<Tag as TestSlot>::slot(&s).len(), 0);
    }

    #[test]
    fn despawn_is_a_noop_on_a_stale_handle() {
        let mut s = TestStorage::default();
        let e = s.push_typed(Position(1));
        s.despawn(e);
        // Second despawn of the same (now stale) handle does nothing.
        s.despawn(e);
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn spawn_makes_a_bare_entity_for_later_inserts() {
        let mut s = TestStorage::default();
        let e = s.spawn();
        assert!(s.is_alive(e));
        assert_eq!(s.len(), 0);
        s.insert_typed(e, Position(9));
        s.insert_typed(e, Velocity(-9));
        let joined: Vec<_> = s.join2::<Position, Velocity>().collect();
        assert_eq!(joined, vec![(e, &Position(9), &Velocity(-9))]);
    }

    #[test]
    fn recycled_entity_index_does_not_report_stale_components() {
        let mut s = TestStorage::default();
        let a = s.push_typed(Position(1));
        s.insert_typed(a, Velocity(1));
        s.despawn(a);
        // Reusing the freed index for a fresh entity must not inherit a's
        // components through the join.
        let b = s.spawn();
        assert_eq!(a.index(), b.index());
        s.insert_typed(b, Position(2));
        let joined: Vec<_> = s.join2::<Position, Velocity>().collect();
        assert!(
            joined.is_empty(),
            "b has no Velocity; stale join must not match"
        );
        let positions: Vec<_> = s
            .join2::<Position, Position>()
            .map(|(e, p, _)| (e, *p))
            .collect();
        assert_eq!(positions, vec![(b, Position(2))]);
    }
}
