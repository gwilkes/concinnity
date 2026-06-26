// concinnity-eas/src/resource.rs
//
// Type-keyed singleton store: each type has at most one instance, fetched by
// type. The home for engine-wide singletons (frame input, the render backend,
// the profiler) that would otherwise be faked as one-element collections.
//
// Values are only required to be `Any`, not `Send`, so a main-thread-only
// resource (the Metal backend) can live here alongside the rest.

use std::any::{Any, TypeId};
use std::collections::HashMap;

#[derive(Default)]
pub struct Resources {
    map: HashMap<TypeId, Box<dyn Any>>,
}

impl Resources {
    pub fn new() -> Resources {
        Resources::default()
    }

    // Insert a resource, returning the previous instance of the same type if
    // one was present.
    pub fn insert<T: Any>(&mut self, value: T) -> Option<T> {
        self.map
            .insert(TypeId::of::<T>(), Box::new(value))
            .and_then(downcast::<T>)
    }

    pub fn get<T: Any>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast_ref::<T>())
    }

    pub fn get_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.map
            .get_mut(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast_mut::<T>())
    }

    pub fn remove<T: Any>(&mut self) -> Option<T> {
        self.map.remove(&TypeId::of::<T>()).and_then(downcast::<T>)
    }

    pub fn contains<T: Any>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<T>())
    }
}

fn downcast<T: Any>(boxed: Box<dyn Any>) -> Option<T> {
    boxed.downcast::<T>().ok().map(|value| *value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct FrameTime(f32);

    #[test]
    fn insert_get_and_remove_by_type() {
        let mut resources = Resources::new();
        assert!(!resources.contains::<FrameTime>());
        assert_eq!(resources.insert(FrameTime(0.016)), None);
        assert!(resources.contains::<FrameTime>());
        assert_eq!(resources.get::<FrameTime>(), Some(&FrameTime(0.016)));
        assert_eq!(resources.remove::<FrameTime>(), Some(FrameTime(0.016)));
        assert!(!resources.contains::<FrameTime>());
    }

    #[test]
    fn insert_returns_previous_value() {
        let mut resources = Resources::new();
        resources.insert(FrameTime(1.0));
        assert_eq!(resources.insert(FrameTime(2.0)), Some(FrameTime(1.0)));
    }

    #[test]
    fn get_mut_edits_in_place() {
        let mut resources = Resources::new();
        resources.insert(FrameTime(1.0));
        resources.get_mut::<FrameTime>().unwrap().0 = 5.0;
        assert_eq!(resources.get::<FrameTime>(), Some(&FrameTime(5.0)));
    }

    #[test]
    fn distinct_types_are_independent() {
        let mut resources = Resources::new();
        resources.insert(FrameTime(1.0));
        resources.insert(7u32);
        assert_eq!(resources.get::<FrameTime>(), Some(&FrameTime(1.0)));
        assert_eq!(resources.get::<u32>(), Some(&7));
    }
}
