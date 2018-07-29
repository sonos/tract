use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;

/// An insert-only HashMap which doesn't require mutable references.
pub struct Cache<K: Eq + Hash, V>(
    // We need to use a RefCell here because we need interior mutability for
    // the cache. This way, the `get` method will only need `&self` (and not
    // `&mut self`) but we'll still be able to insert new items dynamically.
    RefCell<HashMap<K, Box<V>>>,
);

impl<K: Eq + Hash, V> Cache<K, V> {
    /// Creates a new Cache instance.
    pub fn new() -> Cache<K, V> {
        Cache(RefCell::new(HashMap::new()))
    }

    /// Returns a reference to the cached entry for a given key, or stores a
    /// new entry on cache misses and then returns a reference to it.
    pub fn get<F>(&self, index: K, default: F) -> &V
    where
        F: FnOnce() -> V,
    {
        // This is valid because we never remove anything from the cache, so
        // the reference to the items that we return will always exist.
        unsafe {
            let cache = &mut *self.0.as_ptr();
            cache.entry(index).or_insert_with(|| Box::new(default()))
        }
    }
}
