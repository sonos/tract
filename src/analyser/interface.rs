//! A fluent interface for the analyser.
//!
//! This interface provides proxies for the different properties of tensors.
//! This allows inference rules to be stated in a clear, declarative fashion
//! inside the `rules` method of each operator.
//!
//! Take these rules for instance:
//! ```text
//! solver.equals(inputs.len, 2);
//! solver.equals(inputs[0].datatype, outputs[0].datatype);
//! ```
//! Here, `inputs.len`, `inputs[0].datatype` and `outputs[0].datatype` don't
//! actually hold the values of the length and datatypes, but instead act as
//! declarative placeholders for these values.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Index;

/// An insert-only HashMap which doesn't require mutable references.
struct Cache<K: Eq + Hash, V>(
    // We need to use a RefCell here because we need interior mutability for
    // the cache. This way, the `get` method will only need `&self` (and not
    // `&mut self`) but we'll still be able to insert new items dynamically.
    RefCell<HashMap<K, V>>,
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
            cache.entry(index).or_insert_with(default)
        }
    }
}

/// A proxy for a vector of tensors.
///
/// This is used for rules concerning the vector of input or output tensors:
/// ```text
/// solver.equals(inputs.len, 2);
/// ```
/// When the indexing operator is used on a TensorsProxy (e.g. `inputs[0]`),
/// a new TensorProxy is created dynamically and cached in `tensors`.
///
/// The solver should check the coherence of `len` with the indices of every
/// TensorProxy involved in inference rules, to forbid rules like:
/// ```text
/// solver.equals(inputs[i].rank, 2);
/// ```
/// when i >= len.
pub struct TensorsProxy {
    pub len: (),
    tensors: Cache<usize, TensorProxy>,
}

impl TensorsProxy {
    /// Creates a new TensorsProxy instance.
    pub fn new() -> TensorsProxy {
        TensorsProxy {
            len: (),
            tensors: Cache::new(),
        }
    }
}

impl Index<usize> for TensorsProxy {
    type Output = TensorProxy;

    /// Returns the TensorProxy corresponding to the given index.
    ///
    /// When an index is used for the first time, the TensorProxy is created
    /// dynamically and cached inside `self.tensors`. This way, future calls
    /// to `index` will return the same TensorProxy.
    fn index(&self, index: usize) -> &TensorProxy {
        self.tensors.get(index, || TensorProxy::new())
    }
}

/// A proxy for a tensor.
///
/// This is used for rules involving the datatype, rank, shape or value of a
/// tensor. Here are a few examples of constraints that can be expressed:
/// ```text
/// solver.equals(input.datatype, DTYPE_I32)
/// solver.equals(input.rank, 2)
/// solver.equals(input.shape[1], output.value[0][1])
/// ```
pub struct TensorProxy {
    pub datatype: (),
    pub rank: (),
    pub shape: ShapeProxy,
    pub value: ValueProxy,
}

impl TensorProxy {
    /// Creates a new TensorProxy instance.
    pub fn new() -> TensorProxy {
        TensorProxy {
            datatype: (),
            rank: (),
            shape: ShapeProxy::new(),
            value: ValueProxy::new(),
        }
    }
}

/// A proxy for a tensor shape.
pub struct ShapeProxy {
    dims: Cache<usize, DimProxy>,
}

impl ShapeProxy {
    /// Creates a new ShapeProxy instance.
    pub fn new() -> ShapeProxy {
        ShapeProxy { dims: Cache::new() }
    }
}

impl Index<usize> for ShapeProxy {
    type Output = DimProxy;

    /// Returns the DimProxy corresponding to the given index.
    fn index(&self, index: usize) -> &DimProxy {
        self.dims.get(index, || DimProxy::new())
    }
}

/// A proxy for a tensor dimension.
#[derive(PartialEq)]
pub struct DimProxy {}

impl DimProxy {
    /// Creates a new DimProxy instance.
    pub fn new() -> DimProxy {
        DimProxy {}
    }
}

/// A proxy for a tensor value.
///
/// This proxy is a bit special as it allows arbitrarily nested indexing, so
/// that writing something like ```input.value[1][6][2]``` will always work.
/// To make this work, each ValueProxy holds a cache which will generate new
/// ValueProxy for nested items on the fly and store them.
pub struct ValueProxy {
    sub: Cache<usize, ValueProxy>,
}

impl ValueProxy {
    /// Creates a new ValueProxy instance.
    pub fn new() -> ValueProxy {
        ValueProxy { sub: Cache::new() }
    }
}

impl Index<usize> for ValueProxy {
    type Output = ValueProxy;

    /// Returns the ValueProxy corresponding to the given index.
    fn index(&self, index: usize) -> &ValueProxy {
        self.sub.get(index, || ValueProxy::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensors_proxy() {
        let inputs = TensorsProxy::new();
        let _ = &inputs.len;
        let _ = &inputs[0];
        let _ = &inputs[2];
    }

    #[test]
    fn test_tensor_proxy_datatype() {
        let inputs = TensorsProxy::new();
        let input = &inputs[0];
        let _ = input.datatype;
    }

    #[test]
    fn test_tensor_proxy_rank() {
        let inputs = TensorsProxy::new();
        let input = &inputs[0];
        let _ = input.rank;
    }

    #[test]
    fn test_tensor_proxy_shape() {
        let inputs = TensorsProxy::new();
        let input = &inputs[0];
        let _ = input.shape[0];
        let _ = input.shape[2];
    }

    #[test]
    fn test_tensor_proxy_value() {
        let inputs = TensorsProxy::new();
        let input = &inputs[0];
        let _ = input.value[1][3];
        let _ = input.value[0][2];
        let _ = input.value[0][1][2];
    }
}