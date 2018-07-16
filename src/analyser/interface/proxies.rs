use analyser::interface::path::Path;
use analyser::interface::cache::Cache;
use std::ops::Index;

use num_traits::cast::ToPrimitive;

/// A proxy for any value.
pub trait Proxy {
    /// Returns the symbolic path to the value.
    ///
    /// Take the `inputs[0].shape[1]` proxy for instance: it represents the
    /// second dimension of the shape of the first input. Because we encode
    /// the "inputs" vectors as `0`, and the `shape` field as `2`, the path
    /// for this proxy will be `vec![0, 0, 2, 1]`.
    fn get_path(&self) -> &Path;
}

/// A proxy for any Datatype value.
pub trait TypeProxy: Proxy {}

/// A proxy for any DimFact value.
pub trait DimProxy: Proxy {}

/// A proxy for any integer-like value.
pub trait IntProxy: Proxy {}

/// Generates the get_path method for structs which have a `path` field.
macro_rules! impl_proxy(
    ($struct:ident) => {
        impl Proxy for $struct {
            /// Returns the symbolic path to the value.
            fn get_path(&self) -> &Path {
                &self.path
            }
        }
    }
);


/// A simple implementation of a proxy for any Datatype value.
#[derive(new)]
pub struct BaseTypeProxy { path: Path }

impl_proxy!(BaseTypeProxy);
impl TypeProxy for BaseTypeProxy {}


/// A simple implementation of a proxy for any DimFact value.
#[derive(new)]
pub struct BaseDimProxy { path: Path }

impl_proxy!(BaseDimProxy);
impl DimProxy for BaseDimProxy {}


/// A simple implementation of a proxy for any integer-like value.
#[derive(new)]
pub struct BaseIntProxy { path: Path }

impl_proxy!(BaseIntProxy);
impl IntProxy for BaseIntProxy {}


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
    pub len: BaseIntProxy,
    tensors: Cache<usize, TensorProxy>,
    path: Path,
}

impl TensorsProxy {
    /// Creates a new TensorsProxy instance.
    pub fn new(path: Path) -> TensorsProxy {
        TensorsProxy {
            len: BaseIntProxy::new([&path[..], &[-1]].concat()),
            tensors: Cache::new(),
            path,
        }
    }
}

impl_proxy!(TensorsProxy);

impl Index<usize> for TensorsProxy {
    type Output = TensorProxy;

    /// Returns the TensorProxy corresponding to the given index.
    ///
    /// When an index is used for the first time, the TensorProxy is created
    /// dynamically and cached inside `self.tensors`. This way, future calls
    /// to `index` will return the same TensorProxy.
    fn index(&self, index: usize) -> &TensorProxy {
        let path = [&self.path[..], &[index.to_isize().unwrap()]].concat();
        self.tensors.get(index, || TensorProxy::new(path))
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
    pub datatype: BaseTypeProxy,
    pub rank: BaseIntProxy,
    pub shape: ShapeProxy,
    pub value: ValueProxy,
    path: Path,
}

impl TensorProxy {
    /// Creates a new TensorProxy instance.
    pub fn new(path: Path) -> TensorProxy {
        TensorProxy {
            datatype: BaseTypeProxy::new([&path[..], &[0]].concat()),
            rank:     BaseIntProxy::new([&path[..],  &[1]].concat()),
            shape:    ShapeProxy::new([&path[..],    &[2]].concat()),
            value:    ValueProxy::new([&path[..],    &[3]].concat()),
            path,
        }
    }
}

impl_proxy!(TensorProxy);


/// A proxy for a tensor shape.
pub struct ShapeProxy {
    // FIXME(liautaud): Use a BaseDimProxy instead, because we don't handle
    // streaming dimensions with the current approach.
    dims: Cache<usize, BaseIntProxy>,
    path: Path,
}

impl ShapeProxy {
    /// Creates a new ShapeProxy instance.
    pub fn new(path: Path) -> ShapeProxy {
        ShapeProxy { dims: Cache::new(), path }
    }
}

impl_proxy!(ShapeProxy);

impl Index<usize> for ShapeProxy {
    type Output = BaseIntProxy;

    /// Returns the BaseIntProxy corresponding to the given index.
    fn index(&self, index: usize) -> &BaseIntProxy {
        let path = [&self.path[..], &[index.to_isize().unwrap()]].concat();
        self.dims.get(index, || BaseIntProxy::new(path))
    }
}


/// A proxy for a tensor value.
///
/// This proxy is a bit special as it allows arbitrarily nested indexing, so
/// that writing something like ```input.value[1][6][2]``` will always work.
/// To make this work, each ValueProxy holds a cache which will generate new
/// ValueProxys for nested items on the fly and store them.
pub struct ValueProxy {
    sub: Cache<usize, ValueProxy>,
    path: Path,
}

impl ValueProxy {
    /// Creates a new ValueProxy instance.
    pub fn new(path: Path) -> ValueProxy {
        ValueProxy { sub: Cache::new(), path }
    }
}

impl Index<usize> for ValueProxy {
    type Output = ValueProxy;

    /// Returns the ValueProxy corresponding to the given index.
    fn index(&self, index: usize) -> &ValueProxy {
        let path = [&self.path[..], &[index.to_isize().unwrap()]].concat();
        self.sub.get(index, || ValueProxy::new(path))
    }
}

impl_proxy!(ValueProxy);
impl IntProxy for ValueProxy {}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensors_proxy() {
        let inputs = TensorsProxy::new(vec![0]);
        assert_eq!(inputs.len.get_path(), &vec![0, -1]);
        assert_eq!(inputs[0].get_path(),  &vec![0, 0]);
        assert_eq!(inputs[2].get_path(),  &vec![0, 2]);
    }

    #[test]
    fn test_tensor_proxy_datatype() {
        let inputs = TensorsProxy::new(vec![0]);
        let input = &inputs[0];

        assert_eq!(input.datatype.get_path(), &vec![0, 0, 0]);
    }

    #[test]
    fn test_tensor_proxy_rank() {
        let inputs = TensorsProxy::new(vec![0]);
        let input = &inputs[0];

        assert_eq!(input.rank.get_path(), &vec![0, 0, 1]);
    }

    #[test]
    fn test_tensor_proxy_shape() {
        let inputs = TensorsProxy::new(vec![0]);
        let input = &inputs[0];

        assert_eq!(input.shape[0].get_path(), &vec![0, 0, 2, 0]);
        assert_eq!(input.shape[2].get_path(), &vec![0, 0, 2, 2]);
    }

    #[test]
    fn test_tensor_proxy_value() {
        let inputs = TensorsProxy::new(vec![0]);
        let input = &inputs[0];

        assert_eq!(input.value[0][1].get_path(),    &vec![0, 0, 3, 0, 1]);
        assert_eq!(input.value[1][2][3].get_path(), &vec![0, 0, 3, 1, 2, 3]);
    }
}