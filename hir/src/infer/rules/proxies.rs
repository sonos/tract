use std::fmt;
use std::ops::Index;

use crate::tract_num_traits::ToPrimitive;

use crate::infer::factoid::*;

use self::super::cache::Cache;
use self::super::expr::Output;
use self::super::path::Path;

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

/// A proxy which can be used in a solver rule.
pub trait ComparableProxy: Proxy {
    type Output: Output;
}

/// Generates the get_path method for structs which have a `path` field.
macro_rules! impl_proxy {
    ($struct:ident) => {
        impl Proxy for $struct {
            /// Returns the symbolic path to the value.
            fn get_path(&self) -> &Path {
                &self.path
            }
        }

        impl fmt::Debug for $struct {
            fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "{:?}", self.get_path())
            }
        }

        impl<'a> Proxy for &'a $struct {
            /// Returns the symbolic path to the value.
            fn get_path(&self) -> &Path {
                &self.path
            }
        }
    };
}

/// Implements the ComparableProxy trait for the proxy and references to it.
macro_rules! impl_comparable_proxy {
    ($struct:ident, $output:ident) => {
        impl ComparableProxy for $struct {
            type Output = $output;
        }
        impl<'a> ComparableProxy for &'a $struct {
            type Output = $output;
        }
    };
}

/// A proxy for any integer-like value.
#[derive(new)]
pub struct IntProxy {
    path: Path,
}

impl_proxy!(IntProxy);
impl_comparable_proxy!(IntProxy, IntFactoid);

/// A proxy for a tensor.
///
/// This is used for rules involving the datum_type, rank, shape or value of a
/// tensor. Here are a few examples of constraints that can be expressed:
/// ```text
/// solver.equals(input.datum_type, DTYPE_I32)
/// solver.equals(input.rank, 2)
/// solver.equals(input.shape[1], output.value[0][1])
/// ```
pub struct TensorProxy {
    pub datum_type: TypeProxy,
    pub rank: IntProxy,
    pub shape: ShapeProxy,
    pub value: ValueProxy,
    path: Path,
}

impl TensorProxy {
    /// Creates a new TensorProxy instance.
    pub fn new(path: Path) -> TensorProxy {
        TensorProxy {
            datum_type: TypeProxy::new([&path[..], &[0]].concat().into()),
            rank: IntProxy::new([&path[..], &[1]].concat().into()),
            shape: ShapeProxy::new([&path[..], &[2]].concat().into()),
            value: ValueProxy::new([&path[..], &[3]].concat().into()),
            path,
        }
    }
}

impl_proxy!(TensorProxy);

/// A proxy for a tensor datum_type.
#[derive(new)]
pub struct TypeProxy {
    path: Path,
}

impl_proxy!(TypeProxy);
impl_comparable_proxy!(TypeProxy, TypeFactoid);

/// A proxy for a tensor shape.
pub struct ShapeProxy {
    dims: Cache<usize, DimProxy>,
    path: Path,
}

impl ShapeProxy {
    /// Creates a new ShapeProxy instance.
    pub fn new(path: Path) -> ShapeProxy {
        ShapeProxy { dims: Cache::new(), path }
    }
}

impl_proxy!(ShapeProxy);
impl_comparable_proxy!(ShapeProxy, ShapeFactoid);

impl Index<usize> for ShapeProxy {
    type Output = DimProxy;

    /// Returns the DimProxy corresponding to the given index.
    fn index(&self, index: usize) -> &DimProxy {
        let path = [&self.path[..], &[index.to_isize().unwrap()]].concat();
        self.dims.get(index, || DimProxy::new(path.into()))
    }
}

/// A proxy for a dimension of a shape.
#[derive(new)]
pub struct DimProxy {
    path: Path,
}

impl_proxy!(DimProxy);
impl_comparable_proxy!(DimProxy, DimFact);

/// A proxy for the whole tensor value.
///
/// This proxy is a bit special as it allows arbitrarily nested indexing, so
/// that writing something like ```input.value[1][6][2]``` will always work.
/// To make this work, each ValueProxy holds a cache which will generate new
/// ValueProxys for nested items on the fly and store them.
pub struct ValueProxy {
    sub: Cache<usize, ElementProxy>,
    root: IntProxy,
    path: Path,
}

impl ValueProxy {
    /// Creates a new RootValueProxy instance.
    pub fn new(path: Path) -> ValueProxy {
        let root = IntProxy::new([&path[..], &[-1]].concat().into());
        ValueProxy { sub: Cache::new(), root, path }
    }
}

impl Index<()> for ValueProxy {
    type Output = IntProxy;

    /// Returns the RootValueProxy corresponding to the given index.
    fn index(&self, _: ()) -> &IntProxy {
        &self.root
    }
}

impl Index<usize> for ValueProxy {
    type Output = ElementProxy;

    /// Returns the ElementProxy corresponding to the given index.
    fn index(&self, index: usize) -> &ElementProxy {
        let path = [&self.path[..], &[index.to_isize().unwrap()]].concat();
        self.sub.get(index, || ElementProxy::new(path.into()))
    }
}

impl_proxy!(ValueProxy);
impl_comparable_proxy!(ValueProxy, ValueFact);

/// A proxy for a tensor element.
pub struct ElementProxy {
    sub: Cache<usize, ElementProxy>,
    path: Path,
}

impl ElementProxy {
    /// Creates a new ElementProxy instance.
    pub fn new(path: Path) -> ElementProxy {
        ElementProxy { sub: Cache::new(), path }
    }
}

impl Index<usize> for ElementProxy {
    type Output = ElementProxy;

    /// Returns the ElementProxy corresponding to the given index.
    fn index(&self, index: usize) -> &ElementProxy {
        let path = [&self.path[..], &[index.to_isize().unwrap()]].concat();
        self.sub.get(index, || ElementProxy::new(path.into()))
    }
}

impl_proxy!(ElementProxy);
impl_comparable_proxy!(ElementProxy, IntFactoid);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_proxy_datum_type() {
        let input = TensorProxy::new(vec![0, 0].into());
        assert_eq!(input.datum_type.get_path(), &vec![0, 0, 0].into());
    }

    #[test]
    fn test_tensor_proxy_rank() {
        let input = TensorProxy::new(vec![0, 0].into());
        assert_eq!(input.rank.get_path(), &vec![0, 0, 1].into());
    }

    #[test]
    fn test_tensor_proxy_shape() {
        let input = TensorProxy::new(vec![0, 0].into());
        assert_eq!(input.shape[0].get_path(), &vec![0, 0, 2, 0].into());
        assert_eq!(input.shape[2].get_path(), &vec![0, 0, 2, 2].into());
    }

    #[test]
    fn test_tensor_proxy_value() {
        let input = TensorProxy::new(vec![0, 0].into());
        assert_eq!(input.value.get_path(), &vec![0, 0, 3].into());
        assert_eq!(input.value[()].get_path(), &vec![0, 0, 3, -1].into());
        assert_eq!(input.value[0].get_path(), &vec![0, 0, 3, 0].into());
        assert_eq!(input.value[0][1].get_path(), &vec![0, 0, 3, 0, 1].into());
        assert_eq!(input.value[1][2][3].get_path(), &vec![0, 0, 3, 1, 2, 3].into());
    }
}
