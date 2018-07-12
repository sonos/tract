use analyser::interface::cache::Cache;
use std::ops::Index;


/// A proxy for any value.
pub trait Proxy {
    /// Returns the symbolic path to the value.
    ///
    /// Take the `inputs[0].shape[1]` proxy for instance: it represents the
    /// second dimension of the shape of the first input. Because we encode
    /// the "inputs" vectors as `0`, and the `shape` field as `2`, the path
    /// for this proxy will be `&[0, 0, 2, 1]`.
    fn get_path<'a>(&self) -> Vec<usize>;
}

/// A proxy for any Datatype value.
pub trait TypeProxy: Proxy {}

/// A proxy for any integer-like value.
pub trait IntProxy: Proxy {}


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
    pub datatype: DatatypeProxy,
    pub rank: RankProxy,
    pub shape: ShapeProxy,
    pub value: ValueProxy,
}

impl TensorProxy {
    /// Creates a new TensorProxy instance.
    pub fn new() -> TensorProxy {
        TensorProxy {
            datatype: DatatypeProxy::new(),
            rank: RankProxy::new(),
            shape: ShapeProxy::new(),
            value: ValueProxy::new(),
        }
    }
}


/// A proxy for a tensor datatype.
#[derive(PartialEq)]
pub struct DatatypeProxy {}

impl DatatypeProxy {
    /// Creates a new DatatypeProxy instance.
    pub fn new() -> DatatypeProxy {
        DatatypeProxy {}
    }
}

impl Proxy for DatatypeProxy {
    /// Returns the symbolic path to the value.
    fn get_path<'a>(&self) -> Vec<usize> {
        unimplemented!()
    }
}

impl TypeProxy for DatatypeProxy {}


/// A proxy for a tensor rank.
#[derive(PartialEq)]
pub struct RankProxy {}

impl RankProxy {
    /// Creates a new RankProxy instance.
    pub fn new() -> RankProxy {
        RankProxy {}
    }
}

impl Proxy for RankProxy {
    /// Returns the symbolic path to the value.
    fn get_path<'a>(&self) -> Vec<usize> {
        unimplemented!()
    }
}

impl IntProxy for RankProxy {}


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

impl Proxy for DimProxy {
    /// Returns the symbolic path to the value.
    fn get_path<'a>(&self) -> Vec<usize> {
        unimplemented!()
    }
}

impl IntProxy for DimProxy {}


/// A proxy for a tensor value.
///
/// This proxy is a bit special as it allows arbitrarily nested indexing, so
/// that writing something like ```input.value[1][6][2]``` will always work.
/// To make this work, each ValueProxy holds a cache which will generate new
/// ValueProxys for nested items on the fly and store them.
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

impl Proxy for ValueProxy {
    /// Returns the symbolic path to the value.
    fn get_path<'a>(&self) -> Vec<usize> {
        unimplemented!()
    }
}

impl IntProxy for ValueProxy {}


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