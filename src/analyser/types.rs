use std::iter::FromIterator;
use std::ops::*;

use errors::*;
use tfpb::types::DataType;
use Tensor;

/// Partial information about a tensor.
///
/// The task of the analyser is to tag every edge in the graph with information
/// about the tensors that flow through it - specifically their datatype, their
/// shape and possibly their value. During the analysis, however, we might only
/// know some of that information (say, for instance, that an edge only carries
/// tensors of rank 4, but without knowing their precise dimension).
///
/// This is where tensor facts come in: they hold partial information about the
/// datatype, shape and value of tensors that might flow through an edge of the
/// graph. The analyser will first tag each edge with a fact, starting with the
/// most general one and specializing it at each iteration. Eventually, it will
/// reach a fixed point that - hopefully - holds enough information.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct TensorFact {
    pub datatype: TypeFact,
    pub shape: ShapeFact,
    pub value: ValueFact,
}

impl TensorFact {
    /// Constructs the most general tensor fact possible.
    pub fn new() -> TensorFact {
        TensorFact {
            datatype: TypeFact::Any,
            shape: ShapeFact::any(),
            value: ValueFact::Any,
        }
    }
}

/// Partial information about a type.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum TypeFact {
    Any,
    Only(DataType),
}

impl TypeFact {
    /// Tries to transform the type fact into a DataType, or returns None.
    pub fn concretize(&self) -> Option<DataType> {
        match self {
            TypeFact::Any => None,
            TypeFact::Only(d) => Some(*d),
        }
    }
}

/// Partial information about a shape.
///
/// A basic example of a shape fact is `shapefact![1, 2]`, which corresponds to
/// the shape `[1, 2]` in Tensorflow. We can use `_` in facts to denote unknown
/// dimensions (e.g. `shapefact![1, 2, _]` corresponds to any shape `[1, 2, k]`
/// with `k` a non-negative integer). We can also use `..` at the end of a fact
/// to only specify its first dimensions, so `shapefact![1, 2; ..]` matches any
/// shape that starts with `[1, 2]` (e.g. `[1, 2, i]` or `[1, 2, i, j]`), while
/// `shapefact![..]` matches any shape.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeFact {
    pub open: bool,
    pub dims: Vec<DimFact>,
}

impl ShapeFact {
    /// Returns the most general shape fact possible.
    pub fn any() -> ShapeFact {
        ShapeFact::open(vec![])
    }

    /// Constructs an open shape fact.
    pub fn open(dims: Vec<DimFact>) -> ShapeFact {
        ShapeFact { open: true, dims }
    }

    /// Constructs a closed shape fact.
    pub fn closed(dims: Vec<DimFact>) -> ShapeFact {
        ShapeFact { open: false, dims }
    }

    /// Tries to transform the fact into a Vec<usize>, or returns None.
    pub fn concretize(self: &ShapeFact) -> Option<Vec<usize>> {
        if self.open {
            debug!("Impossible to concretize an open shape.");
            return None;
        }

        let dims: Vec<_> = self.dims.iter().filter_map(|d| d.concretize()).collect();

        if dims.len() < self.dims.len() {
            debug!("Impossible to concretize a shape with unknown dimensions.");
            None
        } else {
            Some(dims)
        }
    }
}

impl FromIterator<usize> for ShapeFact {
    /// Converts an iterator over usize into a closed shape.
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> ShapeFact {
        ShapeFact::closed(iter.into_iter().map(|d| DimFact::Only(d)).collect())
    }
}

impl<'a> FromIterator<&'a usize> for ShapeFact {
    /// Converts an iterator over &usize into a closed shape.
    fn from_iter<I: IntoIterator<Item = &'a usize>>(iter: I) -> ShapeFact {
        ShapeFact::closed(iter.into_iter().map(|d| DimFact::Only(*d)).collect())
    }
}

impl<'a> From<&'a [usize]> for ShapeFact {
    /// Converts an usize slice into a closed shape.
    fn from(slice: &'a [usize]) -> ShapeFact {
        slice.iter().collect()
    }
}

/// Partial information about a dimension.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum DimFact {
    Any,
    Only(usize),
}

impl DimFact {
    /// Tries to transform the dimension fact into an usize, or returns None.
    pub fn concretize(&self) -> Option<usize> {
        match self {
            DimFact::Any => None,
            DimFact::Only(i) => Some(*i),
        }
    }

    /// Returns whether the dimension is fully determined.
    pub fn is_concrete(&self) -> bool {
        self.concretize().is_some()
    }
}

/// Implements arithmetic operations over DimFacts.
macro_rules! impl_dim_op {
    ($trait:ident, $method:ident, $i:ident, $j:ident, $res:expr) => {
        impl $trait<Self> for DimFact {
            type Output = Self;

            fn $method(self, other: Self) -> Self {
                match (self, other) {
                    (DimFact::Only($i), DimFact::Only($j)) => DimFact::Only($res),
                    _ => DimFact::Any,
                }
            }
        }

        impl $trait<usize> for DimFact {
            type Output = Self;

            fn $method(self, other: usize) -> Self {
                match (self, other) {
                    (DimFact::Only($i), $j) => DimFact::Only($res),
                    _ => DimFact::Any,
                }
            }
        }
    }
}

impl_dim_op!(Add, add, i, j, i + j);
impl_dim_op!(Sub, sub, i, j, i - j);
impl_dim_op!(Mul, mul, i, j, i * j);
impl_dim_op!(Div, div, i, j, i / j);


/// Partial information about a value.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum ValueFact {
    Any,
    Only(Tensor),
}

impl ValueFact {
    // Tries to transform the value fact into a Tensor, or returns None.
    pub fn concretize(self: &ValueFact) -> Option<&Tensor> {
        match self {
            ValueFact::Any => None,
            ValueFact::Only(m) => Some(m),
        }
    }

    /// Returns whether the value is fully determined.
    pub fn is_concrete(&self) -> bool {
        self.concretize().is_some()
    }

    // Applies fn to a defined value, and leaves an unknown value untouched.
    // Returns an Err if something went wrong during the transformation.
    pub fn map_err<F>(self: &ValueFact, f: F) -> Result<ValueFact>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        match self {
            ValueFact::Any => Ok(ValueFact::Any),
            ValueFact::Only(m) => Ok(ValueFact::Only(f(m)?)),
        }
    }
}

#[cfg(tests)]
mod tests {
    #[test]
    fn new_tensor_fact() {
        assert_eq!(
            TensorFact::new(),
            TensorFact {
                datatype: TypeFact::Any,
                shape: ShapeFact::any(),
                value: ValueFact::Any,
            }
        );
    }
}
