use std::iter::FromIterator;

use errors::*;
use tfpb::types::DataType;
use Matrix;


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
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum TypeFact {
    Any,
    Only(DataType),
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
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeFact {
    Open(Vec<DimFact>),
    Closed(Vec<DimFact>),
}

impl ShapeFact {
    /// Returns the most general shape fact possible.
    pub fn any() -> ShapeFact {
        ShapeFact::Open(vec![])
    }

    /// Returns whether the fact is open.
    pub fn is_open(self: &ShapeFact) -> bool {
        match self {
            ShapeFact::Open(_) => true,
            ShapeFact::Closed(_) => false,
        }
    }

    /// Returns the vector of dimensions defining the fact.
    pub fn inner(self: &ShapeFact) -> &Vec<DimFact> {
        match self {
            ShapeFact::Open(v) | ShapeFact::Closed(v) => v,
        }
    }

    /// Tries to transform the fact into a Vec<usize>, or returns
    /// an Err if some of the dimensions are unknown.
    pub fn concretize(self: &ShapeFact) -> Result<Vec<usize>> {
        match self {
            ShapeFact::Open(_) =>
                bail!("Impossible to concretize an open shape."),
            ShapeFact::Closed(v) => v
                .iter()
                .map(|d| match d {
                    DimFact::Any =>
                        bail!("Impossible to concretize a shape with an unknown dimension."),
                    DimFact::Only(i) =>
                        Ok(*i)
                })
                .collect()
        }
    }
}

impl FromIterator<usize> for ShapeFact {
    /// Converts an iterator over usize into a closed shape.
    fn from_iter<I: IntoIterator<Item=usize>>(iter: I) -> ShapeFact {
        ShapeFact::Closed(iter
            .into_iter()
            .map(|d| DimFact::Only(d))
            .collect())
    }
}

impl<'a> FromIterator<&'a usize> for ShapeFact {
    /// Converts an iterator over &usize into a closed shape.
    fn from_iter<I: IntoIterator<Item=&'a usize>>(iter: I) -> ShapeFact {
        ShapeFact::Closed(iter
            .into_iter()
            .map(|d| DimFact::Only(*d))
            .collect())
    }
}

impl<'a> From<&'a[usize]> for ShapeFact {
    /// Converts an usize slice into a closed shape.
    fn from(slice: &'a[usize]) -> ShapeFact {
        slice.iter().collect()
    }
}

/// Partial information about a dimension.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum DimFact {
    Any,
    Only(usize),
}

impl DimFact {
    /// Returns whether the dimension is known.
    pub fn is_concrete(&self) -> bool {
        match self {
            DimFact::Any => false,
            DimFact::Only(_) => true
        }
    }
}

/// Partial information about a value.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueFact {
    Any,
    Only(Matrix),
}

impl ValueFact {
    // Tries to transform the value fact into a Matrix, or returns an Err.
    pub fn concretize(self: &ValueFact) -> Result<&Matrix> {
        match self {
            ValueFact::Any =>
                bail!("Impossible to concretize an Any value."),
            ValueFact::Only(m) =>
                Ok(m)
        }
    }

    // Applies fn to a defined value, and leaves an unknown value untouched.
    // Returns an Err if something went wrong during the transformation.
    pub fn map_err<F>(self: &ValueFact, f: F) -> Result<ValueFact>
    where F: Fn(&Matrix) -> Result<Matrix> {
        match self {
            ValueFact::Any => Ok(ValueFact::Any),
            ValueFact::Only(m) => Ok(ValueFact::Only(f(m)?))
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
