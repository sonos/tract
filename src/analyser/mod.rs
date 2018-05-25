use std::iter::FromIterator;

use errors::*;
use ops::Op;
use tfpb::types::DataType;
use Matrix;

/// An abstract tensor.
///
/// The task of the analyser is to tag every edge in the graph with information
/// about the tensors that flow through it - specifically their datatype, their
/// shape and possibly their value. During the analysis, however, we might only
/// know some of that information (say, for instance, that an edge only carries
/// tensors of rank 4, but without knowing their precise dimension).
///
/// This is where abstract tensors come in: they hold partial information about
/// the datatype, shape and value of tensors that might flow through an edge of
/// the graph. The analyser will first tag each edge of the graph with the most
/// general abstract tensor possible, and after each iteration of the analysis,
/// the tensors will become more and more specialized - until reaching a fixed
/// point that will hopefully contain enough information for us to work with.
#[derive(Debug, Clone, PartialEq)]
pub struct ATensor {
    pub datatype: AType,
    pub shape: AShape,
    pub value: AValue,
}

impl ATensor {
    /// Constructs a new abstract tensor, which is as general as possible.
    pub fn new() -> ATensor {
        ATensor {
            datatype: AType::Any,
            shape: AShape::any(),
            value: AValue::Any,
        }
    }
}

/// An abstract type.
#[derive(Debug, Clone, PartialEq)]
pub enum AType {
    Any,
    Only(DataType),
}

#[macro_export]
macro_rules! atype {
    (_) =>
        ($crate::analyser::AType::Any);
    ($arg:expr) =>
        ($crate::analyser::AType::Only($arg));
}

/// An abstract shape.
/// They are used to represent partial information about the shapes of tensors.
///
/// A basic example of abstract shape is `ashape![1, 2]` - which corresponds to
/// the shape `[1, 2]` in Tensorflow. We can use unknown dimensions in abstract
/// shapes: `ashape![1, 2, _]` corresponds to any shape `[1, 2, k]`, with `k` a
/// nonnegative integer. We can also use `..` to only describe the beginning of
/// a shape, so `ashape![1; ..]` matches any shape that starts with `[1]` (e.g.
/// `[1]`, `[1, k]`, etc.), and `ashape![..]` matches any shape.
#[derive(Debug, Clone, PartialEq)]
pub enum AShape {
    Open(Vec<ADimension>),
    Closed(Vec<ADimension>),
}

impl AShape {
    /// Returns the most general abstract shape possible.
    pub fn any() -> AShape {
        AShape::Open(vec![])
    }

    /// Returns whether the abstract shape is open.
    pub fn is_open(self: &AShape) -> bool {
        match self {
            AShape::Open(_) => true,
            AShape::Closed(_) => false,
        }
    }

    /// Returns the vector of dimensions defining the abstract shape.
    pub fn inner(self: &AShape) -> &Vec<ADimension> {
        match self {
            AShape::Open(v) | AShape::Closed(v) => v,
        }
    }

    /// Tries to transform the abstract shape into a Vec<usize>, or returns
    /// an Err if some of the dimensions are unknown.
    pub fn concretize(self: &AShape) -> Result<Vec<&usize>> {
        match self {
            AShape::Open(_) =>
                bail!("Impossible to concretize an open shape."),
            AShape::Closed(v) => v
                .iter()
                .map(|d| match d {
                    ADimension::Any =>
                        bail!("Impossible to concretize a shape with an unknown dimension."),
                    ADimension::Only(i) =>
                        Ok(i)
                })
                .collect()
        }
    }
}

impl FromIterator<usize> for AShape {
    /// Converts a Vec<usize> into a closed shape.
    fn from_iter<I: IntoIterator<Item=usize>>(iter: I) -> AShape {
        AShape::Closed(iter
            .into_iter()
            .map(|d| ADimension::Only(d))
            .collect())
    }
}

impl<'a> FromIterator<&'a usize> for AShape {
    /// Converts a Vec<usize> into a closed shape.
    fn from_iter<I: IntoIterator<Item=&'a usize>>(iter: I) -> AShape {
        AShape::Closed(iter
            .into_iter()
            .map(|d| ADimension::Only(*d))
            .collect())
    }
}

impl<'a> From<&'a[usize]> for AShape {
    /// Converts an usize slice into a closed shape.
    fn from(slice: &'a[usize]) -> AShape {
        slice.iter().collect()
    }
}

#[macro_export]
macro_rules! ashape {
    () =>
        ($crate::analyser::AShape::Closed(vec![]));
    (..) =>
        ($crate::analyser::AShape::Open(vec![]));
    ($($arg:tt),+; ..) =>
        ($crate::analyser::AShape::Open(vec![$(adimension!($arg)),+]));
    ($($arg:tt),+) =>
        ($crate::analyser::AShape::Closed(vec![$(adimension!($arg)),+]));
}

/// An abstract dimension.
#[derive(Debug, Clone, PartialEq)]
pub enum ADimension {
    Any,
    Only(usize),
}

#[macro_export]
macro_rules! adimension {
    (_) =>
        ($crate::analyser::ADimension::Any);
    ($arg:expr) =>
        ($crate::analyser::ADimension::Only($arg));
}

/// An abstract value.
#[derive(Debug, Clone, PartialEq)]
pub enum AValue {
    Any,
    Only(Matrix),
}

impl AValue {
    // Tries to transform the abstract value into a Matrix, or returns an Err.
    pub fn concretize(self: &AValue) -> Result<&Matrix> {
        match self {
            AValue::Any =>
                bail!("Impossible to concretize an Any value."),
            AValue::Only(m) =>
                Ok(m)
        }
    }
}

#[macro_export]
macro_rules! avalue {
    (_) =>
        ($crate::analyser::AValue::Any);
    ($arg:expr) =>
        ($crate::analyser::AValue::Only($arg));
}

/// Attempts to unify two abstract tensors into a more specialized one.
pub fn unify(x: &ATensor, y: &ATensor) -> Result<ATensor> {
    let tensor = ATensor {
        datatype: unify_datatype(&x.datatype, &y.datatype)?,
        shape: unify_shape(&x.shape, &y.shape)?,
        value: unify_value(&x.value, &y.value)?,
    };

    Ok(tensor)
}

/// Attempts to unify two abstract datatypes.
fn unify_datatype(x: &AType, y: &AType) -> Result<AType> {
    use self::AType::*;

    let datatype = match (x, y) {
        (_, Any) => x.clone(),
        (Any, _) => y.clone(),
        (Only(a), Only(b)) => if a == b {
            x.clone()
        } else {
            bail!("Impossible to unify datatypes {:?} and {:?}.", x, y);
        },
    };

    Ok(datatype)
}

/// Attempts to unify two abstract shapes.
fn unify_shape(x: &AShape, y: &AShape) -> Result<AShape> {
    use self::ADimension::*;
    use self::AShape::*;
    use itertools::EitherOrBoth::{Both, Left, Right};
    use itertools::Itertools;

    let xi = x.inner().iter();
    let yi = y.inner().iter();

    let dimensions: Vec<_> = xi.zip_longest(yi)
        .map(|r| match r {
            Both(Any, Any) => Ok(Any),

            Both(Only(i), Any) | Both(Any, Only(i)) => Ok(Only(*i)),

            Both(Only(i), Only(j)) if i == j => Ok(Only(*i)),
            Both(Only(i), Only(j)) => bail!("Impossible to unify dimensions {:?} and {:?}.", i, j),

            Left(d) if y.is_open() => Ok(d.clone()),
            Right(d) if x.is_open() => Ok(d.clone()),

            Left(_) | Right(_) => bail!("Impossible to unify closed shapes of different rank."),
        })
        .collect::<Result<_>>()?;

    Ok(Closed(dimensions))
}

/// Attempts to unify two abstract values.
fn unify_value(x: &AValue, y: &AValue) -> Result<AValue> {
    use self::AValue::*;

    let value = match (x, y) {
        (_, Any) => x.clone(),
        (Any, _) => y.clone(),
        (Only(a), Only(b)) => if a == b {
            x.clone()
        } else {
            bail!("Impossible to unify values {:?} and {:?}.", x, y);
        },
    };

    Ok(value)
}

/// Infers basic properties about the output tensors from the input tensors.
pub fn infer_forward_basic(op: &Op, inputs: Vec<&ATensor>) -> Result<Vec<ATensor>> {
    let input_values: Result<Vec<_>> = inputs
        .iter()
        .map(|t| t.value.concretize())
        .collect();

    let output = match input_values {
        // If we know the value of all the inputs, we can deduce everything.
        Ok(v) => {
            let input_inputs: Vec<_> = v
                .into_iter()
                .map(|v| v.clone().into())
                .collect();

            let output_value = op.eval(input_inputs)?.pop().unwrap();

            ATensor {
                datatype: inputs[0].datatype.clone(),
                shape: output_value.shape().into(),
                value: avalue!(output_value.into_matrix())
            }
        }

        // Otherwise we can only deduce the type and shape of the output.
        _ => {
            ATensor {
                datatype: inputs[0].datatype.clone(),
                shape: ashape![..], // todo(romain): Find a way to deal with broadcasting.
                value: avalue!(_)
            }
        }
    };

    Ok(vec![output])
}

#[cfg(test)]
mod tests;