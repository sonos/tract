use errors::*;
use tfpb::types::DataType;
use Matrix;

#[cfg(test)]
mod tests;

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

/// An abstract type.
#[derive(Debug, Clone, PartialEq)]
pub enum AType {
    Any,
    Only(DataType),
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
    /// Returns whether the abstract shape is open.
    pub fn is_open(self: &AShape) -> bool {
        match self {
            AShape::Open(_) => true,
            AShape::Closed(_) => false,
        }
    }

    /// Returns the vector of dimensions defining the abstract shape.
    pub fn unwrap(self: &AShape) -> &Vec<ADimension> {
        match self {
            AShape::Open(v) | AShape::Closed(v) => v,
        }
    }
}

/// An abstract dimension.
#[derive(Debug, Clone, PartialEq)]
pub enum ADimension {
    Any,
    Only(usize),
}

/// An abstract value.
#[derive(Debug, Clone, PartialEq)]
pub enum AValue {
    Any,
    Only(Matrix),
}

impl ATensor {
    /// Constructs a new abstract tensor, which is as general as possible.
    pub fn new() -> ATensor {
        ATensor {
            datatype: AType::Any,
            shape: AShape::Closed(vec![]),
            value: AValue::Any,
        }
    }
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

    let xi = x.unwrap().iter();
    let yi = y.unwrap().iter();

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