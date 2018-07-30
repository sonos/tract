use std::ops::{Add, Sub, Mul, Div};
use std::iter::FromIterator;
use std::fmt;
use Result;

use num_traits::cast::ToPrimitive;
use num_traits::CheckedDiv;

use tfpb::types::DataType;
use Tensor;

/// Partial information about any value.
pub trait Fact: fmt::Debug + Clone + PartialEq + Default {
    type Concrete;

    /// Tries to transform the fact into a concrete value.
    fn concretize(&self) -> Option<Self::Concrete>;

    /// Returns whether the value is fully determined.
    fn is_concrete(&self) -> bool {
        self.concretize().is_some()
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> Result<Self>;
}

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
#[derive(Clone, PartialEq, Default)]
pub struct TensorFact {
    pub datatype: TypeFact,
    pub shape: ShapeFact,
    pub value: ValueFact,
}

impl TensorFact {
    /// Constructs the most general tensor fact possible.
    pub fn new() -> TensorFact {
        TensorFact::default()
    }
}

impl Fact for TensorFact {
    type Concrete = Tensor;

    /// Tries to transform the fact into a concrete value.
    fn concretize(&self) -> Option<Self::Concrete> {
        self.value.concretize()
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> Result<Self> {
        let tensor = TensorFact {
            datatype: self.datatype.unify(&other.datatype)?,
            shape:    self.shape.unify(&other.shape)?,
            value:    self.value.unify(&other.value)?,
        };

        trace!("Unifying {:?} with {:?} into {:?}.", self, other, tensor);

        Ok(tensor)
    }
}

impl From<Tensor> for TensorFact {
    fn from(t: Tensor) -> TensorFact {
        TensorFact {
            datatype: GenericFact::Only(t.datatype()),
            shape: ShapeFact::from(t.shape()),
            value: GenericFact::Only(t),
        }
    }
}

impl fmt::Debug for TensorFact {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        if let Some(t) = self.value.concretize() {
            write!(formatter, "Fully determined tensor: {:?}", t)
        } else {
            write!(formatter, "Tensor")?;
            if let Some(t) = self.datatype.concretize() {
                write!(formatter, " {:?}", t)?;
            }
            write!(formatter, " {:?}", self.shape)?;
            Ok(())
        }
    }
}

/// Partial information about a value of type T.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum GenericFact<T: fmt::Debug + Clone + PartialEq> {
    Any,
    Only(T)
}

impl<T: fmt::Debug + Clone + PartialEq> Fact for GenericFact<T> {
    type Concrete = T;

    /// Tries to transform the fact into a concrete value.
    fn concretize(&self) -> Option<T> {
        match self {
            GenericFact::Any => None,
            GenericFact::Only(m) => Some(m.clone()),
        }
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> Result<Self> {
        let fact = match (self, other) {
            (_, GenericFact::Any) => self.clone(),
            (GenericFact::Any, _) => other.clone(),
            _ if self == other    => self.clone(),
            _ => bail!("Impossible to unify {:?} with {:?}.", self, other),
        };

        Ok(fact)
    }
}

impl<T: fmt::Debug + Clone + PartialEq> Default for GenericFact<T> {
    fn default() -> Self {
        GenericFact::Any
    }
}

impl<T: fmt::Debug + Clone + PartialEq> From<T> for GenericFact<T> {
    fn from(t: T) -> Self {
        GenericFact::Only(t)
    }
}

/// Partial information about a type.
pub type TypeFact = GenericFact<DataType>;

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
#[derive(Clone, PartialEq)]
pub struct ShapeFact {
    pub open: bool,
    pub dims: Vec<DimFact>,
}

impl ShapeFact {
    /// Constructs an open shape fact.
    pub fn open(dims: Vec<DimFact>) -> ShapeFact {
        ShapeFact { open: true, dims }
    }

    /// Constructs a closed shape fact.
    pub fn closed(dims: Vec<DimFact>) -> ShapeFact {
        ShapeFact { open: false, dims }
    }
}

impl Fact for ShapeFact {
    type Concrete = Vec<usize>;

    /// Tries to transform the fact into a `Vec<usize>`, or returns `None`.
    fn concretize(self: &ShapeFact) -> Option<Vec<usize>> {
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

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> Result<Self> {
        let (x, y) = (self, other);

        use itertools::EitherOrBoth::{Both, Left, Right};
        use itertools::Itertools;

        let xi = x.dims.iter();
        let yi = y.dims.iter();

        let dimensions: Vec<_> = xi.zip_longest(yi)
            .map(|r| match r {
                Both(a, b) => a.unify(b),
                Left(d) if y.open => Ok(*d),
                Right(d) if x.open => Ok(*d),

                Left(_) | Right(_) => bail!(
                    "Impossible to unify closed shapes of different rank (found {:?} and {:?}).",
                    x,
                    y
                ),
            })
            .collect::<Result<_>>()?;

        if x.open && y.open {
            Ok(ShapeFact::open(dimensions))
        } else {
            Ok(ShapeFact::closed(dimensions))
        }
    }
}

impl Default for ShapeFact {
    /// Returns the most general shape fact possible.
    fn default() -> ShapeFact {
        ShapeFact::open(vec![])
    }
}

impl FromIterator<usize> for ShapeFact {
    /// Converts an iterator over usize into a closed shape.
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> ShapeFact {
        ShapeFact::closed(iter.into_iter().map(|d| DimFact::Only(d)).collect())
    }
}

impl<'a> From<&'a [usize]> for ShapeFact {
    /// Converts an usize slice into a closed shape.
    fn from(slice: &'a [usize]) -> ShapeFact {
        slice.iter().cloned().collect()
    }
}

impl From<Vec<usize>> for ShapeFact {
    /// Converts an vector of usize into a closed shape.
    fn from(shape: Vec<usize>) -> ShapeFact {
        shape.into_iter().collect()
    }
}

impl FromIterator<Option<usize>> for ShapeFact {
    /// Converts an iterator over `Option<usize>` into a closed shape.
    fn from_iter<I: IntoIterator<Item = Option<usize>>>(iter: I) -> ShapeFact {
        ShapeFact::closed(iter.into_iter().map(|d| match d {
            Some(d) => DimFact::Only(d),
            None => DimFact::Streamed,
        }).collect())
    }
}

impl<'a> From<&'a [Option<usize>]> for ShapeFact {
    /// Converts an `Option<usize>` slice into a closed shape.
    fn from(slice: &'a [Option<usize>]) -> ShapeFact {
        slice.iter().cloned().collect()
    }
}

impl fmt::Debug for ShapeFact {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "[")?;
        for (ix, d) in self.dims.iter().enumerate() {
            if ix != 0 {
                write!(formatter, ",")?
            }
            write!(formatter, "{:?}", d)?;
        }
        if self.open {
            write!(formatter, "..")
        } else {
            write!(formatter, "]")
        }
    }
}

/// Partial information about a dimension.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Clone, Copy, PartialEq)]
pub enum DimFact {
    Any,
    Streamed,
    Only(usize),
}

impl DimFact {
    /// Returns whether the dimension is streamed.
    pub fn is_streamed(&self) -> bool {
        self == &DimFact::Streamed
    }
}

impl Fact for DimFact {
    type Concrete = usize;

    /// Tries to transform the dimension fact into an `usize`, or returns `None`.
    fn concretize(&self) -> Option<usize> {
        match self {
            DimFact::Any => None,
            DimFact::Streamed => None,
            DimFact::Only(i) => Some(*i),
        }
    }

    /// Returns whether the dimension is concrete.
    fn is_concrete(&self) -> bool {
        match self {
            DimFact::Any => false,
            DimFact::Streamed => true,
            DimFact::Only(_) => true,
        }
    }

    /// Tries to unify the fact with another `DimFact`.
    fn unify(&self, other: &Self) -> Result<Self> {
        let fact = match (self, other) {
            (_, DimFact::Any)  => self.clone(),
            (DimFact::Any, _)  => other.clone(),
            _ if self == other => self.clone(),
            _ => bail!("Impossible to unify {:?} with {:?}.", self, other),
        };

        Ok(fact)
    }
}

impl Default for DimFact {
    fn default() -> DimFact {
        DimFact::Any
    }
}

impl From<usize> for DimFact {
    fn from(t: usize) -> DimFact {
        DimFact::Only(t)
    }
}

impl fmt::Debug for DimFact {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DimFact::Any => write!(formatter, "?"),
            DimFact::Streamed => write!(formatter, "S"),
            DimFact::Only(d) => write!(formatter, "{}", d),
        }
    }
}

/// Partial information about a value.
pub type ValueFact = GenericFact<Tensor>;

/// Partial information about any integer-like value.
///
/// This type is mostly used as plumbing for the solver: we convert all
/// `DimFact`, isize and usize values into `IntFact` values so that we
/// can both compare them and do some arithmetic on them. This becomes
/// useful when trying to do something like:
/// ```text
/// solver.equals(&input.rank, &output.shape[0]);
///                ^^^^^^^^^^   ^^^^^^^^^^^^^^^
///                  IntFact        DimFact
///```
///
/// Values using the constructor `Special` are treated as absorbing
/// elements for all arithmetic operations. This is currently used to
/// represent `DimFact::Streamed` values, but could be used for other
/// absorbing values in the future.
#[derive(Debug, Clone, PartialEq)]
pub enum IntFact {
    Any,
    Only(isize),
    Special(SpecialKind),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecialKind {
    Streamed,
}

impl Fact for IntFact {
    type Concrete = isize;

    /// Tries to transform the fact into a concrete value.
    fn concretize(&self) -> Option<isize> {
        match self {
            IntFact::Only(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns whether the IntFact is concrete.
    fn is_concrete(&self) -> bool {
        match self {
            IntFact::Any        => false,
            IntFact::Only(_)    => true,
            IntFact::Special(_) => true,
        }
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &IntFact) -> Result<IntFact> {
        let fact = match (self, other) {
            (_, IntFact::Any)  => self.clone(),
            (IntFact::Any, _)  => other.clone(),
            _ if self == other => self.clone(),
            _ => bail!("Impossible to unify {:?} with {:?}.", self, other),
        };

        Ok(fact)
    }
}

impl Default for IntFact {
    fn default() -> IntFact {
        IntFact::Any
    }
}

impl From<isize> for IntFact {
    fn from(i: isize) -> IntFact {
        IntFact::Only(i)
    }
}

impl From<usize> for IntFact {
    fn from(u: usize) -> IntFact {
        IntFact::Only(u.to_isize().unwrap())
    }
}

impl From<DimFact> for IntFact {
    fn from(d: DimFact) -> IntFact {
        match d {
            DimFact::Any      => IntFact::Any,
            DimFact::Only(d)  => d.into(),
            DimFact::Streamed => IntFact::Special(SpecialKind::Streamed),
        }
    }
}

// /// Implements arithmetic operations over `DimFact`s.
macro_rules! impl_op {
    ($fact:ident, $trait:ident, $method:ident, $i:ident, $j:ident, $res:expr) => {
        impl $trait<Self> for $fact {
            type Output = Self;

            fn $method(self, other: Self) -> Self {
                match (self, other) {
                    (IntFact::Special(a), IntFact::Special(b)) => if a == b {
                        IntFact::Special(a)
                    } else {
                        panic!("Shouldn't perform an arithmetic operation on two\
                                different special values ({:?} and {:?}).", a, b);
                    },

                    (IntFact::Special(s), _) |
                    (_, IntFact::Special(s)) => IntFact::Special(s),

                    (IntFact::Only($i), IntFact::Only($j)) => IntFact::Only($res),
                    _ => IntFact::Any,
                }
            }
        }

        impl $trait<isize> for $fact {
            type Output = Self;

            fn $method(self, other: isize) -> Self {
                match (self, other) {
                    (IntFact::Special(s), _) => IntFact::Special(s),
                    (IntFact::Only($i), $j)  => IntFact::Only($res),
                    _ => IntFact::Any,
                }
            }
        }

        impl $trait<usize> for $fact {
            type Output = Self;

            fn $method(self, other: usize) -> Self {
                match (self, other) {
                    (IntFact::Special(s), _) => IntFact::Special(s),
                    (IntFact::Only($i), $j)  => {
                        let $j = ($j).to_isize().unwrap();
                        IntFact::Only($res)
                    },
                    _ => IntFact::Any,
                }
            }
        }
    }
}

impl_op!(IntFact, Add, add, i, j, i + j);
impl_op!(IntFact, Sub, sub, i, j, i - j);
impl_op!(IntFact, Mul, mul, i, j, i * j);
impl_op!(IntFact, Div, div, i, j, i / j);

impl CheckedDiv for IntFact {
    fn checked_div(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (IntFact::Special(a), IntFact::Special(b)) => if a == b {
                Some(IntFact::Special(*a))
            } else {
                panic!("Shouldn't perform an arithmetic operation on two\
                        different special values ({:?} and {:?}).", a, b);
            },

            (IntFact::Special(s), _) |
            (_, IntFact::Special(s)) => Some(IntFact::Special(*s)),

            (IntFact::Only(i), IntFact::Only(j)) =>
                i.checked_div(j).map(|k| k.into()),

            _ => Some(IntFact::Any),
        }
    }
}
