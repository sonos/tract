use std::fmt;
use std::iter::FromIterator;
use std::ops::{Add, Div, Mul, Sub, Neg};
use Result;

use num::Zero;

use DatumType;
use Tensor;
use linear::LinearDim;

/// Partial information about any value.
pub trait Fact: fmt::Debug + Clone + PartialEq + Default {
    type Concrete: fmt::Debug;

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
/// about the tensors that flow through it - specifically their datum_type, their
/// shape and possibly their value. During the analysis, however, we might only
/// know some of that information (say, for instance, that an edge only carries
/// tensors of rank 4, but without knowing their precise dimension).
///
/// This is where tensor facts come in: they hold partial information about the
/// datum_type, shape and value of tensors that might flow through an edge of the
/// graph. The analyser will first tag each edge with a fact, starting with the
/// most general one and specializing it at each iteration. Eventually, it will
/// reach a fixed point that - hopefully - holds enough information.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Clone, PartialEq, Default)]
pub struct TensorFact {
    pub datum_type: TypeFact,
    pub shape: ShapeFact,
    pub value: ValueFact,
}

impl TensorFact {
    /// Constructs the most general tensor fact possible.
    pub fn new() -> TensorFact {
        TensorFact::default()
    }

    pub fn with_datum_type(self, dt: DatumType) -> TensorFact {
        TensorFact {
            datum_type: dt.into(),
            ..self
        }
    }

    pub fn with_shape<S: Into<ShapeFact>>(self, shape: S) -> TensorFact {
        TensorFact {
            shape: shape.into(),
            ..self
        }
    }

    pub fn stream_dim(&self) -> Result<Option<(usize, LinearDim)>> {
        self.shape.stream_dim()
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
            datum_type: self.datum_type.unify(&other.datum_type)?,
            shape: self.shape.unify(&other.shape)?,
            value: self.value.unify(&other.value)?,
        };

        trace!("Unifying {:?} with {:?} into {:?}.", self, other, tensor);

        Ok(tensor)
    }
}

impl<T: Into<Tensor>> From<T> for TensorFact {
    fn from(t: T) -> TensorFact {
        let t: Tensor = t.into();
        TensorFact {
            datum_type: GenericFact::Only(t.datum_type()),
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
            if let Some(t) = self.datum_type.concretize() {
                write!(formatter, ", {:?}", t)?;
            }
            write!(formatter, ", shape={:?}", self.shape)?;
            Ok(())
        }
    }
}

/// Partial information about a value of type T.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum GenericFact<T: fmt::Debug + Clone + PartialEq> {
    Only(T),
    Any,
}

impl<T:Copy+Clone+fmt::Debug+PartialEq> Copy for GenericFact<T> {}

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
            _ if self == other => self.clone(),
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
pub type TypeFact = GenericFact<DatumType>;

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

    pub fn stream_dim(&self) -> Result<Option<(usize, LinearDim)>> {
        let concrete = self.concretize().ok_or("Shape has unknown dims, can not find streaming dim for sure.")?;
        let count = concrete
            .iter()
            .filter(|&d| d.is_stream())
            .count();
        if count > 1 {
            bail!("Shape has more than one streaming dim. This is terribly wrong.")
        }
        Ok(concrete
            .into_iter()
            .enumerate()
            .find(|(_,d)| d.is_stream()))
    }
}

impl Fact for ShapeFact {
    type Concrete = Vec<LinearDim>;

    /// Tries to transform the fact into a `Vec<usize>`, or returns `None`.
        fn concretize(self: &ShapeFact) -> Option<Vec<LinearDim>> {
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

        let dimensions: Vec<_> = xi
            .zip_longest(yi)
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

impl FromIterator<LinearDim> for ShapeFact {
    /// Converts an iterator over usize into a closed shape.
    fn from_iter<I: IntoIterator<Item = LinearDim>>(iter: I) -> ShapeFact {
        ShapeFact::closed(iter.into_iter().map(|d| GenericFact::Only(d)).collect())
    }
}

impl<'a> From<&'a [usize]> for ShapeFact {
    /// Converts an usize slice into a closed shape.
    fn from(slice: &'a [usize]) -> ShapeFact {
        slice.into_iter().map(|i| LinearDim::from(*i)).collect()
    }
}

impl From<Vec<usize>> for ShapeFact {
    /// Converts an vector of usize into a closed shape.
    fn from(shape: Vec<usize>) -> ShapeFact {
        shape.into_iter().map(|i| LinearDim::from(i)).collect()
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

/*
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
            GenericFact::Any => None,
            DimFact::Streamed => None,
            GenericFact::Only(i) => Some(*i),
        }
    }

    /// Returns whether the dimension is concrete.
    fn is_concrete(&self) -> bool {
        match self {
            GenericFact::Any => false,
            DimFact::Streamed => true,
            GenericFact::Only(_) => true,
        }
    }

    /// Tries to unify the fact with another `DimFact`.
    fn unify(&self, other: &Self) -> Result<Self> {
        let fact = match (self, other) {
            (_, GenericFact::Any) => self.clone(),
            (GenericFact::Any, _) => other.clone(),
            _ if self == other => self.clone(),
            _ => bail!("Impossible to unify {:?} with {:?}.", self, other),
        };

        Ok(fact)
    }
}

impl Default for DimFact {
    fn default() -> DimFact {
        GenericFact::Any
    }
}

impl From<usize> for DimFact {
    fn from(t: usize) -> DimFact {
        GenericFact::Only(t)
    }
}

impl fmt::Debug for DimFact {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GenericFact::Any => write!(formatter, "?"),
            DimFact::Streamed => write!(formatter, "S"),
            GenericFact::Only(d) => write!(formatter, "{}", d),
        }
    }
}
*/

pub type DimFact = GenericFact<LinearDim>;

/// Partial information about a value.
pub type ValueFact = GenericFact<Tensor>;

pub type IntFact = GenericFact<isize>;

/*
impl Fact for IntFact {
    type Concrete = isize;

    /// Tries to transform the fact into a concrete value.
    fn concretize(&self) -> Option<isize> {
        match self {
            GenericFact::Only(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns whether the IntFact is concrete.
    fn is_concrete(&self) -> bool {
        match self {
            GenericFact::Any => false,
            GenericFact::Only(_) => true,
            IntFact::Special(_) => true,
        }
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &IntFact) -> Result<IntFact> {
        let fact = match (self, other) {
            (_, GenericFact::Any) => self.clone(),
            (GenericFact::Any, _) => other.clone(),
            _ if self == other => self.clone(),
            _ => bail!("Impossible to unify {:?} with {:?}.", self, other),
        };

        Ok(fact)
    }
}

impl Default for IntFact {
    fn default() -> IntFact {
        GenericFact::Any
    }
}

impl From<isize> for IntFact {
    fn from(i: isize) -> IntFact {
        GenericFact::Only(i)
    }
}

impl From<usize> for IntFact {
    fn from(u: usize) -> IntFact {
        GenericFact::Only(u.to_isize().unwrap())
    }
}
impl From<DimFact> for IntFact {
    fn from(d: DimFact) -> IntFact {
        match d {
            GenericFact::Any => GenericFact::Any,
            GenericFact::Only(d) => d.into(),
            DimFact::Streamed => IntFact::Special(SpecialKind::Streamed),
        }
    }
}
*/

impl<T> Zero for GenericFact<T>
where T: Add<T, Output=T> + Zero + PartialEq + Copy + Clone + ::std::fmt::Debug,
{
    fn zero() -> GenericFact<T> {
        GenericFact::Only(T::zero())
    }
    fn is_zero(&self) -> bool {
        match self {
            GenericFact::Only(t) => t.is_zero(),
            _ => false
        }
    }
}

impl<T> Neg for GenericFact<T>
where T: Neg<Output=T> + PartialEq + Copy + Clone + ::std::fmt::Debug
{
    type Output=GenericFact<T>;
    fn neg(self) -> GenericFact<T> {
        match self {
            GenericFact::Only(t) => GenericFact::Only(t.neg()),
            any => any,
        }
    }
}

impl<T> Add<GenericFact<T>> for GenericFact<T>
where   T: Add<T, Output=T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
{
    type Output=GenericFact<T>;
    fn add(self, rhs: GenericFact<T>) -> Self::Output {
        match (self.concretize(), rhs.concretize()) {
            (Some(a), Some(b)) => GenericFact::Only(a+b),
            _ => GenericFact::Any
        }
    }
}

impl<T,R> Add<R> for GenericFact<T>
where   T: Add<R, Output=T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
      R: ::num::Num
{
    type Output=GenericFact<T>;
    fn add(self, rhs: R) -> Self::Output {
        if let Some(a) = self.concretize() {
            GenericFact::Only(a+rhs)
        } else {
            GenericFact::Any
        }
    }
}

impl<T> Add<GenericFact<T>> for isize
where   T: Add<isize, Output=T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
{
    type Output=GenericFact<T>;
    fn add(self, rhs: GenericFact<T>) -> Self::Output {
        rhs + self
    }
}


impl<T> Sub<GenericFact<T>> for GenericFact<T>
where   T: Sub<T, Output=T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
{
    type Output=GenericFact<T>;
    fn sub(self, rhs: GenericFact<T>) -> Self::Output {
        match (self.concretize(), rhs.concretize()) {
            (Some(a), Some(b)) => GenericFact::Only(a-b),
            _ => GenericFact::Any
        }
    }
}

impl<T,R> Mul<R> for GenericFact<T>
where   T: Mul<R, Output=T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
      R: ::num::Num
{
    type Output=GenericFact<T>;
    fn mul(self, rhs: R) -> Self::Output {
        if let Some(a) = self.concretize() {
            GenericFact::Only(a*rhs)
        } else {
            GenericFact::Any
        }
    }
}

impl<T> Mul<GenericFact<T>> for GenericFact<T>
where   T: Mul<T, Output=T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
{
    type Output=GenericFact<T>;
    fn mul(self, rhs: GenericFact<T>) -> Self::Output {
        match (self.concretize(), rhs.concretize()) {
            (Some(a), Some(b)) => GenericFact::Only(a*b),
            _ => GenericFact::Any
        }
    }
}

impl<T,R> Div<R> for GenericFact<T>
where   T: Div<R, Output=T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
      R: ::num::Num
{
    type Output=GenericFact<T>;
    fn div(self, rhs: R) -> Self::Output {
        if let Some(a) = self.concretize() {
            GenericFact::Only(a/rhs)
        } else {
            GenericFact::Any
        }
    }
}

impl<T> Div<GenericFact<T>> for GenericFact<T>
where   T: Div<T, Output=T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
{
    type Output=GenericFact<T>;
    fn div(self, rhs: GenericFact<T>) -> Self::Output {
        match (self.concretize(), rhs.concretize()) {
            (Some(a), Some(b)) => GenericFact::Only(a/b),
            _ => GenericFact::Any
        }
    }
}

/*
impl CheckedDiv for IntFact {V
    fn checked_div(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (IntFact::Special(a), IntFact::Special(b)) => if a == b {
                Some(IntFact::Special(*a))
            } else {
                panic!(
                    "Shouldn't perform an arithmetic operation on two\
                     different special values ({:?} and {:?}).",
                    a, b
                );
            },

            (IntFact::Special(s), _) | (_, IntFact::Special(s)) => Some(IntFact::Special(*s)),

            (GenericFact::Only(i), IntFact::Only(j)) => i.checked_div(j).map(|k| k.into()),

            _ => Some(GenericFact::Any),
        }
    }
}
*/
