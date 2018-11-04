use std::fmt;
use std::iter::FromIterator;
use std::ops::{Add, Div, Mul, Neg, Sub};
use TractResult;

use num::Zero;

use ops::prelude::*;

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
    fn unify(&self, other: &Self) -> TractResult<Self>;
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

    pub fn any() -> TensorFact {
        TensorFact::default()
    }

    pub fn dt(dt: DatumType) -> TensorFact {
        TensorFact::default().with_datum_type(dt)
    }

    pub fn dt_shape<S: Into<ShapeFact>>(dt: DatumType, shape: S) -> TensorFact {
        TensorFact::dt(dt).with_shape(shape)
    }

    pub fn shape<S: Into<ShapeFact>>(shape: S) -> TensorFact {
        TensorFact::default().with_shape(shape)
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

    pub fn with_streaming_shape<S: IntoIterator<Item = Option<usize>>>(
        self,
        shape: S,
    ) -> TensorFact {
        use dim::ToDim;
        let shape: ShapeFact = shape
            .into_iter()
            .map(|d| d.map(|d| (d as isize).to_dim()).unwrap_or(TDim::s()))
            .collect();
        self.with_shape(shape)
    }

    pub fn stream_info(&self) -> TractResult<Option<StreamInfo>> {
        self.shape.stream_info()
    }
}

impl Fact for TensorFact {
    type Concrete = Tensor;

    /// Tries to transform the fact into a concrete value.
    fn concretize(&self) -> Option<Self::Concrete> {
        self.value.concretize()
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> TractResult<Self> {
        let tensor = TensorFact {
            datum_type: self.datum_type.unify(&other.datum_type)?,
            shape: self.shape.unify(&other.shape)?,
            value: self.value.unify(&other.value)?,
        };

        trace!("Unifying {:?} with {:?} into {:?}.", self, other, tensor);

        Ok(tensor)
    }
}

impl<V: Into<Tensor>> From<V> for TensorFact {
    fn from(v: V) -> TensorFact {
        let v: Tensor = v.into();
        TensorFact {
            datum_type: GenericFact::Only(v.datum_type()),
            shape: ShapeFact::from(v.shape()),
            value: GenericFact::Only(v),
        }
    }
}

impl fmt::Debug for TensorFact {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        if let Some(t) = self.value.concretize() {
            write!(formatter, "Fully determined tensor: {:?}", t)
        } else {
            write!(formatter, "DtArray")?;
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
#[derive(Clone, PartialEq)]
pub enum GenericFact<T: fmt::Debug + Clone + PartialEq> {
    Only(T),
    Any,
}

impl<T: Copy + Clone + fmt::Debug + PartialEq> Copy for GenericFact<T> {}

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
    fn unify(&self, other: &Self) -> TractResult<Self> {
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

impl<T: fmt::Debug + Clone + PartialEq> fmt::Debug for GenericFact<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GenericFact::Any => write!(formatter, "?"),
            GenericFact::Only(u) => write!(formatter, "{:?}", u),
        }
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
    open: bool,
    dims: TVec<GenericFact<i32>>,
    stream: Option<StreamInfo>,
}

impl ShapeFact {
    /// Constructs an open shape fact.
    pub fn open(dims: TVec<DimFact>) -> ShapeFact {
        if let Some((ix, &d)) = dims
            .iter()
            .enumerate()
            .find(|(_ix, d)| d.concretize().map(|d| d.is_stream()).unwrap_or(false))
        {
            let stream = Some(StreamInfo {
                axis: ix,
                len: d.concretize().unwrap(),
            });
            ShapeFact {
                open: true,
                dims: dims
                    .iter()
                    .map(|d| match d {
                        GenericFact::Only(d) if d.is_stream() => GenericFact::Only(-1),
                        GenericFact::Only(d) => GenericFact::Only(d.to_integer().unwrap()),
                        GenericFact::Any => GenericFact::Any,
                    }).collect(),
                stream,
            }
        } else {
            ShapeFact {
                open: true,
                dims: dims
                    .iter()
                    .map(|d| match d {
                        GenericFact::Only(d) => GenericFact::Only(d.to_integer().unwrap()),
                        GenericFact::Any => GenericFact::Any,
                    }).collect(),
                stream: None,
            }
        }
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Constructs a closed shape fact.
    pub fn closed(dims: TVec<DimFact>) -> ShapeFact {
        ShapeFact {
            open: false,
            ..Self::open(dims)
        }
    }

    pub fn rank(&self) -> IntFact {
        if self.open {
            GenericFact::Any
        } else {
            GenericFact::Only(self.dims.len() as i32)
        }.into()
    }

    pub fn dims(&self) -> impl Iterator<Item = DimFact> {
        let stream = self.stream.clone();
        self.dims.clone().into_iter().map(move |d| match d {
            GenericFact::Only(-1) => {
                assert!(stream.is_some());
                GenericFact::Only(stream.unwrap().len)
            }
            GenericFact::Only(d) => GenericFact::Only(d.to_dim()),
            GenericFact::Any => GenericFact::Any,
        })
    }

    pub fn stream_info(&self) -> TractResult<Option<StreamInfo>> {
        let concrete = self
            .concretize()
            .ok_or("Shape has unknown dims, can not find streaming dim for sure.")?;
        let count = concrete.iter().filter(|&d| d.is_stream()).count();
        if count > 1 {
            bail!("Shape has more than one streaming dim. This is terribly wrong.")
        }
        Ok(concrete
            .into_iter()
            .enumerate()
            .find(|(_, d)| d.is_stream())
            .map(|(axis, len)| StreamInfo { axis, len }))
    }
}

impl Fact for ShapeFact {
    type Concrete = TVec<TDim>;

    /// Tries to transform the fact into a `Vec<usize>`, or returns `None`.
    fn concretize(self: &ShapeFact) -> Option<TVec<TDim>> {
        if self.open {
            debug!("Impossible to concretize an open shape.");
            return None;
        }

        let dims: TVec<_> = self.dims().filter_map(|d| d.concretize()).collect();

        if dims.len() < self.dims.len() {
            debug!("Impossible to concretize a shape with unknown dimensions.");
            None
        } else {
            Some(dims)
        }
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> TractResult<Self> {
        let (x, y) = (self, other);

        use itertools::EitherOrBoth::{Both, Left, Right};
        use itertools::Itertools;

        let xi = x.dims();
        let yi = y.dims();

        let dimensions: TVec<_> = xi
            .zip_longest(yi)
            .map(|r| match r {
                Both(a, b) => a.unify(&b),
                Left(d) if y.open => Ok(d),
                Right(d) if x.open => Ok(d),

                Left(_) | Right(_) => bail!(
                    "Impossible to unify closed shapes of different rank (found {:?} and {:?}).",
                    x,
                    y
                ),
            }).collect::<TractResult<_>>()
            .map_err(|e| format!("Unifying shapes {:?} and {:?}, {}", x, y, e))?;

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
        ShapeFact::open(tvec![])
    }
}

impl FromIterator<TDim> for ShapeFact {
    /// Converts an iterator over usize into a closed shape.
    fn from_iter<I: IntoIterator<Item = TDim>>(iter: I) -> ShapeFact {
        ShapeFact::closed(iter.into_iter().map(|d| GenericFact::Only(d)).collect())
    }
}

impl FromIterator<usize> for ShapeFact {
    /// Converts an iterator over usize into a closed shape.
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> ShapeFact {
        ShapeFact::closed(
            iter.into_iter()
                .map(|d| GenericFact::Only(d.to_dim()))
                .collect(),
        )
    }
}

impl<D: ToDim, I: IntoIterator<Item = D>> From<I> for ShapeFact {
    fn from(it: I) -> ShapeFact {
        ShapeFact::closed(
            it.into_iter()
                .map(|d| GenericFact::Only(d.to_dim()))
                .collect(),
        )
    }
}

/*
impl<'a> From<&'a [usize]> for ShapeFact {
    /// Converts an usize slice into a closed shape.
    fn from(slice: &'a [usize]) -> ShapeFact {
        slice.into_iter().map(|i| TDim::from(*i)).collect()
    }
}

impl From<Option<Vec<usize>>> for ShapeFact {
    /// Converts an vector of usize into a closed shape.
    fn from(shape: Option<Vec<usize>>) -> ShapeFact {
        shape
            .map(|s| ShapeFact::from(s))
            .unwrap_or(ShapeFact::default())
    }
}
*/

/*
impl From<Vec<usize>> for ShapeFact {
    /// Converts an vector of usize into a closed shape.
    fn from(shape: Vec<usize>) -> ShapeFact {
        shape.into_iter().map(|i| TDim::from(i)).collect()
    }
}

impl From<Vec<TDim>> for ShapeFact {
    /// Converts an vector of usize into a closed shape.
    fn from(shape: Vec<TDim>) -> ShapeFact {
        shape.into_iter().collect()
    }
}
*/

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

pub type DimFact = GenericFact<TDim>;

/// Partial information about a value.
pub type ValueFact = GenericFact<Tensor>;

pub type IntFact = GenericFact<i32>;

impl<T> Zero for GenericFact<T>
where
    T: Add<T, Output = T> + Zero + PartialEq + Copy + Clone + ::std::fmt::Debug,
{
    fn zero() -> GenericFact<T> {
        GenericFact::Only(T::zero())
    }
    fn is_zero(&self) -> bool {
        match self {
            GenericFact::Only(t) => t.is_zero(),
            _ => false,
        }
    }
}

impl<T> Neg for GenericFact<T>
where
    T: Neg<Output = T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
{
    type Output = GenericFact<T>;
    fn neg(self) -> GenericFact<T> {
        match self {
            GenericFact::Only(t) => GenericFact::Only(t.neg()),
            any => any,
        }
    }
}

impl<T> Add<GenericFact<T>> for GenericFact<T>
where
    T: Add<T, Output = T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
{
    type Output = GenericFact<T>;
    fn add(self, rhs: GenericFact<T>) -> Self::Output {
        match (self.concretize(), rhs.concretize()) {
            (Some(a), Some(b)) => GenericFact::Only(a + b),
            _ => GenericFact::Any,
        }
    }
}

impl<T, R> Add<R> for GenericFact<T>
where
    T: Add<R, Output = T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
    R: ::num::Num,
{
    type Output = GenericFact<T>;
    fn add(self, rhs: R) -> Self::Output {
        if let Some(a) = self.concretize() {
            GenericFact::Only(a + rhs)
        } else {
            GenericFact::Any
        }
    }
}

impl<T> Add<GenericFact<T>> for isize
where
    T: Add<isize, Output = T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
{
    type Output = GenericFact<T>;
    fn add(self, rhs: GenericFact<T>) -> Self::Output {
        rhs + self
    }
}

impl<T> Sub<GenericFact<T>> for GenericFact<T>
where
    T: Sub<T, Output = T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
{
    type Output = GenericFact<T>;
    fn sub(self, rhs: GenericFact<T>) -> Self::Output {
        match (self.concretize(), rhs.concretize()) {
            (Some(a), Some(b)) => GenericFact::Only(a - b),
            _ => GenericFact::Any,
        }
    }
}

impl<T, R> Mul<R> for GenericFact<T>
where
    T: Mul<R, Output = T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
    R: ::num::Num,
{
    type Output = GenericFact<T>;
    fn mul(self, rhs: R) -> Self::Output {
        if let Some(a) = self.concretize() {
            GenericFact::Only(a * rhs)
        } else {
            GenericFact::Any
        }
    }
}

impl<T> Mul<GenericFact<T>> for GenericFact<T>
where
    T: Mul<T, Output = T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
{
    type Output = GenericFact<T>;
    fn mul(self, rhs: GenericFact<T>) -> Self::Output {
        match (self.concretize(), rhs.concretize()) {
            (Some(a), Some(b)) => GenericFact::Only(a * b),
            _ => GenericFact::Any,
        }
    }
}

impl<T, R> Div<R> for GenericFact<T>
where
    T: Div<R, Output = T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
    R: ::num::Num,
{
    type Output = GenericFact<T>;
    fn div(self, rhs: R) -> Self::Output {
        if let Some(a) = self.concretize() {
            GenericFact::Only(a / rhs)
        } else {
            GenericFact::Any
        }
    }
}

impl<T> Div<GenericFact<T>> for GenericFact<T>
where
    T: Div<T, Output = T> + PartialEq + Copy + Clone + ::std::fmt::Debug,
{
    type Output = GenericFact<T>;
    fn div(self, rhs: GenericFact<T>) -> Self::Output {
        match (self.concretize(), rhs.concretize()) {
            (Some(a), Some(b)) => GenericFact::Only(a / b),
            _ => GenericFact::Any,
        }
    }
}
