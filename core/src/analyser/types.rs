use crate::TractResult;
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use num_traits::Zero;

use crate::internal::*;

/// Partial information about any value.
pub trait Factoid: fmt::Debug + Clone + PartialEq + Default {
    type Concrete: fmt::Debug;

    /// Tries to transform the fact into a concrete value.
    fn concretize(&self) -> Option<Self::Concrete>;

    /// Returns whether the value is fully determined.
    fn is_concrete(&self) -> bool {
        self.concretize().is_some()
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> TractResult<Self>;

    /// Tries to unify the fact with another fact of the same type and update
    /// self.
    ///
    /// Returns true if it actually changed something.
    fn unify_with(&mut self, other: &Self) -> TractResult<bool> {
        let new = self.unify(&other)?;
        let mut changed = false;
        if &new != self {
            changed = true;
            *self = new.clone();
        }
        Ok(changed)
    }

    /// Tries to unify the fact with another fact of the same type and update
    /// both of them.
    ///
    /// Returns true if it actually changed something.
    fn unify_with_mut(&mut self, other: &mut Self) -> TractResult<bool> {
        let new = self.unify(&other)?;
        let mut changed = false;
        if &new != self {
            changed = true;
            *self = new.clone();
        }
        if &new != other {
            changed = true;
            *other = new;
        }
        Ok(changed)
    }

    /// Tries to unify all facts in the list.
    ///
    ///
    /// Returns true if it actually changed something.
    fn unify_all(facts: &mut [&mut Self]) -> TractResult<bool> {
        let mut overall_changed = false;
        loop {
            let mut changed = false;
            for i in 0..facts.len() - 1 {
                for j in i + 1..facts.len() {
                    let (left, right) = facts.split_at_mut(j);
                    let c = left[i].unify_with(right[0])?;
                    changed = changed || c;
                    overall_changed = changed || c;
                }
            }
            if !changed {
                return Ok(overall_changed);
            }
        }
    }
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
pub struct InferenceFact {
    pub datum_type: TypeFact,
    pub shape: ShapeFactoid,
    pub value: ValueFact,
}

impl InferenceFact {
    /// Constructs the most general tensor fact possible.
    pub fn new() -> InferenceFact {
        InferenceFact::default()
    }

    pub fn any() -> InferenceFact {
        InferenceFact::default()
    }

    pub fn dt(dt: DatumType) -> InferenceFact {
        InferenceFact::default().with_datum_type(dt)
    }

    pub fn dt_shape<S: Into<ShapeFactoid>>(dt: DatumType, shape: S) -> InferenceFact {
        InferenceFact::dt(dt).with_shape(shape)
    }

    pub fn shape<S: Into<ShapeFactoid>>(shape: S) -> InferenceFact {
        InferenceFact::default().with_shape(shape)
    }

    pub fn with_datum_type(self, dt: DatumType) -> InferenceFact {
        InferenceFact { datum_type: dt.into(), ..self }
    }

    pub fn with_shape<S: Into<ShapeFactoid>>(self, shape: S) -> InferenceFact {
        InferenceFact { shape: shape.into(), ..self }
    }

    pub fn with_streaming_shape<S: IntoIterator<Item = Option<usize>>>(
        self,
        shape: S,
    ) -> InferenceFact {
        let shape: ShapeFactoid = shape
            .into_iter()
            .map(|d| d.map(|d| (d as isize).to_dim()).unwrap_or(TDim::s()))
            .collect();
        self.with_shape(shape)
    }

    pub fn stream_info(&self) -> TractResult<Option<StreamFact>> {
        self.shape.stream_info()
    }

    pub fn format_dt_shape(&self) -> String {
        if !self.shape.open && self.shape.dims.len() == 0 {
            format!(
                "{}",
                self.datum_type
                    .concretize()
                    .map(|dt| format!("{:?}", dt))
                    .unwrap_or("?".to_string())
            )
        } else {
            format!(
                "{:?}x{}",
                self.shape,
                self.datum_type
                    .concretize()
                    .map(|dt| format!("{:?}", dt))
                    .unwrap_or("?".to_string())
            )
        }
    }

    pub fn dt_shape_from_tensor(t: &Tensor) -> InferenceFact {
        InferenceFact::dt_shape(t.datum_type(), t.shape())
    }

    pub fn without_value(self) -> InferenceFact {
        InferenceFact { value: GenericFact::Any, ..self }
    }
}

impl Factoid for InferenceFact {
    type Concrete = Arc<Tensor>;

    /// Tries to transform the fact into a concrete value.
    fn concretize(&self) -> Option<Self::Concrete> {
        self.value.concretize()
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> TractResult<Self> {
        let tensor = InferenceFact {
            datum_type: self.datum_type.unify(&other.datum_type)?,
            shape: self.shape.unify(&other.shape)?,
            value: self.value.unify(&other.value)?,
        };

        trace!("Unifying {:?} with {:?} into {:?}.", self, other, tensor);

        Ok(tensor)
    }
}

impl<V: Into<Arc<Tensor>>> From<V> for InferenceFact {
    fn from(v: V) -> InferenceFact {
        let v: Arc<Tensor> = v.into();
        InferenceFact {
            datum_type: GenericFact::Only(v.datum_type()),
            shape: ShapeFactoid::from(v.shape()),
            value: GenericFact::Only(v),
        }
    }
}

impl fmt::Debug for InferenceFact {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        if let Some(t) = self.value.concretize() {
            write!(formatter, "{:?}", t)
        } else {
            write!(formatter, "{}", self.format_dt_shape())
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

impl<T: fmt::Debug + Clone + PartialEq> Factoid for GenericFact<T> {
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
/// A basic example of a shape fact is `shapefactoid![1, 2]`, which corresponds to
/// the shape `[1, 2]` in Arc<Tensor>. We can use `_` in facts to denote unknown
/// dimensions (e.g. `shapefactoid![1, 2, _]` corresponds to any shape `[1, 2, k]`
/// with `k` a non-negative integer). We can also use `..` at the end of a fact
/// to only specify its first dimensions, so `shapefactoid![1, 2; ..]` matches any
/// shape that starts with `[1, 2]` (e.g. `[1, 2, i]` or `[1, 2, i, j]`), while
/// `shapefactoid![..]` matches any shape.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Clone, PartialEq)]
pub struct ShapeFactoid {
    open: bool,
    dims: TVec<GenericFact<i32>>,
    stream: Option<StreamFact>,
}

impl ShapeFactoid {
    /// Constructs an open shape fact.
    pub fn open(dims: TVec<DimFact>) -> ShapeFactoid {
        if let Some((ix, d)) = dims
            .iter()
            .enumerate()
            .find(|(_ix, d)| d.concretize().map(|d| d.is_stream()).unwrap_or(false))
        {
            let stream = Some(StreamFact { axis: ix, len: d.concretize().unwrap() });
            ShapeFactoid {
                open: true,
                dims: dims
                    .iter()
                    .map(|d| match d {
                        GenericFact::Only(d) if d.is_stream() => GenericFact::Only(-1),
                        GenericFact::Only(d) => GenericFact::Only(d.to_integer().unwrap()),
                        GenericFact::Any => GenericFact::Any,
                    })
                    .collect(),
                stream,
            }
        } else {
            ShapeFactoid {
                open: true,
                dims: dims
                    .iter()
                    .map(|d| match d {
                        GenericFact::Only(d) => GenericFact::Only(d.to_integer().unwrap()),
                        GenericFact::Any => GenericFact::Any,
                    })
                    .collect(),
                stream: None,
            }
        }
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Constructs a closed shape fact.
    pub fn closed(dims: TVec<DimFact>) -> ShapeFactoid {
        ShapeFactoid { open: false, ..Self::open(dims) }
    }

    pub fn rank(&self) -> IntFact {
        if self.open { GenericFact::Any } else { GenericFact::Only(self.dims.len() as i32) }.into()
    }

    pub fn ensure_rank_at_least(&mut self, n: usize) -> bool {
        let mut changed = false;
        while self.dims.len() <= n {
            self.dims.push(GenericFact::Any);
            changed = true;
        }
        changed
    }

    pub fn dim(&self, i: usize) -> Option<DimFact> {
        self.dims().nth(i)
    }

    pub fn set_dim(&mut self, i: usize, d: TDim) -> bool {
        let fact = GenericFact::Only(d.clone());
        if self.dim(i).as_ref() == Some(&fact) {
            return false;
        }
        match d.to_integer() {
            Ok(n) => self.dims[i] = GenericFact::Only(n),
            Err(_) => {
                self.dims[i] = GenericFact::Only(-1);
                self.stream = Some(StreamFact { axis: i, len: d })
            }
        }
        return true;
    }

    pub fn dims(&self) -> impl Iterator<Item = DimFact> {
        let stream = self.stream.clone();
        self.dims.clone().into_iter().map(move |d| match d {
            GenericFact::Only(-1) => {
                assert!(stream.is_some(), "-1 dim found with no stream. This is a tract bug.");
                GenericFact::Only(stream.as_ref().unwrap().len.clone())
            }
            GenericFact::Only(d) => GenericFact::Only(d.to_dim()),
            GenericFact::Any => GenericFact::Any,
        })
    }

    pub fn stream_info(&self) -> TractResult<Option<StreamFact>> {
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
            .map(|(axis, len)| StreamFact { axis, len }))
    }

    pub fn as_concrete_finite(&self) -> TractResult<Option<TVec<usize>>> {
        if !self.is_concrete() || self.stream_info()?.is_some() {
            return Ok(None);
        }
        Ok(Some(self.dims.iter().map(|i| i.concretize().unwrap() as usize).collect()))
    }
}

impl Factoid for ShapeFactoid {
    type Concrete = TVec<TDim>;

    /// Tries to transform the fact into a `Vec<usize>`, or returns `None`.
    fn concretize(self: &ShapeFactoid) -> Option<TVec<TDim>> {
        if self.open {
            return None;
        }

        let dims: TVec<_> = self.dims().filter_map(|d| d.concretize()).collect();

        if dims.len() < self.dims.len() {
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
                Left(ref d) if y.open => Ok(d.clone()),
                Right(ref d) if x.open => Ok(d.clone()),

                Left(_) | Right(_) => bail!(
                    "Impossible to unify closed shapes of different rank (found {:?} and {:?}).",
                    x,
                    y
                ),
            })
            .collect::<TractResult<_>>()
            .map_err(|e| format!("Unifying shapes {:?} and {:?}, {}", x, y, e))?;

        if x.open && y.open {
            Ok(ShapeFactoid::open(dimensions))
        } else {
            Ok(ShapeFactoid::closed(dimensions))
        }
    }
}

impl Default for ShapeFactoid {
    /// Returns the most general shape fact possible.
    fn default() -> ShapeFactoid {
        ShapeFactoid::open(tvec![])
    }
}

impl FromIterator<TDim> for ShapeFactoid {
    /// Converts an iterator over usize into a closed shape.
    fn from_iter<I: IntoIterator<Item = TDim>>(iter: I) -> ShapeFactoid {
        ShapeFactoid::closed(iter.into_iter().map(|d| GenericFact::Only(d)).collect())
    }
}

impl FromIterator<usize> for ShapeFactoid {
    /// Converts an iterator over usize into a closed shape.
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> ShapeFactoid {
        ShapeFactoid::closed(iter.into_iter().map(|d| GenericFact::Only(d.to_dim())).collect())
    }
}

impl<D: ToDim, I: IntoIterator<Item = D>> From<I> for ShapeFactoid {
    fn from(it: I) -> ShapeFactoid {
        ShapeFactoid::closed(it.into_iter().map(|d| GenericFact::Only(d.to_dim())).collect())
    }
}

impl fmt::Debug for ShapeFactoid {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        for (ix, d) in self.dims.iter().enumerate() {
            if ix != 0 {
                write!(formatter, "x")?
            }
            if let Some(ref stream) = self.stream {
                if stream.axis == ix {
                    write!(formatter, "{:?}", stream.len)?;
                } else {
                    write!(formatter, "{:?}", d)?;
                }
            } else {
                write!(formatter, "{:?}", d)?;
            }
        }
        if self.open {
            if self.dims.len() == 0 {
                write!(formatter, "..")?;
            } else {
                write!(formatter, "x..")?;
            }
        }
        Ok(())
    }
}

pub type DimFact = GenericFact<TDim>;

/// Partial information about a value.
pub type ValueFact = GenericFact<Arc<Tensor>>;

pub type IntFact = GenericFact<i32>;

impl<T> Zero for GenericFact<T>
where
    T: Add<T, Output = T> + Zero + PartialEq + Clone + ::std::fmt::Debug,
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
    T: Neg<Output = T> + PartialEq + Clone + ::std::fmt::Debug,
{
    type Output = GenericFact<T>;
    fn neg(self) -> GenericFact<T> {
        match self {
            GenericFact::Only(t) => GenericFact::Only(t.neg()),
            any => any,
        }
    }
}

impl<T, I> Add<I> for GenericFact<T>
where
    T: Add<T, Output = T> + PartialEq + Clone + ::std::fmt::Debug,
    I: Into<GenericFact<T>>,
{
    type Output = GenericFact<T>;
    fn add(self, rhs: I) -> Self::Output {
        match (self.concretize(), rhs.into().concretize()) {
            (Some(a), Some(b)) => GenericFact::Only(a + b),
            _ => GenericFact::Any,
        }
    }
}

impl<T> Sub<GenericFact<T>> for GenericFact<T>
where
    T: Sub<T, Output = T> + PartialEq + Clone + ::std::fmt::Debug,
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
    T: Mul<R, Output = T> + PartialEq + Clone + ::std::fmt::Debug,
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

impl<T, R> Div<R> for GenericFact<T>
where
    T: Div<R, Output = T> + PartialEq + Clone + ::std::fmt::Debug,
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

impl<T, R> Rem<R> for GenericFact<T>
where
    T: Rem<R, Output = T> + PartialEq + Clone + ::std::fmt::Debug,
{
    type Output = GenericFact<T>;
    fn rem(self, rhs: R) -> Self::Output {
        if let Some(a) = self.concretize() {
            GenericFact::Only(a % rhs)
        } else {
            GenericFact::Any
        }
    }
}
