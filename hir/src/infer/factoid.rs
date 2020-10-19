use std::fmt;
use std::iter::FromIterator;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use tract_num_traits::Zero;

use crate::internal::*;

/// Partial information about any value.
pub trait Factoid: fmt::Debug + Clone + PartialEq + Default + Hash {
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

/// Partial information about a value of type T.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Clone, PartialEq, Hash)]
pub enum GenericFactoid<T: fmt::Debug + Clone + PartialEq + Hash> {
    Only(T),
    Any,
}

impl<T: Copy + Clone + fmt::Debug + PartialEq + Hash> Copy for GenericFactoid<T> {}

impl<T: fmt::Debug + Clone + PartialEq + Hash> Factoid for GenericFactoid<T> {
    type Concrete = T;

    /// Tries to transform the fact into a concrete value.
    fn concretize(&self) -> Option<T> {
        match self {
            GenericFactoid::Any => None,
            GenericFactoid::Only(m) => Some(m.clone()),
        }
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> TractResult<Self> {
        let fact = match (self, other) {
            (_, GenericFactoid::Any) => self.clone(),
            (GenericFactoid::Any, _) => other.clone(),
            _ if self == other => self.clone(),
            _ => bail!("Impossible to unify {:?} with {:?}.", self, other),
        };

        Ok(fact)
    }
}

impl<T: fmt::Debug + Clone + PartialEq + Hash> Default for GenericFactoid<T> {
    fn default() -> Self {
        GenericFactoid::Any
    }
}

impl<T: fmt::Debug + Clone + PartialEq + Hash> From<T> for GenericFactoid<T> {
    fn from(t: T) -> Self {
        GenericFactoid::Only(t)
    }
}

impl<T: fmt::Display + fmt::Debug + Clone + PartialEq + Hash> fmt::Display for GenericFactoid<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GenericFactoid::Any => write!(formatter, "?"),
            GenericFactoid::Only(u) => write!(formatter, "{}", u),
        }
    }
}

impl<T: fmt::Debug + Clone + PartialEq + Hash> fmt::Debug for GenericFactoid<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GenericFactoid::Any => write!(formatter, "?"),
            GenericFactoid::Only(u) => write!(formatter, "{:?}", u),
        }
    }
}

/// Partial information about a type.
pub type TypeFactoid = GenericFactoid<DatumType>;

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
#[derive(Clone, PartialEq, Hash)]
pub struct ShapeFactoid {
    pub(super) open: bool,
    pub(super) dims: TVec<GenericFactoid<TDim>>,
}

impl ShapeFactoid {
    /// Constructs an open shape fact.
    pub fn open(dims: TVec<DimFact>) -> ShapeFactoid {
        ShapeFactoid { open: true, dims }
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Constructs a closed shape fact.
    pub fn closed(dims: TVec<DimFact>) -> ShapeFactoid {
        ShapeFactoid { open: false, dims }
    }

    pub fn rank(&self) -> IntFactoid {
        if self.open { GenericFactoid::Any } else { GenericFactoid::Only(self.dims.len() as i64) }
            .into()
    }

    pub fn ensure_rank_at_least(&mut self, n: usize) -> bool {
        let mut changed = false;
        while self.dims.len() <= n {
            self.dims.push(GenericFactoid::Any);
            changed = true;
        }
        changed
    }

    pub fn dim(&self, i: usize) -> Option<DimFact> {
        self.dims().nth(i).cloned()
    }

    pub fn set_dim(&mut self, i: usize, d: TDim) -> bool {
        let fact = GenericFactoid::Only(d.clone());
        if self.dim(i).as_ref() == Some(&fact) {
            return false;
        }
        self.dims[i] = GenericFactoid::Only(d);
        return true;
    }

    pub fn dims(&self) -> impl Iterator<Item = &DimFact> {
        self.dims.iter()
    }

    pub fn as_concrete_finite(&self) -> TractResult<Option<TVec<usize>>> {
        if self.open {
            return Ok(None);
        }
        Ok(self.dims.iter().map(|d| d.concretize().and_then(|d| d.to_usize().ok())).collect())
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

        use tract_itertools::EitherOrBoth::{Both, Left, Right};
        use tract_itertools::Itertools;

        let xi = x.dims();
        let yi = y.dims();

        let dimensions: TVec<_> = xi
            .zip_longest(yi)
            .map(|r| match r {
                Both(a, b) => a.unify(&b),
                Left(d) if y.open => Ok(d.clone()),
                Right(d) if x.open => Ok(d.clone()),

                Left(_) | Right(_) => bail!(
                    "Impossible to unify closed shapes of different rank (found {:?} and {:?}).",
                    x,
                    y
                ),
            })
            .collect::<TractResult<_>>()
            .with_context(|| format!("Unifying shapes {:?} and {:?}", x, y))?;

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
        ShapeFactoid::closed(iter.into_iter().map(|d| GenericFactoid::Only(d)).collect())
    }
}

impl FromIterator<usize> for ShapeFactoid {
    /// Converts an iterator over usize into a closed shape.
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> ShapeFactoid {
        ShapeFactoid::closed(iter.into_iter().map(|d| GenericFactoid::Only(d.to_dim())).collect())
    }
}

impl<D: ToDim, I: IntoIterator<Item = D>> From<I> for ShapeFactoid {
    fn from(it: I) -> ShapeFactoid {
        ShapeFactoid::closed(it.into_iter().map(|d| GenericFactoid::Only(d.to_dim())).collect())
    }
}

impl fmt::Debug for ShapeFactoid {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        for (ix, d) in self.dims.iter().enumerate() {
            if ix != 0 {
                write!(formatter, "x")?
            }
            write!(formatter, "{}", d)?;
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

pub type DimFact = GenericFactoid<TDim>;

/// Partial information about a value.
pub type ValueFact = GenericFactoid<Arc<Tensor>>;

pub type IntFactoid = GenericFactoid<i64>;

impl<T> Zero for GenericFactoid<T>
where
    T: Add<T, Output = T> + Zero + PartialEq + Clone + ::std::fmt::Debug + Hash,
{
    fn zero() -> GenericFactoid<T> {
        GenericFactoid::Only(T::zero())
    }
    fn is_zero(&self) -> bool {
        match self {
            GenericFactoid::Only(t) => t.is_zero(),
            _ => false,
        }
    }
}

impl<T> Neg for GenericFactoid<T>
where
    T: Neg<Output = T> + PartialEq + Clone + ::std::fmt::Debug + Hash,
{
    type Output = GenericFactoid<T>;
    fn neg(self) -> GenericFactoid<T> {
        match self {
            GenericFactoid::Only(t) => GenericFactoid::Only(t.neg()),
            any => any,
        }
    }
}

impl<T, I> Add<I> for GenericFactoid<T>
where
    T: Add<T, Output = T> + PartialEq + Clone + ::std::fmt::Debug + Hash,
    I: Into<GenericFactoid<T>>,
{
    type Output = GenericFactoid<T>;
    fn add(self, rhs: I) -> Self::Output {
        match (self.concretize(), rhs.into().concretize()) {
            (Some(a), Some(b)) => GenericFactoid::Only(a + b),
            _ => GenericFactoid::Any,
        }
    }
}

impl<T> Sub<GenericFactoid<T>> for GenericFactoid<T>
where
    T: Sub<T, Output = T> + PartialEq + Clone + ::std::fmt::Debug + Hash,
{
    type Output = GenericFactoid<T>;
    fn sub(self, rhs: GenericFactoid<T>) -> Self::Output {
        match (self.concretize(), rhs.concretize()) {
            (Some(a), Some(b)) => GenericFactoid::Only(a - b),
            _ => GenericFactoid::Any,
        }
    }
}

impl<T, R> Mul<R> for GenericFactoid<T>
where
    T: Mul<R, Output = T> + PartialEq + Clone + ::std::fmt::Debug + Hash,
{
    type Output = GenericFactoid<T>;
    fn mul(self, rhs: R) -> Self::Output {
        if let Some(a) = self.concretize() {
            GenericFactoid::Only(a * rhs)
        } else {
            GenericFactoid::Any
        }
    }
}

impl<T, R> Div<R> for GenericFactoid<T>
where
    T: Div<R, Output = T> + PartialEq + Clone + ::std::fmt::Debug + Hash,
{
    type Output = GenericFactoid<T>;
    fn div(self, rhs: R) -> Self::Output {
        if let Some(a) = self.concretize() {
            GenericFactoid::Only(a / rhs)
        } else {
            GenericFactoid::Any
        }
    }
}

impl<T, R> Rem<R> for GenericFactoid<T>
where
    T: Rem<R, Output = T> + PartialEq + Clone + ::std::fmt::Debug + Hash,
{
    type Output = GenericFactoid<T>;
    fn rem(self, rhs: R) -> Self::Output {
        if let Some(a) = self.concretize() {
            GenericFactoid::Only(a % rhs)
        } else {
            GenericFactoid::Any
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GenericFactoid::*;
    use super::*;

    #[test]
    fn unify_same_datum_type() {
        let dt = TypeFactoid::Only(DatumType::F32);
        assert_eq!(dt.unify(&dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_datum_types_only() {
        let dt1 = TypeFactoid::Only(DatumType::F32);
        let dt2 = TypeFactoid::Only(DatumType::F64);
        assert!(dt1.unify(&dt2).is_err());
    }

    #[test]
    fn unify_different_datum_types_any_left() {
        let dt = TypeFactoid::Only(DatumType::F32);
        assert_eq!(TypeFactoid::Any.unify(&dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_datum_types_any_right() {
        let dt = TypeFactoid::Only(DatumType::F32);
        assert_eq!(dt.unify(&TypeFactoid::Any).unwrap(), dt);
    }

    #[test]
    fn unify_same_shape_1() {
        let s = ShapeFactoid::closed(tvec![]);
        assert_eq!(s.unify(&s).unwrap(), s);
    }

    #[test]
    fn unify_same_shape_2() {
        let s = ShapeFactoid::closed(tvec![Any]);
        assert_eq!(s.unify(&s).unwrap(), s);
    }

    #[test]
    fn unify_same_shape_3() {
        let s = ShapeFactoid::closed(tvec![Only(1.into()), Only(2.into())]);
        assert_eq!(s.unify(&s).unwrap(), s);
    }

    #[test]
    fn unify_different_shapes_1() {
        let s1 = ShapeFactoid::closed(tvec![Only(1.into()), Only(2.into())]);
        let s2 = ShapeFactoid::closed(tvec![Only(1.into())]);
        assert!(s1.unify(&s2).is_err());
    }

    #[test]
    fn unify_different_shapes_2() {
        let s1 = ShapeFactoid::closed(tvec![Only(1.into()), Only(2.into())]);
        let s2 = ShapeFactoid::closed(tvec![Any]);
        assert!(s1.unify(&s2).is_err());
    }

    #[test]
    fn unify_different_shapes_3() {
        let s1 = ShapeFactoid::open(tvec![Only(1.into()), Only(2.into())]);
        let s2 = ShapeFactoid::closed(tvec![Any]);
        assert!(s1.unify(&s2).is_err());
    }

    #[test]
    fn unify_different_shapes_4() {
        let s1 = ShapeFactoid::closed(tvec![Any]);
        let s2 = ShapeFactoid::closed(tvec![Any]);
        let sr = ShapeFactoid::closed(tvec![Any]);
        assert_eq!(s1.unify(&s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_5() {
        let s1 = ShapeFactoid::closed(tvec![Any]);
        let s2 = ShapeFactoid::closed(tvec![Only(1.into())]);
        let sr = ShapeFactoid::closed(tvec![Only(1.into())]);
        assert_eq!(s1.unify(&s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_6() {
        let s1 = ShapeFactoid::open(tvec![]);
        let s2 = ShapeFactoid::closed(tvec![Only(1.into())]);
        let sr = ShapeFactoid::closed(tvec![Only(1.into())]);
        assert_eq!(s1.unify(&s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_7() {
        let s1 = ShapeFactoid::open(tvec![Any, Only(2.into())]);
        let s2 = ShapeFactoid::closed(tvec![Only(1.into()), Any, Any]);
        let sr = ShapeFactoid::closed(tvec![Only(1.into()), Only(2.into()), Any]);
        assert_eq!(s1.unify(&s2).unwrap(), sr);
    }

    #[test]
    fn unify_same_value() {
        let t = ValueFact::Only(rctensor0(12f32));
        assert_eq!(t.unify(&t).unwrap(), t);
    }

    #[test]
    fn unify_different_values_only() {
        let t1 = ValueFact::Only(rctensor1(&[12f32]));
        let t2 = ValueFact::Only(rctensor1(&[12f32, 42.0]));
        assert!(t1.unify(&t2).is_err());
    }

    #[test]
    fn unify_different_values_any_left() {
        let t1 = ValueFact::Only(rctensor1(&[12f32]));
        assert_eq!(ValueFact::Any.unify(&t1).unwrap(), t1);
    }

    #[test]
    fn unify_different_values_any_right() {
        let t1 = ValueFact::Only(rctensor1(&[12f32]));
        assert_eq!(t1.unify(&ValueFact::Any).unwrap(), t1);
    }
}
