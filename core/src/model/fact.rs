//! Partial and complete tensor types representations.
use crate::internal::*;
use downcast_rs::Downcast;
use std::convert::{TryFrom, TryInto};
use std::fmt;

/// Type information about a tensor: shape, and element type, in various state
/// of determination.
pub trait Fact: std::fmt::Debug + Downcast + dyn_clone::DynClone + Send + Sync + 'static {
    fn to_typed_fact(&self) -> TractResult<TypedFact>;

    fn matches(&self, t: &Tensor) -> TractResult<bool> {
        self.to_typed_fact()?.matches(t)
    }

    fn same_as(&self, _other: &dyn Fact) -> bool;
}

impl_downcast!(Fact);
dyn_clone::clone_trait_object!(Fact);

/// Fully determined dimension of a tensor.
///
/// Tensors in tract can have one streaming dimension. TDim generalize the
/// regular tensor dimensions (usize) to arithmetic expressions of `S`, the
/// (sometimes hypothetical) tensor length on the streaming axis.
#[derive(Clone, PartialEq, Hash)]
pub struct ShapeFact(TVec<TDim>);

impl std::ops::Deref for ShapeFact {
    type Target = [TDim];
    fn deref(&self) -> &[TDim] {
        &self.0
    }
}

impl std::ops::DerefMut for ShapeFact {
    fn deref_mut(&mut self) -> &mut [TDim] {
        &mut self.0
    }
}

impl ShapeFact {
    /// Rank of the tensor.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn insert_axis(&mut self, axis: usize) -> TractResult<()> {
        self.0.insert(axis, 1.into());
        Ok(())
    }

    pub fn remove_axis(&mut self, axis: usize) -> TractResult<()> {
        self.0.remove(axis);
        Ok(())
    }

    /// Shape of the tensor, unless it is streaming.
    pub fn as_finite(&self) -> Option<TVec<usize>> {
        self.0.iter().map(|d| d.to_usize()).collect::<TractResult<TVec<_>>>().ok()
    }

    /// Iterator over dimension of the shape.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = TDim> + 'a {
        self.0.iter().cloned()
    }

    /// Convert the shape to an array of extended dimensions.
    pub fn to_tvec(&self) -> TVec<TDim> {
        self.iter().collect::<TVec<TDim>>()
    }

    pub fn from_dims<T: AsRef<[TDim]> + std::fmt::Debug>(it: T) -> TractResult<ShapeFact> {
        Ok(ShapeFact(it.as_ref().iter().cloned().collect()))
    }
}

impl TryFrom<()> for ShapeFact {
    type Error = TractError;
    fn try_from(_it: ()) -> TractResult<ShapeFact> {
        ShapeFact::from_dims([0.to_dim(); 0].as_ref())
    }
}

impl TryFrom<&[TDim]> for ShapeFact {
    type Error = TractError;
    fn try_from(it: &[TDim]) -> TractResult<ShapeFact> {
        ShapeFact::from_dims(it)
    }
}

impl TryFrom<&[usize]> for ShapeFact {
    type Error = TractError;
    fn try_from(it: &[usize]) -> TractResult<ShapeFact> {
        Ok(ShapeFact(it.iter().map(|d| d.to_dim()).collect()))
    }
}

impl fmt::Debug for ShapeFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;
        write!(fmt, "{}", self.iter().join("x"))
    }
}

impl<T: AsRef<[usize]>> std::cmp::PartialEq<T> for ShapeFact {
    fn eq(&self, other: &T) -> bool {
        let other = other.as_ref();
        other.len() == self.rank() && other.iter().zip(self.0.iter()).all(|(i, d)| &i.to_dim() == d)
    }
}

/// Fully determined tensor information for TypedModel.
#[derive(Clone, PartialEq, Hash)]
pub struct TypedFact {
    /// tensor element type
    pub datum_type: DatumType,
    /// tensor shape
    pub shape: ShapeFact,
    /// optional constant value
    pub konst: Option<Arc<Tensor>>,
}

tract_data::impl_dyn_hash!(TypedFact);

impl TypedFact {
    pub fn shape<T, S, E>(shape: S) -> TractResult<TypedFact>
    where
        T: Datum,
        S: TryInto<ShapeFact, Error = E>,
        TractError: From<E>,
    {
        Self::dt_shape(T::datum_type(), shape)
    }

    pub fn dt_shape<S, E>(datum_type: DatumType, shape: S) -> TractResult<TypedFact>
    where
        S: TryInto<ShapeFact, Error = E>,
        TractError: From<E>,
    {
        Ok(TypedFact { datum_type, shape: shape.try_into()?, konst: None })
    }

    pub fn rank(&self) -> usize {
        if cfg!(debug_assertions) {
            self.consistent().unwrap();
        }
        self.shape.rank()
    }

    fn format_dt_shape_nocheck(&self) -> String {
        if self.shape.rank() > 0 {
            format!("{:?}x{:?}", self.shape, self.datum_type)
        } else {
            format!("{:?}", self.datum_type)
        }
    }

    pub fn format_dt_shape(&self) -> String {
        if cfg!(debug_assertions) {
            self.consistent().unwrap()
        }
        self.format_dt_shape_nocheck()
    }

    pub fn consistent(&self) -> TractResult<()> {
        if let Some(k) = &self.konst {
            if !self.matches(k.as_ref())? {
                bail!("fact says {}, constant is {:?}", self.format_dt_shape_nocheck(), k);
            }
        }
        Ok(())
    }

    pub fn without_value(&self) -> Self {
        Self::dt_shape(self.datum_type, &*self.shape).unwrap()
    }
}

impl Fact for TypedFact {
    fn to_typed_fact(&self) -> TractResult<TypedFact> {
        if cfg!(debug_assertions) {
            self.consistent()?
        }
        Ok(self.clone())
    }

    fn matches(&self, t: &Tensor) -> TractResult<bool> {
        Ok(self.datum_type == t.datum_type() && self.shape == t.shape())
    }

    fn same_as(&self, other: &dyn Fact) -> bool {
        if cfg!(debug_assertions) {
            self.consistent().unwrap()
        }
        if let Some(other) = other.downcast_ref::<Self>() {
            if cfg!(debug_assertions) {
                other.consistent().unwrap()
            }
            self == other
        } else {
            false
        }
    }
}

impl From<Tensor> for TypedFact {
    fn from(t: Tensor) -> TypedFact {
        TypedFact::from(t.into_arc_tensor())
    }
}

impl<'t> From<&'t Tensor> for TypedFact {
    fn from(t: &'t Tensor) -> TypedFact {
        TypedFact::from(t.clone())
    }
}

impl From<Arc<Tensor>> for TypedFact {
    fn from(t: Arc<Tensor>) -> TypedFact {
        TypedFact {
            datum_type: t.datum_type(),
            shape: ShapeFact(t.shape().iter().map(TDim::from).collect()),
            konst: Some(t),
        }
    }
}

impl<'a> From<&'a TypedFact> for TypedFact {
    fn from(fact: &TypedFact) -> TypedFact {
        fact.clone()
    }
}

impl fmt::Debug for TypedFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self.konst {
            Some(ref k) => write!(fmt, "{:?}", k),
            None if self.rank() > 0 => write!(fmt, "{:?}x{:?}", self.shape, self.datum_type),
            None => write!(fmt, "{:?}", self.datum_type),
        }
    }
}
