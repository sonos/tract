//! Partial and complete tensor types representations.
use crate::internal::*;
use downcast_rs::Downcast;
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
pub struct ShapeFact {
    dims: TVec<TDim>,
    concrete: Option<TVec<usize>>,
}

impl std::ops::Deref for ShapeFact {
    type Target = [TDim];
    fn deref(&self) -> &[TDim] {
        &self.dims
    }
}

impl ShapeFact {
    /// Rank of the tensor.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn insert_axis(&mut self, axis: usize) -> TractResult<()> {
        self.dims.insert(axis, 1.into());
        if let Some(concrete) = &mut self.concrete {
            concrete.insert(axis, 1);
        }
        Ok(())
    }

    pub fn remove_axis(&mut self, axis: usize) -> TractResult<()> {
        self.dims.remove(axis);
        if let Some(concrete) = &mut self.concrete {
            concrete.remove(axis);
        }
        Ok(())
    }

    pub fn set(&mut self, ix: usize, dim: TDim) {
        self.dims[ix] = dim;
        self.compute_concrete();
    }

    /// Shape of the tensor, unless it has symbolic dimensions.
    pub fn as_concrete(&self) -> Option<&[usize]> {
        self.concrete.as_deref()
    }

    /// Do we have a symbol-less value ?
    pub fn is_concrete(&self) -> bool {
        self.concrete.is_some()
    }

    /// Iterator over dimension of the shape.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = TDim> + 'a {
        self.dims.iter().cloned()
    }

    /// Convert the shape to an array of extended dimensions.
    pub fn to_tvec(&self) -> TVec<TDim> {
        self.dims.clone()
    }

    pub fn from_dims<D: ToDim, T: IntoIterator<Item = D>>(it: T) -> ShapeFact {
        let mut fact =
            ShapeFact { dims: it.into_iter().map(|d| d.to_dim()).collect(), concrete: None };
        fact.compute_concrete();
        fact
    }

    fn compute_concrete(&mut self) {
        self.concrete =
            self.dims.iter().map(|d| d.to_usize()).collect::<TractResult<TVec<_>>>().ok()
    }

    pub fn eval(&self, values: &SymbolValues) -> TractResult<Cow<TVec<usize>>> {
        if let Some(c) = &self.concrete {
            Ok(Cow::Borrowed(c))
        } else {
            Ok(Cow::Owned(
                self.iter()
                    .map(|d| d.eval(&values).to_usize())
                    .collect::<TractResult<TVec<_>>>()?,
            ))
        }
    }

    pub fn eval_to_isize(&self, values: &SymbolValues) -> TractResult<TVec<isize>> {
        self.iter().map(|d| d.eval(&values).to_isize()).collect::<TractResult<_>>()
    }
}

impl<D: ToDim, T: IntoIterator<Item = D>> From<T> for ShapeFact {
    fn from(it: T) -> ShapeFact {
        ShapeFact::from_dims(it)
    }
}

impl fmt::Debug for ShapeFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;
        write!(fmt, "{}", self.iter().join(","))
    }
}

impl<T: AsRef<[usize]>> std::cmp::PartialEq<T> for ShapeFact {
    fn eq(&self, other: &T) -> bool {
        let other = other.as_ref();
        other.len() == self.rank() && other.iter().zip(self.iter()).all(|(i, d)| i.to_dim() == d)
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

impl_dyn_hash!(TypedFact);

impl TypedFact {
    pub fn shape<T, S>(shape: S) -> TypedFact
    where
        T: Datum,
        S: Into<ShapeFact>,
    {
        Self::dt_shape(T::datum_type(), shape)
    }

    pub fn dt_shape<S>(datum_type: DatumType, shape: S) -> TypedFact
    where
        S: Into<ShapeFact>,
    {
        TypedFact { datum_type, shape: shape.into(), konst: None }
    }

    pub fn rank(&self) -> usize {
        if cfg!(debug_assertions) {
            self.consistent().unwrap();
        }
        self.shape.rank()
    }

    fn format_dt_shape_nocheck(&self) -> String {
        if self.shape.rank() > 0 {
            format!("{:?},{:?}", self.shape, self.datum_type)
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
        Self::dt_shape(self.datum_type, &*self.shape)
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
            shape: ShapeFact::from_dims(t.shape().iter().map(TDim::from)),
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
            None if self.rank() > 0 => write!(fmt, "{:?},{:?}", self.shape, self.datum_type),
            None => write!(fmt, "{:?}", self.datum_type),
        }
    }
}
