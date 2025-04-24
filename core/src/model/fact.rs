//! Partial and complete tensor types representations.
use crate::internal::*;
use downcast_rs::Downcast;
use std::fmt;
use tract_linalg::block_quant::{BlockQuantFact, BlockQuantValue};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ShapeFact {
    dims: TVec<TDim>,
    concrete: Option<TVec<usize>>,
}

impl ShapeFact {
    #[inline]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    fn compute_concrete(&mut self) {
        assert!(self.dims.iter().all(|d| d.to_isize().map(|d| d >= 0).unwrap_or(true)));
        self.concrete =
            self.dims.iter().map(|d| d.to_usize()).collect::<TractResult<TVec<_>>>().ok()
    }

    /// Shape of the tensor, unless it has symbolic dimensions.
    #[inline]
    pub fn as_concrete(&self) -> Option<&[usize]> {
        self.concrete.as_deref()
    }

    /// Do we have a symbol-less value ?
    #[inline]
    pub fn is_concrete(&self) -> bool {
        self.concrete.is_some()
    }

    /// Convert the shape to an array of extended dimensions.
    #[inline]
    pub fn to_tvec(&self) -> TVec<TDim> {
        self.dims.clone()
    }

    /// Compute the volume of the tensor.
    #[inline]
    pub fn volume(&self) -> TDim {
        self.dims.iter().product()
    }

    #[inline]
    pub fn eval(&self, values: &SymbolValues) -> TractResult<Cow<ShapeFact>> {
        if self.is_concrete() {
            Ok(Cow::Borrowed(self))
        } else {
            Ok(Cow::Owned(self.iter().map(|d| d.eval(values)).collect::<ShapeFact>()))
        }
    }

    #[inline]
    pub fn eval_to_usize(&self, values: &SymbolValues) -> TractResult<Cow<TVec<usize>>> {
        if let Some(c) = &self.concrete {
            Ok(Cow::Borrowed(c))
        } else {
            Ok(Cow::Owned(
                self.iter()
                    .map(|d| d.eval_to_i64(values).map(|d| d as usize))
                    .collect::<TractResult<TVec<_>>>()?,
            ))
        }
    }

    #[inline]
    pub fn eval_to_isize(&self, values: &SymbolValues) -> TractResult<Cow<TVec<isize>>> {
        if let Some(c) = &self.concrete {
            #[allow(unknown_lints, clippy::missing_transmute_annotations)]
            // TVec<usize> -> TVec<isize>
            Ok(unsafe { std::mem::transmute(Cow::Borrowed(c)) })
        } else {
            Ok(Cow::Owned(
                self.iter()
                    .map(|d| d.eval_to_i64(values).map(|d| d as isize))
                    .collect::<TractResult<TVec<_>>>()?,
            ))
        }
    }

    pub fn from_dims<D: ToDim, T: IntoIterator<Item = D>>(it: T) -> ShapeFact {
        let mut dims =
            ShapeFact { dims: it.into_iter().map(|d| d.to_dim()).collect(), concrete: None };
        dims.compute_concrete();
        dims
    }

    pub fn dims(&self) -> &[TDim] {
        self.dims.as_slice()
    }

    pub fn set(&mut self, ix: usize, dim: TDim) {
        self.dims[ix] = dim;
        self.compute_concrete();
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
        } else {
            self.compute_concrete();
        };
        Ok(())
    }

    pub fn compatible_with(&self, _other: &ShapeFact) -> bool {
        if self.rank() == _other.rank() {
            self.dims
                .iter()
                .zip(_other.dims.iter())
                .all(|(dim, other_dim)| dim.compatible_with(other_dim))
        } else {
            false
        }
    }

    pub fn scalar() -> ShapeFact {
        let void: &[usize] = &[];
        Self::from(void)
    }

    pub fn consistent(&self) -> TractResult<()> {
        ensure!(
            self.concrete
                == self.dims.iter().map(|d| d.to_usize()).collect::<TractResult<TVec<_>>>().ok()
        );
        Ok(())
    }
}

impl std::ops::Deref for ShapeFact {
    type Target = [TDim];
    fn deref(&self) -> &[TDim] {
        &self.dims
    }
}

impl<D: ToDim, T: IntoIterator<Item = D>> From<T> for ShapeFact {
    fn from(it: T) -> ShapeFact {
        ShapeFact::from_dims(it)
    }
}

/// Type information about a tensor: shape, and element type, in various state
/// of determination.
pub trait Fact: std::fmt::Debug + Downcast + dyn_clone::DynClone + Send + Sync + 'static {
    fn to_typed_fact(&self) -> TractResult<Cow<TypedFact>>;

    fn matches(&self, t: &Tensor, symbols: Option<&SymbolValues>) -> TractResult<bool> {
        self.to_typed_fact()?.matches(t, symbols)
    }

    fn same_as(&self, _other: &dyn Fact) -> bool;

    /// Ensure that self is same type as another fact or a subtype
    fn compatible_with(&self, _other: &dyn Fact) -> bool;

    fn datum_type(&self) -> Option<DatumType>;
}

impl_downcast!(Fact);
dyn_clone::clone_trait_object!(Fact);

impl<D: ToDim> std::iter::FromIterator<D> for ShapeFact {
    fn from_iter<T: IntoIterator<Item = D>>(iter: T) -> Self {
        ShapeFact::from_dims(iter.into_iter().map(|d| d.to_dim()))
    }
}

impl fmt::Debug for ShapeFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use tract_itertools::Itertools;
        write!(fmt, "{}", self.iter().join(","))
    }
}

impl AsRef<[TDim]> for ShapeFact {
    fn as_ref(&self) -> &[TDim] {
        &self.dims
    }
}

/// Fully determined tensor information for TypedModel.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TypedFact {
    /// tensor element type
    pub datum_type: DatumType,
    /// tensor shape
    pub shape: ShapeFact,
    /// optional constant value
    pub konst: Option<Arc<Tensor>>,
    /// optional uniform value
    pub uniform: Option<Arc<Tensor>>,
    /// optional opaque fact
    pub opaque_fact: Option<Box<dyn OpaqueFact>>,
}

impl TypedFact {
    pub fn scalar<T>() -> TypedFact
    where
        T: Datum,
    {
        Self::dt_scalar(T::datum_type())
    }

    pub fn shape<T, S>(shape: S) -> TypedFact
    where
        T: Datum,
        S: Into<ShapeFact>,
    {
        Self::dt_shape(T::datum_type(), shape)
    }

    pub fn shape_and_dt_of(t: &Tensor) -> TypedFact {
        TypedFact {
            datum_type: t.datum_type(),
            shape: ShapeFact::from_dims(t.shape().iter().map(TDim::from)),
            uniform: None,
            konst: None,
            opaque_fact: None,
        }
    }

    pub fn mem_size(&self) -> TDim {
        self.shape.volume() * self.datum_type.size_of()
            + self.opaque_fact().map(|it| it.mem_size()).unwrap_or(0.into())
    }

    pub fn dt_scalar(datum_type: DatumType) -> TypedFact {
        TypedFact {
            datum_type,
            shape: ShapeFact::scalar(),
            konst: None,
            uniform: None,
            opaque_fact: None,
        }
    }

    pub fn dt_shape<S>(datum_type: DatumType, shape: S) -> TypedFact
    where
        S: Into<ShapeFact>,
    {
        TypedFact { datum_type, shape: shape.into(), konst: None, uniform: None, opaque_fact: None }
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
        self.shape.consistent()?;
        ensure!(self.datum_type.is_opaque() == self.opaque_fact.is_some());
        if let Some(k) = &self.konst {
            if !self.matches(k.as_ref(), None)? {
                bail!("fact says {}, constant is {:?}", self.format_dt_shape_nocheck(), k);
            }
            if let Some(bqf) = self.opaque_fact().and_then(|of| of.downcast_ref::<BlockQuantFact>())
            {
                for o in k.as_slice::<Opaque>().unwrap() {
                    ensure!(o.is::<BlockQuantValue>());
                    ensure!(o.downcast_ref::<BlockQuantValue>().unwrap().fact == *bqf);
                }
            }
        }
        if let Some(u) = &self.uniform {
            if self.datum_type != u.datum_type() {
                bail!("fact as uniform value {:?}, but is of type {:?}", u, self.datum_type);
            }
        }
        if let (Some(u), Some(k)) = (self.uniform.as_deref(), self.konst.as_deref()) {
            if let Some(k) = k.as_uniform() {
                if &k != u {
                    bail!("Uniform value and uniform constant mismatch: {:?}, {:?}", u, k);
                }
            } else {
                bail!("Fact said to be uniform ({:?}) and equal to {:?} which is not.", u, k);
            }
        }
        Ok(())
    }

    pub fn without_value(&self) -> Self {
        let mut new = self.clone();
        new.konst = None;
        new.uniform = None;
        new
    }

    pub fn with_opaque_fact<O: Into<Box<dyn OpaqueFact>>>(mut self, opaque_fact: O) -> Self {
        self.opaque_fact = Some(opaque_fact.into());
        self
    }

    pub fn opaque_fact(&self) -> Option<&dyn OpaqueFact> {
        self.opaque_fact.as_deref()
    }
}

impl Fact for TypedFact {
    fn to_typed_fact(&self) -> TractResult<Cow<TypedFact>> {
        if cfg!(debug_assertions) {
            self.consistent()?
        }
        Ok(Cow::Borrowed(self))
    }

    fn matches(&self, t: &Tensor, symbols: Option<&SymbolValues>) -> TractResult<bool> {
        if self.datum_type != t.datum_type() || self.shape.len() != t.rank() {
            return Ok(false);
        }
        for i in 0..t.rank() {
            if let Ok(dim) =
                self.shape[i].eval(symbols.unwrap_or(&SymbolValues::default())).to_usize()
            {
                if dim != t.shape()[i] {
                    return Ok(false);
                }
            }
        }
        Ok(true)
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

    fn compatible_with(&self, other: &dyn Fact) -> bool {
        if cfg!(debug_assertions) {
            self.consistent().unwrap()
        }
        if let Some(other) = other.downcast_ref::<Self>() {
            if cfg!(debug_assertions) {
                other.consistent().unwrap()
            }
            self.datum_type == other.datum_type
                && self.shape.compatible_with(&other.shape)
                && self
                    .opaque_fact()
                    .zip(other.opaque_fact())
                    .map(|(a, b)| a.compatible_with(b))
                    .unwrap_or(true)
        } else {
            false
        }
    }

    fn datum_type(&self) -> Option<DatumType> {
        Some(self.datum_type)
    }
}

impl From<Tensor> for TypedFact {
    fn from(t: Tensor) -> TypedFact {
        TypedFact::from(t.into_arc_tensor())
    }
}

impl From<Arc<Tensor>> for TypedFact {
    fn from(t: Arc<Tensor>) -> TypedFact {
        TypedFact {
            datum_type: t.datum_type(),
            shape: ShapeFact::from_dims(t.shape().iter().map(TDim::from)),
            uniform: t.as_uniform().map(Arc::new),
            opaque_fact: None,
            konst: Some(t),
        }
    }
}

impl From<&TypedFact> for TypedFact {
    fn from(fact: &TypedFact) -> TypedFact {
        fact.clone()
    }
}

impl<'a> From<&'a Arc<Tensor>> for TypedFact {
    fn from(t: &'a Arc<Tensor>) -> TypedFact {
        Arc::clone(t).into()
    }
}

impl fmt::Debug for TypedFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?},{:?}", self.shape, self.datum_type)?;
        if self.datum_type.is_opaque() {
            if let Some(of) = &self.opaque_fact {
                write!(fmt, " 🔍 {of:?} ")?
            } else {
                write!(fmt, " 🔍 <no opaque fact> ")?
            }
        }
        if let Some(k) = &self.konst {
            write!(fmt, "🟰 {k:?}")?
        }
        Ok(())
    }
}

pub trait DatumExt {
    fn scalar_fact() -> TypedFact;
    fn fact<S>(shape: S) -> TypedFact
    where
        S: Into<ShapeFact>;
}

impl<T: Datum> DatumExt for T {
    #[allow(clippy::needless_borrow)]
    fn scalar_fact() -> TypedFact {
        TypedFact::shape::<Self, &[usize]>(&[])
    }

    fn fact<S>(shape: S) -> TypedFact
    where
        S: Into<ShapeFact>,
    {
        TypedFact::shape::<Self, _>(shape)
    }
}

pub trait DatumTypeExt {
    fn scalar_fact(&self) -> TypedFact;
    fn fact<S>(&self, shape: S) -> TypedFact
    where
        S: Into<ShapeFact>;
}

impl DatumTypeExt for DatumType {
    #[allow(clippy::needless_borrow)]
    fn scalar_fact(&self) -> TypedFact {
        TypedFact::dt_shape::<&[usize]>(*self, &[])
    }

    fn fact<S>(&self, shape: S) -> TypedFact
    where
        S: Into<ShapeFact>,
    {
        TypedFact::dt_shape(*self, shape)
    }
}
