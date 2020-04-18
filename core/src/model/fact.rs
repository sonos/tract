//! Partial and complete tensor types representations.
use crate::internal::*;
use crate::tensor::Tensor;
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

/// Streaming information for a streamed tensor.
#[derive(Debug, Clone, Default, PartialEq, Hash)]
pub struct StreamFact {
    /// Streaming axis
    pub axis: usize,
    /// Streaming length
    pub len: TDim,
}

/// Fully determined dimension of a tensor.
///
/// Tensors in tract can have one streaming dimension. TDim generalize the
/// regular tensor dimensions (usize) to arithmetic expressions of `S`, the
/// (sometimes hypothetical) tensor length on the streaming axis.
#[derive(Clone, PartialEq, Hash)]
pub struct ShapeFact {
    shape: TVec<usize>,
    /// Optional information for streaming tensors. None for regular tensors.
    pub stream_info: Option<StreamFact>,
}

impl ShapeFact {
    /// Rank of the tensor.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Extended dimension of the i-th axis.
    ///
    /// The TDim will wrap a plain integer for regular (non-streaming) tensors.
    pub fn dim(&self, i: usize) -> TDim {
        if let Some(ref stream) = self.stream_info {
            if stream.axis == i {
                return stream.len.clone();
            }
        }
        self.shape[i].to_dim()
    }

    /// Set the i-th axis dimension.
    pub fn set_dim(&mut self, i: usize, dim: TDim) -> TractResult<()> {
        if let Some(ref stream) = self.stream_info {
            if let Ok(int) = dim.to_integer() {
                self.shape[i] = int as _;
                if stream.axis == i {
                    self.stream_info = None;
                }
            } else {
                if stream.axis != i {
                    bail!("Attempt at building a shape with two streaming dim")
                } else {
                    self.stream_info = Some(StreamFact { len: dim, axis: i })
                }
            }
        } else {
            if let Ok(int) = dim.to_integer() {
                self.shape[i] = int as _;
            } else {
                self.shape[i] = 0;
                self.stream_info = Some(StreamFact { len: dim, axis: i })
            }
        }
        Ok(())
    }

    pub fn insert_axis(&mut self, axis: usize) -> TractResult<()> {
        self.shape.insert(axis, 1);
        if let Some(s) = self.stream_info.as_mut() {
            if s.axis >= axis {
                s.axis += 1;
            }
        }
        Ok(())
    }

    pub fn remove_axis(&mut self, axis: usize) -> TractResult<()> {
        self.shape.remove(axis);
        if let Some(s) = self.stream_info.as_mut() {
            if s.axis > axis {
                s.axis -= 1;
            }
        }
        Ok(())
    }

    /// Shape of the tensor, unless it is streaming.
    pub fn as_finite(&self) -> Option<&[usize]> {
        match self.stream_info {
            None => Some(&*self.shape),
            _ => None,
        }
    }

    /// Iterator over dimension of the shape.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = TDim> + 'a {
        self.shape.clone().into_iter().enumerate().map(move |(ix, d)| {
            if let Some(ref info) = self.stream_info {
                if ix == info.axis {
                    return info.len.clone();
                }
            }
            (d as i64).to_dim()
        })
    }

    /// Convert the shape to an array of extended dimensions.
    pub fn to_tvec(&self) -> TVec<TDim> {
        self.iter().collect::<TVec<TDim>>()
    }

    pub fn from_dims<T: AsRef<[TDim]> + std::fmt::Debug>(it: T) -> TractResult<ShapeFact> {
        let count = it.as_ref().iter().filter(|t| t.is_stream()).count();
        if count > 1 {
            bail!("Shape with two streaming dims are invalid: {:?}", it)
        } else {
            let stream_info = it
                .as_ref()
                .iter()
                .enumerate()
                .find(|(_ix, d)| d.is_stream())
                .map(|(ix, d)| StreamFact { axis: ix, len: d.clone() });
            Ok(ShapeFact {
                shape: it
                    .as_ref()
                    .iter()
                    .map(|t| t.to_integer().map(|i| i as usize).unwrap_or(0))
                    .collect(),
                stream_info,
            })
        }
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
        Ok(ShapeFact { shape: it.into(), stream_info: None })
    }
}

impl fmt::Debug for ShapeFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;
        write!(fmt, "{}", self.iter().join("x"))
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


tract_linalg::impl_dyn_hash!(TypedFact);

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
        self.shape.rank()
    }

    pub fn format_dt_shape(&self) -> String {
        format!("{:?}x{:?}", self.shape, self.datum_type)
    }
}

impl Fact for TypedFact {
    fn to_typed_fact(&self) -> TractResult<TypedFact> {
        Ok(self.clone())
    }

    fn matches(&self, t: &Tensor) -> TractResult<bool> {
        Ok(self.datum_type == t.datum_type() && t.shape() == &*self.shape.shape)
    }

    fn same_as(&self, other: &dyn Fact) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
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
            shape: ShapeFact { shape: t.shape().into(), stream_info: None },
            konst: Some(t),
        }
    }
}

impl<'a> TryFrom<&'a TypedFact> for NormalizedFact {
    type Error = TractError;
    fn try_from(fact: &TypedFact) -> TractResult<NormalizedFact> {
        match fact.konst {
            None => Ok(NormalizedFact { shape: fact.shape.clone(), datum_type: fact.datum_type }),
            _ => bail!("Constant tensor are excluded from declutterd stage: {:?}", fact),
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
            None => write!(fmt, "{:?}x{:?}", self.shape, self.datum_type),
        }
    }
}

/// Tensor information for Normalized models.
///
/// Constant value is not allowed, as all tensors in normalized forms are
/// variables.
#[derive(Clone, PartialEq, Hash)]
pub struct NormalizedFact {
    /// tensor element type
    pub datum_type: DatumType,
    /// tensor shape
    pub shape: ShapeFact,
}

tract_linalg::impl_dyn_hash!(NormalizedFact);

impl NormalizedFact {
    pub fn shape<T, S, E>(shape: S) -> TractResult<NormalizedFact>
    where
        T: Datum,
        S: TryInto<ShapeFact, Error = E>,
        TractError: From<E>,
    {
        Self::dt_shape(T::datum_type(), shape)
    }
    pub fn dt_shape<S, E>(datum_type: DatumType, shape: S) -> TractResult<NormalizedFact>
    where
        S: TryInto<ShapeFact, Error = E>,
        TractError: From<E>,
    {
        Ok(NormalizedFact { datum_type, shape: shape.try_into()? })
    }
}

impl Fact for NormalizedFact {
    fn to_typed_fact(&self) -> TractResult<TypedFact> {
        Ok(self.into())
    }

    fn same_as(&self, other: &dyn Fact) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self == other
        } else {
            false
        }
    }
}

impl<'a> From<&'a NormalizedFact> for TypedFact {
    fn from(fact: &NormalizedFact) -> TypedFact {
        TypedFact { shape: fact.shape.clone(), datum_type: fact.datum_type, konst: None }
    }
}

impl fmt::Debug for NormalizedFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}x{:?}", self.shape, self.datum_type)
    }
}

impl<'t> From<&'t Tensor> for NormalizedFact {
    fn from(t: &'t Tensor) -> NormalizedFact {
        NormalizedFact { datum_type: t.datum_type(), shape: t.shape().try_into().unwrap() }
    }
}
