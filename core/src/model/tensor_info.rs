//! Partial and complete tensor types representations.
use crate::internal::*;
use crate::prelude::*;
use crate::tensor::Tensor;
use std::convert::{TryFrom, TryInto};
use std::fmt;

/// Type information about a tensor: shape, and element type, in various state
/// of determination.
pub trait TensorInfo: std::fmt::Debug + objekt::Clone {
    /// Convert to TensorFact, the most accomoding variant of TensorInfo.
    fn to_tensor_fact(&self) -> TensorFact;
}

objekt::clone_trait_object!(TensorInfo);

impl TensorInfo for TensorFact {
    fn to_tensor_fact(&self) -> TensorFact {
        self.clone()
    }
}

impl<'a> TryFrom<&'a TensorFact> for TypedTensorInfo {
    type Error = TractError;
    fn try_from(fact: &TensorFact) -> TractResult<TypedTensorInfo> {
        if let (Some(datum_type), Some(shape)) =
            (fact.datum_type.concretize(), fact.shape.concretize())
        {
            let stream_info = shape
                .iter()
                .cloned()
                .enumerate()
                .find(|d| d.1.to_integer().is_err())
                .map(|(axis, len)| StreamInfo { axis, len });
            let shape = shape.iter().map(|d| d.to_integer().unwrap_or(0) as usize).collect();
            let shape = ShapeInfo { shape, stream_info };
            Ok(TypedTensorInfo { datum_type, shape, konst: fact.value.concretize() })
        } else {
            bail!("Can not make a TypedTensorInfo out of {:?}", fact)
        }
    }
}

impl TryFrom<TensorFact> for TypedTensorInfo {
    type Error = TractError;
    fn try_from(fact: TensorFact) -> TractResult<TypedTensorInfo> {
        (&fact).try_into()
    }
}

impl<'a> From<&'a Tensor> for TensorFact {
    fn from(t: &'a Tensor) -> TensorFact {
        TensorFact::from(t.clone())
    }
}

/// Streaming information for a streamed tensor.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamInfo {
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
#[derive(Clone)]
pub struct ShapeInfo {
    shape: TVec<usize>,
    /// Optional information for streaming tensors. None for regular tensors.
    pub stream_info: Option<StreamInfo>,
}

impl PartialEq for ShapeInfo {
    fn eq(&self, other: &ShapeInfo) -> bool {
        self.shape.len() == other.shape.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl ShapeInfo {
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
                    self.stream_info = Some(StreamInfo { len: dim, axis: i })
                }
            }
        } else {
            if let Ok(int) = dim.to_integer() {
                self.shape[i] = int as _;
            } else {
                self.shape[i] = 0;
                self.stream_info = Some(StreamInfo { len: dim, axis: i })
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

    /// Convert the shape to a fully determined shape fact.
    pub fn to_shape_fact(&self) -> ShapeFact {
        ShapeFact::from(self.iter())
    }

    pub fn from_dims<T: AsRef<[TDim]> + std::fmt::Debug>(it: T) -> TractResult<ShapeInfo> {
        let count = it.as_ref().iter().filter(|t| t.is_stream()).count();
        if count > 1 {
            bail!("Shape with two streaming dims are invalid: {:?}", it)
        } else {
            let stream_info = it
                .as_ref()
                .iter()
                .enumerate()
                .find(|(_ix, d)| d.is_stream())
                .map(|(ix, d)| StreamInfo { axis: ix, len: d.clone() });
            Ok(ShapeInfo {
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

impl TryFrom<&[TDim]> for ShapeInfo {
    type Error = TractError;
    fn try_from(it: &[TDim]) -> TractResult<ShapeInfo> {
        ShapeInfo::from_dims(it)
    }
}

impl TryFrom<&[usize]> for ShapeInfo {
    type Error = TractError;
    fn try_from(it: &[usize]) -> TractResult<ShapeInfo> {
        Ok(ShapeInfo { shape: it.into(), stream_info: None })
    }
}

impl fmt::Debug for ShapeInfo {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;
        write!(fmt, "{}", self.iter().join("x"))
    }
}

/// Fully determined tensor information for TypedModel.
#[derive(Clone)]
pub struct TypedTensorInfo {
    /// tensor element type
    pub datum_type: DatumType,
    /// tensor shape
    pub shape: ShapeInfo,
    /// optional constant value
    pub konst: Option<Arc<Tensor>>,
}

impl TypedTensorInfo {
    pub fn shape<T: Datum, S: TryInto<ShapeInfo, Error = TractError>>(
        shape: S,
    ) -> TractResult<TypedTensorInfo> {
        Self::dt_shape(T::datum_type(), shape)
    }

    pub fn dt_shape<S: TryInto<ShapeInfo, Error = TractError>>(
        datum_type: DatumType,
        shape: S,
    ) -> TractResult<TypedTensorInfo> {
        Ok(TypedTensorInfo { datum_type, shape: shape.try_into()?, konst: None })
    }
}

impl TensorInfo for TypedTensorInfo {
    fn to_tensor_fact(&self) -> TensorFact {
        match self.konst.clone() {
            Some(k) => k.into(),
            None => TensorFact::dt_shape(self.datum_type, self.shape.to_shape_fact()),
        }
    }
}

impl From<Tensor> for TypedTensorInfo {
    fn from(t: Tensor) -> TypedTensorInfo {
        TypedTensorInfo::from(t.into_arc_tensor())
    }
}

impl<'t> From<&'t Tensor> for TypedTensorInfo {
    fn from(t: &'t Tensor) -> TypedTensorInfo {
        TypedTensorInfo::from(t.clone())
    }
}

impl From<Arc<Tensor>> for TypedTensorInfo {
    fn from(t: Arc<Tensor>) -> TypedTensorInfo {
        TypedTensorInfo {
            datum_type: t.datum_type(),
            shape: ShapeInfo { shape: t.shape().into(), stream_info: None },
            konst: Some(t),
        }
    }
}

impl TryFrom<TypedTensorInfo> for NormalizedTensorInfo {
    type Error = TractError;
    fn try_from(fact: TypedTensorInfo) -> TractResult<NormalizedTensorInfo> {
        match fact.konst {
            None => {
                Ok(NormalizedTensorInfo { shape: fact.shape.clone(), datum_type: fact.datum_type })
            }
            _ => bail!("Constant tensor are excluded from declutterd stage: {:?}", fact),
        }
    }
}

impl fmt::Debug for TypedTensorInfo {
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
#[derive(Clone, PartialEq)]
pub struct NormalizedTensorInfo {
    /// tensor element type
    pub datum_type: DatumType,
    /// tensor shape
    pub shape: ShapeInfo,
}

impl NormalizedTensorInfo {
    pub fn shape<T: Datum, S: TryInto<ShapeInfo, Error = TractError>>(
        shape: S,
    ) -> TractResult<NormalizedTensorInfo> {
        Self::dt_shape(T::datum_type(), shape)
    }
    pub fn dt_shape<S: TryInto<ShapeInfo, Error = TractError>>(
        datum_type: DatumType,
        shape: S,
    ) -> TractResult<NormalizedTensorInfo> {
        Ok(NormalizedTensorInfo { datum_type, shape: shape.try_into()? })
    }
}

impl TensorInfo for NormalizedTensorInfo {
    fn to_tensor_fact(&self) -> TensorFact {
        TensorFact::dt_shape(self.datum_type, self.shape.to_shape_fact())
    }
}

impl TryFrom<NormalizedTensorInfo> for TypedTensorInfo {
    type Error = TractError;
    fn try_from(fact: NormalizedTensorInfo) -> TractResult<TypedTensorInfo> {
        Ok(TypedTensorInfo { shape: fact.shape.clone(), datum_type: fact.datum_type, konst: None })
    }
}

impl fmt::Debug for NormalizedTensorInfo {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}x{:?}", self.shape, self.datum_type)
    }
}

impl<'t> From<&'t Tensor> for NormalizedTensorInfo {
    fn from(t: &'t Tensor) -> NormalizedTensorInfo {
        NormalizedTensorInfo { datum_type: t.datum_type(), shape: t.shape().try_into().unwrap() }
    }
}
