use crate::prelude::*;
use crate::internal::*;
use crate::datum::TryInto;
use crate::dim::ToDim;
use crate::tensor::Tensor;
use std::fmt;

pub trait TensorInfo: Clone + std::fmt::Debug {
    fn to_tensor_fact(&self) -> TensorFact;
}

impl<TI: TensorInfo> TryInto<TI> for TI {
    fn try_into(&self) -> TractResult<TI> {
        Ok(self.clone())
    }
}

impl TensorInfo for TensorFact {
    fn to_tensor_fact(&self) -> TensorFact {
        self.clone()
    }
}

impl TryInto<TypedTensorInfo> for TensorFact {
    fn try_into(&self) -> TractResult<TypedTensorInfo> {
        use crate::analyser::types::Fact;
        if let (Some(datum_type), Some(shape)) =
            (self.datum_type.concretize(), self.shape.concretize())
        {
            let stream_info = shape
                .iter()
                .cloned()
                .enumerate()
                .find(|d| d.1.to_integer().is_err())
                .map(|(axis, len)| StreamInfo { axis, len });
            let shape = shape.iter().map(|d| d.to_integer().unwrap_or(0) as usize).collect();
            let shape = ShapeInfo { shape, stream_info };
            Ok(TypedTensorInfo { datum_type, shape, konst: self.value.concretize() })
        } else {
            bail!("Can not make a TypedTensorInfo out of {:?}", self)
        }
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct StreamInfo {
    pub axis: usize,
    pub len: TDim,
}

#[derive(Clone)]
pub struct ShapeInfo {
    shape: TVec<usize>,
    pub stream_info: Option<StreamInfo>,
}

impl PartialEq for ShapeInfo {
    fn eq(&self, other: &ShapeInfo) -> bool {
        self.shape.len() == other.shape.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl ShapeInfo {
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn dim(&self, i: usize) -> TDim {
        if let Some(stream) = self.stream_info {
            if stream.axis == i {
                return stream.len;
            }
        }
        self.shape[i].to_dim()
    }

    pub fn as_finite(&self) -> Option<&[usize]> {
        match self.stream_info {
            None => Some(&*self.shape),
            _ => None,
        }
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = TDim> + 'a {
        self.shape.clone().into_iter().enumerate().map(move |(ix, d)| {
            if let Some(info) = self.stream_info {
                if ix == info.axis {
                    return info.len;
                }
            }
            (d as i64).to_dim()
        })
    }

    pub fn to_shape_fact(&self) -> ShapeFact {
        ShapeFact::from(self.iter())
    }
}

impl<T: AsRef<[usize]>> From<T> for ShapeInfo {
    fn from(it: T) -> ShapeInfo {
        ShapeInfo { shape: it.as_ref().iter().cloned().collect(), stream_info: None }
    }
}

impl fmt::Debug for ShapeInfo {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;
        write!(fmt, "{}", self.iter().join("x"))
    }
}

#[derive(Clone)]
pub struct TypedTensorInfo {
    pub datum_type: DatumType,
    pub shape: ShapeInfo,
    pub konst: Option<SharedTensor>,
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
        TypedTensorInfo::from(SharedTensor::from(t))
    }
}

impl<'t> From<&'t Tensor> for TypedTensorInfo {
    fn from(t: &'t Tensor) -> TypedTensorInfo {
        TypedTensorInfo::from(t.clone())
    }
}

impl From<SharedTensor> for TypedTensorInfo {
    fn from(t: SharedTensor) -> TypedTensorInfo {
        TypedTensorInfo {
            datum_type: t.datum_type(),
            shape: ShapeInfo { shape: t.shape().into(), stream_info: None },
            konst: Some(t),
        }
    }
}

impl TryInto<NormalizedTensorInfo> for TypedTensorInfo {
    fn try_into(&self) -> TractResult<NormalizedTensorInfo> {
        match self.konst {
            None => {
                Ok(NormalizedTensorInfo { shape: self.shape.clone(), datum_type: self.datum_type })
            }
            _ => bail!("Constant tensor are excluded from declutterd stage: {:?}", self),
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

#[derive(Clone, PartialEq)]
pub struct NormalizedTensorInfo {
    pub datum_type: DatumType,
    pub shape: ShapeInfo,
}

impl TensorInfo for NormalizedTensorInfo {
    fn to_tensor_fact(&self) -> TensorFact {
        TensorFact::dt_shape(self.datum_type, self.shape.to_shape_fact())
    }
}

impl TryInto<TypedTensorInfo> for NormalizedTensorInfo {
    fn try_into(&self) -> TractResult<TypedTensorInfo> {
        Ok(TypedTensorInfo { shape: self.shape.clone(), datum_type: self.datum_type, konst: None })
    }
}

impl fmt::Debug for NormalizedTensorInfo {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}x{:?}", self.shape, self.datum_type)
    }
}
