use crate::internal::*;
use std::ops::Deref;

use tract_ndarray::Array;

#[derive(Clone, Eq)]
pub struct TValue(Arc<Tensor>);

impl std::fmt::Debug for TValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

impl PartialEq for TValue {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl TValue {
    pub fn is_exclusive(&self) -> bool {
        Arc::strong_count(&self.0) == 1
    }

    pub fn from_const(t: Arc<Tensor>) -> Self {
        TValue(t)
    }

    pub fn as_arc_tensor(&self) -> Option<&Arc<Tensor>> {
        Some(&self.0)
    }
}

impl From<Tensor> for TValue {
    fn from(t: Tensor) -> Self {
        TValue(Arc::new(t))
    }
}

impl std::ops::Deref for TValue {
    type Target = Tensor;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::borrow::Borrow<Tensor> for TValue {
    fn borrow(&self) -> &Tensor {
        self
    }
}

impl IntoTensor for TValue {
    fn into_tensor(self) -> Tensor {
        self.0.into_tensor()
    }
}

impl IntoArcTensor for TValue {
    fn into_arc_tensor(self) -> Arc<Tensor> {
        self.0
    }
}

pub trait IntoTValue {
    fn into_tvalue(self) -> TValue;
}

impl IntoTValue for Tensor {
    fn into_tvalue(self) -> TValue {
        self.into_tensor().into()
    }
}

impl IntoTValue for Arc<Tensor> {
    fn into_tvalue(self) -> TValue {
        TValue(self)
    }
}

impl<D: ::ndarray::Dimension, T: Datum> IntoTValue for Array<T, D> {
    fn into_tvalue(self) -> TValue {
        Tensor::from(self).into_tvalue()
    }
}
