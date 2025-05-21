use crate::internal::*;
use std::ops::Deref;
use std::rc::Rc;

use tract_ndarray::Array;
use TValue::*;

#[derive(Clone, Eq)]
pub enum TValue {
    Const(Arc<Tensor>),
    Var(Rc<Tensor>),
}

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
        match self {
            Var(it) => Rc::strong_count(it) == 1,
            Const(_) => false,
        }
    }

    pub fn from_const(t: Arc<Tensor>) -> Self {
        Const(t)
    }

    pub fn as_arc_tensor(&self) -> Option<&Arc<Tensor>> {
        match self {
            Const(t) => Some(t),
            Var(_) => None,
        }
    }
}

impl From<Tensor> for TValue {
    fn from(t: Tensor) -> Self {
        TValue::Var(std::rc::Rc::new(t))
    }
}

impl std::ops::Deref for TValue {
    type Target = Tensor;
    fn deref(&self) -> &Self::Target {
        match self {
            Const(it) => it,
            Var(it) => it,
        }
    }
}

impl std::borrow::Borrow<Tensor> for TValue {
    fn borrow(&self) -> &Tensor {
        self
    }
}

impl IntoTensor for TValue {
    fn into_tensor(self) -> Tensor {
        match self {
            Var(it) => Rc::try_unwrap(it).unwrap_or_else(|t| (*t).clone()),
            Const(it) => it.into_tensor(),
        }
    }
}

impl IntoArcTensor for TValue {
    fn into_arc_tensor(self) -> Arc<Tensor> {
        match self {
            Var(ref _it) => self.into_tensor().into_arc_tensor(),
            Const(t) => t,
        }
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
        Const(self)
    }
}

impl<D: ::ndarray::Dimension, T: Datum> IntoTValue for Array<T, D> {
    fn into_tvalue(self) -> TValue {
        Tensor::from(self).into_tvalue()
    }
}
