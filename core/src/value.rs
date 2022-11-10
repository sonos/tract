use crate::internal::*;
use std::rc::Rc;

#[derive(Clone, PartialEq, Eq)]
pub struct TValue(Rc<Tensor>);

impl std::fmt::Debug for TValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl TValue {
    pub fn is_exclusive(&self) -> bool {
        Rc::strong_count(&self.0) == 1
    }
}

impl From<Tensor> for TValue {
    fn from(t: Tensor) -> Self {
        TValue(std::rc::Rc::new(t))
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
        &self.0
    }
}

impl IntoTensor for TValue {
    fn into_tensor(self) -> Tensor {
        Rc::try_unwrap(self.0).unwrap_or_else(|t| (*t).clone())
    }
}

impl IntoArcTensor for TValue {
    fn into_arc_tensor(self) -> Arc<Tensor> {
        Arc::new(self.into_tensor())
    }
}

pub trait IntoTValue {
    fn into_tvalue(self) -> TValue;
}

impl<T: IntoTensor> IntoTValue for T {
    fn into_tvalue(self) -> TValue {
        self.into_tensor().into()
    }
}

