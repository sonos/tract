use std::sync::Arc;

use ops::prelude::*;

#[derive(Debug, Clone)]
pub struct Tensor(Arc<DtArray>);

impl Tensor {
    /// Returns a reference to the DtArray wrapped inside a Tensor.
    pub fn as_tensor(&self) -> &DtArray {
        self.0.as_ref()
    }

    pub fn to_tensor(self) -> DtArray {
        Arc::try_unwrap(self.0).unwrap_or_else(|arc| arc.as_ref().clone())
    }

    pub fn to_array<'a, D: ::datum::Datum>(self) -> TractResult<::ndarray::ArrayD<D>> {
        self.to_tensor().into_array()

    }
}

impl<M> From<M> for Tensor
where
    DtArray: From<M>,
{
    fn from(m: M) -> Tensor {
        Tensor::from(Arc::new(m.into()))
    }
}

impl From<Arc<DtArray>> for Tensor {
    fn from(m: Arc<DtArray>) -> Tensor {
        Tensor(m)
    }
}

impl ::std::ops::Deref for Tensor {
    type Target = DtArray;
    fn deref(&self) -> &DtArray {
        self.0.as_ref()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        self.as_tensor() == other.as_tensor()
    }
}
