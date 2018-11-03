use std::sync::Arc;

use ops::prelude::*;

#[derive(Debug, Clone)]
pub struct Value(Arc<Tensor>);

impl Value {
    /// Returns a reference to the Tensor wrapped inside a Value.
    pub fn as_tensor(&self) -> &Tensor {
        self.0.as_ref()
    }

    pub fn to_tensor(self) -> Tensor {
        Arc::try_unwrap(self.0).unwrap_or_else(|arc| arc.as_ref().clone())
    }

    pub fn to_array<'a, D: ::tensor::Datum>(self) -> TractResult<::ndarray::ArrayD<D>> {
        self.to_tensor().into_array()

    }
}

impl<M> From<M> for Value
where
    Tensor: From<M>,
{
    fn from(m: M) -> Value {
        Value::from(Arc::new(m.into()))
    }
}

impl From<Arc<Tensor>> for Value {
    fn from(m: Arc<Tensor>) -> Value {
        Value(m)
    }
}

impl ::std::ops::Deref for Value {
    type Target = Tensor;
    fn deref(&self) -> &Tensor {
        self.0.as_ref()
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        self.as_tensor() == other.as_tensor()
    }
}
