use std::mem;
use std::sync::Arc;

use ops::prelude::*;

#[derive(Debug, Clone)]
pub enum Value {
    Owned(Tensor),
    Shared(Arc<Tensor>),
}

impl Value {
    /// Creates a shared Value from any Value.
    pub fn into_shared(self) -> Value {
        match self {
            Value::Owned(m) => Value::Shared(Arc::new(m)),
            Value::Shared(_) => self,
        }
    }

    /// Creates a Tensor from a Value.
    pub fn into_tensor(self) -> Tensor {
        match self {
            Value::Owned(m) => m,
            Value::Shared(m) => m.as_ref().clone(),
        }
    }

    /// Returns a reference to the Tensor wrapped inside a Value.
    pub fn as_tensor(&self) -> &Tensor {
        match self {
            &Value::Owned(ref m) => &m,
            &Value::Shared(ref m) => m.as_ref(),
        }
    }

    /// Returns a shared copy of the Value, turning the one passed
    /// as argument into a Value::Shared if necessary.
    pub fn share(&mut self) -> Value {
        // This is somewhat ugly, but sadly we couldn't find any other
        // way to implement it. If we try to write something like:
        //   *self = Value::Shared(Arc::new(*m))
        // the borrow checker will complain about *m being moved out of
        // borrowed content, which makes sense but doesn't apply in our
        // case because we will "give m back" to the Value, except
        // wrapped around an Arc. The only way to get ownership of m is
        // to use mem::replace, which means we have to create a "dummy"
        // value to replace self first.
        if let Value::Owned(_) = self {
            let dummy = Value::Owned(Tensor::i32s(&[], &[0]).unwrap());
            let shared = match mem::replace(self, dummy) {
                Value::Owned(m) => Value::Shared(Arc::new(m)),
                _ => panic!(),
            };

            *self = shared;
        }

        self.clone()
    }

    pub fn into_array<'a, D: ::tensor::Datum>(self) -> TractResult<::ndarray::ArrayD<D>> {
        self.into_tensor().into_array()
    }

    pub fn to_array_view<'a, D: ::tensor::Datum>(
        &'a self,
    ) -> TractResult<::ndarray::ArrayViewD<'a, D>> {
        self.as_tensor().to_array_view()
    }
}

impl<M> From<M> for Value
where
    Tensor: From<M>,
{
    fn from(m: M) -> Value {
        Value::Owned(m.into())
    }
}

impl From<Arc<Tensor>> for Value {
    fn from(m: Arc<Tensor>) -> Value {
        Value::Shared(m)
    }
}

impl ::std::ops::Deref for Value {
    type Target = Tensor;
    fn deref(&self) -> &Tensor {
        match self {
            &Value::Owned(ref m) => &m,
            &Value::Shared(ref m) => m.as_ref(),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        self.as_tensor() == other.as_tensor()
    }
}
