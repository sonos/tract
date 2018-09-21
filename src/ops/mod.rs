//! TensorFlow Ops
use std::collections::VecDeque;
use std::fmt::Debug;
use std::mem;
use std::ops::{Index, IndexMut};
use std::sync::Arc;

use analyser::prelude::*;
use dim::TDim;
use model::TVec;
use {Result, Tensor};

use downcast_rs::Downcast;
use objekt;

#[macro_use]
mod macros;

#[cfg(features = "image_ops")]
pub mod image;
pub mod konst;
pub mod math;
pub mod nn;
pub mod sink;
pub mod source;
pub mod unimpl;

pub mod prelude {
    pub use super::{Op, InferenceOp};
    pub use super::{OpBuffer, QueuesBuffer, StepValue, Stream, StreamInfo, Value};
    pub use dim::TDim;
    pub use model::TVec;
    pub use std::collections::HashMap;
    pub use std::marker::PhantomData;
    pub use tensor::{Datum, DatumType, Tensor};
    pub use Result;
}

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

    pub fn into_array<'a, D: ::tensor::Datum>(self) -> ::Result<::ndarray::ArrayD<D>> {
        self.into_tensor().into_array()
    }

    pub fn to_array_view<'a, D: ::tensor::Datum>(
        &'a self,
    ) -> ::Result<::ndarray::ArrayViewD<'a, D>> {
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

#[derive(Debug, Clone)]
pub enum StepValue {
    Const(Value),
    Stream(Stream),
}

#[derive(Debug, Clone)]
pub struct Stream {
    pub info: StreamInfo,
    pub offset: u64,
    pub chunk: Option<Value>,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct StreamInfo {
    pub axis: usize,
    pub len: TDim,
}

impl StepValue {
    pub fn as_value(&self) -> Option<&Value> {
        match self {
            StepValue::Const(v) => Some(v),
            StepValue::Stream(s) => s.chunk.as_ref(),
        }
    }

    pub fn into_value(self) -> Option<Value> {
        match self {
            StepValue::Const(v) => Some(v),
            StepValue::Stream(s) => s.chunk,
        }
    }

    pub fn as_const(&self) -> Option<&Value> {
        match self {
            StepValue::Const(v) => Some(v),
            _ => None,
        }
    }

    pub fn into_const(self) -> Option<Value> {
        match self {
            StepValue::Const(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_stream(&self) -> Option<&Stream> {
        match self {
            StepValue::Stream(s) => Some(s),
            _ => None,
        }
    }

    pub fn into_stream(self) -> Option<Stream> {
        match self {
            StepValue::Stream(s) => Some(s),
            _ => None,
        }
    }

    pub fn stream_info(&self) -> Option<StreamInfo> {
        self.as_stream().map(|s| s.info)
    }

    pub fn is_const(&self) -> bool {
        match self {
            StepValue::Const(_) => true,
            StepValue::Stream(_) => false,
        }
    }
}

/// A Tensorflow operation.
pub trait Op: Debug + objekt::Clone + Send + Sync + 'static + InferenceOp {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Value>) -> Result<TVec<Value>> {
        bail!("Unexpected call on op.eval(). {:?}", self)
    }

    /// Returns a new streaming buffer for the operation.
    fn new_buffer(&self) -> Box<OpBuffer> {
        Box::new(EmptyBuffer {})
    }

    /// Evaluates one step of the operation on the given input tensors.
    /// This is only implemented for operators which support streaming.
    ///
    /// The input tensors are annotated with an Option<usize>:
    /// - None if the tensor doesn't have a streaming dimension.
    /// - Option(d) if the tensor is being streamed on dimension d.
    ///
    /// If an input tensor has a streaming dimension, the corresponding
    /// Value will only contain a _chunk_ of input of size 1 along
    /// that dimension. Note that each chunk will only be passed once
    /// to the step function, so it should use the provided buffer to
    /// store whichever chunks it needs for future computations.
    ///
    /// The function should return Some(chunks) when it has computed
    /// new chunks, and None if it has computed an intermediary result
    /// successfully but doesn't have new output chunks ready yet.
    ///
    /// For operators like Concat, multiple input tensors might have a
    /// streaming dimension. In that case, at each call to step, only
    /// one of the streaming inputs will receive new chunk while the
    /// others will receive None.
    fn step(
        &self,
        _inputs: TVec<StepValue>,
        _buffer: &mut Box<OpBuffer>,
    ) -> Result<Option<TVec<Value>>> {
        bail!("Streaming is not available for operator {:?}", self)
    }

    /// Infers properties about the input and output tensors.
    ///
    /// The `inputs` and `outputs` arguments correspond to properties about
    /// the input and output tensors that are already known.
    ///
    /// Returns Err in case of an unrecoverable error during the inference,
    /// and the refined properties about the inputs and outputs otherwise.
    fn infer_and_propagate(
        &self,
        inputs: TVec<TensorFact>,
        outputs: TVec<TensorFact>,
    ) -> Result<(TVec<TensorFact>, TVec<TensorFact>)> {
        let (infered_inputs, infered_outputs) = self.infer(inputs, outputs)?;

        if infered_inputs.iter().all(|i| i.value.is_concrete()) {
            let input_values = infered_inputs
                .iter()
                .map(|i| i.value.concretize().unwrap().clone().into())
                .collect(); // checked
            let output_value = self.eval(input_values)?.pop().unwrap();
            Ok((
                infered_inputs,
                tvec![::analyser::helpers::tensor_to_fact(
                    output_value.into_tensor(),
                )],
            ))
        } else {
            Ok((infered_inputs, infered_outputs))
        }
    }

    fn final_prep(
        &self,
        _inputs: TVec<TensorFact>,
        _outputs: TVec<TensorFact>,
    ) -> Result<Option<Box<Op>>> {
        Ok(None)
    }

    fn const_value(&self) -> Option<Value> {
        None
    }

    fn rounding_errors(&self) -> bool {
        false
    }
}

pub trait InferenceOp {
    fn infer(
        &self,
        inputs: TVec<TensorFact>,
        outputs: TVec<TensorFact>,
    ) -> Result<(TVec<TensorFact>, TVec<TensorFact>)>;
}

clone_trait_object!(Op);

/// A streaming buffer for a Tensorflow operation.
///
/// This is used during streaming evaluation of models. Each node is given
/// a mutable reference to a buffer which it can use to store intermediary
/// results between evaluation steps. Every operation must provide its own
/// buffer type (or use one of the general ones defined below), which must
/// implement the OpBuffer trait. It should return a new instance of it in
/// the `Op::new_buffer` method, and downcast it from OpBuffer in `step`.
pub trait OpBuffer: Downcast + Debug + objekt::Clone + Send + 'static {}
clone_trait_object!(OpBuffer);
impl_downcast!(OpBuffer);

/// An empty buffer for operations which don't need one.
#[derive(Debug, Clone)]
pub struct EmptyBuffer {}

impl OpBuffer for EmptyBuffer {}

/// A buffer with a variable number of Value queues.
#[derive(Debug, Clone)]
pub struct QueuesBuffer(TVec<VecDeque<Value>>);

impl OpBuffer for QueuesBuffer {}

impl QueuesBuffer {
    /// Creates a new buffer with a given number of queues.
    pub fn new(size: usize) -> QueuesBuffer {
        QueuesBuffer(tvec![VecDeque::new(); size])
    }

    /// Appends a new Value to each queue in the buffer.
    pub fn append(&mut self, views: TVec<StepValue>) -> Result<()> {
        if views.len() > self.0.len() {
            bail!("There are more input Values than queues in the buffer.");
        }

        for (i, view) in views.into_iter().enumerate() {
            if let Some(v) = view.into_value() {
                self.0[i].push_back(v);
            }
        }

        Ok(())
    }

    /// Returns an iterator over all the queues in the buffer.
    pub fn iter<'a>(&'a mut self) -> impl Iterator<Item = &'a VecDeque<Value>> {
        self.0.iter()
    }

    /// Returns a mutable iterator over all the queues in the buffer.
    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut VecDeque<Value>> {
        self.0.iter_mut()
    }
}

impl Index<usize> for QueuesBuffer {
    type Output = VecDeque<Value>;

    fn index(&self, index: usize) -> &VecDeque<Value> {
        &self.0[index]
    }
}

impl IndexMut<usize> for QueuesBuffer {
    fn index_mut(&mut self, index: usize) -> &mut VecDeque<Value> {
        &mut self.0[index]
    }
}

pub fn arr4<A, V, U, T>(xs: &[V]) -> ::ndarray::Array4<A>
where
    V: ::ndarray::FixedInitializer<Elem = U> + Clone,
    U: ::ndarray::FixedInitializer<Elem = T> + Clone,
    T: ::ndarray::FixedInitializer<Elem = A> + Clone,
    A: Clone,
{
    use ndarray::*;
    let mut xs = xs.to_vec();
    let dim = Ix4(xs.len(), V::len(), U::len(), T::len());
    let ptr = xs.as_mut_ptr();
    let len = xs.len();
    let cap = xs.capacity();
    let expand_len = len * V::len() * U::len() * T::len();
    ::std::mem::forget(xs);
    unsafe {
        let v = if ::std::mem::size_of::<A>() == 0 {
            Vec::from_raw_parts(ptr as *mut A, expand_len, expand_len)
        } else if V::len() == 0 || U::len() == 0 || T::len() == 0 {
            Vec::new()
        } else {
            let expand_cap = cap * V::len() * U::len() * T::len();
            Vec::from_raw_parts(ptr as *mut A, expand_len, expand_cap)
        };
        ArrayBase::from_shape_vec_unchecked(dim, v)
    }
}
