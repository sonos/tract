//! TensorFlow Ops
use std::fmt::Debug;

use downcast_rs::Downcast;

use analyser::prelude::*;
use model::TVec;

use objekt;

#[macro_use]
mod macros;

pub mod array;
#[cfg(features = "image_ops")]
pub mod image;
pub mod identity;
pub mod konst;
pub mod logic;
pub mod math;
pub mod nn;
pub mod sink;
pub mod source;
pub mod unimpl;

mod types;

pub mod prelude {
    pub use super::{InferenceOp, Op, ReducedOpRewire};
    pub use ops::types::Value;
    pub use analyser::types::*;
    pub use streaming::types::{OpBuffer, QueuesBuffer};
    pub use streaming::values::{StepValue, Stream, StreamInfo};
    pub use dim::{TDim, DimLike, ToDim};
    pub use model::TVec;
    pub use std::collections::HashMap;
    pub use std::marker::PhantomData;
    pub use tensor::{Datum, DatumType, Tensor};
    pub use tensor::arr4;
    pub use TfdResult;
}

use TfdResult;
use self::types::{ Value};
pub use streaming::types::{OpBuffer, QueuesBuffer, EmptyBuffer};
pub use streaming::values::StepValue;

/// A Tensorflow operation.
impl_downcast!(Op);
pub trait Op: Debug + objekt::Clone + Send + Sync + 'static + InferenceOp + Downcast {
    fn name(&self) -> &str;

    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Value>) -> TfdResult<TVec<Value>>;

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
    ) -> TfdResult<Option<TVec<Value>>> {
        bail!("Streaming is not available for operator {:?}", self)
    }

    /// Infers properties about the input and output tensors.
    ///
    /// The `inputs` and `outputs` arguments correspond to properties about
    /// the input and output tensors that are already known.
    ///
    /// Returns Err in case of an unrecoverable error during the inference,
    /// and the refined properties about the inputs and outputs otherwise.
    fn infer(
        &self,
        inputs: TVec<TensorFact>,
        outputs: TVec<TensorFact>,
    ) -> TfdResult<(TVec<TensorFact>, TVec<TensorFact>)> {
        let (infered_inputs, infered_outputs) = self.infer_facts(inputs, outputs)?;

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

    fn reduce(
        &self,
        _inputs: TVec<&TensorFact>,
        _outputs: TVec<&TensorFact>,
    ) -> TfdResult<Option<ReducedOpRewire>> {
        Ok(None)
    }

    fn const_value(&self) -> Option<Value> {
        None
    }

    fn rounding_errors(&self) -> bool {
        false
    }

    fn noutputs(&self) -> usize {
        1
    }
}

pub trait InferenceOp {
    fn infer_facts(
        &self,
        inputs: TVec<TensorFact>,
        outputs: TVec<TensorFact>,
    ) -> TfdResult<(TVec<TensorFact>, TVec<TensorFact>)>;
}

clone_trait_object!(Op);

#[derive(Clone, Debug, new)]
pub struct ReducedOpRewire {
    pub new_op: Box<Op>,
    pub rewired: TVec<usize>,
}

