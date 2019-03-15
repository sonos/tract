//! oharedTensorFlow Ops
use std::fmt::Debug;

use downcast_rs::Downcast;

use objekt;

#[macro_use]
mod macros;

pub mod array;
pub mod cast;
pub mod identity;
#[cfg(features = "image_ops")]
pub mod image;
pub mod konst;
pub mod logic;
pub mod math;
pub mod nn;
pub mod source;
pub mod unimpl;

pub mod prelude {
    pub use super::{InferenceOp, Op, OpState, StatefullOp, StatelessOp};
    pub use crate::analyser::rules::expr::{IntoExp, ToDimExp};
    pub use crate::analyser::rules::{InferenceResult, InferenceRulesOp, Solver, TensorProxy};
    pub use crate::analyser::types::TypeFact;
    pub use crate::analyser::types::*;
    pub use crate::datum::{Datum, DatumType};
    pub use crate::dim::{DimLike, TDim, ToDim};
    pub use crate::framework::Framework;
    pub use crate::model::{InletId, Model, Node, OutletId, TVec};
    pub use crate::model::{NormalizedModel, NormalizedNode, TypedModel, TypedNode};
    pub use crate::model::{NormalizedModelPatch, TypedModelPatch};
    pub use crate::model::{NormalizedTensorInfo, TensorInfo, TypedTensorInfo, ShapeInfo, StreamInfo};
    pub use crate::model::dsl::*;
    pub use crate::pulse::{PulsedTensorFact, PulsifiedOp};
    pub use crate::tensor::{arr4, SharedTensor, Tensor};
    pub use crate::ToTract;
    pub use crate::TractResult;
    pub use std::borrow::Cow;
    pub use std::collections::HashMap;
    pub use std::marker::PhantomData;
    pub use tract_linalg::f16::f16;

    pub fn check_input_arity(inputs: &[TensorProxy], expected: usize) -> TractResult<()> {
        if inputs.len() != expected {
            bail!("Wrong input number. Rules expect {}, node has {}.", expected, inputs.len())
        } else {
            Ok(())
        }
    }

    pub fn check_output_arity(outputs: &[TensorProxy], expected: usize) -> TractResult<()> {
        if outputs.len() != expected {
            bail!("Wrong output number. Rules expect {}, node has {}.", expected, outputs.len())
        } else {
            Ok(())
        }
    }
}

use self::prelude::*;

pub trait OpState: Debug + Send + objekt::Clone {
    fn eval(&mut self, op: &Op, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>>;
}

pub trait StatelessOp {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>>;
}

pub trait StatefullOp {
    fn state(&self) -> TractResult<Option<Box<OpState>>>;
    fn as_stateless(&self) -> Option<&StatelessOp> {
        None
    }
}

impl<O: StatelessOp + Clone> StatefullOp for O {
    fn state(&self) -> TractResult<Option<Box<OpState>>> {
        Ok(None)
    }

    fn as_stateless(&self) -> Option<&StatelessOp> {
        Some(self)
    }
}

/// A SharedTensor operation.
impl_downcast!(Op);
pub trait Op:
    Debug + objekt::Clone + Send + Sync + 'static + InferenceOp + Downcast + StatefullOp
{
    fn name(&self) -> Cow<str>;

    /// Infers properties about the input and output tensors.
    ///
    /// The `inputs` and `outputs` arguments correspond to properties about
    /// the input and output tensors that are already known.
    ///
    /// Returns Err in case of an unrecoverable error during the inference,
    /// and the refined properties about the inputs and outputs otherwise.
    fn infer(
        &self,
        inputs: TVec<&TensorFact>,
        outputs: TVec<&TensorFact>,
    ) -> TractResult<(TVec<TensorFact>, TVec<TensorFact>)> {
        let (infered_inputs, infered_outputs) = self.infer_facts(inputs, outputs)?;

        if let Some(stateless) = self.as_stateless() {
            if infered_inputs.iter().all(|i| i.value.is_concrete()) {
                let input_values = infered_inputs
                    .iter()
                    .map(|i| i.value.concretize().unwrap().clone().into())
                    .collect(); // checked
                let output_value = stateless.eval(input_values)?.pop().unwrap();
                return Ok((infered_inputs, tvec![output_value.into(),]));
            }
        }

        Ok((infered_inputs, infered_outputs))
    }

    fn normalize(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    fn pulsify(
        &self,
        _inputs: TVec<&PulsedTensorFact>,
    ) -> TractResult<Vec<crate::pulse::PulsifiedOp>> {
        bail!("Operator {} do not support pulsification", self.name())
    }

    fn codegen(
        &self,
        _model: &NormalizedModel,
        _node: &NormalizedNode,
    ) -> TractResult<Option<NormalizedModelPatch>> {
        Ok(None)
    }

    fn const_value(&self) -> Option<SharedTensor> {
        None
    }

    fn rounding_errors(&self) -> bool {
        false
    }

    fn noutputs(&self) -> usize {
        1
    }

    fn same_as(&self, _other: &Op) -> bool {
        false
    }

    fn info(&self) -> TractResult<Option<String>> {
        Ok(None)
    }
}

pub trait InferenceOp {
    fn infer_facts(
        &self,
        inputs: TVec<&TensorFact>,
        outputs: TVec<&TensorFact>,
    ) -> TractResult<(TVec<TensorFact>, TVec<TensorFact>)>;
}

clone_trait_object!(Op);

impl<O: Op> From<O> for Box<Op> {
    fn from(it: O) -> Box<Op> {
        Box::new(it)
    }
}
