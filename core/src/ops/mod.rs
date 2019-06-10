//! Ops
use std::fmt;

use downcast_rs::Downcast;

use objekt;

#[macro_use]
pub mod macros;

pub mod array;
pub mod cast;
pub mod cnn;
pub mod identity;
pub mod konst;
pub mod logic;
pub mod math;
pub mod nn;
pub mod source;
pub mod unimpl;

pub use source::Source;

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

/// Level of precision to be expected in implementations comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Validation {
    /// Output is random
    Random,
    /// Implementation may induce rounding errors
    Rounding,
    /// Implementation must be accurate
    Accurate
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cost {
    FMA(DatumType),
}

use crate::internal::*;

pub trait OpState: fmt::Debug + Send + objekt::Clone {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &Op,
        inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>>;
}

pub trait StatelessOp: Op {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>>;
}

pub trait StatefullOp {
    fn state(&self, _session: &mut SessionState, node_id: usize) -> TractResult<Option<Box<OpState>>>;
    fn as_stateless(&self) -> Option<&StatelessOp> {
        None
    }
}

impl<O: StatelessOp + Clone> StatefullOp for O {
    fn state(&self, _session: &mut SessionState, _node_id: usize) -> TractResult<Option<Box<OpState>>> {
        Ok(None)
    }

    fn as_stateless(&self) -> Option<&StatelessOp> {
        Some(self)
    }
}

/// A base operation
pub trait Op:
    fmt::Debug + objekt::Clone + Send + Sync + 'static + Downcast + StatefullOp
{
    fn name(&self) -> Cow<str>;

    fn declutter(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        _node: &NormalizedNode,
        _target: &mut PulsedModel,
        _mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        bail!("Operator {} do not support pulsification", self.name())
    }

    fn codegen(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    fn cost(&self, _inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        Ok(tvec!())
    }

    fn validation(&self) -> Validation {
        Validation::Accurate
    }

    fn same_as(&self, _other: &Op) -> bool {
        false
    }

    fn info(&self) -> TractResult<Option<String>> {
        Ok(None)
    }
}


/// An operation with tensor type inference
pub trait InferenceOp:
    Op + fmt::Debug + objekt::Clone + Send + Sync + 'static + Downcast + StatefullOp
{
    /// Infers properties about the input and output tensors.
    ///
    /// The `inputs` and `outputs` arguments correspond to properties about
    /// the input and output tensors that are already known.
    ///
    /// Returns Err in case of an unrecoverable error during the inference,
    /// and the refined properties about the inputs and outputs otherwise.
    fn infer(
        &mut self,
        inputs: TVec<&TensorFact>,
        outputs: TVec<&TensorFact>,
    ) -> TractResult<(TVec<TensorFact>, TVec<TensorFact>)> {
        let (infered_inputs, infered_outputs) = self.infer_facts(inputs, outputs)?;

        if self.as_op().downcast_ref::<crate::ops::source::Source>().is_some() {
            return Ok((infered_inputs, infered_outputs))
        }

        if let Some(stateless) = self.as_stateless() {
            if infered_inputs.iter().all(|i| i.value.is_concrete()) {
                let input_values = infered_inputs
                    .iter()
                    .map(|i| i.value.concretize().unwrap().clone().into())
                    .collect(); // checked
                let output_values = stateless
                    .eval(input_values)?
                    .into_iter()
                    .map(|t| t.into())
                    .collect::<TVec<_>>();
                return Ok((infered_inputs, output_values));
            }
        }

        return Ok((infered_inputs, infered_outputs))
    }

    fn infer_facts(
        &mut self,
        inputs: TVec<&TensorFact>,
        outputs: TVec<&TensorFact>,
    ) -> TractResult<(TVec<TensorFact>, TVec<TensorFact>)>;

    fn as_op(&self) -> &Op;
    fn as_op_mut(&mut self) -> &mut Op;
}

impl_downcast!(Op);

clone_trait_object!(Op);
clone_trait_object!(StatelessOp);
clone_trait_object!(InferenceOp);

impl<O: Op> From<O> for Box<Op> {
    fn from(it: O) -> Box<Op> {
        Box::new(it)
    }
}

impl<O: InferenceOp> From<O> for Box<InferenceOp> {
    fn from(it: O) -> Box<InferenceOp> {
        Box::new(it)
    }
}

impl From<Box<InferenceOp>> for Box<Op> {
    fn from(it: Box<InferenceOp>) -> Box<Op> {
        objekt::clone_box(it.as_op())
    }
}

impl AsRef<Op> for InferenceOp {
    fn as_ref(&self) -> &Op {
        self.as_op()
    }
}

impl AsRef<Op> for Box<InferenceOp> {
    fn as_ref(&self) -> &Op {
        self.as_op()
    }
}

impl AsMut<Op> for InferenceOp {
    fn as_mut(&mut self) -> &mut Op {
        self.as_op_mut()
    }
}

impl AsMut<Op> for Box<InferenceOp> {
    fn as_mut(&mut self) -> &mut Op {
        self.as_op_mut()
    }
}

impl std::fmt::Display for Box<Op> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.name())
    }
}

impl std::fmt::Display for Box<InferenceOp> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.name())
    }
}
