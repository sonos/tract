//! Ops
use std::fmt;

use downcast_rs::Downcast;

use objekt;

#[macro_use]
pub mod macros;
#[macro_use]
pub mod binary;

pub mod axis;

pub mod array;
pub mod cast;
pub mod cnn;
pub mod downsample;
pub mod dummy;
pub mod identity;
pub mod konst;
pub mod logic;
pub mod math;
pub mod nn;
pub mod scan;
pub mod source;
pub mod unimpl;

pub use axis::{AxesInfo, AxisInfo};
pub use downsample::Downsample;

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
    Accurate,
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
        op: &dyn Op,
        inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>>;
}

pub trait StatelessOp: Op {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>>;
}

pub trait StatefullOp {
    #[allow(unused_variables)]
    fn state(
        &self,
        session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>>;
    fn as_stateless(&self) -> Option<&dyn StatelessOp> {
        None
    }
}

impl<O: StatelessOp + Clone> StatefullOp for O {
    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(None)
    }

    fn as_stateless(&self) -> Option<&dyn StatelessOp> {
        Some(self)
    }
}

pub trait Translate<TI1, O1, TI2, O2, Ctx>
where
    TI1: TensorInfo + Clone + 'static,
    TI2: TensorInfo + Clone + 'static,
    O1: fmt::Display + fmt::Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    O2: fmt::Display + fmt::Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn translate(
        &self,
        source: &ModelImpl<TI1, O1>,
        node: &BaseNode<TI1, O1>,
        target: &mut ModelImpl<TI2, O2>,
        mapping: &HashMap<OutletId, OutletId>,
        ctx: &Ctx,
    ) -> TractResult<TVec<OutletId>>;
}

/// A base operation
pub trait Op: fmt::Debug + objekt::Clone + Send + Sync + 'static + Downcast + StatefullOp {
    fn name(&self) -> Cow<str>;

    /// Early pass on inference model, after analyse, but before translation to
    /// typed network. Meant to deal with some framework idiosyncrasies that
    /// manifest with temporaries nodes that can run some form of inference but
    /// require refactoring the network before it can be evaluated.
    ///
    /// Called after succesful analyse, but before translating to typed model.
    fn incorporate(
        &self,
        _model: &InferenceModel,
        _node: &InferenceNode,
    ) -> TractResult<Option<InferenceModelPatch>> {
        Ok(None)
    }

    /// Declutter the op to the tract_core operator set as much as possible.
    fn declutter(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    /// Translate the op into the most efficient form possible for execution.
    ///
    /// This transformation is supposed to be final, no more pass are expected
    /// to be run on the codegen networks.
    fn codegen(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    /// Fuse op after codegen to deal with local optimisations.
    fn fuse(&self, _model: &TypedModel, _node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    /// Computes a cost hint of the operation.
    ///
    /// Each pair is a type of operation and a number per call on eval.
    fn cost(&self, _inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        Ok(tvec!())
    }

    /// Nested models, with label (for audit).
    fn nested_models(&self) -> Vec<(Cow<str>, &dyn Model)> {
        vec![]
    }

    /// The kind of accuracy check that should be performed on operation when
    /// testing them.
    fn validation(&self) -> Validation {
        Validation::Accurate
    }

    fn axes_info(&self, _model: &TypedModel, _node: &TypedNode) -> TractResult<AxesInfo> {
        Ok(tvec![].into())
    }

    /// Compare two ops.
    // Should this one be and Eq or PartialEq impl instead ?
    fn same_as(&self, _other: &dyn Op) -> bool {
        false
    }

    /// Short (one-line) strings giving hints on internal implementation or
    /// important configuration details to be displayed in dumps.
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![])
    }

    fn as_typed(&self) -> Option<&dyn TypedOp>;

    fn is_canonic(&self) -> bool {
        false
    }
}

pub trait TypedOp:
    Op + fmt::Debug + objekt::Clone + Send + Sync + 'static + Downcast + StatefullOp
{
    /// Reinterpret the TypedOp as an Op.
    fn as_op(&self) -> &dyn Op;

    /// Reinterpret the TypedOp as an Op, mutably.
    fn as_op_mut(&mut self) -> &mut dyn Op;

    /// Deduce output facts from input facts.
    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>>;

    /// Transforms the op in an equivalent one, discarding one dummy axis (of dim
    /// assumed to be 1).
    ///
    /// Returns None if the op can be kept as is.
    #[allow(unused_variables)]
    fn dispose_dummy_axis(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        axis: usize,
    ) -> TractResult<Option<Box<dyn TypedOp>>> {
        Ok(None)
    }

    /// Translate an op in a normalized network (no constants) to a pulsing
    /// form, if possible.
    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        _target: &mut PulsedModel,
        _mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        debug!("{:?}", node);
        bail!("Operator {} do not support pulsification", self.name())
    }

    /// Nested model multipliers, with label (for profiling).
    #[allow(unused_variables)]
    fn nested_model_multipliers(&self, inputs: &[&TypedTensorInfo]) -> Vec<(Cow<str>, f32)> {
        vec![]
    }
}

impl
    crate::ops::Translate<
        NormalizedTensorInfo,
        Box<dyn TypedOp>,
        crate::pulse::PulsedTensorFact,
        Box<dyn TypedOp>,
        usize,
    > for Box<dyn TypedOp>
{
    fn translate(
        &self,
        source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        ctx: &usize,
    ) -> TractResult<TVec<OutletId>> {
        self.pulsify(source, node, target, mapping, *ctx)
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
    /// The default implementation will call the private infer_facts method,
    /// which is usually implemented using the InferenceRulesOp trait. It will
    /// also try to eval() the op if its a StatelessOp and if the inputs are
    /// fully determined.
    ///
    /// Returns Err in case of an unrecoverable error during the inference,
    /// and the refined properties about the inputs and outputs otherwise.
    fn infer(
        &mut self,
        inputs: TVec<&TensorFact>,
        outputs: TVec<&TensorFact>,
        observed: TVec<&TensorFact>,
    ) -> TractResult<(TVec<TensorFact>, TVec<TensorFact>, TVec<TensorFact>)> {
        let (infered_inputs, infered_outputs, observed) =
            self.infer_facts(inputs, outputs, observed)?;

        if let Some(stateless) = self.as_stateless() {
            if infered_inputs.iter().all(|i| i.value.is_concrete()) {
                let input_values = infered_inputs
                    .iter()
                    .map(|i| i.value.concretize().unwrap().clone().into())
                    .collect(); // checked
                match stateless.eval(input_values) {
                    Ok(values) => {
                        let output_values =
                            values.into_iter().map(|t| t.into()).collect::<TVec<_>>();
                        return Ok((infered_inputs, output_values, observed));
                    }
                    Err(e) => match e {
                        TractError(TractErrorKind::StreamTensor, _) => (),
                        e => return Err(e),
                    },
                }
            }
        }

        return Ok((infered_inputs, infered_outputs, observed));
    }

    /// Allow an op to specify a supplementary list of outlets facts that
    /// will trigger inference again.
    fn observe_outlets(
        &self,
        _model: &InferenceModel,
        _node: &InferenceNode,
    ) -> TractResult<Vec<OutletId>> {
        Ok(vec![])
    }

    /// Infer properties about inputs and output tensors. This method does not
    /// need to deal with the "trivial" stateless op with fully determined
    /// inputs cases.
    ///
    /// Most of the time, it is implemented using InferenceRulesOp.
    fn infer_facts(
        &mut self,
        inputs: TVec<&TensorFact>,
        outputs: TVec<&TensorFact>,
        observed: TVec<&TensorFact>,
    ) -> TractResult<(TVec<TensorFact>, TVec<TensorFact>, TVec<TensorFact>)>;

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1)
    }

    /// Reinterpret the InferenceOp as an Op.
    fn as_op(&self) -> &dyn Op;

    /// Reinterpret the InferenceOp as an Op, mutably.
    fn as_op_mut(&mut self) -> &mut dyn Op;

    /// Called during translation to TypedModel.
    fn to_typed(
        &self,
        _source: &InferenceModel,
        _node: &InferenceNode,
        _target: &mut TypedModel,
        _mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        bail!("Operator can not be made a TypedOp.")
    }
}

impl crate::ops::Translate<TensorFact, Box<dyn InferenceOp>, TypedTensorInfo, Box<dyn TypedOp>, ()>
    for Box<dyn InferenceOp>
{
    fn translate(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _ctx: &(),
    ) -> TractResult<TVec<OutletId>> {
        self.to_typed(source, node, target, mapping)
    }
}

impl_downcast!(Op);

clone_trait_object!(Op);
clone_trait_object!(StatelessOp);
clone_trait_object!(TypedOp);
clone_trait_object!(InferenceOp);

impl<O: Op> From<O> for Box<dyn Op> {
    fn from(it: O) -> Box<dyn Op> {
        Box::new(it)
    }
}

impl<O: InferenceOp> From<O> for Box<dyn InferenceOp> {
    fn from(it: O) -> Box<dyn InferenceOp> {
        Box::new(it)
    }
}

impl<O: TypedOp> From<O> for Box<dyn TypedOp> {
    fn from(it: O) -> Box<dyn TypedOp> {
        Box::new(it)
    }
}

impl AsRef<dyn Op> for dyn InferenceOp {
    fn as_ref(&self) -> &dyn Op {
        self.as_op()
    }
}

impl AsRef<dyn Op> for Box<dyn InferenceOp> {
    fn as_ref(&self) -> &dyn Op {
        self.as_op()
    }
}

impl AsMut<dyn Op> for dyn InferenceOp {
    fn as_mut(&mut self) -> &mut dyn Op {
        self.as_op_mut()
    }
}

impl AsMut<dyn Op> for Box<dyn InferenceOp> {
    fn as_mut(&mut self) -> &mut dyn Op {
        self.as_op_mut()
    }
}

impl AsRef<dyn Op> for dyn TypedOp {
    fn as_ref(&self) -> &dyn Op {
        self.as_op()
    }
}

impl AsRef<dyn Op> for Box<dyn TypedOp> {
    fn as_ref(&self) -> &dyn Op {
        self.as_op()
    }
}

impl AsMut<dyn Op> for dyn TypedOp {
    fn as_mut(&mut self) -> &mut dyn Op {
        self.as_op_mut()
    }
}

impl AsMut<dyn Op> for Box<dyn TypedOp> {
    fn as_mut(&mut self) -> &mut dyn Op {
        self.as_op_mut()
    }
}

impl std::fmt::Display for Box<dyn Op> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.name())
    }
}

impl std::fmt::Display for Box<dyn InferenceOp> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.name())
    }
}

impl std::fmt::Display for Box<dyn TypedOp> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.name())
    }
}
