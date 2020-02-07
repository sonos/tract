//! Ops
use std::fmt;

use downcast_rs::Downcast;

use dyn_clone;

#[macro_use]
pub mod macros;
#[macro_use]
pub mod element_wise;
#[macro_use]
pub mod binary;

pub mod invariants;

pub mod array;
pub mod cast;
pub mod change_axes;
pub mod cnn;
pub mod downsample;
pub mod dummy;
pub mod identity;
pub mod konst;
pub mod logic;
pub mod math;
pub mod matmul;
pub mod nn;
pub mod quant;
pub mod scan;
pub mod source;
pub mod unimpl;

pub use downsample::Downsample;
pub use invariants::*;

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
    Div(DatumType),
    FMA(DatumType),
    Buffer(DatumType),
}

use crate::internal::*;

pub trait OpState: fmt::Debug + Send + dyn_clone::DynClone {
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

/// A base operation
pub trait Op:
    fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast + StatefullOp
{
    fn name(&self) -> Cow<str>;

    /// Nested models, with label (for audit).
    fn nested_models(&self) -> Vec<(Cow<str>, &dyn Model, Vec<String>, Vec<String>)> {
        vec![]
    }

    /// The kind of accuracy check that should be performed on operation when
    /// testing them.
    fn validation(&self) -> Validation {
        Validation::Accurate
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

    fn as_pulsed(&self) -> Option<&dyn PulsedOp> {
        None
    }

    fn is_canonic(&self) -> bool {
        false
    }
}

pub trait TypedOp:
    Op + fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast + StatefullOp
{
    /// Reinterpret the TypedOp as an Op.
    fn as_op(&self) -> &dyn Op;

    /// Reinterpret the TypedOp as an Op, mutably.
    fn as_op_mut(&mut self) -> &mut dyn Op;

    /// Deduce output facts from input facts.
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>>;

    #[allow(unused_variables)]
    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        Ok(Invariants::default())
    }

    /// Fuse op after codegen to deal with local optimisations.
    fn fuse(&self, _model: &TypedModel, _node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    /// Declutter the op to the tract_core operator set as much as possible.
    #[allow(unused_variables)]
    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    /// Computes a cost hint of the operation.
    ///
    /// Each pair is a type of operation and a number per call on eval.
    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        Ok(tvec!())
    }

    #[allow(unused_variables)]
    fn suggested_axis_changes(&self) -> TractResult<TVec<(InOut, AxisOp)>> {
        Ok(tvec!())
    }

    #[allow(unused_variables)]
    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        Ok(None)
    }

    #[allow(unused_variables)]
    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<OutletId>> {
        let outlet = OutletId::new(node.id, output_slot);
        let output = model.outlet_fact(outlet)?;
        if start == 0 && Some(end as i32) == output.shape.dim(axis).to_integer().ok() {
            return Ok(Some(patch.tap_model(model, outlet)?));
        }
        Ok(None)
    }

    /// Transforms the op in an equivalent one, operating on dt (i8 or u8).
    ///
    /// Returns None if the op can not be translated.
    #[allow(unused_variables)]
    fn quantize(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        dt: DatumType,
        scale: f32,
        zero_point: i32,
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

    /// Nested model multipliers, with label (for profiling).
    #[allow(unused_variables)]
    fn nested_model_multipliers(&self, inputs: &[&TypedFact]) -> Vec<(Cow<str>, f32)> {
        vec![]
    }
}

pub trait PulsedOp:
    Op + fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast + StatefullOp
{
    /// Reinterpret the PulsedOp as an Op.
    fn as_op(&self) -> &dyn Op;

    /// Reinterpret the PulsedOp as an Op, mutably.
    fn as_op_mut(&mut self) -> &mut dyn Op;

    /// Reinterpret the PulsedOp as an TypedOp.
    fn to_typed(&self) -> Box<dyn TypedOp>;

    /// Deduce output facts from input facts.
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>>;
}

/// An operation with tensor type inference
pub trait InferenceOp:
    Op + fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast + StatefullOp
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
        inputs: TVec<&InferenceFact>,
        outputs: TVec<&InferenceFact>,
        observed: TVec<&InferenceFact>,
    ) -> TractResult<(TVec<InferenceFact>, TVec<InferenceFact>, TVec<InferenceFact>)> {
        let (infered_inputs, infered_outputs, observed) =
            self.infer_facts(inputs, outputs, observed).chain_err(|| "Infering facts")?;

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
                        e => return Err(e).chain_err(|| "Eager eval"),
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
        inputs: TVec<&InferenceFact>,
        outputs: TVec<&InferenceFact>,
        observed: TVec<&InferenceFact>,
    ) -> TractResult<(TVec<InferenceFact>, TVec<InferenceFact>, TVec<InferenceFact>)>;

    /// Early pass on inference model, after analyse, but before translation to
    /// typed network. Meant to deal with some framework idiosyncrasies that
    /// manifest with temporaries nodes that can run some form of inference but
    /// require refactoring the network before it can be evaluated.
    ///
    /// Called after succesful analyse, but before translating to typed model.
    #[allow(unused_variables)]
    fn incorporate(
        &self,
        model: &InferenceModel,
        node: &InferenceNode,
    ) -> TractResult<Option<InferenceModelPatch>> {
        Ok(None)
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1)
    }

    /// Reinterpret the InferenceOp as an Op.
    fn as_op(&self) -> &dyn Op;

    /// Reinterpret the InferenceOp as an Op, mutably.
    fn as_op_mut(&mut self) -> &mut dyn Op;

    /// Called during translation to TypedModel.
    #[allow(unused_variables)]
    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        bail!("Operator can not be made a TypedOp.")
    }
}

impl_downcast!(Op);

dyn_clone::clone_trait_object!(Op);
dyn_clone::clone_trait_object!(StatelessOp);
dyn_clone::clone_trait_object!(TypedOp);
dyn_clone::clone_trait_object!(InferenceOp);
dyn_clone::clone_trait_object!(PulsedOp);

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

impl<'a> From<&'a Box<dyn TypedOp>> for Box<dyn TypedOp> {
    fn from(it: &'a Box<dyn TypedOp>) -> Box<dyn TypedOp> {
        it.clone()
    }
}

impl<O: PulsedOp> From<O> for Box<dyn PulsedOp> {
    fn from(it: O) -> Box<dyn PulsedOp> {
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

impl AsMut<dyn Op> for Box<dyn PulsedOp> {
    fn as_mut(&mut self) -> &mut dyn Op {
        self.as_op_mut()
    }
}

impl AsRef<dyn Op> for dyn PulsedOp {
    fn as_ref(&self) -> &dyn Op {
        self.as_op()
    }
}

impl AsRef<dyn Op> for Box<dyn PulsedOp> {
    fn as_ref(&self) -> &dyn Op {
        self.as_op()
    }
}

impl AsMut<dyn Op> for dyn PulsedOp {
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

impl std::fmt::Display for Box<dyn PulsedOp> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.name())
    }
}
