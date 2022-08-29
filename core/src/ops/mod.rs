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

use crate::internal::*;
use crate::optim::OptimizerSession;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Cost {
    Div(DatumType),
    FMA(DatumType),
    Buffer(DatumType),
    Params(DatumType),
}

impl Cost {
    pub fn is_compute(&self) -> bool {
        use Cost::*;
        match self {
            FMA(_) | Div(_) => true,
            Buffer(_) | Params(_) => false,
        }
    }
}

pub trait OpState: fmt::Debug + Send + dyn_clone::DynClone {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>>;
}
dyn_clone::clone_trait_object!(OpState);

pub trait EvalOp {
    #[allow(unused_variables)]
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        bail!("stateless evaluation not implemented")
    }

    #[allow(unused_variables)]
    fn state(
        &self,
        session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(None)
    }

    fn is_stateless(&self) -> bool;
}

/// A base operation
pub trait Op:
    fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast + EvalOp + DynHash
{
    /// Vector of short strings defining what families the op belongs too.
    /// tract-core defines "core", "mir", "lir".
    fn op_families(&self) -> &'static [&'static str];

    fn name(&self) -> Cow<str>;

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
}

pub trait TypedOp:
    Op + fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast + EvalOp + DynHash
{
    /// Reinterpret the TypedOp as an Op.
    fn as_op(&self) -> &dyn Op;

    /// Reinterpret the TypedOp as an Op, mutably.
    fn as_op_mut(&mut self) -> &mut dyn Op;

    /// Deduce output facts from input facts.
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>>;

    #[allow(unused_variables)]
    fn invariants(&self, inputs: &[&TypedFact], outputs: &[&TypedFact]) -> TractResult<Invariants> {
        Ok(Invariants::default())
    }

    /// Fuse op after codegen to deal with local optimisations.
    fn fuse(&self, _model: &TypedModel, _node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    /// Declutter the op to the tract_core operator set as much as possible.
    #[allow(unused_variables)]
    fn declutter_with_session(
        &self,
        session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        self.declutter(model, node)
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
    #[allow(clippy::too_many_arguments)]
    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        slice_suffix: &str,
        output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<(OutletId, bool)>> {
        let outlet = OutletId::new(node.id, output_slot);
        let output = model.outlet_fact(outlet)?;
        if start == 0 && Some(end) == output.shape[axis].to_usize().ok() {
            Ok(Some((patch.tap_model(model, outlet)?, true)))
        } else {
            let wire = patch.tap_model(model, outlet)?;
            let wire = patch.wire_node(
                &format!("{}-{}.{}", node.name, output_slot, slice_suffix),
                crate::ops::array::Slice { start: start.to_dim(), axis, end: end.to_dim() },
                &[wire],
            )?[0];
            Ok(Some((wire, false)))
        }
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

    /// Transform the op into by providing a value to one or more symbols.
    #[allow(unused_variables)]
    fn concretize_dims(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();
        target.wire_node(&node.name, node.op.clone(), &inputs)
    }

    /// Translate the op into the most efficient form possible for execution.
    ///
    /// This transformation is supposed to be final, no more pass are expected
    /// to be run on the codegen networks.
    #[allow(unused_variables)]
    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    /// Nested model multipliers, with label (for profiling).
    #[allow(unused_variables)]
    fn nested_model_multipliers(&self, inputs: &[&TypedFact]) -> Vec<(Cow<str>, f64)> {
        vec![]
    }
}

impl_downcast!(Op);

dyn_clone::clone_trait_object!(Op);
dyn_clone::clone_trait_object!(TypedOp);

impl<O: Op> From<O> for Box<dyn Op> {
    fn from(it: O) -> Box<dyn Op> {
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

impl std::fmt::Display for Box<dyn TypedOp> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.name())
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum AttrOrInput {
    Attr(Arc<Tensor>),
    Input(usize),
}

impl AttrOrInput {
    fn tensor<'t>(&'t self, inputs: &'t [Arc<Tensor>]) -> &'t Arc<Tensor> {
        match self {
            AttrOrInput::Attr(t) => t,
            AttrOrInput::Input(slot) => &inputs[*slot],
        }
    }

    pub fn as_static(&self) -> Option<&Arc<Tensor>> {
        match self {
            AttrOrInput::Attr(t) => Some(t),
            AttrOrInput::Input(_) => None,
        }
    }
}

impl From<usize> for AttrOrInput {
    fn from(input: usize) -> Self {
        AttrOrInput::Input(input)
    }
}

impl From<Tensor> for AttrOrInput {
    fn from(t: Tensor) -> Self {
        Arc::new(t).into()
    }
}

impl From<Arc<Tensor>> for AttrOrInput {
    fn from(t: Arc<Tensor>) -> Self {
        AttrOrInput::Attr(t)
    }
}

impl<'a> From<&'a Arc<Tensor>> for AttrOrInput {
    fn from(t: &'a Arc<Tensor>) -> Self {
        AttrOrInput::Attr(t.clone())
    }
}
