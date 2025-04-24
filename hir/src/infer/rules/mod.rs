//! A fluent interface for the analyser.
//!
//! This interface provides proxies for the different properties of tensors.
//! This allows inference rules to be stated in a clear, declarative fashion
//! inside the `rules` method of each operator.
//!
//! Take these rules for instance:
//! ```text
//! solver.equals(inputs.len(), 2);
//! solver.equals(inputs[0].datum_type, outputs[0].datum_type);
//! ```
//! Here, `inputs.len`, `inputs[0].datum_type` and `outputs[0].datum_type` don't
//! actually hold the values of the length and datum_types, but instead act as
//! declarative placeholders for these values.

#[macro_export]
macro_rules! wrap {
    ($($x:expr),*) => ({
        vec![$( $crate::infer::rules::expr::IntoExp::bex($x) ),*]
    });

    ($($x:expr,)*) => (wrap![$($x),*]);
}

use crate::infer::*;

mod cache;
pub mod expr;
mod path;
mod proxies;
mod solver;

pub use self::proxies::*;
pub use self::solver::Solver;

pub type InferenceResult = TractResult<()>;

pub trait InferenceRulesOp {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult;

    fn as_op(&self) -> &dyn Op;
    fn as_op_mut(&mut self) -> &mut dyn Op;

    #[allow(unused_variables)]
    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        bail!("Node {} can not be typed", node)
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1)
    }

    #[allow(unused_variables)]
    fn incorporate(
        &self,
        model: &InferenceModel,
        node: &InferenceNode,
    ) -> TractResult<Option<InferenceModelPatch>> {
        Ok(None)
    }
}

impl<O: InferenceRulesOp + Op> InferenceOp for O {
    fn infer_facts(
        &mut self,
        inputs: TVec<&InferenceFact>,
        outputs: TVec<&InferenceFact>,
        observed: TVec<&InferenceFact>,
    ) -> TractResult<(TVec<InferenceFact>, TVec<InferenceFact>, TVec<InferenceFact>)> {
        let inputs_proxy: TVec<TensorProxy> =
            (0..inputs.len()).map(|ix| TensorProxy::new(tvec!(0, ix as isize).into())).collect();
        let outputs_proxy: TVec<TensorProxy> =
            (0..outputs.len()).map(|ix| TensorProxy::new(tvec!(1, ix as isize).into())).collect();

        trace!("Building rules for {self:?}");
        let mut solver = Solver::default();
        self.rules(&mut solver, &inputs_proxy, &outputs_proxy)?;
        trace!("Applying rules for {self:?}");
        let (input, output) = solver.infer_facts((inputs, outputs))?;
        trace!("Solver done");
        Ok((input, output, observed.into_iter().cloned().collect()))
    }

    fn nboutputs(&self) -> TractResult<usize> {
        self.nboutputs()
    }

    fn observe_outlets(
        &self,
        _model: &InferenceModel,
        _node: &InferenceNode,
    ) -> TractResult<Vec<OutletId>> {
        Ok(vec![])
    }

    fn as_op(&self) -> &dyn Op {
        self.as_op()
    }

    fn as_op_mut(&mut self) -> &mut dyn Op {
        self.as_op_mut()
    }

    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        self.to_typed(source, node, target, mapping)
    }

    fn incorporate(
        &self,
        model: &InferenceModel,
        node: &InferenceNode,
    ) -> TractResult<Option<InferenceModelPatch>> {
        self.incorporate(model, node)
    }
}
