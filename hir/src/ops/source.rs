use crate::infer::*;
use crate::internal::*;

use tract_core::ops::source::TypedSource;

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct Source;

impl Op for Source {
    fn name(&self) -> StaticName {
        "Source".into()
    }

    not_a_typed_op!();
}

impl EvalOp for Source {
    not_out_of_plan!();
    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(!inputs.is_empty(), "Input for node {} is missing", ctx.node_id);
        Ok(inputs)
    }
}

impl InferenceRulesOp for Source {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 0)?;
        check_output_arity(outputs, 1)?;
        Ok(())
    }

    as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        _mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Ok(fact) = TypedFact::try_from(&node.outputs[0].fact) {
            target.wire_node(&*node.name, TypedSource::new(fact), &[])
        } else {
            bail!(
                "Source node without a determined fact. Help: provide explicit input facts to your model."
            )
        }
    }
}
