use crate::internal::*;
use crate::infer::*;

use crate::ops::source::{ SourceState, TypedSource };

#[derive(Debug, Clone, new)]
pub struct Source;

impl Op for Source {
    fn name(&self) -> Cow<str> {
        "Source".into()
    }

    not_a_typed_op!();
    not_a_pulsed_op!();
}

impl StatefullOp for Source {
    fn state(
        &self,
        _session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(SourceState(node_id))))
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
        check_input_arity(&inputs, 0)?;
        check_output_arity(&outputs, 1)?;
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
        use std::convert::TryFrom;
        if let Ok(fact) = TypedFact::try_from(&node.outputs[0].fact) {
            target.wire_node(&*node.name, TypedSource::new(fact), &[])
        } else {
            bail!("Output type not determined")
        }
    }
}
