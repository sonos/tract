use crate::internal::*;
use tract_core::ops::source::*;

submit_op_pulsifier!(TypedSource, pulsify);

fn pulsify(
    _op: &TypedSource,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    _mapping: &HashMap<OutletId, OutletId>,
    pulse: usize,
) -> TractResult<TVec<OutletId>> {
    let pulsed_fact = PulsedFact::from_tensor_fact_pulse(&node.outputs[0].fact, pulse)?;
    let id = target.add_source(node.name.clone(), pulsed_fact)?;
    Ok(tvec!(id))
}

#[derive(Debug, Clone, Hash)]
pub struct PulsedSource(pub PulsedFact);

tract_data::impl_dyn_hash!(PulsedSource);

impl Op for PulsedSource {
    fn name(&self) -> Cow<str> {
        "PulsedSource".into()
    }
    op_pulse!();
    not_a_typed_op!();
}

impl EvalOp for PulsedSource {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(SourceState(node_id))))
    }
}

impl PulsedOp for PulsedSource {
    fn pulsed_output_facts(&self, _inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        Ok(tvec!(self.0.clone()))
    }

    fn to_typed(&self) -> Box<dyn TypedOp> {
        Box::new(TypedSource::new(TypedFact::dt_shape(self.0.datum_type, &*self.0.shape).unwrap()))
    }

    as_op!();
}
