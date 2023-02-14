use crate::internal::*;
use tract_core::ops::source::*;

register_all!(TypedSource: pulsify);

pub fn pulsify(
    _op: &TypedSource,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    _mapping: &HashMap<OutletId, OutletId>,
    stream_symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let pulsed_fact = PulsedFact::from_tensor_fact_pulse(&node.outputs[0].fact, stream_symbol, pulse)?;
    let id = target.add_source(node.name.clone(), pulsed_fact)?;
    Ok(Some(tvec!(id)))
}

#[derive(Debug, Clone, Hash)]
pub struct PulsedSource(pub PulsedFact);



impl Op for PulsedSource {
    fn name(&self) -> Cow<str> {
        "PulsedSource".into()
    }
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
        Box::new(TypedSource::new(self.0.datum_type.fact(self.0.shape.clone())))
    }

    as_op!();
}
