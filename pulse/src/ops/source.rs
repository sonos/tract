use crate::fact::StreamFact;
use crate::internal::*;
use tract_core::ops::source::*;

register_all!(TypedSource: pulsify);

pub fn pulsify(
    _op: &TypedSource,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    _mapping: &HashMap<OutletId, OutletId>,
    stream_symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let fact = &node.outputs[0].fact;
    let pulsed_fact = if fact.shape.stream_info(stream_symbol).is_some() {
        PulsedFact::from_tensor_fact_pulse(fact, stream_symbol, pulse)?
    } else if source.input_outlets()?.iter().any(|o| {
        source
            .outlet_fact(*o)
            .map(|f| f.shape.stream_info(stream_symbol).is_some())
            .unwrap_or(false)
    }) {
        // This source has no streaming dim, but another model input does.
        // Treat it as a non-streaming (static) input carried through pulsification.
        PulsedFact { datum_type: fact.datum_type, shape: fact.shape.clone(), stream: None }
    } else {
        bail!("Can not pulse a tensor with no streaming dim ({})", stream_symbol)
    };
    let id = target.add_source(node.name.clone(), pulsed_fact)?;
    Ok(Some(tvec!(id)))
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PulsedSource(pub PulsedFact);

impl Op for PulsedSource {
    fn name(&self) -> StaticName {
        "PulsedSource".into()
    }
    not_a_typed_op!();
}

impl EvalOp for PulsedSource {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(&self, _session: &TurnState, node_id: usize) -> TractResult<Option<Box<dyn OpState>>> {
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
