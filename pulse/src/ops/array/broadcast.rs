use crate::fact::StreamInfo;
use crate::internal::*;
use tract_pulse_opl::tract_core::ops::array::MultiBroadcastTo;

register_all!(MultiBroadcastTo: pulsify);

fn pulsify(
    op: &MultiBroadcastTo,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    if let Some(axis) = op.shape.iter().position(|dim| dim.symbols().contains(symbol)) {
        let full_dim = op.shape[axis].clone();
        let fact = PulsedFact {
            datum_type: _source.outlet_fact(node.inputs[0])?.datum_type,
            shape: op.shape.iter().map(|dim| dim.substitute(symbol, pulse)).collect::<TractResult<_>>()?,
            stream: Some(StreamInfo { axis, dim: full_dim, delay: 0 }),
        };
        let new_op = PulsedMultibroadcastTo { fact };
        target
            .wire_node(&node.name, new_op, &[mapping[&node.inputs[0]]])
            .map(Some)
    } else {
        Ok(None)
    }
}

/// Concat with pulse along concat axis
#[derive(Debug, Clone, Hash)]
struct PulsedMultibroadcastTo {
    fact: PulsedFact,
}

impl Op for PulsedMultibroadcastTo {
    fn name(&self) -> Cow<str> {
        "PulsedMultibroadcastTo".into()
    }

    op_as_typed_op!();
}

impl TypedOp for PulsedMultibroadcastTo {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(self.fact.to_pulse_fact().shape)))
    }
    as_op!();
}

impl EvalOp for PulsedMultibroadcastTo {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.to_typed().eval(inputs)
    }
}

impl PulsedOp for PulsedMultibroadcastTo {
    fn pulsed_output_facts(&self, _inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        Ok(tvec!(self.fact.clone()))
    }

    fn to_typed(&self) -> Box<dyn TypedOp> {
        Box::new(MultiBroadcastTo { shape: self.fact.to_pulse_fact().shape })
    }

    as_op!();
}
