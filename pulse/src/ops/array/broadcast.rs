use crate::internal::*;
use crate::model::NonPulsingWrappingOp;
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
    let new_op = MultiBroadcastTo {
        shape: op.shape.iter().map(|d| d.substitute(symbol, pulse)).collect::<TVec<_>>().into(),
    };
    target
        .wire_node(&node.name, NonPulsingWrappingOp(Box::new(new_op)), &[mapping[&node.inputs[0]]])
        .map(Some)
}
