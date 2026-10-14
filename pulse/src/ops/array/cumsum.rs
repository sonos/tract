use crate::internal::*;
use tract_core::ops::array::CumSum;
use tract_pulse_opl::ops::PulsedCumSum;

register_all!(CumSum: pulsify);

fn pulsify(
    op: &CumSum,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?.clone();
    rule_if_some!(stream = fact.stream.as_ref());
    // Off the streaming axis, positions along the stream are independent: returning
    // None hands the op to the generic PulseWrappingOp, which pulses it per frame.
    rule_if!(stream.axis == op.axis);
    ensure!(
        !op.reverse,
        "A reverse CumSum along the streaming axis is not causal, can not pulsify"
    );
    target
        .wire_node(
            &node.name,
            PulsedCumSum { axis: op.axis, exclusive: op.exclusive, skip: stream.delay },
            &[input],
        )
        .map(Some)
}

impl PulsedOp for PulsedCumSum {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
