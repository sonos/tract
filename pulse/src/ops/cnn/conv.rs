use crate::internal::*;
use tract_core::ops::cnn::ConvUnary;

submit_op_pulsifier!(ConvUnary, pulsify);

fn pulsify(
    op: &ConvUnary,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<TVec<OutletId>> {
    fn zero<D: Datum>() -> Tensor {
        tensor0(D::default())
    }
    let fact = target.outlet_fact(mapping[&node.inputs[0]])?;
    let zero = dispatch_numbers!(zero(fact.datum_type)());
    let (wire, pool_spec) = super::pools::pulsify(&op.pool_spec, source, node, target, mapping, Some(zero))?;
    target.wire_node(&node.name, ConvUnary { pool_spec, ..op.clone() }, &[wire])
}

impl PulsedOp for ConvUnary {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        super::pools::pulsed_output_facts(&self.pool_spec, inputs)
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
