use crate::internal::*;
use tract_core::ops::quant::DequantizeLinearF32;

submit_op_pulsifier!(DequantizeLinearF32, pulsify);

fn pulsify(
    op: &DequantizeLinearF32,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<TVec<OutletId>> {
    let input = mapping[&node.inputs[0]];
    target.wire_node(&*node.name, op.clone(), &[input])
}

impl PulsedOp for DequantizeLinearF32 {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.datum_type = f32::datum_type();
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
