use crate::internal::*;
use tract_core::ops::element_wise::ElementWiseOp;

submit_op_pulsifier!(ElementWiseOp, pulsify);

fn pulsify(
    op: &ElementWiseOp,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<TVec<OutletId>> {
    let input = mapping[&node.inputs[0]];
    target.wire_node(&*node.name, op.clone(), &[input])
}

impl PulsedOp for ElementWiseOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        if let Some(dt) = self.0.output_type(fact.datum_type) {
            fact.datum_type = dt;
        }
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
