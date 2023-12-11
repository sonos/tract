use crate::internal::*;
use crate::ops::cnn::pools::pulsify_pooled_input;
use tract_core::ops::cnn::Conv;

register_all!(Conv: pulsify);

fn pulsify(
    op: &Conv,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let fact = target.outlet_fact(mapping[&node.inputs[0]])?;
    let zero = Tensor::zero_scalar_dt(fact.datum_type)?;
    if let Some((wire, pool_spec)) =
        pulsify_pooled_input(&op.pool_spec, source, node, target, mapping, Some(zero))?
    {
        let mut wires: TVec<_> = node.inputs.iter().map(|i| mapping[i]).collect();
        wires[0] = wire;
        Ok(Some(target.wire_node(&node.name, Conv { pool_spec, ..op.clone() }, &wires)?))
    } else {
        Ok(None)
    }
}

impl PulsedOp for Conv {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let dt = self.q_params.unwrap_or(inputs[0].datum_type);
        super::pools::pulsed_output_facts(&self.pool_spec, inputs, dt)
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
