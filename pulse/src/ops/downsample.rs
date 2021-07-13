use crate::internal::*;
use tract_core::ops::Downsample;

register_all!(Downsample: pulsify);

fn pulsify(
    op: &Downsample,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    let pulse = target.outlet_fact(input)?.pulse();
    let stride = if op.stride > 0 {
        op.stride as usize
    } else {
        bail!("Negative strides are not causal, can not pulsify.")
    };
    if pulse % stride != 0 {
        bail!("Pulsificaton requires pulse to be a stride multiple")
    }
    Ok(Some(target.wire_node(&*node.name, op.clone(), &[input])?))
}

impl PulsedOp for Downsample {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape.set(self.axis, fact.shape[self.axis].clone() / self.stride as usize);
        fact.dim = fact.dim.div_ceil(self.stride as _);
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
