use crate::internal::*;
use tract_core::ops::Downsample;
use tract_pulse_opl::tract_nnef::tract_num_traits::Zero;

register_all!(Downsample: pulsify);

fn pulsify(
    op: &Downsample,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    let pulse = target.outlet_fact(input)?.pulse();
    let stride = if op.stride > 0 {
        op.stride as usize
    } else {
        bail!("Negative strides are not causal, can not pulsify.")
    };
    if !(pulse.to_owned() % (stride as i64)).is_zero() {
        bail!("Pulsification requires pulse ({}) to be a stride ({}) multiple", pulse, stride)
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
