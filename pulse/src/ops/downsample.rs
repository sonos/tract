use crate::internal::*;
use tract_core::ops::Downsample;
use tract_pulse_opl::ops::PulsedAxisSlice;
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
    let fact = target.outlet_fact(input)?.clone();
    if let Some(stream) = fact.stream.as_ref() {
        if stream.axis != op.axis {
            return Ok(None);
        }
        let stride = if op.stride > 0 {
            op.stride as usize
        } else {
            bail!("Negative strides are not causal, can not pulsify.")
        };
        let pulse = fact.pulse().unwrap();
        if !(pulse.clone() % stride).is_zero() {
            bail!("Pulsification requires pulse ({}) to be a stride ({}) multiple", pulse, stride)
        }
        let mut wire = tvec!(input);
        let first_offset = stream.delay + op.modulo;
        let new_op = Downsample { modulo: first_offset % stride, axis: op.axis, stride: op.stride };
        wire = target.wire_node(format!("{}.downsample", node.name), new_op, &wire)?;
        wire = target.wire_node(
            &node.name,
            PulsedAxisSlice {
                axis: stream.axis,
                skip: first_offset / stride,
                take: (stream.dim.to_owned() - op.modulo).divceil(stride),
            },
            &wire,
        )?;
        target.rename_node(wire[0].node, &node.name)?;
        Ok(Some(wire))
    } else {
        Ok(None)
    }
}

impl PulsedOp for Downsample {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let stream = fact.stream.as_mut().unwrap();
        fact.shape.set(self.axis, fact.shape[self.axis].clone() / self.stride as usize);
        stream.dim = (stream.dim.clone() + stream.delay).divceil(self.stride as _);
        stream.delay = 0;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
