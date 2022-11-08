use crate::internal::*;
use tract_core::ops::Downsample;
use tract_pulse_opl::ops::PulsedAxisSlice;

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
        if pulse % stride != 0 {
            bail!("Pulsificaton requires pulse to be a stride multiple")
        }
        let mut op = op.clone();
        let modulo = op.modulo + stream.delay;
        op.modulo = modulo % stride;
        let mut wire = target.wire_node(format!("{}.downsample", node.name), op, &[input])?;
        if modulo / stride > 1 {
            wire = target.wire_node(
                format!("{}.skip", node.name),
                PulsedAxisSlice {
                    axis: stream.axis,
                    skip: modulo / stride,
                    take: (stream.dim.to_owned() - modulo).divceil(stride),
                },
                &[input],
            )?;
        }
        target.rename_node(wire[0].node, &node.name)?;
        Ok(Some(wire))
    } else {
        Ok(None)
    }
}

impl PulsedOp for Downsample {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let mut stream = fact.stream.as_mut().unwrap();
        dbg!(&self);
        dbg!(&stream);
        fact.shape.set(self.axis, fact.shape[self.axis].clone() / self.stride as usize);
        stream.dim = (inputs[0].stream.as_ref().unwrap().dim.clone() - self.modulo)
            .div_ceil(self.stride as _);
        dbg!(&stream.dim);
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
