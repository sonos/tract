use crate::internal::*;
use tract_core::ops::array::{Pad, PadMode};
use tract_pulse_opl::ops::{Delay, PulsePad};

register_all!(Pad: pulsify);

fn pulsify(
    op: &Pad,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let mut input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?.clone();
    if !op.pads.iter().enumerate().all(|(ax, &(a, b))| ax == fact.axis || (a == 0 && b == 0)) {
        return Ok(None);
    }
    let (before, after) = op.pads[fact.axis];
    let pulse = fact.pulse();
    let mut extra_delay = before.saturating_sub(fact.delay);
    match op.mode {
        PadMode::Constant(_) => (),
        PadMode::Edge => {
            let pulse = if let Ok(pulse) = pulse.to_usize() {
                pulse
            } else {
                bail!("Edge padding can only by pulsified with concrete integer values")
            };
            if before < pulse {
                let start_offset = (fact.delay + extra_delay) % pulse;
                if before > start_offset {
                    extra_delay += before - start_offset;
                }
            } else {
                bail!(
                    "Edge padding mode needs pulse strictly bigger than left padding (pulse={} padding={})",
                    pulse,
                    before
                    )
            }
        }
        PadMode::Reflect => bail!("Reflect padding mode pulsing is not supported"),
    };
    if extra_delay > 0 {
        input = target.wire_node(
            format!("{}.Delay", node.name),
            Delay::new_typed(&(&fact).into(), fact.axis, extra_delay, 0),
            &[input],
        )?[0];
    }
    let op = PulsePad {
        axis: fact.axis,
        before,
        after: after.into(),
        begin_input: fact.delay + extra_delay,
        end_input: fact.delay.to_dim() + extra_delay + fact.dim,
        mode: op.mode.clone(),
        overlap: 0,
    };
    Ok(Some(target.wire_node(&*node.name, op, &[input])?))
}

impl PulsedOp for PulsePad {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.dim += self.before.to_dim() + &self.after;
        fact.delay -= self.before;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
