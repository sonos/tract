use crate::fact::StreamInfo;
use crate::internal::*;
use tract_core::ops::fft::Stft;
use tract_pulse_opl::ops::Delay;

register_all!(Stft: pulsify);

fn pulsify(
    op: &Stft,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let mut wire = mapping[&node.inputs[0]];
    let input_fact = target.outlet_fact(wire)?.clone();

    let stream = match &input_fact.stream {
        Some(s) => s.clone(),
        None => return Ok(None),
    };

    if stream.axis != op.axis {
        return Ok(None);
    }

    let overlap = op.frame - op.stride;

    // Compute extra delay so that (stream.delay + overlap + extra_delay) % stride == 0
    let delayed = stream.delay + overlap;
    let misalignment = delayed % op.stride;
    let extra_delay = if misalignment > 0 { op.stride - misalignment } else { 0 };

    if overlap > 0 || extra_delay > 0 {
        wire = target.wire_node(
            format!("{}.delay", node.name),
            Delay::new_typed(&(&input_fact).into(), stream.axis, extra_delay, overlap),
            &[wire],
        )?[0];
    }

    Ok(Some(target.wire_node(&node.name, op.clone(), &[wire])?))
}

impl PulsedOp for Stft {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let input = inputs[0];
        let stream = input.stream.as_ref().unwrap();

        // pulse after delay = original_pulse + overlap
        let pulse = &input.shape[stream.axis];
        let out_pulse = (pulse.clone() - self.frame) / self.stride + 1;

        let mut shape = input.shape.to_tvec();
        shape[self.axis] = out_pulse;
        shape.insert(self.axis + 1, self.frame.to_dim());

        Ok(tvec!(PulsedFact {
            datum_type: input.datum_type,
            shape: shape.into(),
            stream: Some(StreamInfo {
                axis: self.axis,
                dim: (stream.dim.clone() - self.frame) / self.stride + 1,
                delay: stream.delay / self.stride,
            }),
        }))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
