use crate::fact::StreamInfo;
use crate::internal::*;
use tract_core::ops::fft::{Fft, Stft};
use tract_pulse_opl::ops::Delay;

register_all!(Stft: stft_pulsify, Fft: fft_pulsify);

/// Pulsifier for `Fft`. The op itself is axes-natural (every axis is
/// 1-to-1 passthrough) so the generic `PulseWrappingOp` fallback in
/// `model.rs` can handle streaming on any non-FFT, non-complex axis
/// once we return `Ok(None)` here. Two cases must be rejected fast:
///
/// 1. Streaming on the FFT axis itself. The natural axes mapping
///    would let `PulseWrappingOp` run the `Fft` per-pulse, but that
///    silently turns `Fft { axis: A }` into `Stft { frame: pulse_size,
///    stride: pulse_size, axis: A }` -- the frame size is dictated by
///    the runtime's choice of `pulse_size`, not by the model. The
///    typed-model semantics ("FFT the whole axis") is unimplementable
///    under streaming because the FFT length needs to be known and
///    that axis is symbolic; the per-pulse interpretation is a real
///    operation but a *different* one. Reject and tell the user to
///    write the STFT they actually want with an explicit frame.
/// 2. Streaming on the trailing `(re, imag)` axis (axis = rank - 1):
///    the trailing 2-axis carries the complex pair, not a time
///    dimension; pulsing it has no semantic meaning.
fn fft_pulsify(
    op: &Fft,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let wire = mapping[&node.inputs[0]];
    let input_fact = target.outlet_fact(wire)?;
    let Some(stream) = input_fact.stream.as_ref() else {
        return Ok(None);
    };
    let rank = input_fact.shape.rank();
    let complex_axis = rank - 1;
    if stream.axis == op.axis {
        bail!(
            "Fft cannot pulsify on its own FFT axis ({}).\n\
             \n\
             The typed-model semantics of `Fft {{ axis: {0} }}` is \
             \"FFT every sample along axis {0}\", which requires the \
             axis length to be known. Once axis {0} is the streaming \
             axis its length is symbolic, so the planned FFT size is \
             undefined.\n\
             \n\
             A per-pulse FFT (FFT of each `pulse_size`-sample chunk) \
             *is* a coherent operation, but it is equivalent to \
             `Stft {{ frame: pulse_size, stride: pulse_size, axis: {0} }}` \
             where the runtime picks `pulse_size`. That silent \
             reinterpretation almost never matches what the user \
             expected from a plain `Fft`, so we refuse it here.\n\
             \n\
             If you want windowed FFT, replace the `Fft` with an \
             explicit `Stft {{ frame: N, stride: N, axis: {0} }}` and \
             choose `N` yourself. If you want to stream a different \
             axis (e.g. batch), keep the `Fft` and mark that other \
             axis as the streaming one instead.",
            op.axis
        );
    }
    if stream.axis == complex_axis {
        bail!(
            "Fft cannot pulsify on its trailing `(re, imag)` axis \
             (axis {} of rank {}). That axis carries the complex pair, \
             not a time dimension -- there is nothing to stream there.",
            stream.axis,
            rank
        );
    }
    // Any other streaming axis: let the generic PulseWrappingOp
    // handle it -- the Fft is rank-preserving and 1-to-1 on every
    // axis, so per-pulse evaluation along a non-FFT, non-complex axis
    // is identical to slicing the input and applying Fft to each
    // slice.
    Ok(None)
}

fn stft_pulsify(
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

    rule_if!(stream.axis == op.axis);

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
