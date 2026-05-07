//! Pulsifier for `tract_core::ops::array::Gather`.
//!
//! Handles the case the rel-pos blockify rewrite produces: a `Gather` whose
//! data input is a session-buffered (non-streaming) tensor and whose indices
//! input is streaming on the chunk axis.  Per pulse the indices reduce to a
//! concrete `[1, 2k-1]` slice of integer pos-row indices, and the gather is
//! a plain indexed read out of the full data tensor.  No cross-pulse state.

use crate::fact::StreamInfo;
use crate::internal::*;
use tract_core::ops::array::Gather;

register_all!(Gather: pulsify);

fn pulsify(
    op: &Gather,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let data = mapping[&node.inputs[0]];
    let indices = mapping[&node.inputs[1]];
    let data_fact = target.outlet_fact(data)?;
    let indices_fact = target.outlet_fact(indices)?;
    // Only handle the rel-pos pattern for now: data session-buffered (no
    // stream info), indices streaming.
    rule_if!(data_fact.stream.is_none());
    rule_if!(indices_fact.stream.is_some());
    Ok(Some(target.wire_node(&*node.name, op.clone(), &[data, indices])?))
}

impl PulsedOp for Gather {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let data = inputs[0];
        let indices = inputs[1];
        ensure!(
            data.stream.is_none(),
            "Gather pulsifier requires non-streaming data input (got stream info)"
        );
        let indices_stream =
            indices.stream.as_ref().context("Gather pulsifier requires streaming indices input")?;
        // Output shape: data.shape[..axis] + indices.shape + data.shape[axis+1..].
        let mut output_shape: TVec<TDim> = data.shape[..self.axis].iter().cloned().collect();
        output_shape.extend(indices.shape.iter().cloned());
        output_shape.extend(data.shape[self.axis + 1..].iter().cloned());
        let output_stream_axis = self.axis + indices_stream.axis;
        Ok(tvec!(PulsedFact {
            datum_type: data.datum_type,
            shape: output_shape.into(),
            stream: Some(StreamInfo {
                axis: output_stream_axis,
                dim: indices_stream.dim.clone(),
                delay: indices_stream.delay,
            }),
        }))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
