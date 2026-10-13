use crate::internal::*;
use tract_pulse_opl::ops::AffineChunkTrim;

register_all!(AffineChunkTrim: pulsify);

fn pulsify(
    op: &AffineChunkTrim,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    let op = AffineChunkTrim { target_per_pulse: op.target_per_pulse * pulse.to_usize()?, ..*op };
    target.wire_node(&*node.name, op, &[input]).map(Some)
}

impl PulsedOp for AffineChunkTrim {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let cur = fact.shape[self.axis].to_usize()?;
        let trim_amount = cur.saturating_sub(self.target_per_pulse);
        if trim_amount > 0 {
            let new_per_pulse = cur - trim_amount;
            let mut shape: TVec<TDim> = fact.shape.iter().cloned().collect();
            shape[self.axis] = new_per_pulse.to_dim();
            fact.shape = shape.into();
        }
        if let Some(stream) = fact.stream.as_mut() {
            stream.dim = stream.dim.clone() - self.typed_trim.to_dim();
        }
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
