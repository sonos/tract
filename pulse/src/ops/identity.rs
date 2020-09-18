

impl PulsedOp for Identity {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
