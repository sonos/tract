use crate::internal::*;
use tract_pulse_opl::ops::PulsedAxisSlice;

impl PulsedOp for PulsedAxisSlice {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.delay += self.skip;
        fact.dim = self.take.clone();
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

