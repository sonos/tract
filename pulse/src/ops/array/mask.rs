use crate::internal::*;
use tract_pulse_opl::ops::PulseMask;

impl PulsedOp for PulseMask {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        Ok(inputs.iter().cloned().cloned().collect())
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
