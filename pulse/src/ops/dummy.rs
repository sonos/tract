use crate::internal::*;
use tract_core::ops::dummy::Dummy;

impl PulsedOp for Dummy {
    fn pulsed_output_facts(&self, _inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        Ok(tvec!())
    }
    as_op!();
    pulsed_op_to_typed_op!();
}
