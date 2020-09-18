use crate::infer::*;
use crate::internal::*;

pub use tract_core::ops::dummy::Dummy;

impl InferenceRulesOp for Dummy {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        _inputs: &'p [TensorProxy],
        _outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        Ok(())
    }

    as_op!();
    to_typed!();
}
