use crate::infer::*;
use crate::internal::*;

pub use tract_core::ops::konst::*;

impl InferenceRulesOp for Const {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 0)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].value, self.val().clone().bex())?;
        Ok(())
    }

    as_op!();
    to_typed!();
}
