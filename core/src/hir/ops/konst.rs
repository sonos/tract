use crate::internal::*;
use crate::infer::*;

pub use crate::ops::konst::*;

impl InferenceRulesOp for Const {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 0)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].value, self.value.clone().bex())?;
        Ok(())
    }

    as_op!();
    to_typed!();
}
