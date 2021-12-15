use tract_core::internal::*;
use crate::infer::*;

use tract_core::ops::cast::Cast;
pub use tract_core::ops::cast::cast;

impl InferenceRulesOp for Cast {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.equals(&outputs[0].datum_type, self.to)?;
        Ok(())
    }

    as_op!();
    to_typed!();
}
