use crate::internal::*;
use crate::infer::*;

pub use crate::ops::array::Size;

impl InferenceRulesOp for Size {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.dt)?;
        s.equals(&outputs[0].rank, 0)?;
        Ok(())
    }

    as_op!();
    to_typed!();
}
