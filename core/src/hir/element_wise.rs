use crate::internal::*;
use crate::infer::*;

use crate::ops::element_wise::ElementWiseOp;

impl InferenceRulesOp for ElementWiseOp {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.given(&inputs[0].datum_type, move |s, dt| {
            if let Some(dt) = self.0.output_type(dt) {
                s.equals(&outputs[0].datum_type, dt)
            } else {
                s.equals(&outputs[0].datum_type, dt)
            }
        })?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }
    to_typed!();
    as_op!();
}
