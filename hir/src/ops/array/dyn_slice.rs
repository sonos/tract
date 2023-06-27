use tract_core::ops::array::DynSlice;
use crate::internal::*;

impl InferenceRulesOp for DynSlice {
    fn rules<'r, 'p: 'r, 's: 'r>(
            &'s self,
            s: &mut crate::infer::Solver<'r>,
            inputs: &'p [crate::infer::TensorProxy],
            outputs: &'p [crate::infer::TensorProxy],
        ) -> InferenceResult {
        check_input_arity(inputs, 3)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[1].rank, 0)?;
        s.equals(&inputs[2].rank, 0)?;
        s.given(&inputs[0].rank, move |s, rank| {
            for axis in 0..rank as usize {
                if axis == self.axis {
                    s.equals(&outputs[0].shape[axis], self.len.clone())?;
                } else {
                    s.equals(&outputs[0].shape[axis], &inputs[0].shape[axis])?;
                }
            }
            Ok(())
        })?;
        Ok(())
    }

    as_op!();
    to_typed!();
}
