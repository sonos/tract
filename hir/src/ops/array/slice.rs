use crate::infer::*;
use crate::internal::*;

pub use tract_core::ops::array::Slice;

impl InferenceRulesOp for Slice {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.given(&inputs[0].rank, move |s, rank| {
            (0..(rank as usize)).try_for_each(move |axis| {
                if self.axis == axis {
                    s.equals(&outputs[0].shape[axis], (self.end.clone() - &self.start).to_dim())
                } else {
                    s.equals(&outputs[0].shape[axis], &inputs[0].shape[axis])
                }
            })
        })?;
        Ok(())
    }

    as_op!();
    to_typed!();
}
