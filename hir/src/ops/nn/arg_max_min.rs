use crate::infer::*;
use crate::internal::*;

pub use tract_core::ops::nn::ArgMaxMin;

impl InferenceRulesOp for ArgMaxMin {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, DatumType::I64)?;
        if self.keepdims {
            s.equals(&outputs[0].rank, &inputs[0].rank)?;
            for i in 0..self.axis {
                s.equals(&outputs[0].shape[i], &inputs[0].shape[i])?;
            }
            s.equals(&outputs[0].shape[self.axis], 1.to_dim())?;
            s.given(&inputs[0].rank, move |s, rank| {
                for i in (self.axis + 1)..(rank as usize) {
                    s.equals(&outputs[0].shape[i], &inputs[0].shape[i])?;
                }
                Ok(())
            })?;
        } else {
            s.equals(&outputs[0].rank, inputs[0].rank.bex() - 1)?;
            for i in 0..self.axis {
                s.equals(&outputs[0].shape[i], &inputs[0].shape[i])?;
            }
            s.given(&inputs[0].rank, move |s, rank| {
                for i in (self.axis + 1)..(rank as usize - 1) {
                    s.equals(&outputs[0].shape[i], &inputs[0].shape[i + 1])?;
                }
                Ok(())
            })?;
        };
        Ok(())
    }

    as_op!();
    to_typed!();
}
