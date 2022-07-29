use crate::infer::*;
use crate::internal::*;

pub use tract_core::ops::downsample::Downsample;

impl InferenceRulesOp for Downsample {
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
        s.given(&inputs[0].rank, move |s, r| {
            for i in 0..(r as usize) {
                if i == self.axis {
                    s.given(&inputs[0].shape[i], move |s, d| {
                        s.equals(
                            &outputs[0].shape[i],
                            (d - self.modulo).div_ceil(self.stride as u64),
                        )
                    })?
                } else {
                    s.equals(&inputs[0].shape[i], &outputs[0].shape[i])?
                }
            }
            Ok(())
        })
    }

    as_op!();
    to_typed!();
}
