use crate::infer::*;
use crate::internal::*;

pub use tract_core::ops::array::{Pad, PadMode};

impl InferenceRulesOp for Pad {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        for (ix, &(a, b)) in self.pads.iter().enumerate() {
            s.equals(&inputs[0].shape[ix], outputs[0].shape[ix].bex() - a.to_dim() - b.to_dim())?;
        }
        Ok(())
    }

    as_op!();
    to_typed!();
}
