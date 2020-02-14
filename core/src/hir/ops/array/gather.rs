use crate::internal::*;
use crate::infer::*;

pub use crate::ops::array::Gather;

impl InferenceRulesOp for Gather {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, i64::datum_type())?;
        s.equals(inputs[0].rank.bex() - 1 + inputs[1].rank.bex(), outputs[0].rank.bex())?;
        s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, input_shape, indices_shape| {
            let output_shape = self.compute_output_shape(&*input_shape, &*indices_shape)?;
            s.equals(&outputs[0].shape, output_shape)?;
            Ok(())
        })?;
        Ok(())
    }

    as_op!();
    to_typed!();
}

