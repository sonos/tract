use crate::internal::*;
use crate::infer::*;

use crate::broadcast::multi_broadcast;
pub use crate::ops::logic::*;

impl InferenceRulesOp for Iff {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, DatumType::Bool)?;
        s.equals(&inputs[1].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[2].datum_type, &outputs[0].datum_type)?;
        s.given_3(&inputs[0].shape, &inputs[1].shape, &inputs[2].shape, move |s, c, t, f| {
            let shape = multi_broadcast(&[&c, &t, &f])
                .ok_or_else(|| format!("Incompatible shapes {:?}, {:?} and {:?}", c, t, f))?;
            s.equals(&outputs[0].shape, shape)
        })?;
        Ok(())
    }

    as_op!();
    to_typed!();
}
