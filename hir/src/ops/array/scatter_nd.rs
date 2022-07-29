
use crate::infer::*;
use crate::internal::*;

pub use tract_core::ops::array::ScatterNd;

impl InferenceRulesOp for ScatterNd {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 3)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&inputs[2].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].shape, &inputs[0].shape)?;

        s.given_2(&inputs[0].rank, &inputs[1].rank, move |s, p ,q| {
            s.given(&inputs[1].shape[q as usize - 1], move |s, r| {
                if let Ok(r) = r.to_i64() {
                    s.equals(&inputs[2].rank, p + q - r - 1)?;
                }
                Ok(())
            })
        })?;
        Ok(())
    }

    as_op!();
    to_typed!();
}
