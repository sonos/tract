use crate::internal::*;
use crate::infer::*;

use crate::ops::array::Flatten;

impl InferenceRulesOp for Flatten {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.given(&inputs[0].shape, move |s, shape| {
            let [shape_0, shape_1] = self.compute_shape(&*shape);
            s.equals(&outputs[0].shape, ShapeFactoid::from(vec![shape_0, shape_1]))
        })
    }

    as_op!();
    to_typed!();
}

