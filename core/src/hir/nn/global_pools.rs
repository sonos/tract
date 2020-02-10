use crate::infer::*;
use crate::internal::*;

use crate::ops::nn::{GlobalAvgPool, GlobalLpPool, GlobalMaxPool};

impl InferenceRulesOp for GlobalAvgPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    as_op!();
    to_typed!();
}

impl InferenceRulesOp for GlobalLpPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    as_op!();
    to_typed!();
}

impl InferenceRulesOp for GlobalMaxPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    as_op!();
    to_typed!();
}

fn rules<'r, 'p: 'r, 's: 'r>(
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    check_input_arity(&inputs, 1)?;
    check_output_arity(&outputs, 1)?;
    s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
    s.equals(&outputs[0].rank, &inputs[0].rank)?;
    s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?;
    s.equals(&outputs[0].shape[1], &inputs[0].shape[1])?;
    s.given(&inputs[0].rank, move |s, rank| {
        for i in 2..rank {
            s.equals(&outputs[0].shape[i as usize], TDim::from(1))?;
        }
        Ok(())
    })
}
