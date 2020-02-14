use crate::infer::*;
use crate::internal::*;

pub use tract_core::ops::matmul::compute_shapes;
pub use tract_core::ops::matmul::MatMul;

impl InferenceRulesOp for MatMul {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        if let Some(qp) = &self.q_params {
            s.equals(&outputs[0].datum_type, &qp.c_datum_type)?;
        } else {
            s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        }
        s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, ashape, bshape| {
            let (_, _, _, cshape) =
                compute_shapes(ashape, bshape, self.a_trans, self.b_trans, self.c_trans)?;
            s.equals(&outputs[0].shape, cshape)
        })?;
        Ok(())
    }

    as_op!();
    to_typed!();
}
