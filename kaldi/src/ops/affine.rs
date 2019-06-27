use tract_core::internal::*;

use crate::parser::*;
use nom::{bytes::complete::*, character::complete::*, combinator::*, sequence::*, IResult};

pub fn fixed_affine_component(i: &[u8]) -> IResult<&[u8], Box<InferenceOp>> {
    let (i, _) = open(i, "LinearParams")?;
    let (i, linear) = matrix(i)?;
    let (i, _) = open(i, "BiasParams")?;
    let (i, bias) = vector(i)?;
    Ok((i, Box::new(Affine::new(linear.into_arc_tensor(), bias.into_arc_tensor()))))
}

#[derive(Clone, Debug, new)]
struct Affine {
    linear_params: Arc<Tensor>,
    bias_params: Arc<Tensor>,
}

impl Op for Affine {
fn name(&self) -> std::borrow::Cow<str> {
        "kaldi.Affine".into()
    }
}

impl StatelessOp for Affine {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unimplemented!();
    }
}

impl InferenceRulesOp for Affine {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        Ok(())
    }

    inference_op_as_op!();
}
