use std::sync::Arc;
use tract_core::internal::*;
use tract_core::ops::binary::BinMiniOp;
use tract_core::ops::math::{Add, Mul};
use tract_core::ops::nn::{Softmax, SoftmaxExp};

#[derive(Clone, Debug, Hash)]
pub struct BasicScaledMaskedSoftmax {
    pub axis: usize,
    pub scale: Arc<Tensor>,
}

impl Op for BasicScaledMaskedSoftmax {
    fn name(&self) -> Cow<str> {
        "BasicScaledMaskedSoftmax".to_string().into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {:?}, scale: {:?}", self.axis, self.scale)])
    }
    op_as_typed_op!();
}

impl EvalOp for BasicScaledMaskedSoftmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, mask) = args_2!(inputs);
        let dt = input.datum_type();
        ensure!(input.shape() == mask.shape());

        let scaled_input = Mul.eval(input, self.scale.clone().into_tvalue(), dt)?;
        let masked_input = Add.eval(scaled_input.into(), mask, dt)?;
        let softmax = Softmax::new(tvec![self.axis], None, SoftmaxExp::Libc)
            .eval(tvec![masked_input.into()])?[0];
        Ok(tvec![softmax.into()])
    }
}

impl TypedOp for BasicScaledMaskedSoftmax {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 2);
        let (input, mask) = (inputs[0], inputs[1]);
        ensure!(input.datum_type == mask.datum_type);
        let dt = input.datum_type;
        let fact = dt.fact(input.shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}
