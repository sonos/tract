use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::binary::BinMiniOp;
use tract_nnef::tract_core::ops::math::Mul;
use tract_nnef::tract_core::ops::nn::Sigmoid;

#[derive(Clone, Debug, Hash)]
pub struct BasicSilu;

impl Op for BasicSilu {
    fn name(&self) -> Cow<str> {
        "BasicSilu".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for BasicSilu {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let dt = input.datum_type();
        let mut a = input.clone().into_tensor();
        Sigmoid {}.eval_in_place(&mut a, None)?;
        let a3 = Mul.eval(input.clone(), a.into_tvalue(), dt)?;
        Ok(tvec![a3.into()])
    }
}

impl TypedOp for BasicSilu {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}