use std::sync::Arc;

use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::binary::BinMiniOp;
use tract_nnef::tract_core::ops::math::{Add, Mul, Rsqrt};
use tract_nnef::tract_core::ops::nn::Reducer;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_transformers_rms_norm",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Scalar.tensor().named("eps"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        rms_norm,
    );
}

// Check with Kali!
pub fn rms_norm(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let eps = invocation.named_arg_as(builder, "eps")?;
    builder.wire(BasicRmsNorm { axis, eps }, &[input])
}

#[derive(Clone, Debug, Hash)]
pub struct BasicRmsNorm {
    pub axis: usize,
    pub eps: Arc<Tensor>,
}

impl Op for BasicRmsNorm {
    fn name(&self) -> Cow<str> {
        "BasicRmsNorm".to_string().into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {:?}, eps: {:?}", self.axis, self.eps)])
    }
    op_as_typed_op!();
}

impl EvalOp for BasicRmsNorm {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let dt = input.datum_type();
        let a1 = Reducer::MeanOfSquares.reduce(&[self.axis], &input)?;
        let mut a2 = Add.eval(a1.into_tvalue(), self.eps.clone().into_tvalue(), dt)?;
        Rsqrt {}.eval_in_place(&mut a2, None)?;
        let a3 = Mul.eval(a2.into_tvalue(), input.clone(), dt)?;

        Ok(tvec![a3.into()])
    }
}

impl TypedOp for BasicRmsNorm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}