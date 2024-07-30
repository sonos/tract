use std::iter::once;

use crate::internal::*;

#[derive(Clone, Debug, Hash)]
pub struct InnerDimToComplex;

impl Op for InnerDimToComplex {
    fn name(&self) -> Cow<str> {
        "InnerDimToComplex".into()
    }

    op_as_typed_op!();
}

impl EvalOp for InnerDimToComplex {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(tvec!(reinterpret_inner_dim_as_complex(inputs.remove(0).into_tensor())?.into_tvalue()))
    }
}

impl TypedOp for InnerDimToComplex {
    #[allow(clippy::into_iter_on_ref)]
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].without_value();
        if fact.shape.last() != Some(&2.to_dim()) {
            bail!("Expect inner tensor dimension to be 2")
        }
        fact.shape = fact.shape.into_iter().rev().skip(1).rev().collect();
        fact.datum_type = fact.datum_type.complexify()?;
        Ok(tvec!(fact))
    }

    as_op!();
}

#[derive(Clone, Debug, Hash)]
pub struct ComplexToInnerDim;

impl Op for ComplexToInnerDim {
    fn name(&self) -> Cow<str> {
        "ComplexToInnerDim".into()
    }

    op_as_typed_op!();
}

impl EvalOp for ComplexToInnerDim {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(tvec!(reinterpret_complex_as_inner_dim(inputs.remove(0).into_tensor())?.into_tvalue()))
    }
}

impl TypedOp for ComplexToInnerDim {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].without_value();
        fact.shape = fact.shape.iter().cloned().chain(once(2.to_dim())).into();
        fact.datum_type = fact.datum_type.decomplexify()?;
        Ok(tvec!(fact))
    }

    as_op!();
}
