use crate::internal::*;

#[derive(Debug, Clone)]
pub struct VPTQGemm {}

impl Op for VPTQGemm {
    fn name(&self) -> Cow<str> {
        "VPTQGemm".into()
    }

    op_as_typed_op!();
}

impl EvalOp for VPTQGemm {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (
            input,
            indices,
            centroids,
            outlier_indices,
            outlier_centroids,
            perm,
            weight_scale,
            weight_bias,
            bias,
        ) = args_9!(inputs);
        let mut input = input.into_tensor();
        // todo:  implement it now!
        Ok(tvec!(input.into_tvalue()))
    }
}

impl TypedOp for VPTQGemm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].without_value()))
    }

    as_op!();
}
