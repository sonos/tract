use crate::kernels::nn::ApplyRope;
use crate::tensor::MetalTensorExt;
use derive_new::new;
use tract_core::internal::*;

#[derive(Clone, Debug, new, Hash)]
pub struct MetalApplyRope;

impl Op for MetalApplyRope {
    fn name(&self) -> Cow<str> {
        "MetalApplyRope".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalApplyRope {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let (opaque_input, opaque_cos, opaque_sin) = args_3!(inputs);
                let input = opaque_input.to_metal_tensor()?;
                let cos = opaque_cos.to_metal_tensor()?;
                let sin = opaque_sin.to_metal_tensor()?;
                Ok(tvec!(ApplyRope
                    .dispatch_eval(context, input, cos, sin)?
                    .into_opaque_tensor()
                    .into_tvalue()))
            })
        })
    }
}

impl TypedOp for MetalApplyRope {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_output_facts(inputs, |facts| {
            let dt = facts[0].datum_type;
            let fact = dt.fact(facts[0].shape.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
