use crate::kernels::nn::Silu;
use crate::tensor::MetalTensorExt;
use derive_new::new;
use tract_core::internal::*;

#[derive(Clone, Debug, new, Hash)]
pub struct MetalSilu;

impl Op for MetalSilu {
    fn name(&self) -> Cow<str> {
        "MetalSilu".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalSilu {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let input = args_1!(inputs);
                let input_metal = input.to_metal_tensor()?;
                Ok(tvec!(Silu
                    .dispatch_eval(context, input_metal)?
                    .into_opaque_tensor()
                    .into_tvalue()))
            })
        })
    }
}

impl TypedOp for MetalSilu {
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
