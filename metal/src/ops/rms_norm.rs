use crate::kernels::nn::RmsNorm;
use crate::{MetalTensor, MetalTensorExt};
use derive_new::new;
use std::sync::Arc;
use tract_core::internal::*;

#[derive(Clone, Debug, new, Hash)]
pub struct MetalRmsNorm {
    pub axis: usize,
    pub eps: Arc<Tensor>,
}

impl Op for MetalRmsNorm {
    fn name(&self) -> Cow<str> {
        "MetalRmsNorm".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {:?}, eps: {:?}", self.axis, self.eps)])
    }
    op_as_typed_op!();
}

impl EvalOp for MetalRmsNorm {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let opaque = args_1!(inputs);
                let input = opaque.to_metal_tensor()?;
                let output =
                    unsafe { MetalTensor::uninitialized_dt(input.datum_type(), input.shape())? };
                RmsNorm.dispatch_eval(context, input, self.axis, &self.eps, &output)?;
                Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
            })
        })
    }
}

impl TypedOp for MetalRmsNorm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_tmp_output_facts(inputs, |facts| {
            let dt = facts[0].datum_type;
            let fact = dt.fact(facts[0].shape.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
