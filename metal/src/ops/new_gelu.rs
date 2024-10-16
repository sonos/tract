use crate::kernels::nn::NewGelu;
use crate::{MetalTensor, MetalTensorExt};
use derive_new::new;
use tract_core::internal::*;

#[derive(Clone, Debug, new, Hash)]
pub struct MetalNewGelu;

impl Op for MetalNewGelu {
    fn name(&self) -> Cow<str> {
        "MetalNewGelu".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalNewGelu {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let input = args_1!(inputs);
                let input_metal = input.to_metal_tensor()?;
                let output = unsafe {
                    MetalTensor::uninitialized_dt(input_metal.datum_type(), input_metal.shape())?
                };
                NewGelu::accurate().dispatch_eval(context, input_metal, &output)?;

                Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
            })
        })
    }
}

impl TypedOp for MetalNewGelu {
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
