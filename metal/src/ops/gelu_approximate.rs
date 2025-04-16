use crate::kernels::nn::GeluApproximate;
use crate::ops::MetalEvalOp;
use crate::MetalContext;
use tract_gpu::tensor::GpuTensorExt;
use tract_core::internal::*;

#[derive(Clone, Debug, Default, Hash)]
pub struct MetalGeluApproximate {
    pub fast_impl: bool,
}

impl Op for MetalGeluApproximate {
    fn name(&self) -> Cow<str> {
        "MetalGeluApproximate".into()
    }

    op_as_typed_op!();
}

crate::impl_eval_op_for_metal_op!(MetalGeluApproximate);

impl MetalEvalOp for MetalGeluApproximate {
    fn metal_eval(
        &self,
        context: &MetalContext,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let input_metal = input.to_gpu_tensor()?;
        let output = crate::ops::make_tensor_for_node(
            session,
            node_id,
            input_metal.datum_type(),
            input_metal.shape(),
        )?;
        GeluApproximate { fast_impl: self.fast_impl }.dispatch_eval(context, input_metal, &output)?;
        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalGeluApproximate {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_facts_from_gpu(inputs, |facts| {
            let dt = facts[0].datum_type;
            let fact = dt.fact(facts[0].shape.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
