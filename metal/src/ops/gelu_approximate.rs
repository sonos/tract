use crate::MetalStream;
use crate::kernels::nn::GeluApproximate;
use crate::utils::with_borrowed_metal_stream;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

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

impl EvalOp for MetalGeluApproximate {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
            &self,
            node_id: usize,
            session: &SessionState,
            inputs: TVec<TValue>,
        ) -> TractResult<TVec<TValue>> {
        with_borrowed_metal_stream(|stream| {
            let input = args_1!(inputs);
            let input_metal = input.to_device_tensor()?;
            let output = tract_gpu::session_handler::make_tensor_for_node(
                session,
                node_id,
                input_metal.datum_type(),
                input_metal.shape(),
            )?;
            GeluApproximate { fast_impl: self.fast_impl }.dispatch_eval(
                stream,
                input_metal,
                &output,
            )?;
            Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
        })
    }
}

impl TypedOp for MetalGeluApproximate {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let dt = facts[0].datum_type;
            let fact = dt.fact(facts[0].shape.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
