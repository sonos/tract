use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

use crate::context::CUDA_STREAM;
use crate::kernels::nn::LeakyRelu;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CudaLeakyRelu {
    pub alpha: f32,
}

impl Op for CudaLeakyRelu {
    fn name(&self) -> StaticName {
        "CudaLeakyRelu".into()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaLeakyRelu {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        CUDA_STREAM.with(|stream| {
            let input = args_1!(inputs);
            let input_cuda = input.to_device_tensor()?;
            let output = tract_gpu::session_handler::make_tensor_for_node(
                session,
                node_id,
                input_cuda.datum_type(),
                input_cuda.shape(),
            )?;
            LeakyRelu.dispatch_eval(stream, input_cuda, self.alpha, &output)?;
            Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
        })
    }
}

impl TypedOp for CudaLeakyRelu {
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
