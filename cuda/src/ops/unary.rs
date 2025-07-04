use derive_new::new;
use tract_core::internal::*;
use tract_gpu::session_handler::make_tensor_for_node;
use tract_gpu::tensor::DeviceTensorExt;

use crate::context::CUDA_STREAM;
use crate::kernels::UnaryOps;

#[derive(Clone, Debug, new, Hash)]
pub struct CudaUnaryOp(pub UnaryOps);

impl Op for CudaUnaryOp {
    fn name(&self) -> StaticName {
        format!("Cuda{}", self.0.name()).into()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaUnaryOp {
    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        CUDA_STREAM.with(|stream| {
            let opaque = args_1!(inputs);
            let input = opaque.to_device_tensor()?;
            let output = make_tensor_for_node(session, node_id, input.datum_type(), input.shape())?;
            self.0.dispatch_eval(stream, input, &output)?;
            Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
        })
    }

    fn is_stateless(&self) -> bool {
        true
    }
}

impl TypedOp for CudaUnaryOp {
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
