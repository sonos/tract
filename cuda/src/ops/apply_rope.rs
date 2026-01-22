use crate::context::CUDA_STREAM;
use crate::kernels::nn::ApplyRope;
use derive_new::new;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Clone, Debug, new, Hash)]
pub struct CudaApplyRope;

impl Op for CudaApplyRope {
    fn name(&self) -> StaticName {
        "CudaApplyRope".into()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaApplyRope {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (opaque_input, opaque_cos, opaque_sin) = args_3!(inputs);
        let input = opaque_input.to_device_tensor()?;
        let cos = opaque_cos.to_device_tensor()?;
        let sin = opaque_sin.to_device_tensor()?;
        let output = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            input.shape(),
        )?;

        CUDA_STREAM.with(|stream| ApplyRope.dispatch_eval(stream, input, cos, sin, &output))?;
        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for CudaApplyRope {
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
