use crate::kernels::nn::ApplyRope;
use crate::utils::with_borrowed_metal_stream;
use derive_new::new;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Clone, Debug, new, Hash)]
pub struct MetalApplyRope;

impl Op for MetalApplyRope {
    fn name(&self) -> StaticName {
        "MetalApplyRope".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalApplyRope {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
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

        with_borrowed_metal_stream(|stream| {
            ApplyRope.dispatch_eval(stream, input, cos, sin, &output)
        })?;
        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalApplyRope {
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
