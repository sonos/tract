use crate::context::StreamExt;
use crate::kernels::array::RotateHalf;
use derive_new::new;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Clone, Debug, new, Hash, PartialEq, Eq)]
pub struct MetalRotateHalf;

impl Op for MetalRotateHalf {
    fn name(&self) -> StaticName {
        "MetalRotateHalf".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalRotateHalf {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        tract_gpu::with_stream(|stream| {
            let stream = stream.metal()?;
            let input_value = args_1!(inputs);
            let input = input_value.to_device_tensor()?;
            let output = tract_gpu::session_handler::make_tensor_for_node(
                session,
                node_id,
                input.datum_type(),
                input.shape(),
            )?;
            RotateHalf.dispatch_eval(stream, input, &output)?;
            Ok(tvec!(output.into_tensor().into_tvalue()))
        })
    }
}

impl TypedOp for MetalRotateHalf {
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
