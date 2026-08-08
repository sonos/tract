use anyhow::ensure;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;
use tract_gpu::utils::facts_to_device_facts;

/// out[route, col] = value[route, col] + bias[expert_ids[route], col], in one
/// pass. Replaces the gather([routes, n] bias matrix) + add pair the MoE
/// lowering previously emitted per expert matmul.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MetalRoutedBiasAdd;

impl MetalRoutedBiasAdd {
    fn output_facts_inner(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3);
        ensure!(inputs[0].rank() == 2, "value must be [routes, n]");
        ensure!(inputs[0].datum_type == f32::datum_type());
        ensure!(inputs[1].datum_type == f32::datum_type());
        ensure!(inputs[2].rank() == 1);
        ensure!(inputs[2].datum_type == i64::datum_type());
        Ok(tvec![f32::datum_type().fact(inputs[0].shape.clone())])
    }
}

impl Op for MetalRoutedBiasAdd {
    fn name(&self) -> StaticName {
        "MetalRoutedBiasAdd".into()
    }
    op_as_typed_op!();
}

impl EvalOp for MetalRoutedBiasAdd {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (value_raw, bias_raw, expert_ids_raw) = args_3!(inputs);
        let value = value_raw
            .to_device_tensor()
            .with_context(|| format!("value is not a Metal tensor: {value_raw:?}"))?;
        let bias = bias_raw
            .to_device_tensor()
            .with_context(|| format!("bias is not a Metal tensor: {bias_raw:?}"))?;
        let expert_ids = expert_ids_raw
            .to_device_tensor()
            .with_context(|| format!("expert ids are not a Metal tensor: {expert_ids_raw:?}"))?;

        let output = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            f32::datum_type(),
            value.shape(),
        )?;

        crate::with_metal_stream(|stream| {
            crate::kernels::moe::dispatch_routed_bias_add_f32(
                stream, value, bias, expert_ids, &output,
            )
        })?;

        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for MetalRoutedBiasAdd {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        facts_to_device_facts(inputs, |input_facts| self.output_facts_inner(input_facts))
            .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    as_op!();
}
