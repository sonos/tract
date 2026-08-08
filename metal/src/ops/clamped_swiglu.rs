use anyhow::ensure;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;
use tract_gpu::utils::facts_to_device_facts;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct MetalClampedSwiGlu {
    pub alpha_bits: u32,
    pub limit_bits: u32,
}

impl MetalClampedSwiGlu {
    pub fn new(alpha: f32, limit: f32) -> Self {
        Self { alpha_bits: alpha.to_bits(), limit_bits: limit.to_bits() }
    }

    fn alpha(&self) -> f32 {
        f32::from_bits(self.alpha_bits)
    }

    fn limit(&self) -> f32 {
        f32::from_bits(self.limit_bits)
    }

    fn output_facts_inner(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 2);
        ensure!(inputs[0].datum_type == f32::datum_type());
        ensure!(inputs[1].datum_type == f32::datum_type());
        ensure!(inputs[0].shape == inputs[1].shape);
        Ok(tvec!(f32::datum_type().fact(inputs[0].shape.clone())))
    }
}

impl Op for MetalClampedSwiGlu {
    fn name(&self) -> StaticName {
        "MetalClampedSwiGlu".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalClampedSwiGlu {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (gate_raw, up_raw) = args_2!(inputs);
        let gate = gate_raw
            .to_device_tensor()
            .with_context(|| format!("gate is not a Metal tensor: {gate_raw:?}"))?;
        let up = up_raw
            .to_device_tensor()
            .with_context(|| format!("up is not a Metal tensor: {up_raw:?}"))?;

        let output = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            f32::datum_type(),
            gate.shape(),
        )?;

        crate::with_metal_stream(|stream| {
            crate::kernels::moe::dispatch_clamped_swiglu_f32(
                stream,
                gate,
                up,
                self.alpha(),
                self.limit(),
                &output,
            )
        })?;

        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for MetalClampedSwiGlu {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        facts_to_device_facts(inputs, |input_facts| self.output_facts_inner(input_facts))
            .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    as_op!();
}
