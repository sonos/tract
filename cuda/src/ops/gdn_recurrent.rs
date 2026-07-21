use crate::kernels::gdn_recurrent::CudaGdnRecurrent;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaGatedDeltaNetRecurrent;

impl Op for CudaGatedDeltaNetRecurrent {
    fn name(&self) -> StaticName {
        "CudaGatedDeltaNetRecurrent".into()
    }
    op_as_typed_op!();
}

impl EvalOp for CudaGatedDeltaNetRecurrent {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        _session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 6, "GDN expects q, k, v, log_decay, beta, state");
        crate::with_cuda_stream(|stream| {
            let (output, final_state) = CudaGdnRecurrent.eval(
                stream,
                inputs[0].to_device_tensor()?,
                inputs[1].to_device_tensor()?,
                inputs[2].to_device_tensor()?,
                inputs[3].to_device_tensor()?,
                inputs[4].to_device_tensor()?,
                inputs[5].to_device_tensor()?,
            )?;
            Ok(tvec![output.into_tensor().into_tvalue(), final_state.into_tensor().into_tvalue()])
        })
    }
}

impl TypedOp for CudaGatedDeltaNetRecurrent {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 6);
            ensure!(facts[0].datum_type == DatumType::F16);
            ensure!(facts[1].datum_type == DatumType::F16);
            ensure!(facts[2].datum_type == DatumType::F16);
            ensure!(facts[3].datum_type == DatumType::F32);
            ensure!(facts[4].datum_type == DatumType::F16);
            ensure!(facts[5].datum_type == DatumType::F32);
            ensure!(facts[0].shape == facts[1].shape && facts[0].shape == facts[2].shape);
            Ok(tvec![facts[0].without_value(), facts[5].without_value()])
        })
        .with_context(|| format!("invalid facts for {}", self.name()))
    }
    as_op!();
}
