use crate::kernels::causal_conv1d_update::CudaCausalConv1dUpdate;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaCausalConv1dUpdateOp;

impl Op for CudaCausalConv1dUpdateOp {
    fn name(&self) -> StaticName {
        "CudaCausalConv1dUpdate".into()
    }
    op_as_typed_op!();
}

impl EvalOp for CudaCausalConv1dUpdateOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        _session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (input, weight, state) = args_3!(inputs);
        crate::with_cuda_stream(|stream| {
            let (output, final_state) = CudaCausalConv1dUpdate.eval(
                stream,
                input.to_device_tensor()?,
                weight.to_device_tensor()?,
                state.to_device_tensor()?,
            )?;
            Ok(tvec![output.into_tensor().into_tvalue(), final_state.into_tensor().into_tvalue()])
        })
    }
}

impl TypedOp for CudaCausalConv1dUpdateOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 3);
            ensure!(facts.iter().all(|f| f.datum_type == DatumType::F16));
            Ok(tvec![facts[0].without_value(), facts[2].without_value()])
        })
        .with_context(|| format!("invalid facts for {}", self.name()))
    }
    as_op!();
}
