use crate::kernels::Iff;
use tract_core::broadcast::multi_broadcast;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaIff;

impl Op for CudaIff {
    fn name(&self) -> StaticName {
        "CudaIff".into()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaIff {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (cond_val, then_val, else_val) = args_3!(inputs);

        let cond = cond_val.to_device_tensor()?;
        let then_t = then_val.to_device_tensor()?;
        let else_t = else_val.to_device_tensor()?;
        ensure!(cond.rank() == then_t.rank());
        ensure!(cond.rank() == else_t.rank());
        ensure!(then_t.datum_type() == else_t.datum_type());

        let out_shape = multi_broadcast(&[cond.shape(), then_t.shape(), else_t.shape()])
            .context("No broadcasting solution found")?;
        let out_dt = then_t.datum_type();
        let output =
            tract_gpu::session_handler::make_tensor_for_node(session, node_id, out_dt, &out_shape)?;

        if output.len() > 0 {
            crate::with_cuda_stream(|stream| {
                Iff.dispatch_eval(stream, cond, then_t, else_t, &output)
                    .with_context(|| "Error while dispatching eval for CudaIff")
            })?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for CudaIff {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |inputs| {
            let out_shape =
                multi_broadcast(&[&*inputs[0].shape, &*inputs[1].shape, &*inputs[2].shape])
                    .context("No broadcasting solution found")?;
            let out_dt = inputs[1].datum_type;
            Ok(tvec!(out_dt.fact(out_shape)))
        })
    }

    as_op!();
}
