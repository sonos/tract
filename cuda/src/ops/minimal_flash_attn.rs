use crate::context::CUDA_STREAM;
use crate::kernels::flash_attn::GgmlFlashAttn;
use crate::kernels::minimal_flash_attn::MinimalFlashAttn;
use derive_new::new;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Clone, Debug, new)]
pub struct CudaMinimalFlashAttention {
    scale: f32,
    is_causal: bool,
}

impl Op for CudaMinimalFlashAttention {
    fn name(&self) -> StaticName {
        "CudaMinimalFlashAttention".into()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaMinimalFlashAttention {
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
            ensure!(inputs.len() >= 4, "flash-attn expects [q, k, v, (mask)]");

            let q = inputs[0].to_device_tensor()?;
            let k = inputs[1].to_device_tensor()?;
            let v = inputs[2].to_device_tensor()?;
            let mask = if inputs.len() == 4 {
                Some(inputs[3].to_device_tensor()?)
            } else { None };

            let output = tract_gpu::session_handler::make_tensor_for_node(
                session,
                node_id,
                q.datum_type(),
                &GgmlFlashAttn.output_shape(q.shape(), k.shape(), v.shape())?,
            )?;
            MinimalFlashAttn { is_causal: self.is_causal }.dispatch_eval(stream, q, k, v, mask, self.scale, &output)?;
            Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
        })
    }
}

impl TypedOp for CudaMinimalFlashAttention {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 4);
            let dt = facts[0].datum_type;

            ensure!(facts.iter().all(|f| f.rank() == 4));
            let shape =
                MinimalFlashAttn { is_causal: self.is_causal }.output_shape(&facts[0].shape, &facts[1].shape, &facts[2].shape)?;
            let fact = dt.fact(shape);
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
