use crate::context::CUDA_STREAM;
use crate::kernels::conv::{ConvKernel, Generic};
use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_gpu::tensor::DeviceTensorExt;

pub fn cuda_conv(model: &TypedModel, node: &TypedNode, op: &Conv) -> TractResult<Option<CudaConv>> {
    let facts = model.node_input_facts(node.id)?;
    if facts.iter().all(|f| f.datum_type.is::<f32>()) && facts[1].rank() <= 6 {
        Ok(Some(CudaConv { op: op.clone(), kernel: Box::new(Generic) }))
    } else {
        Ok(None)
    }
}

#[derive(Debug, Clone)]
pub struct CudaConv {
    op: Conv,
    kernel: Box<dyn ConvKernel>,
}

impl Op for CudaConv {
    fn name(&self) -> StaticName {
        "CudaConv".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut info = self.op.info()?;
        info.push(format!("kernel: {}", self.kernel.name()));
        Ok(info)
    }

    op_as_typed_op!();
}

impl EvalOp for CudaConv {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let inputs =
            inputs.iter().map(|it| it.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let output_shape = self.op.pool_spec.output_shape(inputs[0].shape())?;
        let output = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            inputs[0].datum_type(),
            &output_shape.shape,
        )?;

        CUDA_STREAM.with(|stream| {
            self.kernel.dispatch(&self.op, stream, inputs[0], inputs[1], inputs[2], &output)
        })?;
        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for CudaConv {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| self.op.output_facts(facts))
            .with_context(|| {
                format!("Error while computing facts for Conv/{:?}", self.kernel.name())
            })
    }
}
