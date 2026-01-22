use crate::context::CUDA_STREAM;
use crate::kernels::conv::{ConvGeneric, ConvKernel, ConvKernelScratch};
use crate::kernels::conv_cudnn::ConvCudnn;
use crate::ops::{CudaAxisOp, CudaBinOp};
use tract_core::internal::*;
use tract_core::ops::OpStateFreeze;
use tract_core::ops::cnn::Conv;
use tract_gpu::tensor::DeviceTensorExt;

pub fn wire_cuda_conv(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    inputs: &[OutletId],
    op: &Conv,
) -> TractResult<TVec<OutletId>> {
    let facts = source.node_input_facts(node.id)?;
    let data_shape = op.pool_spec.data_format.shape(&facts[0].shape)?;
    let hw_rank = data_shape.hw_rank();
    if facts.iter().all(|f| f.datum_type.is::<f32>())
        && hw_rank <= 6
        && op
            .pool_spec
            .computed_padding(data_shape.hw_dims())
            .iter()
            .all(|paddings| paddings.pad_before == paddings.pad_after)
    {
        let prefix = &node.name;
        let bias = &facts[2];
        let need_bias = !(bias.konst.is_some() && bias.konst.as_ref().unwrap().is_all_zero()?);
        let conv_name = format!("{prefix}.conv");
        let mut conv_wire = target.wire_node(
            if need_bias { &conv_name } else { &node.name },
            CudaConv { op: op.clone(), kernel: Box::new(ConvCudnn) },
            &inputs[0..2],
        )?[0];
        if need_bias {
            let mut needed_shape = tvec![1.to_dim(); node.outputs[0].fact.rank()];
            needed_shape[data_shape.c_axis()] = op.pool_spec.output_channels.to_dim();
            let reshaped = target.wire_node(
                format!("{prefix}.bias_reshaped"),
                CudaAxisOp(AxisOp::Reshape(0, bias.shape.to_tvec(), needed_shape)),
                &[inputs[2]],
            )?[0];
            conv_wire = target.wire_node(
                prefix,
                CudaBinOp(crate::kernels::BinOps::Add),
                &[conv_wire, reshaped],
            )?[0];
        }
        Ok(tvec!(conv_wire))
    } else {
        target.wire_node(
            &node.name,
            CudaConv { op: op.clone(), kernel: Box::new(ConvGeneric) },
            inputs,
        )
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
        false
    }

    fn state(&self, _session: &TurnState, node_id: usize) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(CudaConvState(node_id, None))))
    }
}

impl TypedOp for CudaConv {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let zero = facts[0].datum_type.scalar_fact();
            let mut facts: TVec<&TypedFact> = facts.into();
            if facts.len() == 2 {
                facts.push(&zero);
            }
            self.op.output_facts(&facts)
        })
        .with_context(|| format!("Error while computing facts for Conv/{:?}", self.kernel.name()))
    }
}

#[derive(Debug)]
struct CudaConvState(usize, Option<Box<dyn ConvKernelScratch>>);

impl Clone for CudaConvState {
    fn clone(&self) -> Self {
        CudaConvState(self.0, None)
    }
}

impl OpState for CudaConvState {
    fn eval(
        &mut self,
        session: &mut TurnState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op: &CudaConv = op.downcast_ref().context("Wrong op")?;
        let inputs =
            inputs.iter().map(|it| it.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let output_shape = op.op.pool_spec.output_shape(inputs[0].shape())?;
        let output = tract_gpu::session_handler::make_tensor_for_node(
            session,
            self.0,
            inputs[0].datum_type(),
            &output_shape.shape,
        )?;

        if self.1.is_none() {
            self.1 = Some(op.kernel.state());
        }

        if output.len() > 0 {
            CUDA_STREAM.with(|stream| {
                op.kernel.dispatch(
                    &mut **self.1.as_mut().unwrap(),
                    self.0,
                    &op.op,
                    stream,
                    inputs[0],
                    inputs[1],
                    inputs.get(2).cloned(),
                    &output,
                )
            })?;
        }
        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

#[derive(Debug, Clone)]
struct FrozenCudaConvState(usize);

impl OpStateFreeze for CudaConvState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenCudaConvState(self.0))
    }
}

impl FrozenOpState for FrozenCudaConvState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(CudaConvState(self.0, None))
    }
}
